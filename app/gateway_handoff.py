from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)


PROJECT_SLUG = "evidenceflow"
EXCHANGE_TIMEOUT_SECONDS = 8
WORKSPACE_IDENTITY_HEADERS = (
    "X-Studio-Workspace-Client-Id",
    "X-Studio-Workspace-Timestamp",
    "X-Studio-Workspace-Signature",
)
WORKSPACE_CLIENT_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
WORKSPACE_TIMESTAMP_PATTERN = re.compile(r"^[0-9]{10}$")
WORKSPACE_SIGNATURE_PATTERN = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)
MAX_RESPONSE_BYTES = 64 * 1024

CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
OWNER_SCOPE_PATTERN = re.compile(r"^[a-f0-9]{64}$")
PROJECT_PATTERN = re.compile(r"^[a-z0-9-]{1,80}$")


class GatewayHandoffError(RuntimeError):
    """Safe-to-display error for a portfolio session exchange."""


class _NoRedirectHandler(HTTPRedirectHandler):
    """Do not forward a one-time code through an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _configured_gateway_url() -> str:
    value = os.environ.get(
        "PORTFOLIO_SESSION_GATEWAY_URL", ""
    ).strip()

    if value:
        return value

    # Support local development using the repository's .env file.
    try:
        from dotenv import dotenv_values

        env_path = Path(__file__).resolve().parent.parent / ".env"
        values = dotenv_values(env_path)

        return str(
            values.get("PORTFOLIO_SESSION_GATEWAY_URL") or ""
        ).strip()
    except Exception:
        return ""


def _validate_gateway_destination(base_url: str) -> None:
    """Reject remote cleartext and private-network destinations before sending a code."""
    try:
        parsed = urlsplit(base_url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise GatewayHandoffError(
            "The portfolio session gateway URL is invalid."
        ) from exc

    if not hostname:
        raise GatewayHandoffError(
            "The portfolio session gateway URL is invalid."
        )

    normalized_host = hostname.rstrip(".").lower()
    is_production = (
        os.environ.get("APP_ENV", "").strip().lower()
        in {"prod", "production"}
        or os.environ.get("RENDER", "").strip().lower() in {"1", "true"}
        or bool(os.environ.get("K_SERVICE"))
    )
    local_hosts = {
        "localhost",
        "localhost.localdomain",
        "127.0.0.1",
        "::1",
        "host.docker.internal",
    }
    is_local_destination = (
        normalized_host in local_hosts
        or normalized_host.endswith(".localhost")
        or normalized_host.endswith(".local")
    )

    if is_local_destination and not is_production:
        if parsed.scheme not in {"http", "https"}:
            raise GatewayHandoffError(
                "The local portfolio session gateway URL is invalid."
            )
        return

    allowed_hosts = {
        item.strip().rstrip(".").lower()
        for item in os.environ.get(
            "PORTFOLIO_SESSION_GATEWAY_ALLOWED_HOSTS",
            "portfolio-llm-gateway.onrender.com",
        ).split(",")
        if item.strip()
    }

    if parsed.scheme != "https":
        raise GatewayHandoffError(
            "The portfolio session gateway must use HTTPS."
        )

    if normalized_host not in allowed_hosts:
        raise GatewayHandoffError(
            "The portfolio session gateway host is not approved."
        )

    if port not in {None, 443}:
        raise GatewayHandoffError(
            "The portfolio session gateway must use the standard HTTPS port."
        )

    try:
        try:
            address = ipaddress.ip_address(normalized_host)
            resolved_addresses = [address]
        except ValueError:
            records = socket.getaddrinfo(
                normalized_host,
                port or 443,
                type=socket.SOCK_STREAM,
            )
            resolved_addresses = [
                ipaddress.ip_address(record[4][0].split("%", 1)[0])
                for record in records
            ]
    except (OSError, ValueError):
        raise GatewayHandoffError(
            "The portfolio session gateway host could not be safely resolved."
        ) from None

    if not resolved_addresses or any(not address.is_global for address in resolved_addresses):
        raise GatewayHandoffError(
            "The portfolio session gateway must resolve only to public addresses."
        )


def _session_endpoint(base_url: str, action: str) -> str:
    if action not in {"exchange", "revoke"}:
        raise GatewayHandoffError("The portfolio session gateway action is invalid.")
    try:
        parsed = urlsplit(base_url.strip())
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise GatewayHandoffError(
            "The portfolio session gateway URL is invalid."
        ) from exc

    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GatewayHandoffError(
            "The portfolio session gateway URL is invalid."
        )

    path = parsed.path.rstrip("/")
    # Accept either a gateway root or an OpenAI-compatible /v1 base URL.
    if path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    endpoint_path = f"{path}/v1/demo/session/{action}"
    return urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def _exchange_endpoint(base_url: str) -> str:
    return _session_endpoint(base_url, "exchange")


def _validated_workspace_identity_headers(
    identity_headers: Mapping[str, str] | None,
) -> dict[str, str]:
    if identity_headers is None:
        return {}

    # Header names are intentionally allowlisted; do not forward caller-
    # supplied headers or arbitrary identity metadata to the Gateway.
    allowed_by_lower = {header.lower(): header for header in WORKSPACE_IDENTITY_HEADERS}
    supplied = {
        allowed_by_lower[name.lower()]: str(value or "").strip()
        for name, value in identity_headers.items()
        if name.lower() in allowed_by_lower
    }
    if not supplied:
        return {}
    if len(supplied) != len(WORKSPACE_IDENTITY_HEADERS) or any(
        not supplied.get(name) for name in WORKSPACE_IDENTITY_HEADERS
    ):
        raise GatewayHandoffError("The workspace identity assertion is incomplete.")

    client_id = supplied[WORKSPACE_IDENTITY_HEADERS[0]]
    timestamp = supplied[WORKSPACE_IDENTITY_HEADERS[1]]
    signature = supplied[WORKSPACE_IDENTITY_HEADERS[2]]
    if (
        not WORKSPACE_CLIENT_ID_PATTERN.fullmatch(client_id)
        or not WORKSPACE_TIMESTAMP_PATTERN.fullmatch(timestamp)
        or int(timestamp) <= 0
        or not WORKSPACE_SIGNATURE_PATTERN.fullmatch(signature)
    ):
        raise GatewayHandoffError("The workspace identity assertion is invalid.")

    return {
        WORKSPACE_IDENTITY_HEADERS[0]: client_id.lower(),
        WORKSPACE_IDENTITY_HEADERS[1]: timestamp,
        WORKSPACE_IDENTITY_HEADERS[2]: signature.lower(),
    }


def revoke_gateway_session(
    access_token: str,
    *,
    gateway_url: str | None = None,
) -> bool:
    """Revoke the delegated EvidenceFlow child session at the Gateway.

    A 401 means the bearer is already inactive (expired or previously revoked),
    so it is safe to complete local cleanup. Other non-confirming responses
    remain retryable and must not cause the local cookie/session to be deleted.
    """
    token = str(access_token or "").strip()
    if (
        not token
        or len(token) > 8192
        or "\n" in token
        or "\r" in token
        or not re.fullmatch(r"[A-Za-z0-9._~-]{16,8192}", token)
    ):
        raise GatewayHandoffError("The stored Gateway session token is invalid.")

    base_url = (
        gateway_url.strip()
        if gateway_url
        else _configured_gateway_url()
    )
    if not base_url:
        raise GatewayHandoffError(
            "Portfolio session revocation is not configured on this app."
        )

    _validate_gateway_destination(base_url)
    endpoint = _session_endpoint(base_url, "revoke")
    request = Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
        },
        method="POST",
    )
    opener = build_opener(_NoRedirectHandler())

    try:
        with opener.open(request, timeout=EXCHANGE_TIMEOUT_SECONDS) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code == 401:
            return True
        raise GatewayHandoffError(
            "The Gateway could not confirm the EvidenceFlow session revocation."
        ) from None
    except (URLError, TimeoutError, OSError):
        raise GatewayHandoffError(
            "The Gateway could not be reached to revoke the EvidenceFlow session."
        ) from None

    if len(raw) > MAX_RESPONSE_BYTES:
        raise GatewayHandoffError(
            "The Gateway returned an invalid revocation response."
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GatewayHandoffError(
            "The Gateway returned an invalid revocation response."
        ) from None
    if not isinstance(payload, dict) or payload.get("revoked") is not True:
        raise GatewayHandoffError(
            "The Gateway could not confirm the EvidenceFlow session revocation."
        )
    return True


def exchange_handoff_code(
    code: str,
    *,
    project: str = PROJECT_SLUG,
    gateway_url: str | None = None,
    identity_headers: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """
    Exchange a one-time portfolio handoff code from the Streamlit server.

    The returned bearer token remains in Streamlit server-side session
    state. It is never appended to a URL or returned to browser JavaScript.
    """
    code = str(code or "").strip()
    project = str(project or "").strip().lower()

    if not CODE_PATTERN.fullmatch(code):
        raise GatewayHandoffError(
            "The demo launch code is invalid or incomplete. "
            "Return to the portfolio and launch EvidenceFlow again."
        )

    if not PROJECT_PATTERN.fullmatch(project):
        raise GatewayHandoffError(
            "The target project identifier is invalid."
        )

    base_url = (
        gateway_url.strip()
        if gateway_url
        else _configured_gateway_url()
    )

    if not base_url:
        raise GatewayHandoffError(
            "Portfolio session exchange is not configured on this app."
        )

    _validate_gateway_destination(base_url)
    endpoint = _exchange_endpoint(base_url)
    trusted_identity_headers = _validated_workspace_identity_headers(identity_headers)

    body = json.dumps(
        {
            "code": code,
            "project": project,
        }
    ).encode("utf-8")

    request = Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            **trusted_identity_headers,
        },
        method="POST",
    )

    opener = build_opener(_NoRedirectHandler())

    try:
        with opener.open(
            request,
            timeout=EXCHANGE_TIMEOUT_SECONDS,
        ) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)

    except HTTPError as exc:
        if exc.code == 401:
            raise GatewayHandoffError(
                "The demo launch code has expired, was already used, "
                "or is not valid for EvidenceFlow. Launch it again "
                "from the portfolio."
            ) from None

        raise GatewayHandoffError(
            "The gateway could not complete the EvidenceFlow session exchange."
        ) from None

    except (URLError, TimeoutError, OSError):
        raise GatewayHandoffError(
            "The AI gateway could not be reached. Please retry "
            "the launch from the portfolio."
        ) from None

    if len(raw) > MAX_RESPONSE_BYTES:
        raise GatewayHandoffError(
            "The gateway returned an invalid session response."
        )

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GatewayHandoffError(
            "The gateway returned an invalid session response."
        ) from None

    if not isinstance(payload, dict):
        raise GatewayHandoffError(
            "The gateway returned an invalid session response."
        )

    token = payload.get("access_token")
    target_project = payload.get("target_project")
    expires_in = payload.get("expires_in")

    if (
        not isinstance(token, str)
        or not token.strip()
        or len(token) > 8192
        or "\n" in token
        or "\r" in token
    ):
        raise GatewayHandoffError(
            "The gateway did not return a valid session."
        )

    if target_project != project:
        raise GatewayHandoffError(
            "The gateway returned a session for an unexpected project."
        )

    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, int)
        or not 1 <= expires_in <= 3600
    ):
        raise GatewayHandoffError(
            "The gateway returned an invalid session expiry."
        )

    owner_scope = payload.get("owner_scope")
    if not isinstance(owner_scope, str) or not OWNER_SCOPE_PATTERN.fullmatch(owner_scope):
        raise GatewayHandoffError(
            "The gateway did not return a valid document access scope."
        )

    return {
        "access_token": token,
        "expires_in": expires_in,
        "target_project": target_project,
        "owner_scope": owner_scope,
        "provider": str(payload.get("provider") or ""),
        "model": str(payload.get("model") or ""),
    }
