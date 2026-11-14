from __future__ import annotations

import json
import os
import re

from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)


PROJECT_SLUG = "evidenceflow"
EXCHANGE_TIMEOUT_SECONDS = 8
MAX_RESPONSE_BYTES = 64 * 1024

CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{32,128}$")
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


def _exchange_endpoint(base_url: str) -> str:
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

    endpoint_path = f"{path}/v1/demo/session/exchange"

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            endpoint_path,
            "",
            "",
        )
    )


def exchange_handoff_code(
    code: str,
    *,
    project: str = PROJECT_SLUG,
    gateway_url: str | None = None,
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

    endpoint = _exchange_endpoint(base_url)

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

    return {
        "access_token": token,
        "expires_in": expires_in,
        "target_project": target_project,
        "provider": str(payload.get("provider") or ""),
        "model": str(payload.get("model") or ""),
    }
