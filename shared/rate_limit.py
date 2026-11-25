"""Owner-scoped Upstash quotas for EvidenceFlow upload and chat actions."""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any


class RateLimitBackendUnavailable(RuntimeError):
    """A production shared quota service is missing or unavailable."""


class RateLimitExceeded(RuntimeError):
    def __init__(self, action: str, retry_after: int) -> None:
        super().__init__(action)
        self.action = action
        self.retry_after = max(1, int(retry_after))


class _LocalWindow:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.events: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = self.clock()
        cutoff = now - max(1, window)
        with self.lock:
            bucket = self.events[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= max(1, limit):
                return False, max(1, int(bucket[0] + window - now + 0.999))
            bucket.append(now)
            if len(self.events) > 20_000:
                stale = [name for name, values in self.events.items() if not values or values[-1] <= cutoff]
                for name in stale[:5_000]:
                    self.events.pop(name, None)
            return True, 0


_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])
if count == 1 or ttl < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
  ttl = tonumber(ARGV[1])
end
return {count, ttl}
"""


class OwnerRateLimiter:
    def __init__(self) -> None:
        self._local = _LocalWindow()
        self._client: Any | None = None
        self._lock = threading.Lock()

    def _is_production(self) -> bool:
        return (
            os.getenv("APP_ENV", "").strip().lower() in {"prod", "production"}
            or os.getenv("RENDER", "").strip().lower() in {"1", "true"}
            or bool(os.getenv("K_SERVICE"))
        )

    def _client_for_production(self):
        url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
        if not url or not token:
            raise RateLimitBackendUnavailable("Shared rate-limit storage is not configured.")
        with self._lock:
            if self._client is None:
                try:
                    from upstash_redis import Redis
                    self._client = Redis(
                        url=url,
                        token=token,
                        rest_retries=0,
                        allow_telemetry=False,
                    )
                except Exception as exc:
                    raise RateLimitBackendUnavailable from exc
        return self._client

    def check(
        self,
        owner_scope: str,
        action: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        scope = str(owner_scope or "").strip().lower()
        if len(scope) != 64 or any(ch not in "0123456789abcdef" for ch in scope):
            raise ValueError("A verified owner scope is required for rate limiting.")
        action_key = "".join(ch for ch in str(action).lower() if ch.isalnum() or ch in "_-")[:32]
        if not action_key:
            raise ValueError("A valid rate-limit action is required.")
        scope_hash = hashlib.sha256(scope.encode("utf-8")).hexdigest()
        key = f"evidenceflow:ratelimit:v1:{action_key}:{scope_hash}"
        safe_limit = max(1, int(limit))
        safe_window = max(1, int(window_seconds))
        if not self._is_production():
            return self._local.check(key, safe_limit, safe_window)
        try:
            result = self._client_for_production().eval(
                _SCRIPT,
                keys=[key],
                args=[safe_window],
            )
            count, ttl = int(result[0]), int(result[1] or safe_window)
            return count <= safe_limit, max(1, ttl)
        except RateLimitBackendUnavailable:
            raise
        except Exception as exc:
            raise RateLimitBackendUnavailable from exc

    def enforce(
        self,
        owner_scope: str,
        action: str,
        *,
        limit: int,
        window_seconds: int,
    ) -> None:
        allowed, retry_after = self.check(
            owner_scope,
            action,
            limit=limit,
            window_seconds=window_seconds,
        )
        if not allowed:
            raise RateLimitExceeded(action, retry_after)


owner_rate_limiter = OwnerRateLimiter()
