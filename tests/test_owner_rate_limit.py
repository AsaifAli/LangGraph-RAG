import pytest

from shared.rate_limit import (
    OwnerRateLimiter,
    RateLimitBackendUnavailable,
    RateLimitExceeded,
)


class FakeRedis:
    def __init__(self):
        self.counts = {}
        self.calls = []

    def eval(self, script, *, keys, args):
        self.calls.append((script, keys, args))
        key = keys[0]
        self.counts[key] = self.counts.get(key, 0) + 1
        return [self.counts[key], args[0]]


def test_production_quotas_use_shared_owner_hashed_keys(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://redis.example")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "test-token")
    limiter = OwnerRateLimiter()
    redis = FakeRedis()
    monkeypatch.setattr(limiter, "_client_for_production", lambda: redis)

    owner = "a" * 64
    assert limiter.check(owner, "upload", limit=1, window_seconds=3600) == (True, 3600)
    assert limiter.check(owner, "upload", limit=1, window_seconds=3600) == (False, 3600)
    script, keys, args = redis.calls[0]
    assert "INCR" in script and "EXPIRE" in script
    assert keys[0].startswith("evidenceflow:ratelimit:v1:upload:")
    assert owner not in keys[0]
    assert args == [3600]


def test_production_quota_fails_closed_if_shared_store_is_missing(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    with pytest.raises(RateLimitBackendUnavailable):
        OwnerRateLimiter().check("a" * 64, "chat", limit=30, window_seconds=60)


def test_excess_owner_requests_raise_retry_after():
    limiter = OwnerRateLimiter()
    owner = "b" * 64
    limiter.enforce(owner, "chat", limit=1, window_seconds=60)
    with pytest.raises(RateLimitExceeded) as error:
        limiter.enforce(owner, "chat", limit=1, window_seconds=60)
    assert error.value.action == "chat"
    assert error.value.retry_after >= 1



def test_local_development_ingestion_scope_remains_usable_outside_production(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    limiter = OwnerRateLimiter()
    assert limiter.check(
        "local-development",
        "upload",
        limit=2,
        window_seconds=60,
    ) == (True, 0)


def test_local_development_scope_is_rejected_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    limiter = OwnerRateLimiter()
    with pytest.raises(ValueError, match="verified owner scope"):
        limiter.check(
            "local-development",
            "upload",
            limit=2,
            window_seconds=60,
        )
