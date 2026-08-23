"""Minimal EvidenceFlow runtime trust guardrails.

These controls are intentionally narrow: protect the user from untrusted
retrieved instructions, obvious secret leakage, oversized inputs, and
unsupported answers without turning EvidenceFlow into a generic LLM-eval
platform.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_QUERY_CHARS = 12_000
MAX_TOOL_QUERY_CHARS = 4_000

# Common credential/token shapes. Deliberately conservative to avoid treating
# ordinary emails, document IDs, or prose as secrets.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._-]{24,}", re.IGNORECASE),
)

_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions", re.I),
    re.compile(r"system\s+prompt\s*[:=]", re.I),
    re.compile(r"reveal\s+(?:your|the)\s+(?:system\s+prompt|hidden\s+instructions|secrets)", re.I),
    re.compile(r"send\s+(?:the|all|this)\s+(?:documents?|data|content)\s+to\s+https?://", re.I),
    re.compile(r"execute\s+(?:this|the following)\s+(?:command|instruction)", re.I),
)


@dataclass(frozen=True)
class GuardrailFinding:
    kind: str
    message: str
    source: str = "system"
    blocked: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "message": self.message,
            "source": self.source,
            "blocked": self.blocked,
        }


def validate_query(query: str) -> GuardrailFinding | None:
    if len(query or "") > MAX_QUERY_CHARS:
        return GuardrailFinding(
            "input_limit",
            f"The request is longer than the {MAX_QUERY_CHARS:,}-character safety limit.",
            blocked=True,
        )
    return None


def validate_tool_query(query: str, *, source: str) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    if len(query or "") > MAX_TOOL_QUERY_CHARS:
        findings.append(
            GuardrailFinding(
                "tool_query_limit",
                f"The {source} search query exceeded the {MAX_TOOL_QUERY_CHARS:,}-character limit and was truncated.",
                source=source,
            )
        )
    return findings


def scan_untrusted_content(text: str, *, source: str) -> list[GuardrailFinding]:
    """Detect instruction-like content in retrieved data.

    Retrieved text is evidence only. Findings are surfaced to the synthesizer
    as warnings; the text is never treated as executable instructions.
    """
    if not text:
        return []
    findings: list[GuardrailFinding] = []
    if any(p.search(text) for p in _INJECTION_PATTERNS):
        findings.append(
            GuardrailFinding(
                "prompt_injection",
                "Retrieved content contained instruction-like text; it was treated as untrusted evidence and not as agent instructions.",
                source=source,
            )
        )
    if any(p.search(text) for p in _SECRET_PATTERNS):
        findings.append(
            GuardrailFinding(
                "secret_in_source",
                "Retrieved content appears to contain a credential or secret pattern; the content remains evidence only and is not executed.",
                source=source,
            )
        )
    return findings


def detect_secret_leakage(text: str) -> list[GuardrailFinding]:
    if not text:
        return []
    if any(p.search(text) for p in _SECRET_PATTERNS):
        return [
            GuardrailFinding(
                "secret_leakage",
                "The generated answer appears to contain credential-like content and was blocked.",
                blocked=True,
            )
        ]
    return []


def build_trust_status(*, grounding_status: str | None, numeric_supported: bool | None,
                       conflicts: list[dict], findings: list[dict], proposed_count: int,
                       verified_count: int) -> dict:
    security_blocked = any(bool(f.get("blocked")) for f in findings)
    if security_blocked:
        label = "BLOCKED"
    elif grounding_status == "VERIFIED" and not conflicts and numeric_supported is not False:
        label = "VERIFIED"
    elif grounding_status == "PARTIAL" or conflicts or numeric_supported is False:
        label = "REVIEW"
    elif grounding_status in {"UNVERIFIED", "NOT_APPLICABLE"} and proposed_count:
        label = "UNVERIFIED"
    else:
        label = "NO_EVIDENCE"
    return {
        "label": label,
        "security_blocked": security_blocked,
        "proposed_count": proposed_count,
        "verified_count": verified_count,
        "finding_count": len(findings),
    }
