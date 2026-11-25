# Security Review — EvidenceFlow (LangGraph-RAG)

This is a code/configuration review, not a penetration test or a declaration of production certification. EvidenceFlow is a Streamlit/LangGraph research app with OpenSearch, optional web research, and a server-side Gateway session.

## Existing safeguards observed
- `.env` is excluded from the repository/container context; secrets are read from server-side environment variables.
- Retrieval requires tenant/document scope filters; an empty document scope does not become an unrestricted query.
- The repository correctly documents that a tenant or document ID is not an authorization credential. A trusted identity/ACL service must approve document IDs before retrieval.
- Citation verification is fail-closed in verified mode.
- Gateway exchange uses a short-lived one-time code, bounded response size, an 8-second timeout, and a no-redirect handler; the exchanged bearer is retained in Streamlit server-side session state rather than the URL or browser JavaScript.
- Tavily and Katzilla keys are loaded server-side; the MCP adapter does not intentionally put the key into model context.
- OpenSearch, model and provider credentials are configured as deployment secrets rather than client-prefixed environment variables.

## Findings requiring attention
1. **User query logging (Medium): PATCHED in this branch.** `tools/web_search_tool.py` no longer prints user queries, provider payloads, response content or source snippets. `tests/test_web_search_logging.py` guards against regression. Other logging/traceback paths still require a broader review.
2. **MCP redirect policy (High):** `tools/primary_source_mcp.py` constructs an authenticated HTTP client with `follow_redirects=True`. The MCP URL is environment-configured rather than user-controlled, but credentials should not be sent through redirects. Disable redirects and test that a redirect fails closed.
3. **Gateway URL trust (High):** `app/gateway_handoff.py` accepts both HTTP and HTTPS for the configured gateway URL and does not pin an expected host or reject loopback/private destinations. Require HTTPS outside local development and restrict the production host to the configured Gateway origin. Do not accept this URL from a chat/user payload.
4. **Container privilege (Medium):** the checked-in Dockerfiles do not declare a non-root runtime user. Use an unprivileged UID and ensure only runtime-writable paths (such as chat history/cache) are owned by it.
5. **Diagnostic disclosure (Medium):** the Streamlit turn failure path displays exception details/tracebacks in an expandable UI block. Keep detailed diagnostics in access-controlled logs with redaction; show a generic correlation identifier to the user.
6. **Upload surface:** Streamlit accepts user files. Enforce size limits before reading all bytes, validate actual file type/signature and archive contents server-side, and reject malformed or decompression-bomb-like inputs.

## 20-control assessment
| # | Control | Result |
|---|---|---|
|01|SQL injection|PARTIAL — OpenSearch queries are constructed through client APIs; review every query/filter builder for raw DSL interpolation|
|02|XSS|PARTIAL — trusted UI templates use escaped HTML for copy actions; review every `unsafe_allow_html` and dynamic Markdown rendering|
|03|CSRF|PARTIAL — Streamlit session model, no app-owned cookie-auth API found; validate Streamlit/proxy origin and cookie settings|
|04|File upload|PARTIAL — Streamlit accepts multiple document/image types; validate actual byte signatures, size and archive expansion limits|
|05|SSRF|PARTIAL — MCP/gateway endpoints are configured server-side; redirect and private-network protections need tightening|
|06|BOLA/IDOR|PARTIAL — scope filters exist, but full user/tenant identity and ACL authorization are intentionally an upstream boundary|
|07|Rate limiting|PARTIAL — bounded agent loops/timeouts exist; add edge and per-session limits for expensive model/search calls|
|08|Password hashing|N/A — no local password authentication identified|
|09|MFA|EXTERNAL — enable for GitHub, hosting, OpenSearch and provider administrator accounts|
|10|Server-side permissions|PARTIAL — document ACL must be enforced before agent execution; client IDs are not authorization|
|11|RLS|N/A/PARTIAL — OpenSearch tenant filters are app-level, not database RLS; isolate tenant credentials/index permissions|
|12|JWT secret|GATEWAY-OWNED — Gateway signs the short-lived session; verify production secret strength/rotation there|
|13|Server-side API keys|PASS by design — provider/MCP keys are server-side environment settings; validate hosting config|
|14|No localStorage tokens|PARTIAL — exchanged Gateway bearer is stored in server-side Streamlit session state; validate browser storage in deployed app|
|15|No default credentials|PARTIAL — no local admin login found; verify all deployment secrets are unique and non-placeholder|
|16|Strict CORS|PARTIAL — Streamlit/proxy origin policy is deployment-owned; configure exact HTTPS hosts|
|17|Webhook signatures|N/A — no webhook receiver identified|
|18|Source maps|N/A — no client bundler/source-map build pipeline identified|
|19|Clean logs|FAIL/PARTIAL — raw query debug prints and expanded traceback display are present; redact/remove before production|
|20|Dependency patching|UNVERIFIED — scan resolved dependencies and container OS packages; pinned versions alone do not prove safety|

## External deployment actions
- Set the exact production HTTPS Gateway URL; do not use localhost, IP literals, or arbitrary internal hosts.
- Set OpenSearch with TLS certificate verification enabled and least-privilege credentials/index permissions.
- Configure unique server-side provider, Tavily, Katzilla and evaluation-ingest secrets; rotate any previously exposed key.
- Enable MFA for GitHub, Render, OpenSearch, and provider accounts.
- Put the app behind HTTPS and verify reverse-proxy headers, upload limits, session cookie policy and access logs.
- Run a dependency and container image vulnerability scan against the exact resolved deployment image.

## AI audit prompt
Act as a Principal Application Security Engineer. Audit every one of the 20 controls against the actual source and deployment configuration. Mark PASS, PARTIAL, FAIL or N/A; cite exact file and line numbers; never infer a pass from absence of a match. For every finding, state exploit preconditions and impact, then provide the smallest safe patch and a regression test. Separate repository controls from hosting/provider settings. Never print secrets, credentials, prompts or raw customer content.