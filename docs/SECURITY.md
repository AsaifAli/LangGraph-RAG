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
1. **User query logging (Medium): PATCHED in main.** `tools/web_search_tool.py` no longer prints user queries, provider payloads, response content or source snippets. `tests/test_web_search_logging.py` guards against regression. Other logging/traceback paths still require a broader review.
2. **MCP redirect policy (High): PATCHED in main.** `tools/primary_source_mcp.py` now sets `follow_redirects=False` on the authenticated client. `tests/test_primary_source_mcp.py` verifies that policy.
3. **Gateway URL trust (High): PATCHED in main.** `app/gateway_handoff.py` requires HTTPS and public DNS destinations for remote gateways, rejects nonstandard external ports, and permits loopback/host.docker.internal only in non-production local development. Gateway URL remains server-side configuration, never a chat payload.
4. **Container privilege (Medium): PATCHED in main.** Both `Dockerfile` and `Dockerfile.render` create an unprivileged `app` account and give it write access only to the chat-history, model-cache and home paths. The Render image was redeployed from the merged commit; keep checking the live deployment after future Dockerfile changes.
5. **Diagnostic disclosure (Medium): PATCHED in main.** Upload and agent-turn failures now emit only a generated incident reference and exception class to application logs. The UI displays a generic retry message with that reference; raw exception messages and tracebacks are no longer rendered or logged.
6. **Upload surface: PATCHED in main.** The central `retrieval/rag_pipeline.py::ingest_document` path calls `shared/upload_security.py` before parsers, image processing, indexing or model calls. It enforces a configurable 200 MB default cap (matching Streamlit's existing cap), validates PDF/DOC/DOCX/text/image signatures, and bounds Office archive entries and expanded size. The Streamlit server upload cap is explicitly set to 200 MB.

## 20-control assessment
| # | Control | Result |
|---|---|---|
|01|SQL injection|PARTIAL — OpenSearch queries are constructed through client APIs; review every query/filter builder for raw DSL interpolation|
|02|XSS|PARTIAL — trusted UI templates use escaped HTML for copy actions; review every `unsafe_allow_html` and dynamic Markdown rendering|
|03|CSRF|PARTIAL — Streamlit session model, no app-owned cookie-auth API found; validate Streamlit/proxy origin and cookie settings|
|04|File upload|PARTIAL — server-side size/signature and Office archive bounds are implemented with compatibility tests; malware scanning and deeper parser sandboxing remain external hardening|
|05|SSRF|PARTIAL — authenticated MCP redirects are disabled and Gateway exchange rejects remote HTTP/private DNS; Katzilla MCP URL remains a trusted operator setting and should be constrained to the approved provider host|
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
|19|Clean logs|PARTIAL — raw search debug prints and user-visible raw exception messages/tracebacks were removed; broader review of all log call sites and deployment log retention remains|
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