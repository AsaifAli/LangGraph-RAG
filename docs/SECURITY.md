# Security Review — EvidenceFlow (LangGraph-RAG)

This is a code/configuration review, not a penetration test or a declaration of production certification. EvidenceFlow is an API-only FastAPI/LangGraph research service with OpenSearch, optional web research, and a server-side Gateway session.

## Existing safeguards observed
- `.env` is excluded from the repository/container context; secrets are read from server-side environment variables.
- Retrieval requires tenant, document and opaque owner-scope filters; lexical, neural-sparse, full-document summary and delete queries all use the same owner term. An empty document list does not become an unrestricted query.
- The repository correctly documents that a tenant or document ID is not an authorization credential. A trusted identity/ACL service must approve document IDs before retrieval.
- Citation verification is fail-closed in verified mode.
- Gateway exchange uses a short-lived one-time code, bounded response size, an 8-second timeout, an approved HTTPS Gateway host allowlist, public-DNS checks and a no-redirect handler; the exchanged bearer and owner scope are retained in the encrypted server-side API session store rather than the URL or browser JavaScript.
- Tavily and Katzilla keys are loaded server-side; the MCP adapter does not intentionally put the key into model context. Katzilla's HTTPS URL is pinned to an approved host/public DNS, proxy trust is disabled, and redirects are not followed.
- OpenSearch, model and provider credentials are configured as deployment secrets rather than client-prefixed environment variables.

## Findings requiring attention
1. **User query logging (Medium): PATCHED in main.** `tools/web_search_tool.py` no longer prints user queries, provider payloads, response content or source snippets. `tests/test_web_search_logging.py` guards against regression. Other logging/traceback paths still require a broader review.
2. **MCP redirect policy (High): PATCHED in main.** `tools/primary_source_mcp.py` now sets `follow_redirects=False` on the authenticated client. `tests/test_primary_source_mcp.py` verifies that policy.
3. **Gateway URL trust (High): PATCHED in main.** `app/gateway_handoff.py` requires HTTPS and public DNS destinations for remote gateways, rejects nonstandard external ports, and permits loopback/host.docker.internal only in non-production local development. Gateway URL remains server-side configuration, never a chat payload.
4. **Container privilege (Medium): PATCHED in main.** Both `Dockerfile` and `Dockerfile.render` create an unprivileged `app` account and give it write access only to the chat-history, model-cache and home paths. The Render image was redeployed from the merged commit; keep checking the live deployment after future Dockerfile changes.
5. **Diagnostic disclosure (Medium): PATCHED in main.** Upload and agent-turn failures now emit only a generated incident reference and exception class to application logs. The UI displays a generic retry message with that reference; raw exception messages and tracebacks are no longer rendered or logged.
6. **Upload surface: PATCHED in main.** The central `retrieval/rag_pipeline.py::ingest_document` path calls `shared/upload_security.py` before parsers, image processing, indexing or model calls. It enforces a configurable 200 MB default cap (used by the API upload layer), validates PDF/DOC/DOCX/text/image signatures, and bounds Office archive entries and expanded size. The API rejects uploads above the configured 200 MB default.

## Additional owner and request hardening
- Supabase-authenticated Gateway sessions share a stable opaque owner hash; guest BYOK sessions get a session-specific owner hash. The website identity attachment is opt-in and the Gateway requires its Supabase URL/publishable key to be configured before the website flag is enabled.
- OpenSearch chunk metadata stores `owner_scope`. API global uploads, chat attachments, chat history, document lists, summaries, retrieval and deletes are owner-filtered. Legacy records without an owner remain inaccessible rather than being assigned to the first visitor.
- The API upload entry point reject declared oversized files before copying bytes; the central ingestion function rechecks actual byte size/signature/archive limits. A per-owner upload quota (8/hour) and chat quota (30/minute) use Upstash in production and fail closed if shared storage is unavailable.
- User-controlled uploaded filenames are escaped before HTML rendering. Trusted templates still use `unsafe_allow_html`; maintain the code-review boundary and do not interpolate unescaped data into those templates.
- Production Upstash credentials are supplied through `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` in Render; local/offline ingestion retains a bounded local-development scope.

## 20-control assessment
| # | Control | Result |
|---|---|---|
|01|SQL injection|PARTIAL — OpenSearch queries are constructed through client APIs; review every query/filter builder for raw DSL interpolation|
|02|XSS|IMPROVED/PARTIAL — dynamic document labels are HTML-escaped and model text is returned as structured JSON/SSE; continue reviewing each trusted `unsafe_allow_html` template|
|03|CSRF|PARTIAL — Same-origin BFF, workspace session and HttpOnly per-client EvidenceFlow cookie; retain CSRF and cookie policy tests|
|04|File upload|PARTIAL — server-side size/signature and Office archive bounds are implemented with compatibility tests; malware scanning and deeper parser sandboxing remain external hardening|
|05|SSRF|IMPROVED/PARTIAL — Gateway and Katzilla destinations are restricted to approved HTTPS hosts/public DNS; redirects and environment proxy trust are disabled|
|06|BOLA/IDOR|IMPROVED/PARTIAL — verified Gateway owner scope is enforced for new documents, chats, retrieval, summaries and deletes; legacy unowned data is hidden. This is owner isolation, not a full document ACL/RBAC system|
|07|Rate limiting|PARTIAL — per-owner Upstash quotas cover upload and chat entry points; add edge/WAF limits and review individual tool-cost quotas|
|08|Password hashing|N/A — no local password authentication identified|
|09|MFA|EXTERNAL — enable for GitHub, hosting, OpenSearch and provider administrator accounts|
|10|Server-side permissions|IMPROVED/PARTIAL — server-derived owner scope filters are enforced before agent retrieval; fine-grained document ACLs and team roles remain out of scope|
|11|RLS|N/A/PARTIAL — OpenSearch tenant filters are app-level, not database RLS; isolate tenant credentials/index permissions|
|12|JWT secret|GATEWAY-OWNED — Gateway signs the short-lived session; verify production secret strength/rotation there|
|13|Server-side API keys|PASS by design — provider/MCP keys are server-side environment settings; validate hosting config|
|14|No localStorage tokens|PARTIAL — exchanged Gateway bearer is stored in the backend session store; validate browser storage in deployed app|
|15|No default credentials|PARTIAL — no local admin login found; verify all deployment secrets are unique and non-placeholder|
|16|Strict CORS|PARTIAL — Studio BFF origin policy is deployment-owned; configure exact HTTPS hosts|
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