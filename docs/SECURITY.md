# Security boundaries

This repository is a portfolio-grade reference implementation, not a complete
enterprise IAM product.

## Implemented safeguards

- Secrets are loaded from environment variables and `.env` is never shipped in
the repository or Docker build context.
- Knowledge-base retrieval always includes tenant and document conditions.
- An empty document scope produces an empty OpenSearch `MatchAny` condition rather
than an unrestricted query.
- Citation verification is fail-closed in `verified` mode.
- Document IDs are treated as scope identifiers, not as proof of authorization.

## Production boundary

The Streamlit app is the trusted caller of the retrieval contract. A production
service should place authentication and authorization **before** the agent and
issue only ACL-approved document IDs to it:

```text
Identity provider -> API auth -> tenant/ACL service -> authorized document IDs
                                                -> LangGraph -> OpenSearch
```

Do not treat a client-supplied `tenant_id` or `document_id` as an authorization
credential. The repository intentionally documents this boundary rather than
claiming to implement a full enterprise identity system.

## Secret rotation

If credentials are ever committed or shared accidentally, rotate them at the
provider immediately. Never replace a leaked key with another key in source.
Use `.env.example` as the template for local configuration.
