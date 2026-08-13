# Portfolio Upgrade — 2026-08-10

## Added

- Evidence-quality conflict candidates and fail-closed KB abstention.
- Evaluation & QA panel that only displays real benchmark output.
- Recruiter-friendly synthetic 2025/2026 policy demo.
- Cross-document benchmark case and explicit abstention case.
- Portfolio positioning, demo script, and evaluation documentation.
- GitHub Actions compile/test workflow and Makefile shortcuts.
- Optional `requirements-experiments.txt` so deepagents reference scripts do not inflate the core runtime dependency path.

## Hardened

- Runtime dependency list now explicitly pins LangGraph and isolates optional reference tooling.
- No secrets or runtime chat state are included in the distributable project.
- Demo seed content is stored under `demo/` rather than embedded in the application path.

## Validation

- Python compile check passes for application, scripts, tests, and support modules.
- Dependency-light evidence-quality checks pass in the build environment.
- Full pytest collection requires the project's pinned dependencies; the current execution sandbox does not have those packages installed.
