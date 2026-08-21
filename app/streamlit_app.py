"""ChatGPT-style chat frontend for the unified deepagents POC agent.

Explicitly "just for now" — a thin UI over `custom_langgraph_poc`'s
build-once / invoke-many API (`build_langgraph_agent`, `astream_langgraph_turn`)
and `rag_pipeline.ingest_document`. No new agent logic lives here; the
actual behavior (routing, retrieval, delegation, citation) is defined in
`custom_langgraph_poc.py` — a lean, custom LangGraph agent (no deepagents),
which replaced the earlier deepagents-based `unified_poc.py` once
comparison testing showed deepagents' planning/filesystem/skills
scaffolding was pure token overhead for this fixed-shape RAG workflow. See
README "Comparison: deepagents/LangGraph vs. custom orchestration" for the
full reasoning trail.

By design, per explicit direction: no pattern picker, no backend selector,
no subagent-count slider in the primary UI. The unified agent decides for
itself which source(s) a question needs (see `skills/query-routing/`) —
that's the whole point of `custom_langgraph_poc.py` over the single-pattern scripts.
Backend defaults to `rag_pipeline.DEFAULT_BACKEND` (`"gemini-api"` unless
overridden via `BACKEND` in `.env` — see README "Backends actually
available" for why that one, not the originally-hardcoded `huggingface`, is
default now); the individual-pattern scripts
(`research_poc.py`/`data_analysis_poc.py`) remain available
from the CLI for anyone who wants to force a specific pattern or backend for
testing/comparison, which is a different use case than this chat UI.

Real conversation memory (checkpointer + thread_id, see README "Context
management"): the agent is built ONCE per browser session
(`st.session_state.built_agent`, survives across turns and across "New
chat") and reused via a `thread_id` (regenerated on "New chat" — the
underlying LangGraph checkpointer is what actually remembers prior turns,
not anything reconstructed here). The checkpointer is a disk-backed
`AsyncSqliteSaver` (`chat_history/checkpoints.db`, gitignored/local), not
`InMemorySaver` — conversation memory survives a server restart, not just
calls within one running process.

Chat history sidebar: every conversation is saved as its own file,
`chat_history/<thread_id>.json` (title + rendered messages, including
citation/token UI metadata the checkpointer's raw LangChain messages don't
carry) so past chats can be listed and switched back into — the
checkpointer persists what the MODEL remembers, this persists what the UI
renders; they're deliberately two different stores serving two different
readers.

Streaming (see README "Frontend / event streaming"): responses stream token
by token via `astream_langgraph_turn`, with a live `st.status()` panel showing
which tool is currently running, instead of a blocking spinner around a
single `ainvoke()` call.

Two upload entry points, calling the same `ingest_document` (no duplicated
ingestion logic) but with DELIBERATELY DIFFERENT scope, by explicit
request:

- **Sidebar uploader** ("load this in before I start asking anything") —
  GLOBAL scope, unchanged from before: available from every chat,
  persisted in its own file (`chat_history/uploaded_documents.json`,
  `st.session_state.global_uploaded_docs`), survives "New chat"/switching/
  deleting chats. Removing one is still only the 🗑️ button next to it,
  which deletes its vectors from Qdrant for real.
- **Attach-in-chat** (`st.chat_input(accept_file=...)`, "here's a document,
  answer about it" in the flow of asking) — PER-CHAT scope: reachable only
  from the chat it was attached in, stored inside that chat's own
  `chat_history/<thread_id>.json` (`thread["uploaded_docs"]`,
  `st.session_state.chat_uploaded_docs`). "New chat" starts with none,
  switching chats loads THAT chat's own list, and — unlike the demo-fixture
  gap `_clear_everything` had to close separately — deleting a chat deletes
  ITS attached documents' vectors from Qdrant too, not just its own JSON
  file; a document attached to a DIFFERENT chat, or uploaded globally via
  the sidebar, is never touched by deleting this one.

`_document_ids_in_scope`/`_document_names_in_scope` return the UNION of
both — global docs plus the current chat's own attached docs — read live on
every turn, so uploading/deleting either kind takes effect immediately
without rebuilding the agent.

Run (from poc/langgraph_rag/): streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import json
import os
import re
import sys
import uuid
import logging
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st


LOGGER = logging.getLogger(__name__)

# This file lives in app/ — _POC_ROOT is one level up (poc/langgraph_rag/),
# where .env/chat_history/ live and where the agents/retrieval/tools/shared/
# etc. top-level packages sit as siblings of app/. Inserted into sys.path so
# `from agents...`/`from retrieval...` below resolve regardless of CWD
# (matches how `streamlit run app/streamlit_app.py` is invoked from
# poc/langgraph_rag/, but doesn't depend on that being the CWD).
_HERE = Path(__file__).resolve().parent
_POC_ROOT = _HERE.parent
if str(_POC_ROOT) not in sys.path:
    sys.path.insert(0, str(_POC_ROOT))

from sidebar_toggle import render_sidebar_toggle  # noqa: E402
from agents.custom_langgraph_poc import astream_langgraph_turn, build_langgraph_agent  # noqa: E402
from citations.verifier import verify_reference  # noqa: E402
from retrieval.rag_pipeline import (  # noqa: E402
    DEFAULT_BACKEND,
    DEFAULT_TURN_TIMEOUT_SECONDS,
    RetrievedChunk,
    delete_document,
    get_gemini_api_key,
    ingest_document,
    load_runtime_config,
)

# Defaults to "gemini-api" (see rag_pipeline.DEFAULT_BACKEND / BACKEND
# in .env) to avoid the HF Inference API's recurring credit exhaustion (see
# README "What was actually tested") — a direct Google AI Studio API key
# (GEMINI_API_KEY). The embedder (`rag_pipeline.build_embedder`) does not
# depend on HF Inference either — it runs sentence-transformers/all-MiniLM-L6-v2
# locally — so HF_TOKEN is only required if "huggingface" is selected
# (directly or via BACKEND).
_BACKEND = DEFAULT_BACKEND
_CHAT_HISTORY_DIR = _POC_ROOT / "chat_history"
_CHECKPOINT_DB_PATH = _CHAT_HISTORY_DIR / "checkpoints.db"
_UPLOADED_DOCS_PATH = _CHAT_HISTORY_DIR / "uploaded_documents.json"

# --- Icon assets -------------------------------------------------------
# Bundled locally (app/assets/icons/*.svg), not hotlinked at runtime — this
# app already hit two flaky-network incidents in one session (HF's xet CDN
# 500s, slow WSL networking), so the UI's own icons shouldn't be one more
# thing that can silently break on a bad connection. Source: Tabler Icons
# (https://github.com/tabler/tabler-icons), MIT licensed, no attribution
# required — license text bundled at app/assets/ICONS_LICENSE.txt.
_ICONS_DIR = _HERE / "assets" / "icons"


def _icon(name: str, *, size: int = 20, color: str = "currentColor") -> str:
    """Inline an SVG icon as an HTML string, with size/color overridden via
    simple string substitution (Tabler SVGs already use
    width="24"/height="24"/stroke="currentColor")."""
    path = _ICONS_DIR / f"{name}.svg"
    if not path.exists():
        return ""
    svg = path.read_text(encoding="utf-8")
    svg = svg.replace('width="24"', f'width="{size}"').replace('height="24"', f'height="{size}"')
    svg = svg.replace("currentColor", color)
    return svg


# Favicon: the bundled robot.svg (see _ICONS_DIR above) rather than a bare
# emoji — `st.set_page_config`'s `page_icon` accepts a local file path and
# serves it through Streamlit's own media file manager (confirmed directly
# in the installed `streamlit.commands.page_config._get_favicon_string`
# source: non-emoji strings go through `image_to_url`, which handles local
# SVG paths). Falls back to the emoji if the file is ever missing, rather
# than a raw exception on page load.
_ROBOT_ICON_PATH = _ICONS_DIR / "robot.svg"
st.set_page_config(
    page_title="Document Research Assistant",
    page_icon=str(_ROBOT_ICON_PATH) if _ROBOT_ICON_PATH.exists() else "🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Shared visual layer. The browser/system appearance drives the visual CSS live;
# Streamlit's theme type is retained only as a first-render fallback for server-side
# components. No rerun is required for the visual palette to switch.

from ui_theme import apply_theme  # noqa: E402
_theme_mode = st.context.theme.type or "light"
apply_theme(
    product="EvidenceFlow",
    subtitle="Verified RAG & research workspace",
    accent="#7C3AED",
    accent2="#2563EB",
    theme=_theme_mode,
)

render_sidebar_toggle()

# Portfolio handoff: accept the temporary gateway JWT from the portfolio launch URL.
portfolio_token = str(st.query_params.get("portfolio_llm_session", "")).strip()
if portfolio_token:
    st.session_state.llm_gateway_session_token = portfolio_token
    try:
        del st.query_params["portfolio_llm_session"]
    except Exception:
        pass

# --- Global styling ------------------------------------------------------
_APP_CSS = """
    <style>
    :root {
        --accent-1: #7c3aed;
        --accent-2: #2563eb;
        --card-bg: var(--ef-surface-2);
        --card-border: var(--ef-border-strong);
        --fg-primary: var(--ef-text);
        --fg-muted: var(--ef-muted);
    }
    @keyframes gradient-shift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes fade-in-up {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse-soft {
        0%, 100% { transform: scale(1); opacity: 1; }
        50%      { transform: scale(1.04); opacity: 0.85; }
    }
    @keyframes bounce-dot {
        0%, 80%, 100% { transform: translateY(0); opacity: 0.5; }
        40%           { transform: translateY(-6px); opacity: 1; }
    }
    @keyframes blink-caret {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0; }
    }

    .ef-capability-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.15rem 0 1.15rem;
    }
    .ef-capability {
        display: inline-flex;
        align-items: center;
        gap: 0.42rem;
        padding: 0.42rem 0.68rem;
        border: 1px solid var(--card-border);
        border-radius: 999px;
        background: color-mix(in srgb, var(--card-bg) 94%, transparent);
        color: var(--fg-primary);
        font-size: 0.76rem;
        font-weight: 650;
        line-height: 1;
    }
    .ef-cap-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent-1); flex: 0 0 auto; }
    .ef-cap-dot.web { background: var(--accent-2); }
    .ef-cap-dot.verify { background: #10b981; }
    .ef-cap-dot.memory { background: #8b5cf6; }

    .app-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.1rem;
    }
    .app-header svg {
        flex-shrink: 0;
        animation: pulse-soft 3.5s ease-in-out infinite;
        color: var(--accent-1);
    }
    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--accent-1), var(--accent-2), var(--accent-1));
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: gradient-shift 6s ease infinite;
    }
    .app-subtitle {
        color: var(--fg-muted);
        font-size: 0.95rem;
        margin-bottom: 1.1rem;
        animation: fade-in-up 0.5s ease both;
    }

    /* Chat bubbles: gentle entrance + rounded card feel */
    [data-testid="stChatMessage"] {
        animation: fade-in-up 0.35s ease both;
        border-radius: 14px;
    }

    /* Buttons: smooth hover lift across sidebar + main area */
    .stButton > button {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        border-radius: 10px;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }

    /* Sidebar: preserve Streamlit's native state and repair the collapsed
       affordance using the same proven pointer-events/z-index strategy used
       by the stable AI Automation Command Center UI. */
    [data-testid="stSidebar"] {
        transition: margin-left 0.25s ease, width 0.25s ease;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] button {
        z-index: 100001 !important;
        pointer-events: auto !important;
    }
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 0.62rem !important;
        left: 0.62rem !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        width: 42px !important;
        height: 42px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        width: 40px !important;
        height: 40px !important;
        border-radius: 12px !important;
        background: color-mix(in srgb, var(--ef-surface) 92%, transparent) !important;
        border: 1px solid var(--ef-border) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,.16) !important;
        color: var(--ef-text) !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        width: 20px !important;
        height: 20px !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        margin-left: 0.2rem !important;
    }


    /* ChatGPT-like active conversation: the hero becomes a compact context
       bar and the conversation itself becomes the scroll surface while the
       native Streamlit composer remains pinned to the viewport bottom. */
    .ef-chat-topbar {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        margin:0 0 .85rem; padding:.65rem .8rem;
        border:1px solid var(--ef-border); border-radius:14px;
        background:color-mix(in srgb,var(--ef-surface) 88%,transparent);
        backdrop-filter:blur(14px);
        box-shadow:0 8px 24px color-mix(in srgb,#0f172a 5%,transparent);
    }
    .ef-chat-topbar strong { font-size:.88rem; color:var(--ef-text); }
    .ef-chat-topbar span { margin-left:.5rem; color:var(--ef-muted); font-size:.72rem; }
    .ef-agent-badge { display:inline-flex; align-items:center; gap:.38rem; padding:.28rem .55rem; border-radius:999px;
        border:1px solid color-mix(in srgb,var(--ef-accent) 24%,var(--ef-border));
        background:color-mix(in srgb,var(--ef-accent) 7%,var(--ef-surface));
        color:var(--ef-accent); font-size:.68rem; font-weight:750; white-space:nowrap; }
    .ef-agent-badge span { width:6px; height:6px; border-radius:50%; background:#10b981; box-shadow:0 0 0 4px rgba(16,185,129,.10); }
    body:has(.ef-chat-active) [data-testid="stAppViewContainer"]>.main {
        height:100vh; overflow-y:auto; overscroll-behavior:contain;
        scrollbar-gutter:stable;
    }
    body:has(.ef-chat-active) [data-testid="stAppViewContainer"]>.main .block-container {
        min-height:100%; padding-bottom:7.5rem!important;
    }
    @media (max-width: 720px) {
        .ef-chat-topbar span { display:none; }
        .ef-chat-topbar { margin-bottom:.55rem; }
    }

    .ef-sidebar-brand {
        display:flex; align-items:center; gap:.65rem; margin:.1rem 0 .75rem;
    }
    .ef-brand-mark {
        display:grid; place-items:center; width:34px; height:34px; border-radius:11px;
        background:linear-gradient(135deg,var(--ef-accent),var(--ef-accent-2));
        color:#fff; font-weight:850; box-shadow:0 8px 22px color-mix(in srgb,var(--ef-accent) 18%,transparent);
    }
    .ef-sidebar-brand strong { display:block; color:var(--ef-text); font-size:.9rem; }
    .ef-sidebar-brand span { display:block; color:var(--ef-muted); font-size:.64rem; margin-top:.08rem; }
    .ef-agent-status {
        display:flex; align-items:center; gap:.55rem; padding:.55rem .65rem; margin-bottom:.8rem;
        border:1px solid color-mix(in srgb,#10b981 22%,var(--ef-border)); border-radius:12px;
        background:color-mix(in srgb,#10b981 5%,var(--ef-surface));
    }
    .ef-agent-status > span { width:7px; height:7px; border-radius:50%; background:#10b981; box-shadow:0 0 0 4px rgba(16,185,129,.10); flex:0 0 auto; }
    .ef-agent-status strong { display:block; color:var(--ef-text); font-size:.7rem; }
    .ef-agent-status small { display:block; color:var(--ef-muted); font-size:.6rem; margin-top:.08rem; }

    /* Sidebar section labels with an inline icon */
    .sidebar-heading {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 0.2rem 0 0.3rem 0;
        color: var(--fg-primary);
    }
    .sidebar-heading svg { color: var(--accent-1); flex-shrink: 0; }

    /* Document chip rows in the sidebar */
    .doc-chip {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.85rem;
        color: var(--fg-muted);
        padding: 0.15rem 0;
    }
    .doc-chip svg { flex-shrink: 0; opacity: 0.7; }

    /* Landing experience — compact, high-signal research product surface. */
    .ef-landing {
        margin: .35rem 0 1.25rem;
        animation: fade-in-up .5s ease both;
    }
    .ef-landing-hero {
        display:grid; grid-template-columns:minmax(0,1.22fr) minmax(330px,.78fr);
        gap:1.1rem; padding:1.35rem 1.4rem; border-radius:22px;
        border:1px solid var(--ef-border-strong);
        background:
          radial-gradient(620px 290px at 95% 0%, color-mix(in srgb,var(--ef-accent-2) 17%,transparent),transparent 60%),
          radial-gradient(420px 260px at 0% 100%, color-mix(in srgb,var(--ef-accent) 10%,transparent),transparent 65%),
          linear-gradient(135deg,color-mix(in srgb,var(--ef-accent) 9%,var(--ef-surface)),var(--ef-surface));
        box-shadow:0 22px 65px color-mix(in srgb,#0f172a 13%,transparent);
        overflow:hidden; position:relative;
    }
    .ef-landing-hero::before {
        content:""; position:absolute; inset:0;
        background:linear-gradient(120deg,transparent 25%,rgba(255,255,255,.035) 50%,transparent 75%);
        transform:translateX(-100%); animation: ef-sheen 7s ease-in-out infinite; pointer-events:none;
    }
    @keyframes ef-sheen { 0%,55%{transform:translateX(-100%)} 72%,100%{transform:translateX(100%)} }
    .ef-landing-copy { position:relative; z-index:1; padding:.25rem .15rem; }
    .ef-landing-kicker { color:var(--ef-accent); font-size:.65rem; font-weight:850; letter-spacing:.17em; }
    .ef-landing-title { margin:.5rem 0 .55rem; color:var(--ef-text); font-size:clamp(2rem,3.4vw,3rem); line-height:1.02; font-weight:900; letter-spacing:-.045em; }
    .ef-landing-title span { background:linear-gradient(90deg,var(--ef-accent),var(--ef-accent-2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .ef-landing-sub { max-width:720px; color:var(--ef-muted); font-size:.88rem; line-height:1.58; }
    .ef-landing-pills { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:.8rem; }
    .ef-landing-pill { display:inline-flex; align-items:center; gap:.35rem; padding:.38rem .58rem; border:1px solid var(--ef-border); border-radius:999px; background:color-mix(in srgb,var(--ef-surface-2) 86%,transparent); color:var(--ef-text-2); font-size:.67rem; font-weight:750; }
    .ef-landing-pill i { width:6px; height:6px; border-radius:50%; background:var(--ef-accent); display:inline-block; }
    .ef-landing-pill:nth-child(2) i { background:var(--ef-accent-2); }
    .ef-landing-pill:nth-child(3) i { background:#10b981; }
    .ef-landing-pill:nth-child(4) i { background:#f59e0b; }

    .ef-route-card { position:relative; z-index:1; padding:1rem; border-radius:17px; border:1px solid var(--ef-border); background:color-mix(in srgb,var(--ef-surface-2) 82%,transparent); box-shadow:inset 0 1px 0 rgba(255,255,255,.04),0 12px 30px rgba(15,23,42,.08); }
    .ef-route-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:.65rem; }
    .ef-route-head strong { color:var(--ef-text); font-size:.69rem; letter-spacing:.1em; }
    .ef-ready { display:inline-flex; align-items:center; gap:.35rem; padding:.22rem .48rem; border-radius:999px; color:#10b981; border:1px solid rgba(16,185,129,.22); background:rgba(16,185,129,.06); font-size:.59rem; font-weight:850; }
    .ef-ready::before { content:""; width:6px; height:6px; border-radius:50%; background:#10b981; box-shadow:0 0 0 4px rgba(16,185,129,.08); }
    .ef-route-flow { display:flex; flex-direction:column; gap:.28rem; }
    .ef-route-node { display:flex; align-items:center; gap:.55rem; padding:.5rem .58rem; border:1px solid var(--ef-border); border-radius:11px; background:var(--ef-surface); }
    .ef-route-node b { width:23px; height:23px; display:grid; place-items:center; border-radius:8px; color:var(--ef-accent); background:color-mix(in srgb,var(--ef-accent) 10%,var(--ef-surface)); font-size:.6rem; }
    .ef-route-node span { color:var(--ef-text-2); font-size:.66rem; font-weight:740; }
    .ef-route-arrow { color:var(--ef-muted); font-size:.58rem; margin-left:.7rem; line-height:.45; }

    .ef-benchmark-strip { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin-top:.7rem; }
    .ef-benchmark-card { position:relative; padding:.78rem .9rem; border:1px solid var(--ef-border); border-radius:15px; background:var(--ef-surface); box-shadow:0 10px 28px color-mix(in srgb,#0f172a 5%,transparent); }
    .ef-benchmark-card::before { content:""; display:block; width:28px; height:3px; border-radius:999px; margin-bottom:.55rem; background:linear-gradient(90deg,var(--ef-accent),var(--ef-accent-2)); }
    .ef-benchmark-card strong { display:block; color:var(--ef-text); font-size:.72rem; }
    .ef-benchmark-card p { margin:.18rem 0 0; color:var(--ef-muted); font-size:.62rem; line-height:1.4; }

    .ef-workspace-ready { margin-top:.7rem; display:grid; grid-template-columns:1.6fr repeat(3,.8fr); gap:.65rem; padding:.72rem; border:1px solid var(--ef-border); border-radius:16px; background:color-mix(in srgb,var(--ef-surface-2) 72%,transparent); }
    .ef-ready-copy { padding:.2rem .35rem; }
    .ef-ready-copy strong { color:var(--ef-text); font-size:.78rem; }
    .ef-ready-copy p { margin:.15rem 0 0; color:var(--ef-muted); font-size:.63rem; }
    .ef-live-metric { padding:.55rem .65rem; border:1px solid var(--ef-border); border-radius:12px; background:var(--ef-surface); }
    .ef-live-metric span { display:block; color:var(--ef-muted); font-size:.56rem; text-transform:uppercase; letter-spacing:.08em; }
    .ef-live-metric strong { display:block; margin-top:.12rem; color:var(--ef-text); font-size:.8rem; }

    @media (max-width:900px) {
        .ef-landing-hero { grid-template-columns:1fr; }
        .ef-benchmark-strip { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .ef-workspace-ready { grid-template-columns:1fr 1fr; }
        .ef-ready-copy { grid-column:1 / -1; }
    }
    @media (max-width:560px) {
        .ef-landing-hero { padding:1rem; }
        .ef-benchmark-strip { grid-template-columns:1fr; }
        .ef-workspace-ready { grid-template-columns:1fr; }
    }

    /* Typing indicator (three bouncing dots), shown while a turn is in
    flight before the first token arrives — replaces a bare "Working..."
    text with something that actually communicates "alive, not stuck". */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0;
    }
    .typing-indicator span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--accent-1);
        animation: bounce-dot 1.2s infinite ease-in-out;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

    /* Streaming cursor: appended to the answer text while tokens are still
    arriving (see `_consume` below), removed once the turn completes. */
    .stream-caret {
        animation: blink-caret 0.9s step-end infinite;
        color: var(--accent-1);
    }

    /* KB citation chips: accent border matching the header gradient's
    first color, so KB- vs web-sourced evidence reads as visually distinct
    at a glance without needing to read the label. */
    .citation-row.citation-row-kb {
        border-left: 3px solid var(--accent-1);
        padding-left: 8px;
    }

    /* Web-sources expander: same idea, second gradient color. Targeted via
    an attribute-selector substring match (`[class*=...]`) rather than an
    exact class name — `st.expander(key=...)` becomes a per-instance
    `st-key-<key>` class, and this needs to match EVERY message's own
    uniquely-keyed expander (`web-src-0`, `web-src-1`, ...), not just one. */
    [class*="st-key-web-src-"] {
        border-left: 3px solid var(--accent-2);
        padding-left: 4px;
        border-radius: 4px;
    }

    /* Per-message action row (Copy/Regenerate) — tighter spacing than
    Streamlit's default column gap, so it reads as one small toolbar
    rather than two full-width buttons. */
    /* Targeted via the `st.button(key="regen_...")`-derived `st-key-regen_*`
    class (substring match — the key includes each message's own `idx`,
    so this must match every one of them, not just a single instance),
    NOT `.msg-actions .stButton > button` — that selector never actually
    matched anything: Regenerate isn't a descendant of the `.msg-actions`
    div at all, only the HTML Copy button is (found live, not a guess: it
    silently rendered with Streamlit's plain default button size this
    whole time instead of the smaller size this rule intended). Padding
    matches `.copy-native-btn`'s own so the two end up the same height. */
    [class*="st-key-regen_"] button {
        /* Forced to the EXACT same fixed square as `.copy-native-btn`
        below, not just a similar font-size/padding — Streamlit's own
        default button padding (sized for a text label) is noticeably
        wider than a tight icon-only button needs, even with the label
        left empty, so matching padding alone wasn't enough to make the
        two buttons actually the same physical size. */
        width: 38px;
        height: 38px;
        padding: 0;
        font-size: 1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }
    /* Copy button — plain HTML, not a Streamlit widget (see the history
    loop's own comment on why), so it needs its own rule rather than
    inheriting `.stButton > button`'s styling. Matches that styling by
    hand: same radius/hover-lift/transition as every other button in this
    app, same muted color scheme as the rest of the per-message toolbar. */
    .copy-native-btn {
        /* Small fixed square, icon-only — matches Streamlit's own
        icon-only buttons elsewhere in this app (the 🗑️ delete buttons,
        `st.button("", icon=...)`), rather than trying to match a
        text-labeled button's size like an earlier version of this rule
        did (moot now that Regenerate is icon-only too, see the history
        loop). Fixed, not percentage-based, since there's no longer a long
        label whose width needs a flexible column to avoid overflowing. */
        width: 38px;
        height: 38px;
        font-size: 1rem;
        padding: 0;
        border-radius: 10px;
        border: 1px solid var(--card-border);
        background: transparent;
        color: var(--fg-muted);
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .copy-native-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }


    .ef-route-pill { display:inline-flex;align-items:center;gap:.38rem;padding:.30rem .58rem;border-radius:999px;border:1px solid var(--ef-border);background:color-mix(in srgb,var(--ef-accent) 7%,var(--ef-surface));color:var(--ef-text-2);font-size:.70rem;font-weight:700;margin:.05rem 0 .65rem; }
    .ef-route-dot { width:6px;height:6px;border-radius:50%;background:var(--ef-accent);box-shadow:0 0 0 4px color-mix(in srgb,var(--ef-accent) 10%,transparent); }
    .ef-answer-summary {
        display:flex; align-items:center; justify-content:space-between; gap:.7rem;
        flex-wrap:wrap; margin:0 0 .65rem; padding:.58rem .68rem;
        border:1px solid var(--ef-border); border-radius:13px;
        background:linear-gradient(180deg,color-mix(in srgb,var(--ef-accent) 4%,var(--ef-surface-2)),var(--ef-surface-2));
    }
    .ef-answer-summary-main { display:flex; align-items:center; gap:.45rem; min-width:0; flex-wrap:wrap; }
    .ef-answer-kicker { color:var(--ef-accent); font-size:.58rem; font-weight:900; letter-spacing:.13em; text-transform:uppercase; }
    .ef-answer-route { color:var(--ef-text-2); font-size:.69rem; font-weight:760; }
    .ef-answer-dot { width:7px; height:7px; border-radius:50%; background:#10b981; box-shadow:0 0 0 4px color-mix(in srgb,#10b981 10%,transparent); }
    .ef-answer-dot.warn { background:#f59e0b; box-shadow:0 0 0 4px color-mix(in srgb,#f59e0b 10%,transparent); }
    .ef-answer-dot.muted { background:var(--ef-muted); box-shadow:none; }
    .ef-answer-metrics { display:flex; align-items:center; gap:.35rem; flex-wrap:wrap; }
    .ef-answer-metric { display:inline-flex; align-items:center; gap:.28rem; padding:.24rem .42rem; border:1px solid var(--ef-border); border-radius:999px; color:var(--ef-muted); font-size:.6rem; background:var(--ef-surface); }
    .ef-answer-metric strong { color:var(--ef-text-2); font-size:.61rem; }
    .ef-answer-quality { color:#10b981; font-weight:850; }
    .ef-answer-quality.warn { color:#f59e0b; }
    .ef-answer-quality.muted { color:var(--ef-muted); }
    .ef-answer-body { padding:0 .1rem .15rem; }
    .ef-evidence-mini {
        margin:.45rem 0 .2rem; padding:.52rem .62rem; border-left:3px solid var(--ef-accent);
        border-radius:0 10px 10px 0; background:color-mix(in srgb,var(--ef-accent) 4%,var(--ef-surface-2));
        color:var(--ef-muted); font-size:.66rem;
    }
    .ef-evidence-mini strong { color:var(--ef-text-2); }
    .ef-evidence-summary { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.4rem;margin:.55rem 0 .75rem; }
    .ef-evidence-stat { padding:.52rem .55rem;border:1px solid var(--ef-border);border-radius:10px;background:var(--ef-surface-2); }
    .ef-evidence-stat strong { display:block;color:var(--ef-text)!important;font-size:.85rem; }
    .ef-evidence-stat span { color:var(--ef-muted);font-size:.62rem; }
    .ef-trace { border-left:2px solid color-mix(in srgb,var(--ef-accent) 30%,var(--ef-border));padding-left:.7rem; }
    .ef-trace-step { padding:.28rem 0;color:var(--ef-text-2);font-size:.72rem; }
    .ef-trace-step::before { content:'✓';color:#10b981;font-weight:800;margin-right:.45rem; }
    .ef-evidence-card { transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease; }
    .ef-evidence-card:hover { transform:translateY(-1px);border-color:color-mix(in srgb,var(--ef-accent) 36%,var(--ef-border)); }
    .ef-evidence-source { color:var(--ef-muted);font-size:.65rem;margin-top:.35rem; }

    .ef-section-kicker {
        margin: .7rem 0 .35rem; font-size:.67rem; font-weight:850; letter-spacing:.16em;
        color:var(--ef-accent); text-transform:uppercase;
    }
    .ef-mode-caption { color:var(--ef-muted); font-size:.73rem; margin-bottom:.35rem; }
    .ef-evidence-panel {
        position:sticky; top:1rem; align-self:flex-start;
        max-height:calc(100vh - 7rem); overflow:auto; padding:1rem;
        border:1px solid var(--ef-border); border-radius:18px;
        background:linear-gradient(180deg,color-mix(in srgb,var(--ef-accent) 3%,var(--ef-surface)),var(--ef-surface));
        box-shadow:var(--ef-shadow);
    }
    .ef-evidence-title { font-size:.9rem; font-weight:850; color:var(--ef-text); margin-bottom:.12rem; }
    .ef-evidence-sub { font-size:.72rem; color:var(--ef-muted); margin-bottom:.8rem; }
    .ef-evidence-card { border:1px solid var(--ef-border); border-radius:13px; padding:.72rem; margin:.55rem 0; background:var(--ef-surface-2); }
    .ef-evidence-card.selected { border-color:color-mix(in srgb,var(--ef-accent) 55%,var(--ef-border)); box-shadow:0 0 0 3px color-mix(in srgb,var(--ef-accent) 9%,transparent); }
    .ef-evidence-card strong { color:var(--ef-text)!important; font-size:.75rem; }
    .ef-evidence-card p { color:var(--ef-text-2)!important; font-size:.74rem; line-height:1.5; margin:.4rem 0 0; }
    .ef-evidence-badge { display:inline-flex; margin-top:.42rem; padding:.18rem .45rem; border-radius:999px; border:1px solid color-mix(in srgb,var(--ef-accent) 28%,var(--ef-border)); color:var(--ef-accent); font-size:.64rem; font-weight:750; }
    .ef-evidence-empty { padding:1rem .3rem; color:var(--ef-muted); font-size:.74rem; line-height:1.55; }
    .ef-evidence-panel .stButton>button { min-height:34px!important; padding:.2rem .55rem!important; font-size:.72rem!important; }
    .ef-open-marker { display:none!important; }
    body:has(.ef-evidence-open) [data-testid="stChatInput"]>div { width:min(1060px,calc(100% - 430px))!important; margin-left:0!important; margin-right:390px!important; }
    @media (max-width: 1100px) {
        body:has(.ef-evidence-open) [data-testid="stChatInput"]>div { width:calc(100% - 2rem)!important; margin:0 auto!important; }
    }

    </style>
    """
st.markdown(_APP_CSS, unsafe_allow_html=True)

# Streamlit 1.58+ can leave the native collapsed control unreachable; this
# trusted same-origin iframe is a fallback only and hides itself when expanded.


def _chat_file_path(thread_id: str) -> Path:
    return _CHAT_HISTORY_DIR / f"{thread_id}.json"


def _load_chat_store() -> dict:
    """Each conversation is its own file under chat_history/ (by request —
    a real, separately-readable file per chat, not one combined blob) —
    <thread_id>.json holds that thread's title/messages; checkpoints.db (the
    LangGraph checkpointer, see `_build_checkpointer`) and
    uploaded_documents.json (see `_load_uploaded_docs`) live in the same
    folder. Older chat files from before document scope went global may
    still carry their own stale "uploaded_docs" field — harmless, just
    unused now; `_load_uploaded_docs` is what migrates that old per-chat
    data into the new global registry, once, the first time it's missing."""
    _CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    threads = []
    for f in _CHAT_HISTORY_DIR.glob("*.json"):
        if f == _UPLOADED_DOCS_PATH:
            continue
        try:
            threads.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - corrupt/partial file, skip rather than crash the app
            continue
    return {"threads": threads}


def _load_uploaded_docs() -> list[dict]:
    """Global document registry, `chat_history/uploaded_documents.json` —
    by explicit request, uploads are no longer scoped to the chat they
    happened in (see module docstring for why). One-time migration on first
    load if this file doesn't exist yet: earlier versions stored each
    chat's uploads in that chat's OWN `uploaded_docs` field, which would
    otherwise strand any already-uploaded document with no path back into
    scope — aggregated here instead (deduped by document_id) so nothing
    already in Qdrant becomes silently unreachable."""
    if _UPLOADED_DOCS_PATH.exists():
        try:
            return json.loads(_UPLOADED_DOCS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt/partial file, start fresh rather than crash the app
            return []

    migrated: dict[str, dict] = {}
    for f in _CHAT_HISTORY_DIR.glob("*.json"):
        if f == _UPLOADED_DOCS_PATH:
            continue
        try:
            thread = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for doc in thread.get("uploaded_docs", []) or []:
            migrated.setdefault(doc["document_id"], doc)
    docs = list(migrated.values())
    if docs:
        _CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        _UPLOADED_DOCS_PATH.write_text(json.dumps(docs, indent=2), encoding="utf-8")
    return docs


def _save_uploaded_docs() -> None:
    _CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _UPLOADED_DOCS_PATH.write_text(
        json.dumps(st.session_state.uploaded_docs, indent=2), encoding="utf-8"
    )


def _messages_to_jsonable(messages: list[dict]) -> list[dict]:
    """`meta["chunks"]` holds `RetrievedChunk` dataclass instances (see
    rag_pipeline.py), not JSON-serializable as-is."""
    out = []
    for m in messages:
        m2 = {"role": m["role"], "content": m["content"]}
        meta = m.get("meta")
        if meta:
            meta2 = dict(meta)
            if meta2.get("chunks"):
                meta2["chunks"] = [asdict(c) for c in meta2["chunks"]]
            m2["meta"] = meta2
        out.append(m2)
    return out


def _messages_from_jsonable(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m2 = {"role": m["role"], "content": m["content"]}
        meta = m.get("meta")
        if meta:
            meta2 = dict(meta)
            if meta2.get("chunks"):
                meta2["chunks"] = [RetrievedChunk(**c) for c in meta2["chunks"]]
            m2["meta"] = meta2
        out.append(m2)
    return out


# Moved here (ABOVE the session-state init block below) rather than left
# next to `_persist_current_chat`/`_switch_to_chat` further down, which is
# where they conceptually belong — the query-param resume logic in that
# init block calls `_messages_from_jsonable` directly at MODULE level
# (not inside a function of its own), so it needs the name to already
# exist by the time Python executes that line, not just by the time some
# LATER function gets called. Left both defined together up here rather
# than only relocating one, so the pair stays next to each other.
st.session_state.setdefault("messages", [])  # [{"role": "user"|"assistant", "content": str}]
st.session_state.setdefault("evidence_open", False)
st.session_state.setdefault("selected_evidence_id", None)
st.session_state.setdefault("evidence_checks", {})
st.session_state.setdefault("uploaded_docs", _load_uploaded_docs())  # GLOBAL (sidebar uploader) — see module docstring
st.session_state.setdefault("sidebar_uploader_key", 0)  # bumped after each ingest to reset the widget
st.session_state.setdefault("built_agent", None)
# Seeded from the persisted registry, not just an empty set — a file already
# in the global registry should stay deduped across a server restart too,
# now that uploads are meant to be permanent rather than session-local.
st.session_state.setdefault(
    "ingested_hashes",
    {d["content_hash"] for d in st.session_state.uploaded_docs if d.get("content_hash")},
)
st.session_state.setdefault("chat_store", _load_chat_store())  # {"threads": [{thread_id, title, messages, ...}]}

# Active thread persisted across a real browser REFRESH, not just a
# same-tab rerun. `st.session_state` does not survive a hard reload — a
# fresh browser request starts a brand-new server-side session — so
# without this, refreshing mid-conversation always dropped back to an
# empty "New chat", with the only way back being a manual click on that
# chat in the sidebar list (the chat itself was never actually lost — see
# `chat_history/<thread_id>.json` — just no longer the one showing).
# `st.query_params` DOES survive a refresh (it's part of the URL), so the
# active thread_id is mirrored there and read back here whenever this is a
# genuinely fresh session (`"thread_id" not in st.session_state`) — a
# same-tab rerun (button click, etc.) already has it in session_state and
# skips straight past this.
if "thread_id" not in st.session_state:
    _resume_thread = next(
        (t for t in st.session_state.chat_store["threads"] if t["thread_id"] == st.query_params.get("chat")),
        None,
    )
    if _resume_thread is not None:
        st.session_state.thread_id = _resume_thread["thread_id"]
        st.session_state.messages = _messages_from_jsonable(_resume_thread.get("messages", []))
    else:
        st.session_state.thread_id = uuid.uuid4().hex
st.query_params["chat"] = st.session_state.thread_id

# PER-CHAT (attach-in-chat) — the current thread's own uploaded_docs, a
# property of the specific chat, not global (see module docstring). Sourced
# from that thread's own chat_store entry, same "read from the persisted
# thread dict" pattern `_switch_to_chat` uses — a brand-new thread_id (the
# common case: every fresh browser session starts on one) simply has no
# matching entry yet, so this correctly starts empty.
if "chat_uploaded_docs" not in st.session_state:
    _current_thread = next(
        (t for t in st.session_state.chat_store["threads"] if t["thread_id"] == st.session_state.thread_id),
        None,
    )
    st.session_state.chat_uploaded_docs = list((_current_thread or {}).get("uploaded_docs", []))
st.session_state.setdefault(
    "chat_ingested_hashes",
    {d["content_hash"] for d in st.session_state.chat_uploaded_docs if d.get("content_hash")},
)


_TURN_TIMEOUT_SECONDS = DEFAULT_TURN_TIMEOUT_SECONDS


def _run_async(coro, *, timeout: float = _TURN_TIMEOUT_SECONDS):
    """Run a coroutine on ONE event loop kept alive for the whole browser
    session, instead of `asyncio.run()`'s fresh-loop-per-call. Required
    because the agent (and its model client) is built ONCE and reused across
    turns (see module docstring "Real conversation memory") — `ChatHuggingFace`
    /`HuggingFaceEndpoint` lazily creates and CACHES an `AsyncInferenceClient`
    on itself the first time it's used (confirmed directly in the installed
    `langchain_huggingface` source), so that client stays bound to whichever
    event loop was running on its first call. `asyncio.run()` tears its loop
    down when it returns, so turn 2 (a new `asyncio.run()`, therefore a new
    loop) would hand that stale, already-bound client to a loop it was never
    created for — this is what live testing showed as the retriever "working
    once, then stuck" on every turn after the first.

    `asyncio.set_event_loop(loop)` on every call, not just at creation: some
    async libraries in this dependency chain (httpx/huggingface_hub/qdrant's
    async paths) fall back to `asyncio.get_event_loop()` rather than
    `get_running_loop()` in places; without explicitly marking our persisted
    loop as "the" event loop for this thread every time, such a call could
    resolve to a different (never-run) default loop and hang waiting on it
    forever instead of raising. `timeout` is the actual hang backstop: no
    single turn should be able to freeze the UI indefinitely regardless of
    which dependency stalls and why — see `run_langgraph_turn`'s caller for the
    user-facing message this produces on expiry.

    Defensive fallback for `loop.is_running()`: seen LIVE, not theorized —
    clicking the 🗑️ delete-document button raised
    `RuntimeError: This event loop is already running` against this exact
    persisted loop. Streamlit (1.60+) runs its server on Uvicorn/asyncio now,
    and Streamlit's own rerun-on-interrupt model can start a NEW script
    execution on a new thread while an OLDER one for the same session is
    still blocked inside a previous `_run_async` call — `run_until_complete`
    is plain asyncio, not a `st.*` call, so Streamlit's cooperative
    stop-signal can't preempt it, and two script runs can end up racing to
    use the SAME loop object. Rather than crash the whole rerun, fall back to
    a throwaway loop for just this one call when that happens. Safe for
    anything that doesn't depend on the persisted loop's cached async client
    (e.g. `delete_document`, which only touches Qdrant via
    `asyncio.to_thread`); for a chat turn specifically hitting this race, the
    fallback could in principle reintroduce the original "stuck client"
    failure mode this function exists to prevent — but that's still strictly
    better than an unhandled crash, and only the abnormal race path pays for it."""
    if "event_loop" not in st.session_state:
        st.session_state.event_loop = asyncio.new_event_loop()
    loop = st.session_state.event_loop
    # Streamlit/Uvicorn can invalidate an event loop during a rerun.  Reusing
    # a closed loop is what produces secondary "client has been closed"
    # ingestion failures.  Recreate it before any Qdrant/model coroutine is
    # scheduled.  The normal path still keeps one loop alive for cached async
    # model clients.
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        st.session_state.event_loop = loop
        # Cached model clients may have been bound to the old loop.
        # Rebuild the agent so those clients are created on the new loop.
        st.session_state.built_agent = None
    if loop.is_running():
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))


async def _build_checkpointer():
    """Disk-backed LangGraph checkpointer (`chat_history/checkpoints.db`), replacing
    `InMemorySaver` — conversation memory now survives a server restart, not
    just calls within one running process. `AsyncSqliteSaver` (not the sync
    `SqliteSaver`) because every call site here (`ainvoke`/`astream_events`/
    `aget_state`) is async. Built from a plain `aiosqlite.Connection` kept
    open for the session's lifetime rather than `from_conn_string`'s `async
    with` form, which would close the connection as soon as that block
    exited — this needs to stay open across every turn, like the agent
    itself (see `_get_or_build_agent`)."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    conn = await aiosqlite.connect(str(_CHECKPOINT_DB_PATH))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


def _persist_current_chat() -> None:
    """Write the active thread to its own file, `chat_history/<thread_id>.json`.
    Deliberately separate from the LangGraph checkpointer (`_build_checkpointer`):
    that persists what the MODEL uses for context (raw LangChain messages);
    this persists what the UI renders (including citation/token metadata the
    checkpointer's messages don't carry), so a past chat re-opened from the
    sidebar looks exactly like it did live, not just "the model still
    remembers it." Called after every turn/upload — every real chat gets a
    title from its first message, and empty threads (a "New chat" nobody
    typed into yet) never get written, so the folder never fills up with
    blank files."""
    if not st.session_state.messages:
        return
    threads = st.session_state.chat_store["threads"]
    thread = next((t for t in threads if t["thread_id"] == st.session_state.thread_id), None)
    if thread is None:
        thread = {"thread_id": st.session_state.thread_id, "title": None}
        threads.append(thread)
    if not thread.get("title"):
        first_user = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), None)
        if first_user:
            thread["title"] = first_user[:50] + ("…" if len(first_user) > 50 else "")
    thread["messages"] = _messages_to_jsonable(st.session_state.messages)
    # This chat's own PER-CHAT attached documents (not the global sidebar
    # ones — see module docstring) — a property of this specific thread,
    # persisted alongside its messages so `_switch_to_chat` can restore it.
    thread["uploaded_docs"] = list(st.session_state.chat_uploaded_docs)
    thread["updated_at"] = datetime.now(timezone.utc).isoformat()
    _CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _chat_file_path(st.session_state.thread_id).write_text(
        json.dumps(thread, indent=2), encoding="utf-8"
    )


def _switch_to_chat(thread_id: str) -> None:
    thread = next(
        (t for t in st.session_state.chat_store["threads"] if t["thread_id"] == thread_id), None
    )
    if thread is None:
        return
    st.session_state.thread_id = thread_id
    st.session_state.messages = _messages_from_jsonable(thread.get("messages", []))
    # PER-CHAT documents load with the chat they belong to (module docstring)
    # — the GLOBAL sidebar list is untouched, it's the same regardless of
    # which chat is active.
    st.session_state.chat_uploaded_docs = list(thread.get("uploaded_docs", []))
    st.session_state.chat_ingested_hashes = {
        d["content_hash"] for d in st.session_state.chat_uploaded_docs if d.get("content_hash")
    }


def _delete_chat(thread_id: str) -> None:
    """Delete a chat for real, not just from the sidebar list: removes every
    store a chat lives in (see `_persist_current_chat`'s "two stores" note,
    plus now a third) — `chat_history/<thread_id>.json` (what the UI
    renders), the LangGraph checkpointer's own thread state (what the model
    remembers, via `agent.checkpointer.adelete_thread(thread_id)` — confirmed
    present on the compiled graph), AND — by explicit request — every
    document that was attached IN this chat, deleted from Qdrant for real
    (`rag_pipeline.delete_document`), the same way the 🗑️ per-document button
    does it. GLOBAL (sidebar-uploaded) documents are never touched here —
    only `thread["uploaded_docs"]`, this chat's own per-chat list, is read
    for what to delete.

    Looked up from `chat_store` (not `st.session_state.chat_uploaded_docs`)
    since the chat being deleted isn't necessarily the active one — deleting
    a past chat from the sidebar must delete THAT chat's documents, not
    whatever's currently loaded for the chat you're looking at."""
    target = next(
        (t for t in st.session_state.chat_store["threads"] if t["thread_id"] == thread_id), None
    )
    for d in (target or {}).get("uploaded_docs", []):
        _run_async(delete_document(d["document_id"]))

    st.session_state.chat_store["threads"] = [
        t for t in st.session_state.chat_store["threads"] if t["thread_id"] != thread_id
    ]
    _chat_file_path(thread_id).unlink(missing_ok=True)
    if st.session_state.built_agent is not None:
        _run_async(st.session_state.built_agent["agent"].checkpointer.adelete_thread(thread_id))
    if st.session_state.thread_id == thread_id:
        st.session_state.messages = []
        st.session_state.chat_uploaded_docs = []
        st.session_state.chat_ingested_hashes = set()
        st.session_state.thread_id = uuid.uuid4().hex


def _delete_document(document_id: str) -> None:
    """Delete a document for real, not just from the sidebar/scope list:
    removes its chunks from Qdrant (`rag_pipeline.delete_document` — see
    that function's docstring for how the delete filter is built), then
    drops it from the GLOBAL `uploaded_docs` registry (persisted immediately
    via `_save_uploaded_docs` — this is the only action that removes a
    document from scope now that uploads are global, see module docstring)
    and its content hash from `ingested_hashes` (so re-uploading the exact
    same file later isn't silently skipped as "already ingested" — it
    genuinely isn't, anymore)."""
    doc = next((d for d in st.session_state.uploaded_docs if d["document_id"] == document_id), None)
    _run_async(delete_document(document_id))
    st.session_state.uploaded_docs = [
        d for d in st.session_state.uploaded_docs if d["document_id"] != document_id
    ]
    _save_uploaded_docs()
    if doc and doc.get("content_hash"):
        st.session_state.ingested_hashes.discard(doc["content_hash"])
    if doc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"🗑️ Removed **{doc['name']}** from the knowledge base.",
                "kind": "notice",
            }
        )


def _delete_chat_document(document_id: str) -> None:
    """PER-CHAT counterpart to `_delete_document` — same real Qdrant
    deletion, but removes from `st.session_state.chat_uploaded_docs` (this
    chat's own list) and `chat_ingested_hashes` instead of the global ones.
    Caller is responsible for `_persist_current_chat()` afterward, same as
    the global version's callers do."""
    doc = next((d for d in st.session_state.chat_uploaded_docs if d["document_id"] == document_id), None)
    _run_async(delete_document(document_id))
    st.session_state.chat_uploaded_docs = [
        d for d in st.session_state.chat_uploaded_docs if d["document_id"] != document_id
    ]
    if doc and doc.get("content_hash"):
        st.session_state.chat_ingested_hashes.discard(doc["content_hash"])
    if doc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"🗑️ Removed **{doc['name']}** from this chat's knowledge base.",
                "kind": "notice",
            }
        )


def _clear_everything() -> None:
    """Wipe every chat and every document — explicit-action only, no
    automatic trigger (Streamlit has no officially supported "browser tab
    closed" hook — `st.session_state` is documented as just dropped from
    server memory on disconnect, nothing fires user code; a real
    close-detection setup needs an ASGI `st.App` + a client-side JS beacon,
    and even then a network blip or a backgrounded mobile tab can look
    identical to "closed" and trigger it by accident — a real risk for
    something this destructive. Explicit button + confirmation dialog was
    chosen over that, and over a quieter inactivity-timeout, by request).

    Real deletion, matching `_delete_document`/`_delete_chat`'s own honesty
    about what "delete" means here: every document's vectors actually leave
    Qdrant (not just the sidebar list), and every chat's checkpointer thread
    is actually dropped (not just its `chat_history/<thread_id>.json` file).

        Also deletes every PER-CHAT attached document across every chat, not
    just the currently active one — reads each thread's own
    `uploaded_docs` out of `chat_store` (plus the active thread's current
    `chat_uploaded_docs`, which may not be persisted into `chat_store` yet)
    BEFORE deleting the chat files themselves, since the files being wiped
    below are the only record of which per-chat documents belong to which
    (now-deleted) chat."""
    for d in list(st.session_state.uploaded_docs):
        _run_async(delete_document(d["document_id"]))
    st.session_state.uploaded_docs = []
    st.session_state.ingested_hashes = set()
    _save_uploaded_docs()

    per_chat_ids = {d["document_id"] for d in st.session_state.chat_uploaded_docs}
    for t in st.session_state.chat_store["threads"]:
        per_chat_ids.update(d["document_id"] for d in t.get("uploaded_docs", []))
    for doc_id in per_chat_ids:
        _run_async(delete_document(doc_id))
    st.session_state.chat_uploaded_docs = []
    st.session_state.chat_ingested_hashes = set()

    if st.session_state.built_agent is not None:
        for t in list(st.session_state.chat_store["threads"]):
            _run_async(st.session_state.built_agent["agent"].checkpointer.adelete_thread(t["thread_id"]))
    for f in _CHAT_HISTORY_DIR.glob("*.json"):
        if f != _UPLOADED_DOCS_PATH:
            f.unlink(missing_ok=True)
    st.session_state.chat_store = {"threads": []}
    st.session_state.messages = []
    st.session_state.thread_id = uuid.uuid4().hex


@st.dialog("Clear everything?")
def _confirm_clear_everything() -> None:
    st.warning(
        "This permanently deletes every chat and every document's vectors from "
        "Qdrant. This cannot be undone.",
        icon="⚠️",
    )
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("Cancel", width="stretch"):
            st.rerun()
    with col_confirm:
        if st.button("Yes, delete everything", type="primary", width="stretch"):
            _clear_everything()
            st.rerun()


def _document_ids_in_scope() -> list[str]:
    """The UNION of GLOBAL (sidebar-uploaded) and PER-CHAT (attached-in-this-chat)
    documents — by explicit request, the seeded demo fixture
    The demo fixture is not implicitly included here.
    An empty result correctly yields an empty knowledge base: verified
    directly against `build_kb_filter_expr` that an empty document_ids list
    builds a Qdrant `MatchAny(any=[])` filter, which matches zero points,
    not "unrestricted" — no separate empty-list guard needed here or in
    retrieval."""
    return [d["document_id"] for d in st.session_state.uploaded_docs] + [
        d["document_id"] for d in st.session_state.chat_uploaded_docs
    ]


def _document_names_in_scope() -> dict[str, str]:
    """`{document_id: display name}` for every document currently in
    scope (global + this chat's own, see `_document_ids_in_scope`) — used by
    the agent's routing prompt (so it reasons about a real uploaded
    document's name, not a bare UUID) and to target a specific document for
    a whole-document summarization request. Read live, same as
    `_document_ids_in_scope`, so a newly uploaded document's name is
    available immediately."""
    names = {d["document_id"]: d["name"] for d in st.session_state.uploaded_docs}
    names.update({d["document_id"]: d["name"] for d in st.session_state.chat_uploaded_docs})
    return names


_EVIDENCE_TAG_RE = re.compile(r"\s*\[EVID:\s*E\d+\]")


def _strip_evidence_tags(text: str) -> str:
    """Strips the model's raw `[EVID: E<n>]` markers (including malformed
    space-separated runs of them, e.g. `[EVID: E14] [EVID: E15] [EVID: E16]`)
    out of the user-visible answer text — display-layer only. The tags stay
    intact in `outcome["final_answer"]` itself; custom_langgraph_poc.py's
    _finalize_outcome parses them out of that SAME raw text to build the
    citation-verification registry, and _meta_from_outcome below extracts the
    cited-evidence-id set from it too — both must see the tags, so this strip
    happens only once, right before the text is stored as chat content."""
    return _EVIDENCE_TAG_RE.sub("", text).strip()


def _meta_from_outcome(outcome: dict) -> dict:
    """Pull the bits of a run_langgraph_turn/astream_langgraph_turn
    outcome worth showing under an assistant message — token usage, both
    KB and web citations, and the orchestrator's plan. Stored alongside the
    message in session_state (not fed back to the model — the checkpointer
    handles real memory, this is purely for the UI), so it renders every
    time this message is drawn, not just right after the run.

    Subagent delegations are deliberately NOT surfaced here anymore (by
    request) — the underlying mechanism (parallel chunk-analyst/
    document-rollup calls) is unchanged in custom_langgraph_poc.py, this
    only stops the UI from displaying it.

    `cited_evidence_ids` is parsed straight from the raw final_answer text
    (same regex _finalize_outcome already uses to build its verification
    registry) — it's the set of evidence ids the model actually CITED in the
    answer, as opposed to `chunks`, which is every chunk that was retrieved
    /analyzed that turn whether or not it ended up referenced. A chunk can be
    retrieved and still be irrelevant to what the model chose to say; the
    citation-chip UI filters against this set so it only shows chunks that
    genuinely backed the answer."""
    final_answer = outcome.get("final_answer", "") or ""
    # sorted list, not a set — meta gets JSON-dumped verbatim by
    # _persist_current_chat() (chat history saved to disk), and json.dumps
    # can't serialize a set. `in` works the same on a list for the chip
    # filter below.
    cited_evidence_ids = sorted(set(re.findall(r"\[EVID:\s*(E\d+)\]", final_answer)))
    # Only mark the turn as KB-grounded when the final answer actually
    # contains verified evidence citations. The router may perform web
    # research after a previous/document-aware turn, and the raw outcome
    # can still carry KB state even when the current answer is web-only.
    # The UI must not render a stale/unrelated document citation for a
    # web-only answer.
    has_cited_kb_evidence = bool(cited_evidence_ids)
    used_knowledge_base = bool(outcome.get("used_knowledge_base", False)) and has_cited_kb_evidence

    return {
        "token_usage": outcome.get("token_usage"),
        "used_knowledge_base": used_knowledge_base,
        "chunks": outcome.get("chunks", []) if used_knowledge_base else [],
        "cited_evidence_ids": cited_evidence_ids if used_knowledge_base else [],
        "grounding_status": outcome.get("grounding_status"),
        "reasons": outcome.get("reasons", {}),
        "quality": outcome.get("quality", {}),
        "abstained": outcome.get("abstained", False),
        "abstention_reason": outcome.get("abstention_reason"),
        "used_web": outcome.get("used_web", False),
        "web_sources": outcome.get("web_sources", []),
        "used_primary_source": outcome.get("used_primary_source", False),
        "primary_source": outcome.get("primary_source", {}),
        "plan": outcome.get("plan", []),
    }


_TODO_STATUS_ICON = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}

# Light/dark pairs for the citation-chip hover tooltip. `st.html` content is
# NOT iframed (confirmed against the installed Streamlit's own docstring),
# so a pure-CSS `:hover` reveal can position/overflow freely with no JS and
# no iframe-clipping risk — the only native-widget alternative (st.popover)
# is click-triggered, not hover, so this is the "no native element provides
# this behavior" case the styling skill's own routing table carves out for
# custom HTML. Colors are picked here (not pulled from the user's real
# config.toml values) because `st.context.theme` deliberately exposes only
# `.type` ("light"/"dark"), not the actual configured colors — confirmed
# directly in the installed `streamlit.runtime.context.StreamlitTheme`
# docstring ("Theme information is restricted to the type of the theme").
_CITATION_CHIP_COLORS = {
    "light": {"chip_bg": "#eef2f1", "chip_text": "#3c4a4d", "tooltip_bg": "#ffffff",
              "tooltip_text": "#1a2226", "tooltip_border": "#d8e0de"},
    "dark": {"chip_bg": "#25302f", "chip_text": "#c9d1d3", "tooltip_bg": "#1c2428",
             "tooltip_text": "#e6edf1", "tooltip_border": "#3d4a50"},
}


def _effective_theme() -> str:
    """Return the current Streamlit theme type for server-rendered widgets.

    The visual shell can switch live via CSS/media queries, while server-side
    generated markup such as citation chips needs a concrete palette at each
    script run. Streamlit exposes the active theme as `st.context.theme.type`.
    """
    try:
        return (st.context.theme.type or "light").lower()
    except Exception:
        return "light"


def _render_citation_chips(
    chunks: list, *, grounding_status: str | None, cited_evidence_ids: list[str] | None = None
) -> None:
    """Citation chips grouped BY DOCUMENT — one chip per source document, not
    per chunk, labeled with that document's real name instead of a bare
    evidence tag. If document A backed the answer with 4 cited chunks, that's
    ONE "document A" chip whose hover tooltip lists all 4 (each with its own
    evidence id, score, and content) rather than 4 separate near-identical
    chips. The label is resolved live via `_document_names_in_scope()` (the
    same global+per-chat lookup the agent's own routing prompt uses); falls
    back to a short id fragment only for a document no longer in scope (e.g.
    since deleted) that a PAST message still cites.

    `chunks` is every chunk retrieved/analyzed this turn, not every chunk
    actually used — a chunk can be fetched and still be irrelevant to what
    the model chose to answer with (live-observed: a query retrieved 2
    chunks but the answer only cited 1). When `cited_evidence_ids` is given,
    chunks are filtered down to just those before grouping, so a chunk
    retrieved but never cited doesn't show up as if it were a citation, and
    doesn't pad out its document's tooltip either. Falls back to showing
    every chunk only if the filter would leave nothing (e.g. cited id
    parsing failed for some reason) — never hide every citation outright."""
    import html as html_lib

    colors = _CITATION_CHIP_COLORS[_effective_theme()]
    document_names = _document_names_in_scope()

    if cited_evidence_ids:
        cited_chunks = [c for c in chunks if c.evidence_id in cited_evidence_ids]
        chunks = cited_chunks or chunks

    # Group by document_id, preserving first-seen order so chip order tracks
    # the order chunks were originally retrieved/cited in, not an arbitrary
    # dict order.
    groups: dict[str, list] = {}
    doc_order: list[str] = []
    for c in chunks:
        if c.document_id not in groups:
            groups[c.document_id] = []
            doc_order.append(c.document_id)
        groups[c.document_id].append(c)

    doc_word = "document" if len(doc_order) == 1 else "documents"
    st.caption(f"📚 Citations — {grounding_status} ({len(doc_order)} {doc_word}) · hover a tag for its sources")

    chips = []
    for doc_id in doc_order:
        doc_chunks = groups[doc_id]
        label = html_lib.escape(document_names.get(doc_id) or f"doc-{str(doc_id)[:8]}")
        # Tooltip content is plain evidence text, never user-authored HTML,
        # but escaped anyway on principle — st.html also runs DOMPurify as
        # a second safety net regardless. Each chunk in the group gets its
        # own evidence id/score/content, separated by a rule so multiple
        # chunks from the same document stay visually distinct in one tooltip.
        entries = []
        for c in doc_chunks:
            content = html_lib.escape(c.content[:400])
            entries.append(
                f"<strong>{html_lib.escape(c.evidence_id)}</strong> · score={c.score:.3f}<br>{content}"
            )
        tooltip_body = '<hr class="citation-tooltip-rule">'.join(entries)
        chips.append(
            f'<span class="citation-chip">{label}'
            f'<span class="citation-tooltip">{tooltip_body}</span></span>'
        )

    st.html(f"""
<style>
.citation-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 2px; }}
.citation-chip {{
  position: relative;
  display: inline-block;
  background: {colors["chip_bg"]};
  color: {colors["chip_text"]};
  padding: 2px 9px;
  border-radius: 999px;
  font-family: ui-monospace, "SF Mono", Consolas, monospace;
  font-size: 12px;
  cursor: default;
}}
.citation-chip .citation-tooltip {{
  visibility: hidden;
  opacity: 0;
  position: absolute;
  bottom: 130%;
  left: 0;
  width: 320px;
  max-height: 220px;
  overflow-y: auto;
  background: {colors["tooltip_bg"]};
  color: {colors["tooltip_text"]};
  border: 1px solid {colors["tooltip_border"]};
  border-radius: 8px;
  padding: 10px 12px;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 12.5px;
  line-height: 1.5;
  white-space: pre-wrap;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
  transition: opacity 0.12s ease;
  z-index: 999;
}}
.citation-tooltip-rule {{
  border: none;
  border-top: 1px solid {colors["tooltip_border"]};
  margin: 8px 0;
}}
.citation-chip:hover .citation-tooltip {{ visibility: visible; opacity: 1; }}
</style>
<div class="citation-row citation-row-kb">{"".join(chips)}</div>
""")


def _render_answer_summary(meta: dict, *, idx: int, is_latest: bool) -> None:
    """Compact evidence-first summary shown above each real assistant answer."""
    quality = meta.get("quality") or {}
    grounding = str(meta.get("grounding_status") or ("Abstained" if meta.get("abstained") else "Grounded"))
    route = _route_summary(meta)
    cited_ids = set(meta.get("cited_evidence_ids") or [])
    all_chunks = meta.get("chunks") or []
    cited_chunks = [c for c in all_chunks if getattr(c, "evidence_id", None) in cited_ids] if cited_ids else []
    if not cited_chunks and all_chunks and not cited_ids:
        cited_chunks = all_chunks
    kb_count = len(cited_chunks)
    doc_count = len({str(getattr(c, "document_id", "") or "") for c in cited_chunks if getattr(c, "document_id", None)})
    web_count = len(meta.get("web_sources") or [])
    primary_used = bool(meta.get("used_primary_source"))
    coverage = quality.get("citation_coverage")
    if isinstance(coverage, (int, float)):
        coverage_label = f"{coverage:.0%}" if 0 <= coverage <= 1 else f"{coverage:.0f}%"
    else:
        coverage_label = None

    confidence = quality.get("confidence")
    confidence_text = str(confidence).strip().title() if confidence not in (None, "", "n/a", "N/A") else None
    conflicts = quality.get("evidence_conflicts") or []

    if meta.get("abstained") or grounding.lower() in {"abstained", "blocked", "failed"}:
        dot_class = "warn"
        quality_text = "Needs review"
        quality_class = "warn"
    elif grounding.lower() in {"grounded", "verified", "high"}:
        dot_class = ""
        quality_text = confidence_text or quality.get("quality_label") or "Grounded"
        quality_class = ""
    else:
        dot_class = "muted"
        quality_text = confidence_text or quality.get("quality_label") or grounding
        quality_class = "muted"

    metrics = []
    if kb_count:
        label = "passage" if kb_count == 1 else "passages"
        metrics.append(f'<span class="ef-answer-metric"><strong>{kb_count}</strong> {label}</span>')
    if doc_count:
        label = "document" if doc_count == 1 else "documents"
        metrics.append(f'<span class="ef-answer-metric"><strong>{doc_count}</strong> {label}</span>')
    if web_count:
        suffix = "" if web_count == 1 else "s"
        metrics.append(f'<span class="ef-answer-metric"><strong>{web_count}</strong> web source{suffix}</span>')
    if primary_used:
        metrics.append('<span class="ef-answer-metric"><strong>1</strong> primary source</span>')
    if coverage_label:
        metrics.append(f'<span class="ef-answer-metric"><strong>{html_lib.escape(coverage_label)}</strong> coverage</span>')
    if conflicts:
        suffix = "" if len(conflicts) == 1 else "s"
        metrics.append(f'<span class="ef-answer-metric"><strong>{len(conflicts)}</strong> conflict{suffix}</span>')

    st.markdown(
        f'''<div class="ef-answer-summary">
              <div class="ef-answer-summary-main">
                <span class="ef-answer-dot {dot_class}"></span>
                <span class="ef-answer-kicker">Evidence check</span>
                <span class="ef-answer-route">{html_lib.escape(route)}</span>
                <span class="ef-answer-quality {quality_class}">{html_lib.escape(str(quality_text))}</span>
              </div>
              <div class="ef-answer-metrics">{"".join(metrics)}</div>
            </div>''',
        unsafe_allow_html=True,
    )

    if conflicts:
        suffix = "" if len(conflicts) == 1 else "s"
        st.markdown(
            f'<div class="ef-evidence-mini"><strong>Potential evidence conflict</strong> · {len(conflicts)} cited claim{suffix} need source-level review.</div>',
            unsafe_allow_html=True,
        )

    if is_latest and (kb_count or web_count or primary_used):
        if st.button(
            "Open evidence workspace",
            icon="🔎",
            key=f"answer-evidence-{idx}",
            help="Inspect the cited passages, web sources, and agent trace for this answer",
        ):
            st.session_state.evidence_open = True
            st.rerun()


def _render_message_meta(meta: dict, idx: int) -> None:
    """`idx` (the message's position in `st.session_state.messages`) gives
    the Plan/Web-sources expanders below a stable, unique `key=` per
    message — required since every past message renders its own copy of
    these widgets in the same script run, and Streamlit requires unique
    keys. The `web-src-{idx}` key also drives the accent-border CSS rule
    (`[class*="st-key-web-src-"]`, see the global stylesheet) — same key,
    doing double duty as both a uniqueness guarantee and a CSS hook."""
    usage = meta.get("token_usage")
    if usage:
        st.caption(
            f"🔢 {usage['total_tokens']} tokens "
            f"(input: {usage['input_tokens']}, output: {usage['output_tokens']})"
        )

    plan = meta.get("plan")
    if plan:
        done = sum(1 for t in plan if t.get("status") == "completed")
        with st.expander(f"Plan ({done}/{len(plan)} done)", icon="📝", key=f"plan-{idx}"):
            for t in plan:
                icon = _TODO_STATUS_ICON.get(t.get("status"), "⬜")
                st.markdown(f"{icon} {t.get('content', '')}")

    if meta.get("used_knowledge_base"):
        _render_citation_chips(
            meta.get("chunks", []),
            grounding_status=meta.get("grounding_status"),
            cited_evidence_ids=meta.get("cited_evidence_ids"),
        )
        quality = meta.get("quality") or {}
        coverage = quality.get("citation_coverage")
        numeric = (quality.get("numeric_support") or {}).get("numeric_claims_supported")
        if meta.get("abstained"):
            st.warning("Evidence QA blocked the answer because the KB route produced no verified support.", icon="🛡️")
        elif coverage is not None:
            numeric_label = "✓ numeric/date support" if numeric is True else ("⚠ numeric/date review" if numeric is False else "— numeric/date check not applicable")
            st.caption(
                f"🛡️ Citation QA: {quality.get('quality_label', 'UNKNOWN')} · "
                f"{coverage:.1f}% cited evidence verified · {numeric_label}"
            )
            conflicts = quality.get("evidence_conflicts") or []
            if conflicts:
                st.warning(
                    f"Possible evidence conflict detected across {len(conflicts)} cited claim(s). "
                    "Review the source passages before relying on the value.",
                    icon="⚠️",
                )

    if meta.get("used_web"):
        sources = meta.get("web_sources", [])
        with st.expander(f"Web sources ({len(sources)})", icon="🌐", key=f"web-src-{idx}"):
            for s in sources:
                st.markdown(f"- [{s['title'] or s['url']}]({s['url']}) — score={s['score']:.3f}")

    if meta.get("used_primary_source"):
        primary = meta.get("primary_source", {}) or {}
        citation = primary.get("citation") or {}
        quality = primary.get("quality") or {}
        with st.expander("Primary-source MCP evidence", icon="🏛️", key=f"primary-src-{idx}"):
            st.markdown(f"**Agent:** `{primary.get('agent', 'unknown')}`  \n**Action:** `{primary.get('action', 'unknown')}`")
            if citation:
                source_name = citation.get("source_name") or citation.get("source") or "Primary source"
                source_url = citation.get("source_url") or citation.get("url") or ""
                st.markdown(f"**Source:** {source_name}")
                if source_url:
                    st.markdown(f"**URL:** {source_url}")
                if citation.get("retrieved_at"):
                    st.markdown(f"**Retrieved:** {citation['retrieved_at']}")
                if citation.get("data_hash"):
                    st.code(str(citation["data_hash"]), language="text")
            if quality:
                st.caption(f"Freshness: {quality.get('freshness_seconds', 'n/a')}s · Confidence: {quality.get('confidence', 'n/a')}")


def _record_upload(name: str, document_id: str, chunk_count: int, content_hash: str | None = None) -> None:
    """GLOBAL upload record (sidebar uploader only — see module docstring).
    Persisted immediately to its own file, independent of any one chat's
    own save."""
    st.session_state.uploaded_docs.append(
        {"name": name, "document_id": document_id, "chunk_count": chunk_count, "content_hash": content_hash}
    )
    _save_uploaded_docs()  # global registry — persisted immediately, not tied to any one chat's save
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"📄 Added **{name}** to the knowledge base ({chunk_count} chunks). "
            "You can ask about it now.",
            # "notice", not a real turn — see the history loop's own comment
            # on why Copy/Regenerate are hidden for these (there's no
            # question here to "regenerate", and nothing meaningful to copy).
            "kind": "notice",
        }
    )


def _record_chat_upload(name: str, document_id: str, chunk_count: int, content_hash: str | None = None) -> None:
    """PER-CHAT upload record (attach-in-chat only — see module docstring).
    Not persisted here directly — the caller already calls
    `_persist_current_chat()` after every turn/attach, which is what
    actually writes `chat_uploaded_docs` into this thread's own file."""
    st.session_state.chat_uploaded_docs.append(
        {"name": name, "document_id": document_id, "chunk_count": chunk_count, "content_hash": content_hash}
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"📄 Added **{name}** to this chat's knowledge base ({chunk_count} chunks). "
            "You can ask about it now.",
            "kind": "notice",
        }
    )


def _run_status_check() -> dict:
    """Cheap, timeout-bounded self-check for the sidebar "Status" section —
    a raw TCP connect to Qdrant, not a full API round trip, so a slow or
    unreachable instance can't stall page load itself. Added after this
    session's own real incidents (a Docker port conflict, a misconfigured
    env var) that only ever surfaced as a raw exception mid-chat — this
    surfaces the same class of problem before the user asks anything."""
    import socket
    from urllib.parse import urlparse

    config = load_runtime_config()
    parsed = urlparse(config.vector_db.qdrant_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    qdrant_ok = False
    qdrant_error = "unreachable"
    try:
        with socket.create_connection((host, port), timeout=1.5):
            qdrant_ok = True
            qdrant_error = ""
    except OSError as exc:
        qdrant_error = f"unreachable ({exc.__class__.__name__})"
    return {
        "qdrant_ok": qdrant_ok,
        "qdrant_error": qdrant_error,
        "web_search_configured": bool((config.web_search.web_search_api_key or "").strip()),
        "katzilla_configured": bool(os.environ.get("KATZILLA_API_KEY", "").strip()) and os.environ.get("KATZILLA_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
    }


with st.sidebar:
    st.markdown(
        '<div class="ef-sidebar-brand"><div class="ef-brand-mark">✦</div><div><strong>EvidenceFlow</strong><span>Agentic RAG workspace</span></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ef-agent-status"><span></span><div><strong>Agent routing active</strong><small>Tools selected automatically</small></div></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "New chat",
        type="primary",
        use_container_width=True,
        icon="➕",
        shortcut="Alt+N",  # not Ctrl+N: reserved by every major browser for a new window
    ):
        st.session_state.messages = []
        # PER-CHAT documents reset — a new chat genuinely has none yet (see
        # module docstring). The GLOBAL sidebar-uploaded list is untouched;
        # it's not a property of any one chat. New thread_id, NOT a new
        # agent: the checkpointer's memory is scoped by thread_id, so the
        # same built agent can start a fresh conversation just by talking to
        # it on a new thread — no need to rebuild tools for every "New chat"
        # click. The OLD thread stays fully intact (both in the checkpointer
        # and in its own chat_history/<thread_id>.json — it was persisted
        # after its own last turn), so switching away doesn't lose it.
        st.session_state.chat_uploaded_docs = []
        st.session_state.chat_ingested_hashes = set()
        st.session_state.thread_id = uuid.uuid4().hex
        st.rerun()

    st.divider()
    st.markdown(f'<div class="sidebar-heading">{_icon("history", size=18)}Chats</div>', unsafe_allow_html=True)
    _threads = sorted(
        st.session_state.chat_store["threads"],
        key=lambda t: t.get("updated_at", ""),
        reverse=True,
    )
    if _threads:
        for _t in _threads:
            _is_current = _t["thread_id"] == st.session_state.thread_id
            _col_chat, _col_del = st.columns([5, 1])
            with _col_chat:
                if st.button(
                    ("💬 " if _is_current else "") + (_t.get("title") or "Untitled chat"),
                    key=f"chat_{_t['thread_id']}",
                    use_container_width=True,
                    disabled=_is_current,
                ):
                    _switch_to_chat(_t["thread_id"])
                    st.rerun()
            with _col_del:
                if st.button("", key=f"del_chat_{_t['thread_id']}", icon="🗑️", help="Delete this chat"):
                    _delete_chat(_t["thread_id"])
                    st.rerun()
    else:
        st.caption("No past chats yet.")

    st.divider()
    st.markdown(
        f'<div class="sidebar-heading">{_icon("books", size=18)}Documents (all chats)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Uploaded here: available from every chat, survives deleting chats.")
    sidebar_file = st.file_uploader(
        "Upload a .txt/.md/.csv/.pdf/.docx document",
        type=["txt", "md", "csv", "pdf", "docx", "doc"],
        key=f"sidebar_uploader_{st.session_state.sidebar_uploader_key}",
    )
    # Auto-ingests on selection, no separate confirm click — matches the
    # chat-input attach path, which was already automatic; the extra button
    # here used to be the one inconsistent step between the two upload
    # entry points. Bumping the widget key + rerun afterwards (same
    # mechanism as before) is what stops this from re-ingesting the same
    # file again on the next unrelated rerun — except the file_uploader
    # widget can still re-fire with its old value across a hot-reload (seen
    # live: editing this file while a doc sat in the uploader caused it to
    # re-submit once on the next reconnect), so content-hash dedup below is
    # the actual guarantee, not just the key bump.
    if sidebar_file is not None:
        content = sidebar_file.read()
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash not in st.session_state.ingested_hashes:
            with st.spinner(f"Adding {sidebar_file.name}..."):
                try:
                    result = _run_async(ingest_document(content, source_name=sidebar_file.name))
                except Exception as exc:  # noqa: BLE001 - POC diagnostic surface
                    # Appended as a real message, not `st.error(...)` — this
                    # path always ends in `st.rerun()` a few lines down,
                    # which redraws the whole page from scratch; a bare
                    # `st.error` shown THIS run vanishes the instant that
                    # rerun happens (seen live: "an error but it disappeared
                    # instantly"). Only something in session_state survives
                    # a rerun.
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"⚠️ Couldn't ingest {sidebar_file.name}: {exc!r}",
                            "kind": "notice",
                        }
                    )
                else:
                    st.session_state.ingested_hashes.add(content_hash)
                    _record_upload(
                        sidebar_file.name, result["document_id"], result["chunk_count"], content_hash
                    )
                    # st.toast survives exactly one rerun (Streamlit's own
                    # documented behavior) — the `st.rerun()` a few lines
                    # down is that one rerun, so this still shows.
                    st.toast(f"Added {sidebar_file.name} ✓", icon="✅")
        st.session_state.sidebar_uploader_key += 1  # fresh, empty uploader next run either way
        _persist_current_chat()  # this path reruns immediately, before reaching the bottom of the script
        st.rerun()

    if st.session_state.uploaded_docs:
        for d in st.session_state.uploaded_docs:
            _col_doc, _col_del_doc = st.columns([5, 1])
            with _col_doc:
                st.markdown(
                    f'<div class="doc-chip">{_icon("file-text", size=15)}'
                    f"{d['name']} — {d['chunk_count']} chunks</div>",
                    unsafe_allow_html=True,
                )
            with _col_del_doc:
                if st.button("", key=f"del_doc_{d['document_id']}", icon="🗑️", help="Delete from the knowledge base"):
                    _delete_document(d["document_id"])
                    _persist_current_chat()
                    st.rerun()
    else:
        st.caption("None uploaded yet.")

    st.divider()
    st.markdown(
        f'<div class="sidebar-heading">{_icon("file-text", size=18)}Documents (this chat)</div>',
        unsafe_allow_html=True,
    )
    st.caption("Attached in chat: only visible here, deleted with this chat.")
    if st.session_state.chat_uploaded_docs:
        for d in st.session_state.chat_uploaded_docs:
            _col_cdoc, _col_del_cdoc = st.columns([5, 1])
            with _col_cdoc:
                st.markdown(
                    f'<div class="doc-chip">{_icon("file-text", size=15)}'
                    f"{d['name']} — {d['chunk_count']} chunks</div>",
                    unsafe_allow_html=True,
                )
            with _col_del_cdoc:
                if st.button(
                    "",
                    key=f"del_chat_doc_{d['document_id']}",
                    icon="🗑️",
                    help="Delete from this chat's knowledge base",
                ):
                    _delete_chat_document(d["document_id"])
                    _persist_current_chat()
                    st.rerun()
    else:
        st.caption("None attached in this chat yet.")

    st.divider()
    # Deliberately away from "New chat" (which is a frequent, low-stakes
    # action) and behind a confirmation dialog — explicit-action only, no
    # automatic "on browser close" trigger (see _clear_everything's
    # docstring for why that was ruled out).
    if st.button(
        "Clear everything",
        width="stretch",
        icon="🗑️",
        help="Delete every chat and every document, permanently",
    ):
        _confirm_clear_everything()

    st.divider()
    st.markdown(f'<div class="sidebar-heading">{_icon("search", size=18)}Status</div>', unsafe_allow_html=True)
    # Worth having after today's real incidents: a Qdrant/backend hiccup
    # used to only surface mid-chat, as a raw exception in the middle of a
    # turn. A cheap self-check at page load surfaces it here instead,
    # before the user even asks a question. Cached in session_state (not
    # re-checked on every rerun/keystroke) since it's a real network call —
    # a manual "Recheck" button below re-runs it on demand.
    if "status_check" not in st.session_state or st.button("Recheck", icon="🔄", help="Re-run the connectivity check"):
        st.session_state.status_check = _run_status_check()
    _status = st.session_state.status_check
    st.markdown(
        f'<div class="doc-chip">{_icon("robot", size=15)}Backend: <code>{_BACKEND}</code></div>'
        f'<div class="doc-chip">{_icon("search" if _status["qdrant_ok"] else "search-off", size=15)}'
        f'Qdrant: {"connected" if _status["qdrant_ok"] else _status["qdrant_error"]}</div>'
        f'<div class="doc-chip">{_icon("world", size=15)}'
        f'Web search: {"configured" if _status["web_search_configured"] else "not configured"}</div>'
        f'<div class="doc-chip">🏛️ Primary-source MCP: {"configured" if _status.get("katzilla_configured") else "not configured"}</div>',
        unsafe_allow_html=True,
    )



def _get_or_build_agent() -> dict:
    """Build once per session and cache — see module docstring "Real
    conversation memory". Checkpointer is the disk-backed `AsyncSqliteSaver`
    from `_build_checkpointer`, not `InMemorySaver` — an existing thread_id
    (from a past chat, even from a previous server run) resumes its real
    memory instead of starting empty. Unlike the old `InMemorySaver` version,
    this does NOT reset `thread_id` on build: with a disk-backed
    checkpointer, `st.session_state.thread_id` (set at startup, or by
    "New chat"/switching to a past chat in the sidebar) already names a
    valid thread to resume — forcing a fresh one here would silently
    discard that choice."""
    gateway_token = str(st.session_state.get("llm_gateway_session_token", "")).strip()
    if st.session_state.built_agent is None or st.session_state.get("built_agent_gateway_token", "") != gateway_token:
        checkpointer = _run_async(_build_checkpointer())
        st.session_state.built_agent = build_langgraph_agent(
            document_ids_provider=_document_ids_in_scope,
            document_names_provider=_document_names_in_scope,
            backend=_BACKEND,
            checkpointer=checkpointer,
            gateway_token=gateway_token,
        )
        st.session_state.built_agent_gateway_token = gateway_token
    return st.session_state.built_agent


# Interactive evidence workspace. It is a presentation/action layer over the
# existing citation metadata: inspect, locally re-verify, or challenge an item
# with a follow-up research turn. The underlying LangGraph/RAG behavior stays unchanged.
def _latest_meta() -> dict | None:
    return next(
        (m.get("meta") for m in reversed(st.session_state.messages)
         if m.get("role") == "assistant" and m.get("meta")),
        None,
    )


def _evidence_registry_for_meta(meta: dict) -> dict:
    chunks = meta.get("chunks") or []
    eid_to_doc = {}
    eid_to_content = {}
    eid_to_meta = {}
    for c in chunks:
        eid = getattr(c, "evidence_id", None)
        if not eid:
            continue
        eid_to_doc[eid] = getattr(c, "document_id", "") or ""
        eid_to_content[eid] = getattr(c, "content", "") or ""
        eid_to_meta[eid] = {"retrieval_score": getattr(c, "score", None)}
    return {"eid_to_doc": eid_to_doc, "eid_to_content": eid_to_content, "eid_to_meta": eid_to_meta}


def _route_summary(meta: dict | None) -> str:
    if not meta:
        return "Awaiting a research turn"
    parts=[]
    if meta.get("used_knowledge_base"): parts.append("Knowledge base")
    if meta.get("used_web"): parts.append("Web")
    if meta.get("used_primary_source"): parts.append("MCP")
    return " + ".join(parts) if parts else "No external source route"


def _render_evidence_panel() -> None:
    meta = _latest_meta()
    if not meta or not st.session_state.get("evidence_open", False):
        return

    st.markdown('<div class="ef-evidence-open"></div>', unsafe_allow_html=True)
    header_col, close_col = st.columns([8, 1])

    with header_col:
        st.markdown(
            '<div class="ef-evidence-title">Verified evidence</div>',
            unsafe_allow_html=True,
        )
        grounding = meta.get("grounding_status") or (
            "Abstained" if meta.get("abstained") else "Grounded"
        )
        st.markdown(
            f'<div class="ef-evidence-sub">Claim → evidence → source · '
            f'<strong>{html_lib.escape(str(grounding))}</strong></div>',
            unsafe_allow_html=True,
        )

    with close_col:
        if st.button(
            "",
            icon=":material/close:",
            help="Close evidence workspace",
            key="close-evidence-panel",
            use_container_width=True,
        ):
            st.session_state.evidence_open = False
            st.rerun()

    cited=set(meta.get("cited_evidence_ids") or [])
    chunks=[c for c in (meta.get("chunks") or []) if not cited or getattr(c,"evidence_id",None) in cited]
    web_sources=meta.get("web_sources") or []
    quality=meta.get("quality") or {}
    coverage=quality.get("citation_coverage")
    verified_count=meta.get("verified_count")
    proposed_count=meta.get("proposed_count")
    if coverage is not None:
        try: cov=f"{float(coverage):.0%}"
        except (TypeError,ValueError): cov="—"
    else: cov=str(verified_count if verified_count is not None else 0)
    stats=[(str(len(chunks)),"cited passages"),(str(len(web_sources)),"web sources"),(cov,"coverage")]
    st.markdown('<div class="ef-evidence-summary">'+''.join(f'<div class="ef-evidence-stat"><strong>{html_lib.escape(v)}</strong><span>{html_lib.escape(lbl)}</span></div>' for v,lbl in stats)+'</div>',unsafe_allow_html=True)

    tab_evidence,tab_sources,tab_trace=st.tabs(["Evidence","Sources","Trace"])
    with tab_evidence:
        if not chunks:
            st.info("No knowledge-base evidence was cited in the latest answer.")
        for idx,c in enumerate(chunks[:8]):
            eid=getattr(c,"evidence_id",None) or f"E{idx+1}"
            content=str(getattr(c,"content","") or "").strip()
            selected=st.session_state.get("selected_evidence_id")==eid
            check=st.session_state.get("evidence_checks",{}).get(eid)
            badge="Verified" if (check or {}).get("status")=="verified" else ("Needs review" if check else "Backend verified")
            doc_id=str(getattr(c,"document_id","") or "")
            st.markdown(f'<div class="ef-evidence-card {"selected" if selected else ""}"><strong>{html_lib.escape(eid)}</strong><p>{html_lib.escape(content[:420])}{"…" if len(content)>420 else ""}</p><span class="ef-evidence-badge">{html_lib.escape(badge)}</span><div class="ef-evidence-source">{html_lib.escape(doc_id or "Source passage")}</div></div>',unsafe_allow_html=True)
            if st.button("Inspect passage",key=f"inspect-evidence-{eid}",use_container_width=True):
                st.session_state.selected_evidence_id=eid; st.rerun()
            if selected:
                st.caption(f"Evidence {eid} · retrieval score: {getattr(c,'score',0.0):.3f}")
                st.markdown(content)
        if quality.get("evidence_conflicts"):
            st.warning("Potential evidence conflict detected. The agent has flagged this for review; inspect the cited passages below for the supporting context.")
        if meta.get("abstained"):
            st.error("The answer was withheld because sufficient verified evidence was not available.")

    with tab_sources:
        if chunks:
            st.markdown("**Knowledge-base sources**")
            for idx,c in enumerate(chunks[:8]):
                doc_id=str(getattr(c,"document_id","") or f"Document {idx+1}")
                eid=str(getattr(c,"evidence_id",None) or f"E{idx+1}")
                st.markdown(f'<div class="ef-evidence-card"><strong>{html_lib.escape(doc_id)}</strong><div class="ef-evidence-source">Referenced by {html_lib.escape(eid)}</div></div>',unsafe_allow_html=True)
        if web_sources:
            st.markdown("**Web sources**")
            for idx,src in enumerate(web_sources[:6]):
                title=str(src.get("title") or src.get("url") or f"Web source {idx+1}"); url=str(src.get("url") or "")
                st.markdown(f'<div class="ef-evidence-card"><strong>WEB {idx+1}</strong><p>{html_lib.escape(title)}</p></div>',unsafe_allow_html=True)
                if url: st.link_button("Open source ↗",url,use_container_width=True)
        if not chunks and not web_sources: st.info("No source records are available for this turn.")

    with tab_trace:
        st.markdown("<div class='ef-trace'>",unsafe_allow_html=True)
        route=_route_summary(meta)
        steps=[f"Route selected: {route}",f"Retrieved {len(meta.get('chunks') or [])} candidate passages",f"Cited passages: {len(chunks)}",f"Verification: {meta.get('grounding_status') or ('Abstained' if meta.get('abstained') else 'Grounded')}",f"Web research: {'used' if meta.get('used_web') else 'not used'}"]
        for step in steps: st.markdown(f'<div class="ef-trace-step">{html_lib.escape(step)}</div>',unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)
        if verified_count is not None or proposed_count is not None: st.caption(f"{verified_count or 0} verified / {proposed_count or 0} proposed evidence references")


# Main workspace + conditional evidence workspace. The app's left sidebar remains
# the navigation/document control plane; this area is the product workspace.
# Main workspace + conditional evidence workspace. The agent owns routing: the UI intentionally
# does not expose competing research modes.
_latest = _latest_meta()
_has_messages = bool(st.session_state.messages)
_show_evidence = bool(_latest and st.session_state.get("evidence_open", False))
if _show_evidence:
    _main_col, _evidence_col = st.columns([7.5, 2.8], gap="large")
else:
    _main_col = st.container()
    _evidence_col = None

if _has_messages:
    st.markdown('<div class="ef-chat-active"></div>', unsafe_allow_html=True)
else:
    # The shared EvidenceFlow header is rendered by ui_theme. Keep the empty
    # state focused on the product action rather than repeating the brand.
    pass

with _main_col:
    if _has_messages:
        st.markdown(
            '<div class="ef-chat-topbar"><div><strong>EvidenceFlow</strong><span>Verified research workspace</span></div>'
            '<div class="ef-agent-badge"><span></span>Agent decides the route</div></div>',
            unsafe_allow_html=True,
        )
    if not _has_messages:
        _doc_count = len(st.session_state.uploaded_docs) + len(st.session_state.chat_uploaded_docs)
        _thread_count = len(st.session_state.chat_store.get("threads", []))
        _status_ok = st.session_state.get("status_check", {}).get("qdrant_ok", True)
        _status_label = "Ready" if _status_ok else "Check backend"
        st.markdown(
            f"""
            <section class="ef-landing">
              <div class="ef-landing-hero">
                <div class="ef-landing-copy">
                  <div class="ef-landing-kicker">AGENTIC RESEARCH ENGINE</div>
                  <div class="ef-landing-title">Ask anything.<br><span>Let the agent find the evidence.</span></div>
                  <div class="ef-landing-sub">EvidenceFlow routes each question across your knowledge base, the web, and trusted primary sources, then carries evidence verification into the final answer.</div>
                  <div class="ef-landing-pills">
                    <span class="ef-landing-pill"><i></i>Agentic routing</span>
                    <span class="ef-landing-pill"><i></i>Hybrid retrieval</span>
                    <span class="ef-landing-pill"><i></i>Evidence verification</span>
                    <span class="ef-landing-pill"><i></i>Persistent threads</span>
                  </div>
                </div>
                <div class="ef-route-card">
                  <div class="ef-route-head"><strong>HOW EVIDENCEFLOW WORKS</strong><span class="ef-ready">{html_lib.escape(_status_label.upper())}</span></div>
                  <div class="ef-route-flow">
                    <div class="ef-route-node"><b>01</b><span>Understand the question</span></div>
                    <div class="ef-route-arrow">↓</div>
                    <div class="ef-route-node"><b>02</b><span>Select the best evidence path</span></div>
                    <div class="ef-route-arrow">↓</div>
                    <div class="ef-route-node"><b>03</b><span>Retrieve + verify evidence</span></div>
                    <div class="ef-route-arrow">↓</div>
                    <div class="ef-route-node"><b>04</b><span>Synthesize a grounded answer</span></div>
                  </div>
                </div>
              </div>
              <div class="ef-benchmark-strip">
                <div class="ef-benchmark-card"><strong>Adaptive retrieval</strong><p>Tools are chosen per question instead of by a manual mode.</p></div>
                <div class="ef-benchmark-card"><strong>Parallel evidence</strong><p>Relevant passages can be analyzed concurrently for grounding.</p></div>
                <div class="ef-benchmark-card"><strong>Verified answers</strong><p>Citations and evidence checks stay attached to the response.</p></div>
                <div class="ef-benchmark-card"><strong>Research memory</strong><p>Persistent threads let investigations continue across sessions.</p></div>
              </div>
              <div class="ef-workspace-ready">
                <div class="ef-ready-copy"><strong>Research workspace ready</strong><p>Upload a document, or start with a question. The agent decides what to use.</p></div>
                <div class="ef-live-metric"><span>Knowledge sources</span><strong>{_doc_count}</strong></div>
                <div class="ef-live-metric"><span>Research threads</span><strong>{_thread_count}</strong></div>
                <div class="ef-live-metric"><span>Backend</span><strong>{html_lib.escape(_BACKEND)}</strong></div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    _ASSISTANT_AVATAR = "🤖"
    _last_idx = len(st.session_state.messages) - 1

    for idx, msg in enumerate(st.session_state.messages):
        avatar = _ASSISTANT_AVATAR if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("meta"):
                _render_answer_summary(msg["meta"], idx=idx, is_latest=(idx == _last_idx))
                st.markdown('<div class="ef-answer-body">', unsafe_allow_html=True)
                st.write(msg["content"])
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.write(msg["content"])
            if msg.get("meta"):
                _render_message_meta(msg["meta"], idx)

            # ALLOW-list, deliberately — not "show buttons unless this looks
            # like a notice" (a deny-list), which is what this used to be and
            # kept needing another patch every time one more notice-message
            # site turned up untagged (upload confirmations, then delete
            # confirmations, ...): a real question sits behind a genuine
            # answer or a turn-level error, so those two cases are enumerated
            # explicitly, and everything else — including any FUTURE system
            # notice this file adds and forgets to tag — safely defaults to no
            # action row, rather than defaulting to showing one.
            # `"meta"` marks a real successful answer; `"kind": "turn_error"`
            # marks a real turn that failed (see the `except` block above) —
            # both appended only from inside the actual turn-handling flow,
            # never from an upload/delete helper. The content-prefix check
            # covers turn errors saved to `chat_history/<thread_id>.json`
            # before `"kind": "turn_error"` existed (old wording, no tag).
            _is_real_turn = bool(msg.get("meta")) or msg.get("kind") == "turn_error" or msg["content"].startswith(
                "⚠️ Sorry,"
            )
            if msg["role"] == "assistant" and _is_real_turn:
                # Regenerate: only offered on the LAST message, and deliberately
                # NOT a true history rewrite — the checkpointer (what the MODEL
                # remembers, see module docstring "Real conversation memory")
                # still has the original turn in it; this just asks the same
                # question again as a fresh turn and swaps the UI-visible
                # answer. Offering it on an EARLIER message would leave the
                # checkpointer's memory permanently out of sync with what the
                # UI shows from that point on, so it's scoped out rather than
                # built half-right.
                is_last_assistant = idx == _last_idx

                # Icon-only, hover-tooltip design — same "icon + native title
                # attribute" pattern the browser already gives Streamlit's own
                # `help=` tooltip on Regenerate, just written by hand for the
                # Copy button since it isn't a Streamlit widget. `title="..."`
                # is a plain HTML attribute, not JS — no extra library or
                # component needed for the hover behavior.
                #
                # Real one-click clipboard copy via a plain HTML button with an
                # `onclick` handler — not a Streamlit widget at all, so there's
                # no rerun/callback involved. Inline event-handler ATTRIBUTES
                # (onclick="...") execute normally wherever the browser parses
                # them, unlike a `<script>` TAG injected via innerHTML (which
                # browsers block) — confirmed this is how Streamlit inserts
                # `unsafe_allow_html` content (a direct DOM insertion, not a
                # sandboxed component iframe), so this isn't the fragile case
                # an iframe-based `st.html` clipboard call would be.
                # Double-escaped, deliberately: `json.dumps` first produces a
                # valid JS string literal (handles quotes/newlines/backslashes
                # inside the message text), then `html.escape` makes THAT safe
                # to sit inside an HTML attribute (handles a literal `"` in the
                # JS-string output from breaking out of the attribute early) —
                # skipping either step is a real injection risk, not a
                # theoretical one, given this text can contain a model's or a
                # document's own content. Click feedback swaps the ICON itself
                # (📋 -> ✅ -> back), not button text, since there's no text
                # label anymore to swap.
                _copy_payload = html_lib.escape(json.dumps(msg["content"]), quote=True)
                _copy_html = (
                    '<div class="msg-actions">'
                    f'<button class="copy-native-btn" title="Copy to clipboard" '
                    f"onclick=\"navigator.clipboard.writeText({_copy_payload}).then(()=>{{"
                    f"this.innerText='✅';setTimeout(()=>this.innerText='📋',1400)}})"
                    f'.catch(()=>{{this.innerText=\'⚠️\'}})">📋</button></div>'
                )
                if is_last_assistant:
                    # Explicit `st.columns` (not just two stacked elements) —
                    # Streamlit lays out top-level calls vertically by default,
                    # so without this the HTML Copy button and the Regenerate
                    # widget would each land on their own row regardless of how
                    # narrow either one is. Both are small, icon-only, content-
                    # width buttons now (no long label text to size a column
                    # around anymore), so a narrow shared column pair followed
                    # by a wide spacer keeps them snug together on the left
                    # rather than stretched across the row.
                    col_copy, col_regen, _col_spacer = st.columns([1, 1, 10])
                    with col_copy:
                        st.markdown(_copy_html, unsafe_allow_html=True)
                    with col_regen:
                        if st.button("", key=f"regen_{idx}", icon="🔁", help="Ask this again"):
                            prev_user_text = next(
                                (m["content"] for m in reversed(st.session_state.messages[:idx]) if m["role"] == "user"),
                                None,
                            )
                            if prev_user_text:
                                st.session_state.messages.pop(idx)
                                st.session_state["_pending_regenerate"] = prev_user_text
                                st.rerun()
                else:
                    st.markdown(_copy_html, unsafe_allow_html=True)

    user_input = st.chat_input(
        "Message the assistant, or attach a document to add it to the knowledge base...",
        accept_file="multiple",
        file_type=["txt", "md", "csv", "pdf", "docx", "doc"],
    )

    # A "Regenerate" click on the last message (see the history loop above)
    # stashes the preceding user text here and reruns — picked up as if it were
    # fresh chat_input, EXCEPT the user bubble/message-append below is skipped
    # for it (that turn's user message is already in session_state; regenerate
    # only replaces the assistant's reply, never duplicates the question).
    _pending_regenerate = st.session_state.pop("_pending_regenerate", None)
    is_regenerate = _pending_regenerate is not None
    if is_regenerate:
        text, files = _pending_regenerate, []
    elif user_input:
        text = user_input if isinstance(user_input, str) else user_input.text
        files = [] if isinstance(user_input, str) else user_input.files
    else:
        text, files = None, []

    if text or files:
        # Attach-in-chat is PER-CHAT scope (see module docstring) — uses
        # chat_ingested_hashes/_record_chat_upload, not the global ones the
        # sidebar uploader uses.
        for f in files:
            content = f.read()
            content_hash = hashlib.sha256(content).hexdigest()
            if content_hash in st.session_state.chat_ingested_hashes:
                continue  # already attached in this chat — skip silently
            with st.spinner(f"Adding {f.name} to this chat's knowledge base..."):
                try:
                    result = _run_async(ingest_document(content, source_name=f.name))
                except Exception as exc:  # noqa: BLE001 - POC diagnostic surface
                    st.session_state.messages.append(
                        {"role": "assistant", "content": f"Couldn't ingest {f.name}: {exc!r}", "kind": "notice"}
                    )
                    continue
            st.session_state.chat_ingested_hashes.add(content_hash)
            _record_chat_upload(f.name, result["document_id"], result["chunk_count"], content_hash)
            st.toast(f"Added {f.name} ✓", icon="✅")

        if text and text.strip():
            _display_text = text
            # The graph is genuinely agentic: the router chooses KB, web,
            # primary-source MCP, multiple tools, or no tool at all. The UI
            # deliberately passes the user's request through unchanged.
            _agent_text = text
            if not is_regenerate:
                st.session_state.messages.append({"role": "user", "content": _display_text})
                with st.chat_message("user"):
                    st.write(_display_text)

            _TYPING_HTML = (
                '<div class="typing-indicator"><span></span><span></span><span></span></div>'
            )
            # Rough, deliberately approximate step count for the progress bar
            # below — turns don't all hit the same nodes (a pure web-search
            # turn never touches `analyze_chunk`, a single-document answer
            # skips `summarize_document`), so there's no single "total steps"
            # to compute honestly ahead of time. Treated as a visual "still
            # moving forward" signal, not a literal percentage — capped short
            # of 100% until the turn actually finishes.
            _PROGRESS_STEP_GUESS = 4

            async def _consume(built, agent_query):
                import html as html_lib

                status_box = st.status("Working...", expanded=False)
                progress_box = st.empty()
                text_box = st.empty()
                progress_box.progress(0.05, text=None)
                # Animated dots instead of a bare blank area — communicates
                # "alive, still working" for the gap between the status updates
                # above and the first real token arriving below.
                text_box.markdown(_TYPING_HTML, unsafe_allow_html=True)
                accumulated = ""
                steps_seen = 0
                final_outcome = None
                async for event in astream_langgraph_turn(built, agent_query, thread_id=st.session_state.thread_id):
                    if event["type"] == "status":
                        status_box.update(label=event["text"])
                        steps_seen += 1
                        progress_box.progress(min(0.9, steps_seen / _PROGRESS_STEP_GUESS))
                    elif event["type"] == "token":
                        accumulated += event["text"]
                        # Escaped (HTML-special chars only, quotes left alone so
                        # rare markdown title-attribute quotes still render) +
                        # a blinking caret, for the LIVE in-progress view only —
                        # the final render below reverts to the exact original
                        # plain `st.markdown(accumulated)` call, unescaped, so
                        # the persisted message's formatting is byte-for-byte
                        # what it always was.
                        text_box.markdown(
                            html_lib.escape(accumulated, quote=False) + ' <span class="stream-caret">▌</span>',
                            unsafe_allow_html=True,
                        )
                    elif event["type"] == "final":
                        final_outcome = event
                progress_box.progress(1.0)
                progress_box.empty()
                text_box.markdown(accumulated)  # final, unescaped render — see comment above
                status_box.update(label="Done", state="complete")
                return final_outcome

            with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
                outcome = None
                error_text = None
                try:
                    # Agent/checkpointer construction moved inside this same
                    # try/except (and this same visible chat bubble) — it used
                    # to run before this block, unguarded and with no status
                    # indicator, so a real failure or just a slow first-ever
                    # `chat_history/checkpoints.db` setup looked like "nothing happens" with
                    # zero feedback. First call after a fresh install/restart
                    # does real disk I/O now (AsyncSqliteSaver), unlike the old
                    # instant, zero-I/O `InMemorySaver`.
                    with st.spinner("Setting up..."):
                        built = _get_or_build_agent()
                    outcome = _run_async(_consume(built, _agent_text))
                except asyncio.TimeoutError:
                    error_text = (
                        f"⚠️ Sorry, this took longer than {_TURN_TIMEOUT_SECONDS}s and was stopped "
                        "(backend or retrieval likely stalled) — please try again."
                    )
                except Exception as exc:  # noqa: BLE001 - POC diagnostic surface
                    LOGGER.exception("LangGraph turn failed", exc_info=exc)
                    tb = traceback.format_exc()
                    error_text = f"⚠️ Sorry, that run failed: {exc!r}"
                    # Keep the user-facing message clean, but expose the actual
                    # exception type/traceback in an expandable diagnostic area
                    # so hosted deployments don't collapse everything into the
                    # unhelpful `UnexpectedResponse()` string.
                    with st.expander("Technical error details", expanded=True):
                        st.code(tb, language="text")

                # Appended to session_state, not left as a bare `st.error(...)`
                # call — this block always ends in `st.rerun()` a few lines
                # down, which redraws the whole page from scratch; anything
                # rendered only in THIS run's DOM (a transient `st.error`)
                # vanishes the instant that rerun fires. Seen live: "an error
                # but it disappeared instantly." A message in session_state
                # survives the rerun and renders every time, like any other
                # chat bubble.
                if error_text is not None:
                    st.error(error_text)  # still shown immediately in this run, for zero-latency feedback
                    # "kind": "turn_error" — a REAL turn failure (timeout,
                    # exception), not a system notice: there IS a real question
                    # behind it, so unlike upload/delete notices this correctly
                    # keeps Copy/Regenerate (retry) available — see the history
                    # loop's own comment on why this is now an allow-list, not
                    # a deny-list.
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_text, "kind": "turn_error"}
                    )

            if outcome is not None:
                # meta is built from the RAW (tagged) final_answer first — it
                # needs the `[EVID: E<n>]` markers intact to know which chunks
                # were actually cited (see _meta_from_outcome) — only the stored
                # chat `content` gets the tags stripped, so the bubble itself
                # never shows raw "[EVID: E1]" text to the user.
                meta = _meta_from_outcome(outcome)
                if (meta.get("chunks") or meta.get("web_sources") or meta.get("used_primary_source")):
                    st.session_state.evidence_open = True
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": _strip_evidence_tags(outcome["final_answer"]),
                        "meta": meta,
                    }
                )

        _persist_current_chat()  # covers both the upload-only and text-message paths above
        st.rerun()


if _evidence_col is not None:
    with _evidence_col:
        _render_evidence_panel()
