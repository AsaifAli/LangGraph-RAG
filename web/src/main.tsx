import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Evidence = { evidence_id?: string; document_id?: string; content?: string; score?: number; source?: string };
type Meta = {
  token_usage?: { total_tokens?: number; input_tokens?: number; output_tokens?: number } | null;
  used_knowledge_base?: boolean;
  chunks?: Evidence[];
  cited_evidence_ids?: string[];
  grounding_status?: string | null;
  quality?: { quality_label?: string; citation_coverage?: number; numeric_support?: { numeric_claims_supported?: boolean | null }; evidence_conflicts?: unknown[] };
  abstained?: boolean;
  used_web?: boolean;
  web_sources?: { title?: string; url: string; score?: number }[];
  used_primary_source?: boolean;
  primary_source?: any;
  plan?: { content?: string; status?: string }[];
  verified_count?: number;
  proposed_count?: number;
  retrieved_count?: number;
};
type Message = { role: "user" | "assistant"; content: string; meta?: Meta; notice?: boolean };
type Doc = { document_id: string; name: string; content_hash?: string; chunk_count?: number };
type Chat = { thread_id: string; title: string; updated_at?: string; message_count?: number };
type Session = { status: string; authenticated?: boolean; expires_at?: number | null; provider?: string; model?: string; project?: string };
type Status = { qdrant_ok: boolean; qdrant_error?: string | null; web_search_configured: boolean; primary_source_configured: boolean };
type NoticeKind = "success" | "info" | "error";
type Notice = { kind: NoticeKind; text: string } | null;

const API = window.location.origin;
const SESSION_KEY = "portfolio_llm_session";
const THREAD_KEY = "evidenceflow_thread";

function getToken() {
  const q = new URLSearchParams(window.location.search);
  const h = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const token = h.get("portfolio_llm_session") || q.get("portfolio_llm_session");
  if (token) {
    sessionStorage.setItem(SESSION_KEY, token);
    q.delete("portfolio_llm_session"); h.delete("portfolio_llm_session");
    const qs = q.toString();
    window.history.replaceState({}, "", `${window.location.pathname}${qs ? `?${qs}` : ""}`);
  }
  return sessionStorage.getItem(SESSION_KEY) || "";
}
function newThread() { const id = crypto.randomUUID(); sessionStorage.setItem(THREAD_KEY, id); return id; }
function getThread() { return sessionStorage.getItem(THREAD_KEY) || newThread(); }

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const p: Record<string, React.ReactNode> = {
    plus: <><path d="M12 5v14"/><path d="M5 12h14"/></>,
    search: <><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    trash: <><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M7 7l1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></>,
    copy: <><rect x="8" y="8" width="11" height="12" rx="2"/><path d="M5 16V6a2 2 0 0 1 2-2h8"/></>,
    rotate: <><path d="M20 11a8 8 0 1 0 1 4"/><path d="M20 4v7h-7"/></>,
    globe: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c2.2 2.4 3.3 5.4 3.3 9s-1.1 6.6-3.3 9c-2.2-2.4-3.3-5.4-3.3-9S9.8 5.4 12 3Z"/></>,
    shield: <><path d="M12 3 20 6v5c0 4.7-3 8-8 10-5-2-8-5.3-8-10V6l8-3Z"/><path d="m9 12 2 2 4-4"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
    save: <><path d="M5 5h14v14H5z"/><path d="M8 5v6h8V5"/><path d="M8 19v-5h8v5"/></>,
    external: <><path d="M14 5h5v5"/><path d="m19 5-8 8"/><path d="M19 13v5H6V6h5"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    menu: <><path d="M4 6h16M4 12h16M4 18h16"/></>,
  };
  return <svg {...common}>{p[name] ?? p.check}</svg>;
}

function AnswerRenderer({ text, evidenceMap, onEvidence, onHover }: { text: string; evidenceMap: Map<string, Evidence>; onEvidence: (id: string) => void; onHover: (id: string | null) => void }) {
  const parts = text.split(/(\[EVID:\s*E\d+\]|https?:\/\/[^\s)]+|\*\*[^*]+\*\*|`[^`]+`)/g);
  return <div className="answer-text">{parts.map((part, i) => {
    const m = part.match(/^\[EVID:\s*(E\d+)\]$/);
    if (m) {
      const id = m[1];
      const evidence = evidenceMap.get(id);
      return <span key={i} className="cite-wrap" onMouseEnter={() => onHover(id)} onMouseLeave={() => onHover(null)}>
        <button className="cite-chip-inline" onClick={() => onEvidence(id)} title="Inspect verified evidence">{id} ✓</button>
        {evidence && <span className="cite-tooltip">
          <span className="cite-tooltip-head"><b>{id}</b><span>Verified evidence</span></span>
          <span className="cite-tooltip-body">{evidence.content || "No excerpt available."}</span>
          <span className="cite-tooltip-foot">Score {typeof evidence.score === "number" ? evidence.score.toFixed(3) : "—"} · Click to inspect</span>
        </span>}
      </span>;
    }
    if (/^https?:\/\//.test(part)) return <a key={i} href={part} target="_blank" rel="noreferrer">{part}</a>;
    if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (/^`[^`]+`$/.test(part)) return <code key={i}>{part.slice(1, -1)}</code>;
    return <React.Fragment key={i}>{part}</React.Fragment>;
  })}</div>;
}

function CitationPopover({ evidence, onOpen, active, onHover }: { evidence: Evidence; onOpen: () => void; active: boolean; onHover: (id: string | null) => void }) {
  return <button className={`evidence-card citation-popover ${active ? "highlighted" : ""}`} onMouseEnter={() => onHover(evidence.evidence_id || null)} onMouseLeave={() => onHover(null)} onClick={onOpen}>
    <div className="evidence-top"><span>{evidence.evidence_id}</span><span className="verified-pill">Verified</span></div>
    <div className="evidence-excerpt">{evidence.content || "No excerpt available."}</div>
    <div className="evidence-foot"><span>Score {typeof evidence.score === "number" ? evidence.score.toFixed(3) : "—"}</span><span>Open evidence <Icon name="external" size={13}/></span></div>
  </button>;
}

function App() {
  const [token] = useState(getToken);
  const [session, setSession] = useState<Session>({ status: token ? "validating" : "missing" });
  const [threadId, setThreadId] = useState(getThread);
  const [chats, setChats] = useState<Chat[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [docs, setDocs] = useState<{global: Doc[]; chat: Doc[]}>({ global: [], chat: [] });
  const [status, setStatus] = useState<Status | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [composer, setComposer] = useState("");
  const [busy, setBusy] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [notice, setNotice] = useState<Notice>(null);
  const [mode, setMode] = useState("Answer");
  const [showSources, setShowSources] = useState(false);
  const [showPlan, setShowPlan] = useState(false);
  const [highlightedEvidence, setHighlightedEvidence] = useState<string | null>(null);
  const [showSaved, setShowSaved] = useState(false);
  const [saved, setSaved] = useState<Evidence[]>(() => JSON.parse(localStorage.getItem("evidenceflow_saved") || "[]"));
  const [uploading, setUploading] = useState<string[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [rightOpen, setRightOpen] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  const authHeaders = useMemo(() => ({ Authorization: `Bearer ${token}` }), [token]);

  function showNotice(kind: NoticeKind, text: string) { setNotice({ kind, text }); window.setTimeout(() => setNotice(null), 3400); }

  async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    Object.entries(authHeaders).forEach(([k, v]) => headers.set(k, v));
    const res = await fetch(`${API}${path}`, { ...init, headers });
    if (res.status === 401) { setSession(s => ({ ...s, status: "expired" })); throw new Error("BYOK session expired. Return to the portfolio to start a new session."); }
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  }

  async function validateSession() {
    if (!token) { setSession({ status: "missing" }); return; }
    try { setSession(await api<Session>("/api/session")); }
    catch { setSession({ status: "expired" }); }
  }
  async function refreshChats() { try { const r = await api<{chats: Chat[]}>("/api/chats"); setChats(r.chats); } catch (error) { console.debug("EvidenceFlow refresh failed", error); } }
  async function loadThread(id: string) { try { const r = await api<any>(`/api/chats/${encodeURIComponent(id)}`); setMessages(r.messages || []); setThreadId(id); sessionStorage.setItem(THREAD_KEY, id); setDocs(await api(`/api/documents?thread_id=${encodeURIComponent(id)}`)); } catch (e: any) { showNotice("error", e.message || "Could not load chat"); } }
  async function refreshDocs(id = threadId) { try { setDocs(await api(`/api/documents?thread_id=${encodeURIComponent(id)}`)); } catch (error) { console.debug("EvidenceFlow refresh failed", error); } }
  async function refreshStatus() { try { setStatus(await api<Status>("/api/status")); } catch (error) { console.debug("EvidenceFlow refresh failed", error); } }

  useEffect(() => { validateSession(); if (token) { refreshChats(); refreshDocs(); refreshStatus(); } }, [token]);
  useEffect(() => { if (autoScroll) endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, streamingText, autoScroll]);
  useEffect(() => { if (!token) return; const t = window.setInterval(validateSession, 30000); return () => window.clearInterval(t); }, [token]);

  async function newChat() {
    try { const r = await api<Chat>("/api/chats", { method: "POST" }); setThreadId(r.thread_id); sessionStorage.setItem(THREAD_KEY, r.thread_id); setMessages([]); setDocs({ global: docs.global, chat: [] }); setSelectedEvidence(null); setStreamingText(""); await refreshChats(); showNotice("success", "New research thread created"); } catch (e: any) { showNotice("error", e.message); }
  }
  async function deleteChat(id: string) {
    if (!confirm("Delete this chat and its attached document vectors?")) return;
    try { await api(`/api/chats/${encodeURIComponent(id)}`, { method: "DELETE" }); if (id === threadId) await newChat(); await refreshChats(); showNotice("success", "Chat deleted"); } catch (e: any) { showNotice("error", e.message); }
  }
  async function clearAll() {
    if (!confirm("Permanently delete all chats and all indexed documents?")) return;
    try { await api("/api/clear", { method: "POST" }); const id = newThread(); setThreadId(id); setMessages([]); setDocs({ global: [], chat: [] }); setChats([]); showNotice("success", "Everything cleared"); } catch (e: any) { showNotice("error", e.message); }
  }

  async function upload(file: File, scope: "global" | "chat") {
    setUploading(v => [...v, file.name]);
    try {
      const fd = new FormData(); fd.append("file", file);
      const r = await api<any>(`/api/documents?thread_id=${encodeURIComponent(threadId)}&scope=${scope}`, { method: "POST", body: fd });
      await refreshDocs();
      showNotice(r.duplicate ? "info" : "success", r.duplicate ? `${file.name} was already indexed.` : `${file.name} indexed successfully.`);
    } catch (e: any) { showNotice("error", `Upload failed: ${e.message}`); }
    finally { setUploading(v => v.filter(x => x !== file.name)); }
  }
  async function removeDoc(doc: Doc, scope: "global" | "chat") { try { await api(`/api/documents/${encodeURIComponent(doc.document_id)}?thread_id=${encodeURIComponent(threadId)}&scope=${scope}`, { method: "DELETE" }); await refreshDocs(); showNotice("success", `${doc.name} removed`); } catch (e: any) { showNotice("error", e.message); } }

  function modeInstruction(text: string) {
    if (mode === "Compare") return `Compare the relevant sources and present a concise comparison. ${text}`;
    if (mode === "Contradictions") return `Look specifically for contradictions or conflicting evidence. ${text}`;
    if (mode === "Summarize") return `Summarize the available evidence and explain the key takeaways. ${text}`;
    if (mode === "Deep research") return `Perform a thorough research pass using the best available sources and verify important claims. ${text}`;
    return text;
  }

  async function send(textOverride?: string, regenerate = false) {
    const q = (textOverride ?? composer).trim();
    if (!q || busy) return;
    if (session.status !== "active") { showNotice("error", "Your portfolio BYOK session is not active."); return; }
    setComposer(""); setBusy(true); setStreamingText(""); setAutoScroll(true); setStatus({ ...(status || { qdrant_ok: false, web_search_configured: false, primary_source_configured: false }) });
    if (!regenerate) setMessages(m => [...m, { role: "user", content: q }, { role: "assistant", content: "", meta: undefined }]);
    else setMessages(m => { const copy = [...m]; const i = copy.length - 1; copy[i] = { ...copy[i], content: "", meta: undefined }; return copy; });
    const controller = new AbortController(); abortRef.current = controller;
    let final: any = null; let accumulated = "";
    try {
      const res = await fetch(`${API}/api/chat/stream`, { method: "POST", headers: { ...authHeaders, "Content-Type": "application/json" }, body: JSON.stringify({ thread_id: threadId, query: modeInstruction(q) }), signal: controller.signal });
      if (res.status === 401) throw new Error("BYOK session expired. Return to the portfolio to start a new session.");
      if (!res.ok) throw new Error(await res.text());
      const reader = res.body?.getReader(); if (!reader) throw new Error("Streaming is unavailable.");
      const decoder = new TextDecoder(); let buffer = "";
      while (true) {
        const { value, done } = await reader.read(); if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n"); buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find(x => x.startsWith("data: ")); if (!line) continue;
          const raw = line.slice(6); if (raw === "[DONE]") continue;
          const ev = JSON.parse(raw);
          if (ev.type === "status") setNotice({ kind: "info", text: ev.text });
          if (ev.type === "token") { accumulated += ev.text || ""; setStreamingText(accumulated); setMessages(m => { const copy=[...m]; copy[copy.length-1]={...copy[copy.length-1], content: accumulated}; return copy; }); }
          if (ev.type === "final") final = ev;
          if (ev.type === "error") throw new Error(ev.message || "Turn failed");
        }
      }
      if (final) { const display = (final.final_answer || accumulated).replace(/\[EVID:\s*E\d+\]/g, "").trim(); const meta: Meta = { token_usage: final.token_usage, used_knowledge_base: final.used_knowledge_base, chunks: final.chunks || [], cited_evidence_ids: Array.from(new Set((final.final_answer || "").match(/\[EVID:\s*(E\d+)\]/g)?.map((m: string) => m.replace(/[^E\d]/g, "")) || [])), grounding_status: final.grounding_status, quality: final.quality, abstained: final.abstained, used_web: final.used_web, web_sources: final.web_sources || [], used_primary_source: final.used_primary_source, primary_source: final.primary_source || {}, plan: final.plan || [], verified_count: final.verified_count, proposed_count: final.proposed_count, retrieved_count: (final.chunks || []).length }; setMessages(m => { const copy=[...m]; copy[copy.length-1]={...copy[copy.length-1], content:display, meta}; return copy; }); }
      await refreshChats();
      setNotice(null);
    } catch (e: any) { setMessages(m => { const copy=[...m]; copy[copy.length-1]={role:"assistant",content:`I couldn't complete that turn. ${e.message || e}`}; return copy; }); setNotice({kind:"error", text:e.message || "Turn failed"}); }
    finally { abortRef.current = null; setBusy(false); setStreamingText(""); }
  }

  function copy(text: string) { navigator.clipboard.writeText(text).then(() => showNotice("success", "Answer copied")); }
  function saveEvidence(e: Evidence) { const next = saved.some(x => x.evidence_id === e.evidence_id) ? saved : [...saved, e]; setSaved(next); localStorage.setItem("evidenceflow_saved", JSON.stringify(next)); showNotice("success", "Evidence saved"); }
  function exportBrief() { const lines = ["# EvidenceFlow research brief", "", `Thread: ${threadId}`, "", ...messages.map(m => `${m.role.toUpperCase()}: ${m.content}\n${m.meta?.quality ? `Grounding: ${m.meta.quality.quality_label || m.meta.grounding_status || "N/A"}` : ""}`)]; const blob = new Blob([lines.join("\n\n")], { type: "text/markdown" }); const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="evidenceflow-research-brief.md"; a.click(); URL.revokeObjectURL(a.href); }

  const allMessages = messages;
  const latestMeta = [...messages].reverse().find(m => m.role === "assistant" && m.meta)?.meta;
  const evidenceMap = new Map<string, Evidence>();
  (latestMeta?.chunks || []).forEach((e) => { if (e.evidence_id) evidenceMap.set(e.evidence_id, e); });
  const latestWeb = latestMeta?.web_sources || [];
  const latestPrimary = latestMeta?.primary_source;
  const latestPlan = latestMeta?.plan || [];

  function renderMeta(meta: Meta) {
    if (!meta) return null;
    return <div className="message-meta">
      <div className="meta-row">
        {meta.token_usage && <span>🔢 {meta.token_usage.total_tokens ?? 0} tokens</span>}
        {meta.grounding_status && <span className="soft-pill"><Icon name="shield" size={13}/> {meta.grounding_status}</span>}
        {meta.verified_count !== undefined && <span>{meta.verified_count}/{meta.proposed_count ?? meta.verified_count} claims verified</span>}
      </div>
      {meta.quality?.citation_coverage !== undefined && <div className="coverage"><div className="coverage-label"><span>Evidence coverage</span><span>{meta.quality.citation_coverage.toFixed(1)}%</span></div><div className="coverage-bar"><span style={{width:`${Math.max(0, Math.min(100, meta.quality.citation_coverage))}%`}}/></div></div>}
      {(meta.quality?.numeric_support?.numeric_claims_supported !== null && meta.quality?.numeric_support?.numeric_claims_supported !== undefined) && <span className={`qa-chip ${meta.quality.numeric_support.numeric_claims_supported ? "ok" : "warn"}`}>{meta.quality.numeric_support.numeric_claims_supported ? "✓ numeric/date support" : "⚠ numeric/date review"}</span>}
      {meta.quality?.evidence_conflicts?.length ? <div className="conflict-banner">⚠ Possible evidence conflict detected. Review the source passages.</div> : null}
      {meta.abstained && <div className="abstain-banner"><Icon name="shield"/> Evidence QA blocked the answer because verified support was insufficient.</div>}
      {meta.plan?.length ? <button className="trace-toggle" onClick={() => setShowPlan(v=>!v)}>Plan · {meta.plan.filter(p=>p.status==="completed").length}/{meta.plan.length} done</button> : null}
    </div>
  }

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><div className="brand-mark">EF</div><div><div className="brand-title">EvidenceFlow</div><div className="brand-sub">Verified RAG & Research</div></div></div>
      <div className="top-actions"><span className={`session-pill ${session.status}`}>{session.status === "active" ? "BYOK session active" : session.status === "expired" ? "BYOK session expired" : session.status === "validating" ? "Validating session…" : "BYOK session required"}</span><button className="ghost-btn" onClick={() => setSidebarOpen(v=>!v)}><Icon name="menu"/></button><button className="ghost-btn" onClick={newChat}>New thread</button></div>
    </header>
    <div className="workspace">
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="side-top"><div className="side-title">Research</div><button className="icon-btn" onClick={newChat}><Icon name="plus" size={16}/></button></div>
        <div className="chat-history">
          {chats.map(c => <div className={`chat-row ${c.thread_id === threadId ? "active" : ""}`} key={c.thread_id}><button className="chat-open" onClick={() => loadThread(c.thread_id)}><span className="chat-title">{c.title}</span><span className="chat-count">{c.message_count || 0} messages</span></button><button className="icon-btn danger" onClick={() => deleteChat(c.thread_id)}><Icon name="trash" size={15}/></button></div>)}
          {!chats.length && <div className="muted">No saved chats yet.</div>}
        </div>
        <section className="side-section"><div className="side-title">Documents · all chats</div><label className="upload-btn"><Icon name="upload" size={15}/><span>Add global document</span><input type="file" multiple hidden onChange={e => [...(e.target.files||[])].forEach(f=>upload(f,"global"))}/></label>{docs.global.map(d => <div className="doc-row" key={d.document_id}><button onClick={()=>setSelectedEvidence({document_id:d.document_id, content:d.name})}><Icon name="save" size={14}/><span>{d.name}</span></button><button className="icon-btn danger" onClick={()=>removeDoc(d,"global")}><Icon name="trash" size={14}/></button></div>)}</section>
        <section className="side-section"><div className="side-title">Documents · this chat</div><label className="upload-btn secondary"><Icon name="upload" size={15}/><span>Attach to this chat</span><input type="file" multiple hidden onChange={e => [...(e.target.files||[])].forEach(f=>upload(f,"chat"))}/></label>{docs.chat.length ? docs.chat.map(d => <div className="doc-row" key={d.document_id}><button onClick={()=>setSelectedEvidence({document_id:d.document_id, content:d.name})}><Icon name="file" size={14}/><span>{d.name}</span></button><button className="icon-btn danger" onClick={()=>removeDoc(d,"chat")}><Icon name="trash" size={14}/></button></div>) : <div className="muted">None attached in this chat.</div>}</section>
        <section className="side-section"><div className="side-title">System</div><div className="status-grid"><span className={status?.qdrant_ok ? "ok-dot" : "bad-dot"}>Qdrant {status?.qdrant_ok ? "connected" : "offline"}</span><span>{status?.web_search_configured ? "Web search configured" : "Web search unavailable"}</span><span>{status?.primary_source_configured ? "Primary-source MCP configured" : "Primary-source MCP unavailable"}</span></div><button className="secondary-action" onClick={refreshStatus}>Recheck connectivity</button><button className="secondary-action" onClick={clearAll}>Clear everything</button></section>
      </aside>

      <main className="main-pane">
        <div className="hero-row"><div><div className="eyebrow">AGENTIC RESEARCH WORKSPACE</div><h1>Ask questions. Inspect evidence. Trust less by default.</h1><p>Dynamic routing across your knowledge base and web research, with retrieval, reranking, citation verification and fail-closed answers.</p></div><div className="hero-actions"><button className="secondary-action" onClick={exportBrief}>Export brief</button><button className={`secondary-action ${showSaved ? "active" : ""}`} onClick={()=>setShowSaved(v=>!v)}><Icon name="save" size={15}/> Saved evidence</button></div></div>
        <div className="mode-row"><div className="mode-label">Research mode</div>{["Answer","Deep research","Compare","Contradictions","Summarize"].map(m => <button key={m} className={`mode-chip ${mode===m ? "active" : ""}`} onClick={()=>setMode(m)}>{m}</button>)}</div>
        {showSaved && <div className="saved-strip">{saved.length ? saved.map(e => <button key={e.evidence_id} onClick={()=>setSelectedEvidence(e)} className="saved-card"><span>{e.evidence_id}</span><span>{(e.content||"").slice(0,120)}…</span></button>) : <span className="muted">No saved evidence yet.</span>}</div>}
        <section className="messages">
          {!allMessages.length && <div className="empty-state"><div className="empty-badge"><Icon name="search" size={18}/></div><h2>Start a research turn</h2><p>Try comparing two documents, asking for the strongest supported conclusion, or finding contradictions across your sources.</p><div className="prompt-grid">{["Summarize the uploaded evidence", "Compare the strongest supported conclusions", "Find contradictions in the source set"].map(p => <button key={p} onClick={()=>setComposer(p)}>{p}<Icon name="arrow" size={15}/></button>)}</div></div>}
          {allMessages.map((m, i) => <article className={`message ${m.role} ${m.notice ? "notice" : ""}`} key={`${i}-${m.content.slice(0,8)}`}><div className="avatar">{m.role === "user" ? "You" : "EF"}</div><div className="bubble-wrap"><div className="bubble">{m.role === "assistant" ? <AnswerRenderer text={m.content} evidenceMap={evidenceMap} onEvidence={(id)=>{setSelectedEvidence(evidenceMap.get(id) || {evidence_id:id, content:"Evidence unavailable in the current scope."}); setRightOpen(true)}} onHover={setHighlightedEvidence}/> : <div className="answer-text">{m.content}</div>}{m.role === "assistant" && m.meta && renderMeta(m.meta)}{m.role === "assistant" && m.meta && <div className="answer-actions"><button className="icon-btn" title="Copy" onClick={()=>copy(m.content)}><Icon name="copy" size={15}/></button>{i===allMessages.length-1 && <button className="icon-btn" title="Regenerate" onClick={()=>send(allMessages[i-1]?.content || "", true)}><Icon name="rotate" size={15}/></button>}</div>}</div></div></article>)}
          {busy && <div className="streaming"><div className="avatar">EF</div><div className="streaming-card"><span className="typing-dot"/><span className="typing-dot"/><span className="typing-dot"/><span className="streaming-text">{notice?.text || "Working…"}</span></div></div>}
          <div ref={endRef}/>
        </section>
        <div className="composer"><textarea value={composer} onChange={e=>setComposer(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send();}}} placeholder={session.status === "active" ? "Message the assistant, or attach a document to add it to the knowledge base…" : "Launch this app from the portfolio to activate BYOK."}/><div className="composer-row"><span className="composer-hint">Enter to send · Shift+Enter for new line</span><div><button className="ghost-btn" onClick={()=>setAutoScroll(v=>!v)}>{autoScroll ? "Auto-scroll on" : "Auto-scroll off"}</button><button className="send-btn" disabled={busy || session.status!=="active" || !composer.trim()} onClick={()=>send()}>{busy ? "Working…" : "Send"}</button></div></div></div>
      </main>

      <aside className={`evidence-panel ${rightOpen ? "" : "closed"}`}>
        <div className="panel-head"><div><div className="side-title">Verified evidence</div><div className="panel-sub">Claim → evidence → source</div></div><button className="icon-btn" onClick={()=>setRightOpen(v=>!v)}><Icon name="close" size={16}/></button></div>
        {selectedEvidence ? <div className="selected-evidence"><div className="evidence-kicker">{selectedEvidence.evidence_id || "Document"}</div><h3>{selectedEvidence.document_id || "Source"}</h3><p>{selectedEvidence.content}</p><div className="selected-actions"><button className="secondary-action" onClick={()=>saveEvidence(selectedEvidence)}><Icon name="save" size={14}/> Save evidence</button><button className="secondary-action" onClick={()=>setComposer(`Explain this evidence in the context of my question: ${selectedEvidence.content || ""}`)}>Ask about evidence</button><button className="secondary-action" onClick={()=>setComposer(`Challenge this evidence and look for contradicting sources: ${selectedEvidence.content || ""}`)}>Challenge evidence</button></div></div> : <div className="empty-evidence"><div className="empty-badge"><Icon name="shield" size={17}/></div><h3>Evidence will appear here</h3><p>Hover or click inline citation chips to inspect the exact passage used to support an answer.</p></div>}
        {latestMeta?.used_knowledge_base && <section className="panel-section"><div className="side-title">Cited evidence</div>{(latestMeta.chunks||[]).map(e => <CitationPopover key={e.evidence_id} evidence={e} active={highlightedEvidence===e.evidence_id} onHover={setHighlightedEvidence} onOpen={()=>{setSelectedEvidence(e);setRightOpen(true)}}/>)}</section>}
        {latestMeta?.used_web && <section className="panel-section"><button className="section-toggle" onClick={()=>setShowSources(v=>!v)}><span><Icon name="globe" size={15}/> Web sources ({latestWeb.length})</span><span>{showSources ? "−" : "+"}</span></button>{showSources && latestWeb.map((s,i)=><a key={i} className="source-link" href={s.url} target="_blank" rel="noreferrer"><span>{s.title || s.url}</span><span>{typeof s.score === "number" ? s.score.toFixed(3) : ""}</span></a>)}</section>}
        {latestMeta?.used_primary_source && latestPrimary && <section className="panel-section"><div className="side-title">Primary-source MCP</div><div className="primary-card"><div><strong>{latestPrimary.agent || "unknown agent"}</strong> · {latestPrimary.action || "action"}</div>{latestPrimary.citation?.source_name && <div>Source: {latestPrimary.citation.source_name}</div>}{latestPrimary.citation?.source_url && <a href={latestPrimary.citation.source_url} target="_blank" rel="noreferrer">Open source ↗</a>}{latestPrimary.quality && <div>Freshness {latestPrimary.quality.freshness_seconds ?? "n/a"}s · Confidence {latestPrimary.quality.confidence ?? "n/a"}</div>}</div></section>}
        {showPlan && latestPlan.length > 0 && <section className="panel-section plan-panel">{latestPlan.map((p,i)=><div key={i} className="plan-row"><span>{p.status === "completed" ? "✓" : p.status === "in_progress" ? "◔" : "○"}</span><span>{p.content}</span></div>)}</section>}
      </aside>
    </div>
    {uploading.length > 0 && <div className="upload-overlay"><div className="upload-card"><div className="spinner"/><div><strong>Indexing documents</strong><span>{uploading.join(", ")}</span></div></div></div>}
    {notice && <div className={`toast ${notice.kind}`}>{notice.text}</div>}
    {session.status !== "active" && <div className="session-banner"><div><strong>{session.status === "expired" ? "Your BYOK session has expired." : session.status === "missing" ? "BYOK session required." : "Validating BYOK session…"}</strong><span>{session.status === "active" ? "" : "Launch EvidenceFlow from the portfolio after entering your provider key."}</span></div>{session.status !== "validating" && <a href="https://asaifali-portfolio.vercel.app">Return to portfolio</a>}</div>}
  </div>;
}

createRoot(document.getElementById("root")!).render(<App />);
