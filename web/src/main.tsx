import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Evidence = {
  evidence_id?: string;
  document_id?: string;
  content?: string;
  score?: number;
  source?: string;
};
type FinalResult = {
  final_answer?: string;
  verified_count?: number;
  chunks?: Evidence[];
  abstained?: boolean;
  grounding_status?: string;
  used_web?: boolean;
};
type Message = { role: "user" | "assistant"; content: string; final?: FinalResult | null };
type Doc = { document_id: string; name: string; content_hash?: string };
type DocsState = { global: Doc[]; chat: Doc[] };
type Notice = { kind: "success" | "info" | "error"; text: string } | null;

type UploadState = { id: string; name: string };

const API = window.location.origin;
const SESSION_KEY = "portfolio_llm_session";
const THREAD_KEY = "evidenceflow_thread";

function getToken() {
  const query = new URLSearchParams(window.location.search);
  const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  const token = fragment.get("portfolio_llm_session") || query.get("portfolio_llm_session");

  if (token) {
    sessionStorage.setItem(SESSION_KEY, token);
    query.delete("portfolio_llm_session");
    fragment.delete("portfolio_llm_session");
    const queryText = query.toString();
    const clean = `${window.location.pathname}${queryText ? `?${queryText}` : ""}`;
    window.history.replaceState({}, "", clean);
  }
  return sessionStorage.getItem(SESSION_KEY) || "";
}

function newThread() {
  const id = crypto.randomUUID();
  sessionStorage.setItem(THREAD_KEY, id);
  return id;
}

function getThread() {
  return sessionStorage.getItem(THREAD_KEY) || newThread();
}

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  const paths: Record<string, React.ReactNode> = {
    search: <><circle cx="11" cy="11" r="6.5"/><path d="m16 16 4.5 4.5"/></>,
    upload: <><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    plus: <><path d="M12 5v14"/><path d="M5 12h14"/></>,
    spark: <><path d="m12 2 1.8 5.2L19 9l-5.2 1.8L12 16l-1.8-5.2L5 9l5.2-1.8L12 2Z"/><path d="m19 15 .8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"/></>,
    trash: <><path d="M4 7h16"/><path d="M9 7V4h6v3"/><path d="M7 7l1 13h8l1-13"/><path d="M10 11v5M14 11v5"/></>,
    arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    shield: <><path d="M12 3 20 6v5c0 4.7-3 8-8 10-5-2-8-5.3-8-10V6l8-3Z"/><path d="m9 12 2 2 4-4"/></>,
    web: <><circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3c2.2 2.4 3.3 5.4 3.3 9s-1.1 6.6-3.3 9c-2.2-2.4-3.3-5.4-3.3-9S9.8 5.4 12 3Z"/></>,
    file: <><path d="M7 3h7l4 4v14H7z"/><path d="M14 3v5h5"/></>,
    close: <><path d="m6 6 12 12M18 6 6 18"/></>,
  };
  return <svg {...common}>{paths[name] ?? paths.spark}</svg>;
}

function MarkdownLite({ text }: { text: string }) {
  const parts = text.split(/(https?:\/\/[^\s]+|`[^`]+`|\*\*[^*]+\*\*)/g);
  return <>{parts.map((part, i) => {
    if (/^https?:\/\//.test(part)) return <a key={i} href={part} target="_blank" rel="noreferrer">{part}</a>;
    if (/^`[^`]+`$/.test(part)) return <code key={i}>{part.slice(1, -1)}</code>;
    if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={i}>{part.slice(2, -2)}</strong>;
    return <React.Fragment key={i}>{part}</React.Fragment>;
  })}</>;
}

function App() {
  const [token] = useState(getToken);
  const [threadId, setThreadId] = useState(getThread);
  const [messages, setMessages] = useState<Message[]>([]);
  const [docs, setDocs] = useState<DocsState>({ global: [], chat: [] });
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Ready for research");
  const [uploading, setUploading] = useState<UploadState[]>([]);
  const [removingIds, setRemovingIds] = useState<Set<string>>(new Set());
  const [dragActive, setDragActive] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  const [threadPulse, setThreadPulse] = useState(false);
  const [sourcePulse, setSourcePulse] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const chatEndRef = useRef<HTMLDivElement | null>(null);
  const noticeTimerRef = useRef<number | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const allDocs = useMemo(() => [...docs.global, ...docs.chat], [docs]);
  const latestFinal = useMemo(() => messages[messages.length - 1]?.final ?? null, [messages]);
  const evidence = useMemo<Evidence[]>(() => latestFinal?.chunks ?? [], [latestFinal]);

  function announce(kind: "success" | "info" | "error", text: string) {
    if (noticeTimerRef.current) window.clearTimeout(noticeTimerRef.current);
    setNotice({ kind, text });
    noticeTimerRef.current = window.setTimeout(() => setNotice(null), 3200);
  }

  useEffect(() => () => {
    if (noticeTimerRef.current) window.clearTimeout(noticeTimerRef.current);
  }, []);

  useEffect(() => {
    if (!token) return;
    void refreshDocs();
  }, [token, threadId]);

  useEffect(() => {
    const node = chatEndRef.current;
    if (!node) return;
    node.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    textareaRef.current?.focus({ preventScroll: true });
  }, [threadId]);

  useEffect(() => {
    if (allDocs.length === 0) return;
    setSourcePulse(true);
    const timer = window.setTimeout(() => setSourcePulse(false), 700);
    return () => window.clearTimeout(timer);
  }, [allDocs.length]);

  async function refreshDocs() {
    if (!token) return;
    const r = await fetch(`${API}/api/documents?thread_id=${encodeURIComponent(threadId)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (r.ok) setDocs(await r.json());
  }

  function openEvidence(item: Evidence) {
    setSelectedEvidence(item);
    setShowEvidence(true);
    setShowSources(false);
  }

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    if (!token) {
      setMessages((m) => [...m, { role: "assistant", content: "Launch this demo from the portfolio after entering your BYOK provider key. The temporary portfolio session is required." }]);
      announce("info", "Portfolio BYOK session required");
      return;
    }
    setInput("");
    setBusy(true);
    setStatus("Routing your research request…");
    setMessages((m) => [...m, { role: "user", content: q }, { role: "assistant", content: "" }]);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const r = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ thread_id: threadId, query: q }),
        signal: controller.signal,
      });
      if (!r.ok) throw new Error(await r.text());
      if (!r.body) throw new Error("Streaming response unavailable.");
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let final: FinalResult | null = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find((x) => x.startsWith("data: "));
          if (!line) continue;
          const raw = line.slice(6);
          if (raw === "[DONE]") continue;
          const event = JSON.parse(raw) as { type?: string; text?: string; message?: string; final_answer?: string; [key: string]: unknown };
          if (event.type === "status") setStatus(String(event.text ?? "Working…"));
          if (event.type === "token") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = { ...copy[copy.length - 1], content: copy[copy.length - 1].content + String(event.text ?? "") };
              return copy;
            });
          }
          if (event.type === "final") final = event as FinalResult;
          if (event.type === "error") throw new Error(String(event.message ?? "Research turn failed."));
        }
      }
      if (final) {
        setMessages((m) => {
          const copy = [...m];
          copy[copy.length - 1] = { ...copy[copy.length - 1], content: final?.final_answer || copy[copy.length - 1].content, final };
          return copy;
        });
      }
      setStatus(final?.abstained ? "Insufficient verified evidence" : "Verified response ready");
      announce(final?.abstained ? "info" : "success", final?.abstained ? "No sufficient verified evidence" : "Verified response ready");
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: `I couldn't complete that turn. ${message}` };
        return copy;
      });
      setStatus("Turn failed");
      announce("error", "Research turn failed");
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  async function upload(file: File, scope: "global" | "chat") {
    if (!token) {
      setStatus("BYOK session required");
      announce("info", "Portfolio BYOK session required");
      return;
    }
    const uploadId = `${file.name}-${file.lastModified}-${Math.random().toString(36).slice(2)}`;
    setUploading((items) => [...items, { id: uploadId, name: file.name }]);
    setStatus(`Indexing ${file.name}…`);
    const body = new FormData();
    body.append("file", file);
    try {
      const r = await fetch(`${API}/api/documents?thread_id=${encodeURIComponent(threadId)}&scope=${scope}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
        body,
      });
      if (!r.ok) throw new Error(await r.text());
      await refreshDocs();
      setStatus(`${file.name} is ready for research`);
      announce("success", `${file.name} indexed successfully`);
    } catch (e) {
      setStatus(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
      announce("error", `Couldn't index ${file.name}`);
    } finally {
      setUploading((items) => items.filter((item) => item.id !== uploadId));
    }
  }

  function handleFiles(files: FileList | File[]) {
    const accepted = Array.from(files).filter((file) => /\.(pdf|docx|txt|md|csv)$/i.test(file.name));
    if (!accepted.length) {
      announce("error", "Supported formats: PDF, DOCX, TXT, MD, CSV");
      return;
    }
    void Promise.all(accepted.map((file) => upload(file, "chat")));
  }

  async function removeDoc(documentId: string) {
    if (!token || removingIds.has(documentId)) return;
    setRemovingIds((set) => new Set(set).add(documentId));
    try {
      const r = await fetch(`${API}/api/documents/${encodeURIComponent(documentId)}?thread_id=${encodeURIComponent(threadId)}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!r.ok) throw new Error(await r.text());
      setDocs((current) => ({
        global: current.global.filter((doc) => doc.document_id !== documentId),
        chat: current.chat.filter((doc) => doc.document_id !== documentId),
      }));
      if (selectedEvidence?.document_id === documentId) setSelectedEvidence(null);
      setStatus("Document removed");
      announce("success", "Source removed");
    } catch (e) {
      setStatus(`Remove failed: ${e instanceof Error ? e.message : String(e)}`);
      announce("error", "Couldn't remove the source");
    } finally {
      setRemovingIds((set) => {
        const next = new Set(set);
        next.delete(documentId);
        return next;
      });
    }
  }

  function resetThread() {
    abortRef.current?.abort();
    const id = newThread();
    setThreadPulse(true);
    setThreadId(id);
    setMessages([]);
    setSelectedEvidence(null);
    setShowEvidence(false);
    setShowSources(false);
    setStatus("New research thread");
    setTimeout(() => setThreadPulse(false), 500);
    announce("success", "New research thread created");
  }

  function resizeTextarea(event: React.ChangeEvent<HTMLTextAreaElement>) {
    const el = event.currentTarget;
    setInput(el.value);
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }

  function selectPrompt(copy: string) {
    setInput(copy);
    requestAnimationFrame(() => textareaRef.current?.focus({ preventScroll: true }));
  }

  const promptCards = [
    ["Compare sources", "Compare two sources and identify the strongest supported conclusion.", "file"],
    ["Find contradictions", "Find conflicting claims and explain which evidence is stronger.", "shield"],
    ["Research a topic", "Research this topic and cite only verified evidence.", "search"],
  ] as const;

  return (
    <div className={`app ${threadPulse ? "thread-pulse" : ""}`}>
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      {notice && <div className={`toast toast-${notice.kind}`} role="status"><span className="toast-dot" />{notice.text}</div>}
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><span>EF</span><i /></div>
          <div><div className="brand-title">EvidenceFlow</div><div className="brand-sub">Verified RAG & Research</div></div>
        </div>
        <div className="top-actions">
          <div className={`session-pill ${token ? "active" : "required"}`}><span className="status-dot" />{token ? "BYOK session active" : "BYOK session required"}</div>
          <button className={`ghost-button source-count ${sourcePulse ? "pulse" : ""}`} onClick={() => { setShowSources((v) => !v); setShowEvidence(false); }}><Icon name="file" size={16} />{allDocs.length} sources</button>
          <button className="primary-button compact" onClick={resetThread}><Icon name="plus" size={16} />New thread</button>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar panel-glass">
          <div className="sidebar-head"><div><div className="section-kicker">Workspace</div><h2>Research scope</h2></div><span className={`mini-count ${sourcePulse ? "pulse-ring" : ""}`}>{allDocs.length}</span></div>
          <label
            className={`upload-card ${uploading.length ? "busy" : ""} ${dragActive ? "drag-active" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => { e.preventDefault(); setDragActive(false); handleFiles(e.dataTransfer.files); }}
          >
            <input type="file" multiple accept=".pdf,.docx,.txt,.md,.csv" disabled={uploading.length > 0} onChange={(e) => { if (e.target.files) handleFiles(e.target.files); e.currentTarget.value = ""; }} />
            <div className={`upload-icon ${uploading.length ? "spin-soft" : ""}`}>{uploading.length ? <span className="upload-spinner" /> : <Icon name="upload" size={18} />}</div>
            <div><strong>{uploading.length ? `Indexing ${uploading.length} source${uploading.length > 1 ? "s" : ""}…` : dragActive ? "Drop to index" : "Add research"}</strong><span>PDF, DOCX, TXT, MD, CSV · drag & drop</span></div>
            <div className="upload-arrow"><Icon name="plus" size={16} /></div>
          </label>

          {uploading.length > 0 && <div className="upload-queue">{uploading.map((item) => <div className="upload-item" key={item.id}><span className="upload-wave" /><span>{item.name}</span><span className="upload-dots"><i /><i /><i /></span></div>)}</div>}

          <div className="doc-group">
            <div className="group-label"><span>Active sources</span><span>{allDocs.length}</span></div>
            <div className="doc-list">
              {allDocs.map((d) => (
                <div className={`doc-row ${removingIds.has(d.document_id) ? "removing" : ""}`} key={d.document_id}>
                  <button className="doc-button" onClick={() => openEvidence({ document_id: d.document_id, content: d.name, source: d.name })}>
                    <span className="doc-icon"><Icon name="file" size={14} /></span>
                    <span className="doc-name">{d.name}</span><span className="doc-chevron"><Icon name="arrow" size={12} /></span>
                  </button>
                  <button className="icon-button danger" title="Remove" disabled={removingIds.has(d.document_id)} onClick={() => void removeDoc(d.document_id)}><Icon name="trash" size={14} /></button>
                </div>
              ))}
              {!allDocs.length && <div className="empty-docs"><span>No sources yet</span><small>Upload a document to ground the next research turn.</small></div>}
            </div>
          </div>

          <div className="stack-card">
            <div className="group-label"><span>Retrieval stack</span><span className="live-label"><i />Live</span></div>
            <div className="stack-list">
              <div><span className="stack-bullet violet" />Dense + BM25</div>
              <div><span className="stack-bullet blue" />RRF fusion</div>
              <div><span className="stack-bullet cyan" />Jina reranking</div>
              <div><span className="stack-bullet green" />Evidence verification</div>
            </div>
          </div>
        </aside>

        <main className="chat-panel">
          <div className="hero-block">
            <div className="hero-badge"><span className="pulse" />AGENTIC RESEARCH WORKSPACE</div>
            <h1>Research with evidence,<br /><em>not assumptions.</em></h1>
            <p>EvidenceFlow dynamically routes between knowledge-base retrieval and web research, then verifies whether the answer is actually supported.</p>
            <div className="hero-metrics">
              <span><Icon name="shield" size={14} /> Fail-closed citations</span>
              <span><Icon name="spark" size={14} /> Hybrid retrieval</span>
              <span><Icon name="web" size={14} /> Web-aware routing</span>
            </div>
          </div>

          <div className="chat-scroll">
            {messages.length === 0 ? (
              <div className="welcome-grid">
                {promptCards.map(([title, copy, icon]) => (
                  <button key={title} className="prompt-card" onClick={() => selectPrompt(copy)}><div className="prompt-icon"><Icon name={icon} size={18} /></div><div><strong>{title}</strong><span>{copy}</span></div><Icon name="arrow" size={16} /></button>
                ))}
              </div>
            ) : (
              <div className="message-stack">
                {messages.map((m, i) => (
                  <div key={i} className={`message-row ${m.role}`} style={{ "--message-index": i } as React.CSSProperties}>
                    <div className={`message-avatar ${m.role}`}>{m.role === "user" ? "You" : "EF"}</div>
                    <div className="message-column">
                      <div className="message-label">{m.role === "user" ? "You" : "EvidenceFlow"}</div>
                      <div className={`bubble ${m.role} ${busy && i === messages.length - 1 ? "streaming" : ""}`}>
                        {m.content ? <MarkdownLite text={m.content} /> : <span className="typing"><i /><i /><i /></span>}
                        {busy && i === messages.length - 1 && m.content && <span className="stream-caret" aria-hidden="true" />}
                      </div>
                      {m.final && (
                        <div className="answer-meta">
                          <span className={`answer-status ${m.final.abstained ? "warn" : "good"}`}><i />{m.final.abstained ? "Insufficient evidence" : "Grounded response"}</span>
                          <span>{m.final.verified_count ?? 0} verified citations</span>
                          {m.final.used_web && <span>Web research</span>}
                          {(m.final.chunks?.length ?? 0) > 0 && <button onClick={() => { setShowEvidence(true); setShowSources(false); }}>Inspect evidence <Icon name="arrow" size={12} /></button>}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} className="chat-end" />
              </div>
            )}
          </div>

          <div className="composer-wrap">
            <div className={`composer ${busy ? "active" : ""}`}>
              <textarea ref={textareaRef} value={input} onChange={resizeTextarea} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }} placeholder="Ask about your sources or research the web…" aria-label="Research question" />
              <div className="composer-bottom">
                <div className={`status-line ${busy ? "active" : ""}`}><span className="status-spinner" />{status}</div>
                <button className="send-button" disabled={busy || !input.trim()} onClick={() => void send()}><span className="send-icon-wrap">{busy ? <span className="button-spinner" /> : <Icon name="arrow" size={16} />}</span><span>{busy ? "Working" : "Send"}</span></button>
              </div>
            </div>
            <div className="composer-note">EvidenceFlow can abstain when available evidence is insufficient. Press Enter to send · Shift + Enter for a new line.</div>
          </div>
        </main>

        <aside className={`evidence-panel panel-glass ${showEvidence || showSources ? "open" : ""}`}>
          <div className="evidence-head">
            <div><div className="section-kicker">Evidence layer</div><h2>{showSources ? "Sources" : "Verified evidence"}</h2></div>
            <button className="icon-button" onClick={() => { setShowEvidence(false); setShowSources(false); }}><Icon name="close" size={16} /></button>
          </div>
          {showSources ? (
            <div className="source-view">
              <div className="source-summary"><span>{allDocs.length}</span><div><strong>Active sources</strong><small>Documents currently visible to the thread</small></div></div>
              {allDocs.map((d) => <button className="source-card" key={d.document_id} onClick={() => { openEvidence({ document_id: d.document_id, content: d.name, source: d.name }); }}><Icon name="file" size={15} /><span>{d.name}</span><Icon name="arrow" size={14} /></button>)}
              {!allDocs.length && <div className="panel-empty">Upload a source to start grounding your research.</div>}
            </div>
          ) : selectedEvidence ? (
            <div className="detail-card detail-in">
              <div className="detail-tag">{selectedEvidence.evidence_id || "SOURCE"}</div>
              <h3>{selectedEvidence.source || "Evidence fragment"}</h3>
              <p>{selectedEvidence.content || "No evidence content available."}</p>
              <button className="ghost-button full" onClick={() => setSelectedEvidence(null)}>Back to evidence</button>
            </div>
          ) : (
            <>
              <div className="verification-card"><div className="verification-icon"><Icon name="shield" size={18} /></div><div><strong>{latestFinal?.abstained ? "Fail-closed response" : latestFinal ? "Evidence verified" : "Verification ready"}</strong><span>{latestFinal ? `${latestFinal.verified_count ?? 0} citations authorized for this turn.` : "Complete a research turn to inspect the evidence registry."}</span></div></div>
              <div className="evidence-list">
                {evidence.slice(0, 8).map((e, i) => <button key={`${e.evidence_id || "e"}-${i}`} className="evidence-card" style={{ "--evidence-index": i } as React.CSSProperties} onClick={() => openEvidence(e)}><div className="evidence-id">{e.evidence_id || `E${i + 1}`}<span>{typeof e.score === "number" ? e.score.toFixed(2) : "verified"}</span></div><span>{(e.content || "").slice(0, 160)}{(e.content || "").length > 160 ? "…" : ""}</span></button>)}
                {!evidence.length && <div className="panel-empty">Your verified evidence will appear here after a research turn.</div>}
              </div>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
