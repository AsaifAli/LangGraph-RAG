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

const API = window.location.origin;
const SESSION_KEY = "portfolio_llm_session";
const THREAD_KEY = "evidenceflow_thread";

function getToken() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("portfolio_llm_session");
  if (token) {
    sessionStorage.setItem(SESSION_KEY, token);
    params.delete("portfolio_llm_session");
    const clean = `${window.location.pathname}${params.toString() ? `?${params}` : ""}`;
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
    external: <><path d="M14 4h6v6"/><path d="M10 14 20 4"/><path d="M20 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h5"/></>,
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
  const [fileBusy, setFileBusy] = useState(false);
  const [status, setStatus] = useState("Ready for research");
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showSources, setShowSources] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const allDocs = useMemo(() => [...docs.global, ...docs.chat], [docs]);
  const latestFinal = useMemo(() => messages[messages.length - 1]?.final ?? null, [messages]);
  const evidence = useMemo<Evidence[]>(() => latestFinal?.chunks ?? [], [latestFinal]);

  async function refreshDocs() {
    if (!token) return;
    const r = await fetch(`${API}/api/documents?thread_id=${encodeURIComponent(threadId)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (r.ok) setDocs(await r.json());
  }

  useEffect(() => {
    void refreshDocs();
  }, [token, threadId]);

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    if (!token) {
      setMessages((m) => [...m, { role: "assistant", content: "Launch this demo from the portfolio after entering your BYOK provider key. The temporary portfolio session is required." }]);
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
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { role: "assistant", content: `I couldn't complete that turn. ${message}` };
        return copy;
      });
      setStatus("Turn failed");
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  async function upload(file: File, scope: "global" | "chat") {
    if (!token) {
      setStatus("BYOK session required");
      return;
    }
    setFileBusy(true);
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
      setStatus("Document indexed and ready");
    } catch (e) {
      setStatus(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setFileBusy(false);
    }
  }

  async function removeDoc(documentId: string) {
    if (!token) return;
    const r = await fetch(`${API}/api/documents/${encodeURIComponent(documentId)}?thread_id=${encodeURIComponent(threadId)}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    });
    if (r.ok) {
      await refreshDocs();
      setSelectedEvidence(null);
      setStatus("Document removed");
    }
  }

  function resetThread() {
    abortRef.current?.abort();
    const id = newThread();
    setThreadId(id);
    setMessages([]);
    setSelectedEvidence(null);
    setStatus("New research thread");
  }

  const promptCards = [
    ["Compare sources", "Compare two sources and identify the strongest supported conclusion."],
    ["Find contradictions", "Find conflicting claims and explain which evidence is stronger."],
    ["Research a topic", "Research this topic and cite only verified evidence."],
  ];

  return (
    <div className="app">
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><span>EF</span><i /></div>
          <div>
            <div className="brand-title">EvidenceFlow</div>
            <div className="brand-sub">Verified RAG & Research</div>
          </div>
        </div>
        <div className="top-actions">
          <div className={`session-pill ${token ? "active" : "required"}`}><span className="status-dot" />{token ? "BYOK session active" : "BYOK session required"}</div>
          <button className="ghost-button" onClick={() => setShowSources((v) => !v)}><Icon name="file" size={16} />{allDocs.length} sources</button>
          <button className="primary-button compact" onClick={resetThread}><Icon name="plus" size={16} />New thread</button>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar panel-glass">
          <div className="sidebar-head"><div><div className="section-kicker">Workspace</div><h2>Research scope</h2></div><span className="mini-count">{allDocs.length}</span></div>
          <label className={`upload-card ${fileBusy ? "busy" : ""}`}>
            <input type="file" multiple accept=".pdf,.docx,.txt,.md,.csv" disabled={fileBusy} onChange={(e) => [...(e.target.files || [])].forEach((f) => void upload(f, "chat"))} />
            <div className="upload-icon"><Icon name="upload" size={18} /></div>
            <div><strong>{fileBusy ? "Indexing source…" : "Add research"}</strong><span>PDF, DOCX, TXT, MD, CSV</span></div>
            <div className="upload-arrow"><Icon name="plus" size={16} /></div>
          </label>

          <div className="doc-group">
            <div className="group-label"><span>Active sources</span><span>{allDocs.length}</span></div>
            <div className="doc-list">
              {allDocs.map((d) => (
                <div className="doc-row" key={d.document_id}>
                  <button className="doc-button" onClick={() => setSelectedEvidence({ document_id: d.document_id, content: d.name, source: d.name })}>
                    <span className="doc-icon"><Icon name="file" size={14} /></span>
                    <span className="doc-name">{d.name}</span>
                  </button>
                  <button className="icon-button danger" title="Remove" onClick={() => void removeDoc(d.document_id)}><Icon name="trash" size={14} /></button>
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
                {promptCards.map(([title, copy]) => (
                  <button key={title} className="prompt-card" onClick={() => setInput(copy)}>
                    <div className="prompt-icon"><Icon name={title === "Compare sources" ? "file" : title === "Find contradictions" ? "shield" : "search"} size={18} /></div>
                    <div><strong>{title}</strong><span>{copy}</span></div>
                    <Icon name="arrow" size={16} />
                  </button>
                ))}
              </div>
            ) : (
              <div className="message-stack">
                {messages.map((m, i) => (
                  <div key={i} className={`message-row ${m.role}`}>
                    <div className={`message-avatar ${m.role}`}>{m.role === "user" ? "You" : "EF"}</div>
                    <div className="message-column">
                      <div className="message-label">{m.role === "user" ? "You" : "EvidenceFlow"}</div>
                      <div className={`bubble ${m.role}`}>
                        {m.content ? <MarkdownLite text={m.content} /> : <span className="typing"><i /><i /><i /></span>}
                      </div>
                      {m.final && (
                        <div className="answer-meta">
                          <span className={`answer-status ${m.final.abstained ? "warn" : "good"}`}><i />{m.final.abstained ? "Insufficient evidence" : "Grounded response"}</span>
                          <span>{m.final.verified_count ?? 0} verified citations</span>
                          {m.final.used_web && <span>Web research</span>}
                          {(m.final.chunks?.length ?? 0) > 0 && <button onClick={() => setShowEvidence(true)}>Inspect evidence <Icon name="arrow" size={12} /></button>}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="composer-wrap">
            <div className={`composer ${busy ? "active" : ""}`}>
              <textarea value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }} placeholder="Ask about your sources or research the web…" aria-label="Research question" />
              <div className="composer-bottom">
                <div className="status-line"><span className="status-spinner" />{status}</div>
                <button className="send-button" disabled={busy || !input.trim()} onClick={() => void send()}><span>{busy ? "Working" : "Send"}</span><Icon name="arrow" size={16} /></button>
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
              {allDocs.map((d) => <button className="source-card" key={d.document_id} onClick={() => { setSelectedEvidence({ document_id: d.document_id, content: d.name, source: d.name }); setShowSources(false); }}><Icon name="file" size={15} /><span>{d.name}</span><Icon name="arrow" size={14} /></button>)}
              {!allDocs.length && <div className="panel-empty">Upload a source to start grounding your research.</div>}
            </div>
          ) : selectedEvidence ? (
            <div className="detail-card">
              <div className="detail-tag">{selectedEvidence.evidence_id || "SOURCE"}</div>
              <h3>{selectedEvidence.source || "Evidence fragment"}</h3>
              <p>{selectedEvidence.content || "No evidence content available."}</p>
              <button className="ghost-button full" onClick={() => setSelectedEvidence(null)}>Back to evidence</button>
            </div>
          ) : (
            <>
              <div className="verification-card"><div className="verification-icon"><Icon name="shield" size={18} /></div><div><strong>{latestFinal?.abstained ? "Fail-closed response" : latestFinal ? "Evidence verified" : "Verification ready"}</strong><span>{latestFinal ? `${latestFinal.verified_count ?? 0} citations authorized for this turn.` : "Complete a research turn to inspect the evidence registry."}</span></div></div>
              <div className="evidence-list">
                {evidence.slice(0, 8).map((e, i) => <button key={`${e.evidence_id || "e"}-${i}`} className="evidence-card" onClick={() => setSelectedEvidence(e)}><div className="evidence-id">{e.evidence_id || `E${i + 1}`}<span>{typeof e.score === "number" ? e.score.toFixed(2) : "verified"}</span></div><span>{(e.content || "").slice(0, 160)}{(e.content || "").length > 160 ? "…" : ""}</span></button>)}
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
