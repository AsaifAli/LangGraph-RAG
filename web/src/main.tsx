import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type Evidence = {
  evidence_id?: string;
  document_id?: string;
  content?: string;
};
type Message = { role: "user" | "assistant"; content: string; final?: any };
type Doc = { document_id: string; name: string; content_hash?: string };

const API = window.location.origin;
const SESSION_KEY = "portfolio_llm_session";
const THREAD_KEY = "evidenceflow_thread";

function getToken() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("portfolio_llm_session");
  if (token) {
    sessionStorage.setItem(SESSION_KEY, token);
    params.delete("portfolio_llm_session");
    const clean = `${window.location.pathname}${params.toString() ? "?" + params : ""}`;
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
function App() {
  const [token] = useState(getToken);
  const [threadId, setThreadId] = useState(getThread);
  const [messages, setMessages] = useState<Message[]>([]);
  const [docs, setDocs] = useState<{global: Doc[]; chat: Doc[]}>({global:[], chat:[]});
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [selectedEvidence, setSelectedEvidence] = useState<Evidence | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  async function refreshDocs() {
    if (!token) return;
    const r = await fetch(`${API}/api/documents?thread_id=${encodeURIComponent(threadId)}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (r.ok) setDocs(await r.json());
  }
  useEffect(() => { void refreshDocs(); }, [token, threadId]);

  async function send() {
    const q = input.trim();
    if (!q || busy) return;
    if (!token) {
      setMessages(m => [...m, {role:"assistant", content:"Launch this demo from the portfolio after entering your BYOK provider key. The temporary portfolio session is required."}]);
      return;
    }
    setInput(""); setBusy(true); setStatus("Starting research…");
    setMessages(m => [...m, {role:"user", content:q}, {role:"assistant", content:""}]);
    const controller = new AbortController(); abortRef.current = controller;
    try {
      const r = await fetch(`${API}/api/chat/stream`, {
        method: "POST",
        headers: {"Content-Type":"application/json", Authorization:`Bearer ${token}`},
        body: JSON.stringify({thread_id:threadId, query:q}),
        signal: controller.signal,
      });
      if (!r.ok) throw new Error(await r.text());
      const reader = r.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let final: any = null;
      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream:true});
        const parts = buffer.split("\n\n"); buffer = parts.pop() || "";
        for (const part of parts) {
          const line = part.split("\n").find(x => x.startsWith("data: "));
          if (!line) continue;
          const raw = line.slice(6);
          if (raw === "[DONE]") continue;
          const event = JSON.parse(raw);
          if (event.type === "status") setStatus(event.text);
          if (event.type === "token") {
            setMessages(m => {
              const copy = [...m]; copy[copy.length-1] = {...copy[copy.length-1], content: copy[copy.length-1].content + event.text}; return copy;
            });
          }
          if (event.type === "final") final = event;
          if (event.type === "error") throw new Error(event.message);
        }
      }
      if (final) {
        setMessages(m => {
          const copy = [...m]; copy[copy.length-1] = {...copy[copy.length-1], content: final.final_answer || copy[copy.length-1].content, final}; return copy;
        });
      }
      setStatus("Verified response ready");
    } catch (e:any) {
      setMessages(m => {
        const copy=[...m]; copy[copy.length-1]={role:"assistant",content:`I couldn't complete that turn. ${e.message || e}`}; return copy;
      });
      setStatus("Turn failed");
    } finally {
      abortRef.current = null; setBusy(false);
    }
  }

  async function upload(file: File, scope: "global"|"chat") {
    if (!token) return setStatus("BYOK session required");
    const body = new FormData(); body.append("file", file);
    try {
      const r = await fetch(`${API}/api/documents?thread_id=${encodeURIComponent(threadId)}&scope=${scope}`, {
        method:"POST", headers:{Authorization:`Bearer ${token}`}, body
      });
      if (!r.ok) throw new Error(await r.text());
      await refreshDocs(); setStatus("Document indexed");
    } catch(e:any) { setStatus(`Upload failed: ${e.message || e}`); }
  }

  function resetThread() {
    abortRef.current?.abort();
    const id = newThread(); setThreadId(id); setMessages([]); setSelectedEvidence(null); setStatus("New research thread");
  }

  const evidence = useMemo<Evidence[]>(() => {
    const final = messages[messages.length-1]?.final;
    return (final?.verified_count ? final?.chunks || [] : final?.chunks || []) as Evidence[];
  }, [messages]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><div className="brand-mark">EF</div><div><div className="brand-title">EvidenceFlow</div><div className="brand-sub">Verified RAG & Research</div></div></div>
        <div className="top-actions">
          <span className={`pill ${token ? "good" : "warn"}`}>{token ? "BYOK session active" : "BYOK session required"}</span>
          <button onClick={resetThread}>New thread</button>
        </div>
      </header>

      <div className="layout">
        <aside className="sidebar">
          <div className="side-section">
            <div className="side-title">Knowledge base</div>
            <label className="upload">
              <input type="file" multiple accept=".pdf,.docx,.txt,.md,.csv" onChange={e => [...(e.target.files || [])].forEach(f => upload(f,"chat"))} />
              <span>＋ Add research document</span>
            </label>
            <div className="doc-list">
              {docs.global.concat(docs.chat).map(d => <button key={d.document_id} className="doc" onClick={() => setSelectedEvidence({document_id:d.document_id, content:d.name})}>{d.name}</button>)}
              {!docs.global.length && !docs.chat.length && <div className="muted">No documents in scope.</div>}
            </div>
          </div>
          <div className="side-section compact"><div className="side-title">Retrieval stack</div><div className="stack"><span>Dense + BM25</span><span>RRF fusion</span><span>Jina reranking</span><span>Evidence verification</span></div></div>
        </aside>

        <main className="chat">
          <div className="hero">
            <div className="eyebrow">AGENTIC RESEARCH WORKSPACE</div>
            <h1>Ask questions. Inspect evidence. Trust less by default.</h1>
            <p>EvidenceFlow dynamically routes between your knowledge base and web research, then verifies citations against the evidence actually supplied to the model.</p>
          </div>
          <div className="messages">
            {messages.length === 0 && <div className="empty"><div className="empty-title">Start a research turn</div><div className="empty-copy">Try: “Compare the two uploaded documents and identify the strongest supported conclusion.”</div></div>}
            {messages.map((m,i) => <div key={i} className={`msg ${m.role}`}>
              <div className="avatar">{m.role==="user" ? "You" : "EF"}</div>
              <div className="bubble">{m.content || (busy && i===messages.length-1 ? "Thinking…" : "")}
                {m.final && <div className="meta-row">
                  <span className={`status-chip ${m.final.abstained ? "warn" : "good"}`}>{m.final.abstained ? "Abstained" : m.final.grounding_status || "Grounded"}</span>
                  <span>{m.final.verified_count ?? 0} verified citations</span>
                  {m.final.used_web && <span>Web research used</span>}
                </div>}
              </div>
            </div>)}
          </div>
          <div className="composer">
            <textarea value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();void send();}}} placeholder="Ask a question about your documents or the web…" />
            <div className="composer-row"><span>{status}</span><button className="send" disabled={busy || !input.trim()} onClick={() => void send()}>{busy ? "Working…" : "Send"}</button></div>
          </div>
        </main>

        <aside className="evidence">
          <div className="side-title">Evidence</div>
          {selectedEvidence ? <div><div className="evidence-title">{selectedEvidence.evidence_id || "Document"}</div><div className="evidence-body">{selectedEvidence.content}</div><button onClick={()=>setSelectedEvidence(null)}>Close</button></div> : <div className="muted">Select a document or use a completed answer to inspect evidence.</div>}
          <div className="evidence-list">{evidence.slice(0,8).map((e:any,i)=><button key={i} onClick={()=>setSelectedEvidence(e)}><strong>{e.evidence_id || `E${i+1}`}</strong><span>{(e.content||"").slice(0,100)}…</span></button>)}</div>
        </aside>
      </div>
    </div>
  )
}

createRoot(document.getElementById("root")!).render(<App />);
