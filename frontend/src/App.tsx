import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowDown, ArrowUpRight, BookOpen, Check, ChevronDown,
  CircleHelp, Clock3, FileText, FolderOpen, History, LoaderCircle,
  LogOut, Menu, MessageSquare, MoreHorizontal, Plus, Search, Send,
  ShieldCheck, Sparkles, Trash2, Upload, X, PanelRightClose, PanelRightOpen,
} from "lucide-react";
import { AppearanceControl } from "./AppearanceControl";
import { applyThemePreference, getInitialThemePreference } from "./theme";
import type { ThemePreference } from "./theme";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  ChatMessage, ChatRecord, ChatSummary, DocumentRecord, EvidenceChunk,
  SessionInfo, createChat, deleteChat, deleteDocument, exchangeSession,
  getChat, getSession, listChats, listDocuments, revokeSession,
  streamChatMessage, uploadDocument,
} from "./api";

type WorkspaceView = "research" | "knowledge" | "history";

function formatEvidenceCitations(content: string): string {
  return content.replace(
    /\[EVID:\s*(E\d+)\]/gi,
    (_marker, id: string) => `[${id.toUpperCase()}](#evidence-${id.toUpperCase()})`,
  );
}

function formatDate(value?: string | null): string {
  if (!value) return "Just now";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Just now";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

function sourceName(chunk: EvidenceChunk): string {
  const metadata = chunk.meta_data || {};
  return String(metadata.source_name || metadata.file_name || metadata.document_name || "Knowledge-base document");
}

function AuthGate({ error }: { error: string }) {
  return (
    <main className="auth-screen">
      <div className="auth-card">
        <div className="brand-mark brand-mark-large"><Sparkles size={23} /></div>
        <p className="eyebrow">EVIDENCEFLOW · VERIFIED RESEARCH</p>
        <h1>A quieter way to work with evidence.</h1>
        <p className="auth-copy">
          EvidenceFlow uses your portfolio AI session to protect access to your research workspace.
          Start from the AI Engineering Studio portfolio to connect a provider and open this app.
        </p>
        {error ? <div className="notice notice-error">{error}</div> : null}
        <a className="primary-button auth-link" href="https://studio.asaifali-studio.workers.dev">
          Return to portfolio <ArrowUpRight size={16} />
        </a>
        <div className="auth-foot"><ShieldCheck size={14} /> No provider key is stored in this browser.</div>
      </div>
    </main>
  );
}

function DocumentRow({
  document, onDelete, compact = false,
}: {
  document: DocumentRecord;
  onDelete: (document: DocumentRecord) => void;
  compact?: boolean;
}) {
  return (
    <div className={compact ? "document-row document-row-compact" : "document-row"}>
      <div className="document-icon"><FileText size={16} /></div>
      <div className="document-copy">
        <strong title={document.name}>{document.name}</strong>
        <span>{document.chunk_count} evidence segments · {formatDate(document.created_at)}</span>
      </div>
      <button className="icon-button subtle-icon" title="Remove document" onClick={() => onDelete(document)}>
        <Trash2 size={14} />
      </button>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <article className={isUser ? "message-row message-user" : "message-row message-assistant"}>
      {!isUser ? <div className="assistant-avatar"><Sparkles size={15} /></div> : null}
      <div className="message-stack">
        <div className="message-label">{isUser ? "You" : "EvidenceFlow"}</div>
        <div className={isUser ? "message-bubble user-bubble" : "message-bubble assistant-bubble"}>
          {isUser
            ? <p>{message.content}</p>
            : <div className="markdown-content"><ReactMarkdown remarkPlugins={[remarkGfm]}>{formatEvidenceCitations(message.content)}</ReactMarkdown></div>}
        </div>
      </div>
    </article>
  );
}

function EvidenceInspector({
  meta, collapsed, mobileOpen, onToggle,
}: {
  meta: ChatMessage["meta"] | undefined;
  collapsed: boolean;
  mobileOpen: boolean;
  onToggle: () => void;
}) {
  const chunks = meta?.chunks || [];
  const webSources = meta?.web_sources || [];
  const trace = meta?.delegations || meta?.agent_trace || [];
  const [tab, setTab] = useState<"sources" | "trace">("sources");

  return (
    <aside className={["inspector", collapsed ? "inspector-collapsed" : "", mobileOpen ? "inspector-mobile-open" : ""].filter(Boolean).join(" ")}>
      <div className="inspector-head">
        <div>
          <p className="eyebrow">RESEARCH CONTEXT</p>
          <h2>Evidence</h2>
        </div>
        <button className="icon-button" onClick={onToggle} title={collapsed ? "Open evidence panel" : "Close evidence panel"}>
          {collapsed ? <PanelRightOpen size={17} /> : <PanelRightClose size={17} />}
        </button>
      </div>
      {!collapsed ? (
        <>
          <div className="inspector-tabs">
            <button className={tab === "sources" ? "inspector-tab active" : "inspector-tab"} onClick={() => setTab("sources")}>
              Sources <span>{chunks.length + webSources.length}</span>
            </button>
            <button className={tab === "trace" ? "inspector-tab active" : "inspector-tab"} onClick={() => setTab("trace")}>
              Activity <span>{trace.length}</span>
            </button>
          </div>
          {tab === "sources" ? (
            <div className="inspector-scroll">
              {chunks.length === 0 && webSources.length === 0 ? (
                <div className="inspector-empty">
                  <div className="empty-orbit"><BookOpen size={19} /></div>
                  <strong>Evidence appears here</strong>
                  <p>Ask a question about your documents or request external research. Retrieved passages and source details will be shown here.</p>
                </div>
              ) : (
                <>
                  {chunks.map((chunk, index) => (
                    <div className="evidence-card" id={`evidence-${chunk.evidence_id}`} key={chunk.evidence_id || chunk.document_id + index}>
                      <div className="evidence-card-top">
                        <span className="evidence-index">{chunk.evidence_id || "SOURCE"}</span>
                        <span className="evidence-score">{typeof chunk.score === "number" ? Math.round(chunk.score * 100) + " relevance" : "Knowledge base"}</span>
                      </div>
                      <strong>{sourceName(chunk)}</strong>
                      <p>{chunk.content}</p>
                      <div className="evidence-foot">
                        <span>{String(chunk.meta_data?.page_no ? "Page " + chunk.meta_data.page_no : chunk.meta_data?.section || "Retrieved passage")}</span>
                      </div>
                    </div>
                  ))}
                  {webSources.map((source, index) => (
                    <a className="evidence-card web-evidence" key={String(source.url || index)} href={String(source.url || "#")} target="_blank" rel="noreferrer">
                      <div className="evidence-card-top"><span className="evidence-index">WEB {index + 1}</span><ArrowUpRight size={13} /></div>
                      <strong>{String(source.title || source.name || source.url || "Web source")}</strong>
                      <p>{String(source.snippet || source.content || "External research source")}</p>
                    </a>
                  ))}
                </>
              )}
            </div>
          ) : (
            <div className="inspector-scroll">
              {trace.length === 0 ? (
                <div className="inspector-empty"><Activity size={20} /><strong>No activity yet</strong><p>Agent activity will appear after a research turn completes.</p></div>
              ) : trace.map((step, index) => (
                <div className="trace-item" key={String(step.agent || index)}>
                  <div className="trace-marker"><Check size={12} /></div>
                  <div><strong>{String(step.description || step.subagent_type || "Research step")}</strong><p>{String(step.result || step.responsibility || "")}</p></div>
                </div>
              ))}
            </div>
          )}
          <div className="inspector-foot">
            <ShieldCheck size={14} />
            <span>Evidence-first answers. Unsupported claims are withheld.</span>
          </div>
        </>
      ) : null}
    </aside>
  );
}

function App() {
  const [themePreference, setThemePreference] = useState<ThemePreference>(getInitialThemePreference);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [authError, setAuthError] = useState("");
  const [booting, setBooting] = useState(true);
  const [view, setView] = useState<WorkspaceView>("research");
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [globalDocuments, setGlobalDocuments] = useState<DocumentRecord[]>([]);
  const [chatDocuments, setChatDocuments] = useState<DocumentRecord[]>([]);
  const [threadId, setThreadId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [streamText, setStreamText] = useState("");
  const [statusText, setStatusText] = useState("Preparing your research workspace");
  const [notice, setNotice] = useState("");
  const [uploading, setUploading] = useState(false);
  const [inspectorCollapsed, setInspectorCollapsed] = useState(false);
  const [inspectorMobileOpen, setInspectorMobileOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [search, setSearch] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const knowledgeInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    try {
      applyThemePreference(themePreference, document.documentElement, window.localStorage);
    } catch {
      applyThemePreference(themePreference, document.documentElement);
    }
  }, [themePreference]);

  const activeChat = chats.find((chat) => chat.thread_id === threadId);
  const latestMeta = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant" && message.meta)?.meta,
    [messages],
  );
  const filteredChats = chats.filter((chat) =>
    (chat.title || "").toLowerCase().includes(search.toLowerCase()),
  );

  const refreshLists = useCallback(async (currentThread?: string) => {
    const [chatItems, docs, scopedDocs] = await Promise.all([
      listChats(),
      listDocuments("global"),
      currentThread ? listDocuments("chat", currentThread) : Promise.resolve([]),
    ]);
    setChats(chatItems);
    setGlobalDocuments(docs);
    setChatDocuments(scopedDocs);
  }, []);

  const openConversation = useCallback(async (id: string) => {
    const chat = await getChat(id);
    setThreadId(id);
    setMessages(chat.messages || []);
    setChatDocuments(chat.uploaded_docs || []);
    setNotice("");
    setView("research");
    setSidebarOpen(false);
  }, []);

  const initializeWorkspace = useCallback(async (info: SessionInfo) => {
    const chatItems = await listChats();
    const docs = await listDocuments("global");
    setChats(chatItems);
    setGlobalDocuments(docs);
    setSession(info);
    const preferred = chatItems[0]?.thread_id;
    if (preferred) {
      await openConversation(preferred);
    } else {
      const created = await createChat();
      await refreshLists(created.thread_id);
      await openConversation(created.thread_id);
    }
  }, [openConversation, refreshLists]);

  useEffect(() => {
    let alive = true;
    const initialize = async () => {
      const url = new URL(window.location.href);
      const handoff = url.searchParams.get("portfolio_llm_handoff");
      if (handoff) {
        url.searchParams.delete("portfolio_llm_handoff");
        window.history.replaceState({}, "", url.pathname + url.search + url.hash);
        try {
          const info = await exchangeSession(handoff);
          if (alive) await initializeWorkspace(info);
        } catch (error) {
          if (alive) setAuthError(error instanceof Error ? error.message : "Could not connect your portfolio session.");
        } finally {
          if (alive) setBooting(false);
        }
        return;
      }
      try {
        const info = await getSession();
        if (alive) await initializeWorkspace(info);
      } catch (error) {
        if (alive && error instanceof Error && !error.message.includes("missing or expired")) {
          setAuthError(error.message);
        }
      } finally {
        if (alive) setBooting(false);
      }
    };
    void initialize();
    return () => { alive = false; };
  }, [initializeWorkspace]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: streaming ? "auto" : "smooth", block: "end" });
  }, [messages, streamText, streaming]);

  const startNewChat = async () => {
    try {
      const chat = await createChat();
      await refreshLists(chat.thread_id);
      await openConversation(chat.thread_id);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not create a conversation.");
    }
  };

  const handleGlobalUpload = async (selected: FileList | File[] | null) => {
    const picked = selected ? Array.from(selected) : [];
    if (!picked.length) return;
    setUploading(true);
    setNotice("");
    try {
      for (const file of picked) await uploadDocument(file, "global");
      await refreshLists(threadId || undefined);
      setNotice(picked.length === 1 ? "Document added to your knowledge base." : picked.length + " documents added to your knowledge base.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Document upload failed.");
    } finally {
      setUploading(false);
      if (knowledgeInputRef.current) knowledgeInputRef.current.value = "";
    }
  };

  const handleDeleteDocument = async (document: DocumentRecord) => {
    try {
      await deleteDocument(document.document_id, document.scope, document.scope === "chat" ? threadId : undefined);
      await refreshLists(threadId || undefined);
      setNotice("Removed " + document.name + ".");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Document could not be removed.");
    }
  };

  const handleDeleteChat = async (id: string) => {
    try {
      await deleteChat(id);
      const remaining = chats.filter((chat) => chat.thread_id !== id);
      setChats(remaining);
      if (id === threadId) {
        if (remaining[0]) await openConversation(remaining[0].thread_id);
        else {
          const created = await createChat();
          await refreshLists(created.thread_id);
          await openConversation(created.thread_id);
        }
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Conversation could not be deleted.");
    }
  };

  const sendMessage = async () => {
    const text = draft.trim();
    if ((!text && files.length === 0) || streaming || uploading) return;
    let activeId = threadId;
    try {
      if (!activeId) {
        const chat = await createChat();
        activeId = chat.thread_id;
        setThreadId(activeId);
      }
      if (files.length) {
        setUploading(true);
        for (const file of files) await uploadDocument(file, "chat", activeId);
        setFiles([]);
        await refreshLists(activeId);
        setUploading(false);
      }
      if (!text) {
        await openConversation(activeId);
        setNotice("Attachments added to this conversation.");
        return;
      }

      setNotice("");
      setMessages((previous) => [...previous, {
        id: "optimistic-" + Date.now(),
        role: "user",
        content: text,
        created_at: new Date().toISOString(),
      }]);
      setDraft("");
      setStreaming(true);
      setStreamText("");
      setStatusText("Starting research");
      const streamState: { finalMessage: ChatMessage | null } = { finalMessage: null };
      let streamError = "";
      await streamChatMessage(activeId, text, (event) => {
        const payload = event.data as Record<string, unknown>;
        if (event.event === "status") setStatusText(String(payload.text || "Working"));
        if (event.event === "token") setStreamText((previous) => previous + String(payload.text || ""));
        if (event.event === "final") {
          streamState.finalMessage = payload.message as ChatMessage;
          if (streamState.finalMessage) {
            setMessages((previous) => [...previous, streamState.finalMessage as ChatMessage]);
          }
        }
        if (event.event === "error") {
          streamError = String(payload.message || "Research failed.");
          setNotice(streamError);
        }
      });
      if (!streamState.finalMessage) {
        const chat = await getChat(activeId);
        setMessages(chat.messages || []);
      }
      await refreshLists(activeId);
      setStreamText("");
      if (streamError) setNotice(streamError);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Research request failed.");
      if (activeId) {
        try {
          const chat = await getChat(activeId);
          setMessages(chat.messages || []);
        } catch {
          // Preserve the visible error if the history refresh also fails.
        }
      }
    } finally {
      setUploading(false);
      setStreaming(false);
      setStreamText("");
      setStatusText("Ready");
    }
  };

  const toggleInspector = () => {
    if (window.matchMedia("(max-width: 700px)").matches) {
      setInspectorMobileOpen((value) => !value);
    } else {
      setInspectorCollapsed((value) => !value);
    }
  };

  const closeInspector = () => {
    if (window.matchMedia("(max-width: 700px)").matches) {
      setInspectorMobileOpen(false);
    } else {
      setInspectorCollapsed(true);
    }
  };

  const handleLogout = async () => {
    try { await revokeSession(); } catch { /* Expired sessions can still be cleared locally. */ }
    setSession(null);
    setMessages([]);
    setChats([]);
    setGlobalDocuments([]);
    setAuthError("Your EvidenceFlow session has been disconnected. Launch the app again from the portfolio.");
  };

  if (booting) {
    return <main className="boot-screen"><div className="boot-mark"><Sparkles size={20} /></div><div className="boot-line" /><p>Preparing EvidenceFlow</p></main>;
  }
  if (!session) return <AuthGate error={authError} />;

  const chatDocs = chatDocuments;
  const allScopedDocs = [...globalDocuments, ...chatDocs];

  return (
    <div className="app-shell">
      <div className={sidebarOpen ? "mobile-scrim visible" : "mobile-scrim"} onClick={() => setSidebarOpen(false)} />
      <aside className={sidebarOpen ? "sidebar sidebar-open" : "sidebar"}>
        <div className="sidebar-brand">
          <div className="brand-mark"><Sparkles size={17} /></div>
          <div><strong>EvidenceFlow</strong><span>Verified research</span></div>
          <button className="icon-button mobile-close" onClick={() => setSidebarOpen(false)}><X size={17} /></button>
        </div>
        <button className="new-chat-button" onClick={startNewChat}><Plus size={16} /> New research</button>
        <nav className="primary-nav" aria-label="Main navigation">
          <button className={view === "research" ? "nav-item active" : "nav-item"} onClick={() => setView("research")}><MessageSquare size={17} /> Research</button>
          <button className={view === "knowledge" ? "nav-item active" : "nav-item"} onClick={() => setView("knowledge")}><FolderOpen size={17} /> Knowledge base <span className="nav-count">{globalDocuments.length}</span></button>
          <button className={view === "history" ? "nav-item active" : "nav-item"} onClick={() => setView("history")}><History size={17} /> History</button>
        </nav>
        <div className="sidebar-divider" />
        <div className="sidebar-section-head">
          <span>YOUR LIBRARY</span>
          <button className="icon-button" onClick={() => knowledgeInputRef.current?.click()} title="Add documents"><Plus size={15} /></button>
        </div>
        <div className="sidebar-doc-list">
          {globalDocuments.length ? globalDocuments.slice(0, 8).map((document) => (
            <button className="sidebar-doc" key={document.document_id} onClick={() => setView("knowledge")}>
              <FileText size={14} /><span>{document.name}</span>
            </button>
          )) : <p className="sidebar-muted">Your sources will live here.</p>}
        </div>
        <input ref={knowledgeInputRef} type="file" multiple hidden accept=".pdf,.doc,.docx,.txt,.md,.csv,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff" onChange={(event) => void handleGlobalUpload(event.target.files)} />
        <div className="sidebar-bottom">
          <div className="provider-card">
            <div className="provider-dot"><ShieldCheck size={14} /></div>
            <div><span>AI session connected</span><strong>{session.provider || "Portfolio Gateway"}</strong></div>
            <span className="connected-dot" />
          </div>
          <button className="account-button" onClick={handleLogout}>
            <div className="account-avatar">EF</div>
            <div className="account-copy"><strong>Research session</strong><span>{session.model || "Connected through portfolio"}</span></div>
            <LogOut size={15} />
          </button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" onClick={() => setSidebarOpen(true)}><Menu size={19} /></button>
            <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><strong>{view === "research" ? "Research" : view === "knowledge" ? "Knowledge base" : "History"}</strong></div>
          </div>
          <div className="topbar-actions">
            <AppearanceControl value={themePreference} onChange={setThemePreference} />
            <span className="privacy-pill"><span className="connected-dot" /> Private workspace</span>
            <button className="icon-button desktop-evidence-toggle" onClick={toggleInspector} title="Toggle evidence panel">
              {inspectorCollapsed ? <PanelRightOpen size={17} /> : <PanelRightClose size={17} />}
            </button>
            <button className="icon-button mobile-evidence-toggle" onClick={() => setInspectorMobileOpen(true)} title="Open evidence panel">
              <BookOpen size={17} />
            </button>
            <a className="help-link" href="https://github.com/AsaifAli/LangGraph-RAG" target="_blank" rel="noreferrer"><CircleHelp size={16} /></a>
          </div>
        </header>

        <div className="workspace-layout">
          <section className="content-column">
            {view === "research" ? (
              <div className="research-view">
                <div className="research-heading">
                  <div>
                    <div className="eyebrow-row"><span className="eyebrow">RESEARCH WORKSPACE</span><span className="soft-status"><span className="connected-dot" /> Online</span></div>
                    <h1>{activeChat?.title || "What would you like to understand?"}</h1>
                    <p>Ask across your documents, explore external sources, and inspect the evidence behind every answer.</p>
                  </div>
                  <button className="secondary-button heading-new" onClick={startNewChat}><Plus size={15} /> New chat</button>
                </div>
                <div className="research-canvas">
                  {messages.length === 0 && !streaming ? (
                    <div className="welcome-state">
                      <div className="welcome-orbit"><div className="welcome-core"><Sparkles size={24} /></div><span className="orbit-dot orbit-dot-one" /><span className="orbit-dot orbit-dot-two" /></div>
                      <p className="eyebrow">START WITH A QUESTION</p>
                      <h2>Make your next decision<br />with better evidence.</h2>
                      <p className="welcome-copy">Search your knowledge base, compare ideas across documents, or ask for current external research. EvidenceFlow will show where its answer comes from.</p>
                      <div className="suggestion-grid">
                        <button className="suggestion-card" onClick={() => setDraft("Summarize the key findings in my uploaded documents.")}><BookOpen size={17} /><span><strong>Explore my documents</strong><small>Find themes and key findings</small></span><ArrowUpRight size={14} /></button>
                        <button className="suggestion-card" onClick={() => setDraft("Compare the strongest evidence for and against this topic.")}><Search size={17} /><span><strong>Compare evidence</strong><small>Surface agreement and conflict</small></span><ArrowUpRight size={14} /></button>
                      </div>
                    </div>
                  ) : (
                    <div className="message-list">
                      {messages.map((message) => <MessageBubble key={message.id} message={message} />)}
                      {streaming ? (
                        <article className="message-row message-assistant">
                          <div className="assistant-avatar"><Sparkles size={15} /></div>
                          <div className="message-stack">
                            <div className="message-label">EvidenceFlow <span className="thinking-status"><LoaderCircle size={12} /> {statusText}</span></div>
                            <div className="message-bubble assistant-bubble">
                              {streamText ? <div className="markdown-content"><ReactMarkdown remarkPlugins={[remarkGfm]}>{streamText.replace(/\[EVID:\s*(E\d+)\]/gi, (_marker, id: string) => `[${id.toUpperCase()}](#evidence-${id.toUpperCase()})`)}</ReactMarkdown></div> : <div className="thinking-dots"><i /><i /><i /></div>}
                            </div>
                          </div>
                        </article>
                      ) : null}
                      <div ref={bottomRef} />
                    </div>
                  )}
                </div>
                <div className="composer-wrap">
                  {files.length ? (
                    <div className="attachment-strip">
                      {files.map((file, index) => <span className="attachment-chip" key={file.name + index}><FileText size={13} />{file.name}<button onClick={() => setFiles((items) => items.filter((_, i) => i !== index))}><X size={12} /></button></span>)}
                    </div>
                  ) : null}
                  <div className="composer">
                    <button className="composer-add" title="Attach document to this conversation" onClick={() => fileInputRef.current?.click()}><Plus size={18} /></button>
                    <input ref={fileInputRef} hidden type="file" multiple accept=".pdf,.doc,.docx,.txt,.md,.csv,.png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff" onChange={(event) => setFiles((items) => [...items, ...Array.from(event.target.files || [])])} />
                    <textarea
                      value={draft}
                      onChange={(event) => setDraft(event.target.value)}
                      onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage(); } }}
                      placeholder="Ask a question about your evidence..."
                      rows={1}
                      disabled={streaming || uploading}
                    />
                    <div className="composer-right"><span className="composer-hint">Enter to send · Shift + Enter for a new line</span><button className="send-button" disabled={(!draft.trim() && files.length === 0) || streaming || uploading} onClick={() => void sendMessage()} title="Send message">{uploading ? <LoaderCircle size={17} className="spin" /> : streaming ? <LoaderCircle size={17} className="spin" /> : <Send size={16} />}</button></div>
                  </div>
                  <div className="composer-foot"><span><ShieldCheck size={12} /> Your workspace is scoped to your verified session.</span><span>{allScopedDocs.length} source{allScopedDocs.length === 1 ? "" : "s"} in scope</span></div>
                </div>
              </div>
            ) : view === "knowledge" ? (
              <div className="page-view">
                <div className="page-heading">
                  <div><p className="eyebrow">YOUR RESEARCH LIBRARY</p><h1>Knowledge base</h1><p>Documents available to your research conversations.</p></div>
                  <button className="primary-button" onClick={() => knowledgeInputRef.current?.click()} disabled={uploading}>{uploading ? <LoaderCircle size={15} className="spin" /> : <Upload size={15} />} Add documents</button>
                </div>
                <div className="storage-card">
                  <div className="storage-icon"><FolderOpen size={19} /></div>
                  <div className="storage-copy"><strong>{globalDocuments.length} documents in your library</strong><span>Global sources are available across all your research conversations.</span></div>
                  <span className="storage-status"><span className="connected-dot" /> Indexed</span>
                </div>
                <div className="section-title"><div><h2>Library documents</h2><p>Uploaded and indexed sources</p></div><span>{globalDocuments.length} items</span></div>
                <div className="document-list-panel">
                  {globalDocuments.length ? globalDocuments.map((document) => <DocumentRow key={document.document_id} document={document} onDelete={(item) => void handleDeleteDocument(item)} />) : (
                    <div className="empty-panel"><div className="empty-orbit"><Upload size={19} /></div><strong>Your library is ready for its first source.</strong><p>Add PDFs, Office documents, text files, CSVs or supported images. EvidenceFlow will index them for grounded research.</p><button className="secondary-button" onClick={() => knowledgeInputRef.current?.click()}><Upload size={15} /> Upload a document</button></div>
                  )}
                </div>
                {chatDocs.length ? <>
                  <div className="section-title section-title-spaced"><div><h2>Attached to this conversation</h2><p>These documents are only available inside the current chat.</p></div><span>{chatDocs.length} items</span></div>
                  <div className="document-list-panel">{chatDocs.map((document) => <DocumentRow key={document.document_id} document={document} onDelete={(item) => void handleDeleteDocument(item)} />)}</div>
                </> : null}
              </div>
            ) : (
              <div className="page-view">
                <div className="page-heading"><div><p className="eyebrow">YOUR WORK</p><h1>Research history</h1><p>Return to a previous line of investigation.</p></div><button className="primary-button" onClick={startNewChat}><Plus size={15} /> New research</button></div>
                <div className="history-tools"><div className="search-field"><Search size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search conversations" /></div><span>{filteredChats.length} conversations</span></div>
                <div className="history-list">
                  {filteredChats.length ? filteredChats.map((chat) => (
                    <div className={chat.thread_id === threadId ? "history-card history-card-active" : "history-card"} key={chat.thread_id}>
                      <button className="history-main" onClick={() => void openConversation(chat.thread_id)}>
                        <div className="history-icon"><MessageSquare size={16} /></div>
                        <div><strong>{chat.title || "New conversation"}</strong><span>{chat.message_count} messages · Updated {formatDate(chat.updated_at)}</span></div>
                      </button>
                      <button className="icon-button" title="Delete conversation" onClick={() => void handleDeleteChat(chat.thread_id)}><Trash2 size={15} /></button>
                    </div>
                  )) : <div className="empty-panel"><div className="empty-orbit"><History size={18} /></div><strong>No conversations found</strong><p>Start a research thread and it will appear here.</p></div>}
                </div>
              </div>
            )}
            {notice ? <div className="notice-toast" role="status"><span>{notice}</span><button onClick={() => setNotice("")}><X size={14} /></button></div> : null}
          </section>

          {inspectorMobileOpen ? <button className="inspector-mobile-scrim" aria-label="Close evidence panel" onClick={() => setInspectorMobileOpen(false)} /> : null}
          <EvidenceInspector
            meta={latestMeta}
            collapsed={inspectorCollapsed && !inspectorMobileOpen}
            mobileOpen={inspectorMobileOpen}
            onToggle={closeInspector}
          />
        </div>
      </main>
    </div>
  );
}

export default App;
