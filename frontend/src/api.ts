export interface SessionInfo {
  active: boolean;
  project: string;
  provider: string;
  model: string;
  expires_at: number;
}

export interface ChatSummary {
  thread_id: string;
  title: string;
  updated_at: string | null;
  message_count: number;
}

export interface EvidenceChunk {
  evidence_id: string;
  document_id: string;
  content: string;
  score: number;
  meta_data: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  kind?: string;
  meta?: {
    chunks?: EvidenceChunk[];
    web_sources?: Array<Record<string, unknown>>;
    agent_trace?: Array<Record<string, unknown>>;
    quality?: Record<string, unknown>;
    grounding_status?: string | null;
    abstained?: boolean;
    [key: string]: unknown;
  };
}

export interface ChatRecord extends ChatSummary {
  messages: ChatMessage[];
  uploaded_docs: DocumentRecord[];
  created_at: string;
}

export interface DocumentRecord {
  name: string;
  document_id: string;
  chunk_count: number;
  content_hash: string;
  owner_scope: string;
  scope: "global" | "chat";
  created_at: string;
  visual_pages_processed?: number;
  visual_pages_skipped?: number;
  modalities?: string[];
}

export interface UploadResult {
  document: DocumentRecord;
  duplicate: boolean;
}

export interface SseEvent {
  event: string;
  data: unknown;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("X-Requested-With", "fetch");
  const response = await fetch(path, {
    ...init,
    headers,
    credentials: "same-origin",
    cache: "no-store",
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
      else if (typeof body?.message === "string") detail = body.message;
    } catch {
      // Keep the generic status message for non-JSON responses.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function decodeSseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim() || "message";
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (data.length === 0) return null;
  const raw = data.join("\n");
  try {
    return { event, data: JSON.parse(raw) as unknown };
  } catch {
    return { event, data: raw };
  }
}

export async function getSession(): Promise<SessionInfo> {
  return request<SessionInfo>("/api/v1/session");
}

export async function exchangeSession(code: string): Promise<SessionInfo> {
  return request<SessionInfo>("/api/v1/session/exchange", {
    method: "POST",
    body: JSON.stringify({ code }),
  });
}

export async function revokeSession(): Promise<void> {
  await request<{ revoked: boolean }>("/api/v1/session/revoke", { method: "POST" });
}

export async function listChats(): Promise<ChatSummary[]> {
  const result = await request<{ data: ChatSummary[] }>("/api/v1/chats");
  return result.data;
}

export async function createChat(): Promise<ChatRecord> {
  return request<ChatRecord>("/api/v1/chats", { method: "POST" });
}

export async function getChat(threadId: string): Promise<ChatRecord> {
  return request<ChatRecord>(`/api/v1/chats/${encodeURIComponent(threadId)}`);
}

export async function deleteChat(threadId: string): Promise<void> {
  await request<{ deleted: boolean }>(`/api/v1/chats/${encodeURIComponent(threadId)}`, {
    method: "DELETE",
  });
}

export async function listDocuments(
  scope: "global" | "chat" | "all" = "all",
  threadId?: string,
): Promise<DocumentRecord[]> {
  const params = new URLSearchParams({ scope });
  if (threadId) params.set("thread_id", threadId);
  const result = await request<{ data: DocumentRecord[] }>(
    `/api/v1/documents?${params.toString()}`,
  );
  return result.data;
}

export async function uploadDocument(
  file: File,
  scope: "global" | "chat",
  threadId?: string,
): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("scope", scope);
  if (threadId) form.append("thread_id", threadId);
  return request<UploadResult>("/api/v1/documents", {
    method: "POST",
    body: form,
  });
}

export async function deleteDocument(
  documentId: string,
  scope: "global" | "chat" = "global",
  threadId?: string,
): Promise<void> {
  const params = new URLSearchParams({ scope });
  if (threadId) params.set("thread_id", threadId);
  await request<{ deleted: boolean }>(
    `/api/v1/documents/${encodeURIComponent(documentId)}?${params.toString()}`,
    { method: "DELETE" },
  );
}

export async function streamChatMessage(
  threadId: string,
  message: string,
  onEvent: (event: SseEvent) => void,
): Promise<void> {
  const response = await fetch(
    `/api/v1/chats/${encodeURIComponent(threadId)}/messages`,
    {
      method: "POST",
      credentials: "same-origin",
      cache: "no-store",
      headers: {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "X-Requested-With": "fetch",
      },
      body: JSON.stringify({ message }),
    },
  );
  if (!response.ok) {
    let detail = `Research request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // Preserve generic status text for non-JSON errors.
    }
    throw new Error(detail);
  }
  if (!response.body) throw new Error("Streaming is not available in this browser.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const dispatchBlocks = () => {
    let boundary = buffer.match(/\r?\n\r?\n/);
    while (boundary && boundary.index !== undefined) {
      const block = buffer.slice(0, boundary.index);
      buffer = buffer.slice(boundary.index + boundary[0].length);
      const parsed = decodeSseBlock(block);
      if (parsed) onEvent(parsed);
      boundary = buffer.match(/\r?\n\r?\n/);
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      dispatchBlocks();
      if (done) break;
    }
    if (buffer.trim()) {
      const parsed = decodeSseBlock(buffer);
      if (parsed) onEvent(parsed);
    }
  } finally {
    reader.releaseLock();
  }
}
