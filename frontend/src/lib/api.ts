export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export const WS_BASE = API_BASE.replace(/^http/, "ws");

// ---- wire types (mirror chimera/api/server.py) ----
export type TokenMsg = {
  type: "token";
  kind: "prefill" | "decode";
  token: string;
  token_id: number;
  latency_ms: number;
  cache_tokens: number;
  cache_bytes: number;
  index: number;
  probability: number;
  blocks: number[] | null;
};

export type StartMsg = {
  type: "start";
  request_id: number;
  prompt_tokens: { id: number; text: string }[];
  use_cache: boolean;
  paged: boolean;
};

export type Metrics = {
  wall_s: number;
  total_tokens: number;
  throughput_tok_s: number;
  completed_requests: number;
  avg_ttft_s: number;
  avg_tpot_ms: number;
  peak_cache_mb: number;
};

export type DoneMsg = { type: "done"; metrics: Metrics };
export type ErrorMsg = { type: "error"; message: string };
export type GenMsg = StartMsg | TokenMsg | DoneMsg | ErrorMsg;

export type Seat = { id: number; tokens: number; target: number };
export type PoolStats = {
  num_blocks: number;
  blocks_used: number;
  blocks_free: number;
  block_utilization: number;
  pool_bytes: number;
};
export type TickMsg = {
  type: "tick";
  tick: number;
  seated: Seat[];
  queued: number;
  completed: number;
  admitted: number[];
  evicted: number[];
  capacity: number;
  utilization: number;
  pool: PoolStats | null;
};
export type SimDone = {
  type: "done";
  stats: {
    wall_s: number;
    throughput_tok_s: number;
    avg_ttft_s: number;
    utilization: number;
    completed: number;
  };
  outputs: { id: number; text: string }[];
};
export type SimMsg =
  | { type: "start"; policy: string; max_batch_size: number; num_requests: number; paged: boolean }
  | TickMsg
  | SimDone
  | ErrorMsg;

export type AttentionResp = {
  tokens: { id: number; text: string }[];
  layer: number;
  head: number;
  n_layer: number;
  n_head: number;
  attention: number[][];
};

export type CacheBench = {
  device: string;
  rows: { seq_len: number; naive_ms: number; cached_ms: number; speedup: number }[];
};

export type PagedBench = {
  budget_mb: number;
  workload: number;
  naive: { fit: number; used_mb: number; reserved_mb: number; waste: number };
  paged: {
    fit: number;
    used_mb: number;
    allocated_mb: number;
    waste: number;
    num_blocks: number;
  };
};

export async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export async function getConfig() {
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) throw new Error("config failed");
  return res.json();
}

/** Open a websocket, send the request, and hand every message to onMsg. */
export function stream<T>(
  path: string,
  request: unknown,
  onMsg: (m: T) => void,
  onClose?: () => void
): () => void {
  const ws = new WebSocket(`${WS_BASE}${path}`);
  ws.onopen = () => ws.send(JSON.stringify(request));
  ws.onmessage = (e) => onMsg(JSON.parse(e.data) as T);
  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();
  return () => ws.close();
}
