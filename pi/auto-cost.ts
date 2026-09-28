import type { MessageEndEvent } from "@earendil-works/pi-coding-agent";

type Assistant = Extract<MessageEndEvent["message"], { role: "assistant" }>;
type Captures = Map<string, Promise<number | undefined>>;

function isRouter(model: string): boolean {
  return model === "openrouter/auto" || model === "openrouter/auto-beta";
}

function isRouterRequest(input: string | URL, init?: RequestInit): boolean {
  // Matches the (url, init) call form pi-ai's transport uses. A Request-object
  // first argument carries its body as a stream and is not recognized.
  if (typeof init?.body !== "string") return false;
  try {
    const raw = typeof input === "string" ? input : input.href;
    const url = new URL(raw);
    return url.hostname === "openrouter.ai" && url.pathname.endsWith("/chat/completions")
      && isRouter(JSON.parse(init.body).model);
  } catch {
    return false;
  }
}

async function captureCost(body: ReadableStream<Uint8Array>, captures: Captures): Promise<void> {
  let id: string | undefined;
  let cost: number | undefined;
  let resolve!: (cost: number | undefined) => void;
  const captured = new Promise<number | undefined>((done) => { resolve = done; });
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) !== -1) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith("data:")) continue;
        let chunk: any;
        try { chunk = JSON.parse(line.slice(5).trim()); } catch { continue; }
        if (!id && typeof chunk.id === "string" && chunk.id) {
          const responseId: string = chunk.id;
          id = responseId;
          captures.set(responseId, captured);
          if (captures.size > 200) captures.delete(captures.keys().next().value!);
        }
        if (typeof chunk.usage?.cost === "number" && Number.isFinite(chunk.usage.cost) && chunk.usage.cost >= 0) {
          cost = chunk.usage.cost;
        }
      }
    }
  } catch {
    // Cost accounting must never interrupt the model stream.
  } finally {
    resolve(cost);
  }
}

export function wrapRouterCostFetch(originalFetch: typeof fetch, captures: Captures): typeof fetch {
  return async (input, init) => {
    if (!isRouterRequest(input as string | URL, init)) return originalFetch(input, init);
    const response = await originalFetch(input, init);
    if (!response.ok || !response.body || !response.headers.get("content-type")?.includes("text/event-stream")) return response;
    const [forPi, forCost] = response.body.tee();
    void captureCost(forCost, captures);
    return new Response(forPi, { status: response.status, statusText: response.statusText, headers: response.headers });
  };
}

export async function restoreRouterCost(message: Assistant, captures: Captures): Promise<Assistant | undefined> {
  if (message.provider !== "openrouter" || !isRouter(message.model) || !message.responseId ||
      message.stopReason === "error" || message.stopReason === "aborted" || message.usage.cost.total > 0) return;
  let captured = captures.get(message.responseId);
  if (!captured) {
    // The tee coroutine usually registers the capture during stream reading,
    // but message_end can win the race; yield once and look again.
    await new Promise((resolve) => setTimeout(resolve, 0));
    captured = captures.get(message.responseId);
  }
  if (!captured) return;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const billed = await Promise.race([
    captured,
    new Promise<undefined>((resolve) => { timer = setTimeout(() => resolve(undefined), 250); }),
  ]);
  clearTimeout(timer);
  captures.delete(message.responseId);
  if (billed === undefined) return;
  return { ...message, usage: { ...message.usage, cost: { ...message.usage.cost, total: billed } } };
}
