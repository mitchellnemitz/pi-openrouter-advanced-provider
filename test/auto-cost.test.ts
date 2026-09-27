import assert from "node:assert/strict";
import { it } from "node:test";

import * as cost from "../pi/auto-cost.ts";

const message = (model: string, responseId = "gen-a") => ({
  role: "assistant" as const,
  provider: "openrouter",
  model,
  responseId,
  content: [{ type: "text" as const, text: "ok" }],
  api: "openai-completions",
  timestamp: 1,
  stopReason: "stop" as const,
  usage: {
    input: 23, output: 7, cacheRead: 0, cacheWrite: 0, totalTokens: 30,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  },
});

it("captures actual cost from a split OpenRouter SSE stream without changing the response", async () => {
  assert.equal(typeof cost.wrapRouterCostFetch, "function");
  assert.equal(typeof cost.restoreRouterCost, "function");
  const captures = new Map<string, Promise<number | undefined>>();
  const sse = [
    'data: {"id":"gen-a","model":"z-ai/glm-5.3-flash","choices":[]}\n\n',
    'data: {"id":"gen-a","usage":{"prompt_tokens":23,"completion_tokens":7,"cost":0.000123},"choices":[]}\n\n',
    "data: [DONE]\n\n",
  ].join("");
  const originalFetch: typeof fetch = async () => new Response(new ReadableStream({
    start(controller) {
      const bytes = new TextEncoder().encode(sse);
      controller.enqueue(bytes.slice(0, 9));
      controller.enqueue(bytes.slice(9, 64));
      controller.enqueue(bytes.slice(64));
      controller.close();
    },
  }), { headers: { "content-type": "text/event-stream" } });
  const wrapped = cost.wrapRouterCostFetch(originalFetch, captures);
  const response = await wrapped("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST", body: JSON.stringify({ model: "openrouter/auto" }),
  });
  assert.equal(await response.text(), sse);
  const original = message("openrouter/auto");
  const replaced = await cost.restoreRouterCost(original, captures);
  assert.equal(replaced?.usage.cost.total, 0.000123);
  assert.equal(replaced?.usage.input, 23);
  assert.equal(original.usage.cost.total, 0);
});

it("restores cost for auto-beta but not concrete models or unrelated responses", async () => {
  assert.equal(typeof cost.restoreRouterCost, "function");
  const captures = new Map([["gen-a", Promise.resolve(0.0042)]]);
  assert.equal((await cost.restoreRouterCost(message("openrouter/auto-beta"), captures))?.usage.cost.total, 0.0042);
  assert.equal(await cost.restoreRouterCost(message("z-ai/glm-5.3"), captures), undefined);
  assert.equal(await cost.restoreRouterCost(message("openrouter/auto", "other-gen"), captures), undefined);
});

it("overrides the built-in negative router price sentinel", async () => {
  const original = message("openrouter/auto");
  original.usage.cost.total = -42;
  const captures = new Map([["gen-a", Promise.resolve(0.001)]]);
  assert.equal((await cost.restoreRouterCost(original, captures))?.usage.cost.total, 0.001);
});

it("leaves the message unchanged when the stream has no billed cost", async () => {
  assert.equal(typeof cost.restoreRouterCost, "function");
  const captures = new Map([["gen-a", Promise.resolve(undefined)]]);
  assert.equal(await cost.restoreRouterCost(message("openrouter/auto"), captures), undefined);
});

it("does not inspect concrete-model requests", async () => {
  assert.equal(typeof cost.wrapRouterCostFetch, "function");
  const captures = new Map<string, Promise<number | undefined>>();
  const response = new Response("unchanged", { headers: { "content-type": "text/event-stream" } });
  const wrapped = cost.wrapRouterCostFetch(async () => response, captures);
  const result = await wrapped("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST", body: JSON.stringify({ model: "z-ai/glm-5.3" }),
  });
  assert.equal(result, response);
  assert.equal(captures.size, 0);
});
