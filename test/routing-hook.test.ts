/**
 * Regression tests for the per-request routing hook and the final output
 * guard (pi/index.ts + pi/stream.ts).
 *
 * The bug these cover: the routing hook capped the output budget by writing
 * `max_tokens`, but pi's OpenAI Completions transport writes the budget into
 * `max_completion_tokens` (the legacy field is only used by providers that
 * pin `maxTokensField: "max_tokens"`). The hook therefore added a SECOND,
 * competing budget field instead of clamping the one the transport reads:
 *
 *   - a request whose host budget exceeded the endpoint ceiling kept the
 *     over-ceiling value, so the final guard rejected it even though the hook
 *     was supposed to clamp it;
 *   - a request whose host budget was SMALLER than the ceiling gained a
 *     competing, larger cap, and the guard then reserved that larger output
 *     allowance against the endpoint's context window — rejecting a request
 *     that fits.
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  clampOutputBudget,
  resolveOutputBudgetField,
  routeRequestPayload,
  type RequestRoutingDeps,
} from "../pi/index.ts";
import { estimateWireContextTokens } from "../pi/limits.ts";
import { streamOpenRouterWithRecovery, type StreamSimpleDelegate } from "../pi/stream.ts";
import { clearEndpointLimits, setEndpointLimits } from "../pi/selection-limits-state.ts";
import { setRuntimeOverride } from "../pi/platform.ts";
import {
  createAssistantMessageEventStream,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai/compat";

const MODEL_ID = "vendor/model";
const ENDPOINT_CEILING = 512;

function catalogModel(overrides: Partial<{
  maxTokens: number;
  compat: Record<string, unknown>;
}> = {}) {
  return {
    id: MODEL_ID,
    name: "Vendor Model",
    reasoning: false,
    input: ["text"] as ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128_000,
    maxTokens: overrides.maxTokens ?? 4_096,
    ...(overrides.compat ? { compat: overrides.compat } : {}),
  };
}

/**
 * Deps for a model whose selection pins endpoint `lambda` with a 512-token
 * completion ceiling. The catalog's own maxTokens (4096) is above it, so the
 * effective ceiling is 512.
 */
function deps(overrides: Partial<RequestRoutingDeps> = {}): RequestRoutingDeps {
  const selection = {
    result: {
      pick: "lambda",
      ignore: [],
      contextLength: 4_096,
      maxCompletionTokens: ENDPOINT_CEILING,
      table: [],
      sampledCount: 1,
    },
    timestamp: Date.now(),
  };
  return {
    requestConfig: {},
    selectionTuning: {
      enabled: true,
      priceAnchor: 3,
      budgetWeights: { throughput: 0.55, latency: 0.2, toolCall: 0.15, price: 0.1 },
      flagshipWeights: { throughput: 0.3, latency: 0.2, toolCall: 0.15, price: 0.3 },
    },
    catalogModels: () => [catalogModel()],
    baseMaxTokens: () => 4_096,
    // The selection is per model — a lookup for any other model id finds
    // nothing, exactly as pi/state.ts caches them.
    getSelection: (modelId) => (modelId === MODEL_ID ? selection : undefined),
    getStaleSelection: (modelId) => (modelId === MODEL_ID ? selection : undefined),
    selectProvider: async () => undefined,
    tier: () => null,
    ...overrides,
  };
}

/** The wire body pi's transport builds: the budget lives in one field. */
function wireBody(extra: Record<string, unknown> = {}, contentChars = 0): Record<string, unknown> {
  return {
    model: MODEL_ID,
    messages: [{ role: "user", content: "x".repeat(contentChars) }],
    max_completion_tokens: 65_536,
    ...extra,
  };
}

function fakeModel(maxTokens = ENDPOINT_CEILING): Model<"openai-completions"> {
  return {
    id: MODEL_ID,
    name: "Vendor Model",
    provider: "openrouter",
    api: "openai-completions",
    baseUrl: "https://openrouter.ai/api/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 4_096,
    maxTokens,
  };
}

/**
 * Run the final guard (the onPayload the recovery wrapper installs) over a
 * body, and report whether it accepted the request.
 *
 * The routing hook is wired in as the guard's inner onPayload — exactly how
 * pi's extension runner invokes it — so the hook and the guard run together
 * on the same body.
 */
async function hookThenGuard(
  body: Record<string, unknown>,
  hookDeps: RequestRoutingDeps = deps(),
  contextLength = 8_192,
): Promise<{ ok: boolean; error?: unknown; body?: unknown }> {
  const model = fakeModel();
  const context: TranscriptContext = { messages: [] };
  setRuntimeOverride("pi");
  setEndpointLimits(model.id, { contextLength, maxCompletionTokens: ENDPOINT_CEILING });
  let error: unknown;
  let ok = false;
  let accepted: unknown;
  const delegate: StreamSimpleDelegate = (_model, _context, options?: SimpleStreamOptions) => {
    const stream = createAssistantMessageEventStream();
    void Promise.resolve(options?.onPayload?.(body, model))
      .then((result) => { ok = true; accepted = result; })
      .catch((err: unknown) => { error = err; })
      .then(() => stream.end());
    return stream;
  };
  try {
    const stream = streamOpenRouterWithRecovery(
      model,
      context,
      {
        onPayload: async (payload) => routeRequestPayload(payload, hookDeps),
      },
      delegate,
    );
    for await (const _event of stream) {}
    return { ok, error, body: accepted };
  } finally {
    clearEndpointLimits();
    setRuntimeOverride(undefined);
  }
}

describe("resolveOutputBudgetField", () => {
  it("defaults to max_completion_tokens (pi's OpenAI Completions default)", () => {
    assert.equal(resolveOutputBudgetField(undefined, {}), "max_completion_tokens");
    assert.equal(resolveOutputBudgetField({}, {}), "max_completion_tokens");
  });

  it("honors a compat pin", () => {
    assert.equal(
      resolveOutputBudgetField({ maxTokensField: "max_tokens" }, {}),
      "max_tokens",
    );
    assert.equal(
      resolveOutputBudgetField({ maxTokensField: "max_completion_tokens" }, {}),
      "max_completion_tokens",
    );
  });

  it("falls back to the field the payload already carries", () => {
    assert.equal(
      resolveOutputBudgetField(undefined, { max_completion_tokens: 100 }),
      "max_completion_tokens",
    );
    assert.equal(resolveOutputBudgetField(undefined, { max_tokens: 100 }), "max_tokens");
  });

  it("prefers the compat pin over the payload evidence", () => {
    assert.equal(
      resolveOutputBudgetField({ maxTokensField: "max_tokens" }, { max_completion_tokens: 100 }),
      "max_tokens",
    );
  });
});

describe("clampOutputBudget", () => {
  it("clamps an over-ceiling budget in the transport's own field", () => {
    const result = clampOutputBudget({ max_completion_tokens: 32_768 }, 512, "max_completion_tokens");
    assert.deepEqual(result, { max_completion_tokens: 512 });
  });

  it("never adds the legacy field next to the transport's field", () => {
    const result = clampOutputBudget({ max_completion_tokens: 32_768 }, 512, "max_completion_tokens");
    assert.equal("max_tokens" in (result ?? {}), false);
  });

  it("preserves a budget below the ceiling", () => {
    const result = clampOutputBudget({ max_completion_tokens: 64 }, 512, "max_completion_tokens");
    assert.equal(result, undefined);
  });

  it("bounds a request that carries no budget at all", () => {
    const result = clampOutputBudget({ model: "vendor/model" }, 512, "max_completion_tokens");
    assert.deepEqual(result, { model: "vendor/model", max_completion_tokens: 512 });
  });

  it("clamps the legacy field when the transport uses it", () => {
    const result = clampOutputBudget({ max_tokens: 32_768 }, 512, "max_tokens");
    assert.deepEqual(result, { max_tokens: 512 });
    assert.equal("max_completion_tokens" in (result ?? {}), false);
  });

  it("reads a numeric string budget as a budget", () => {
    const result = clampOutputBudget({ max_completion_tokens: "32768" }, 512, "max_completion_tokens");
    assert.deepEqual(result, { max_completion_tokens: 512 });
  });

  it("replaces an unreadable budget in the transport's field with the ceiling", () => {
    const result = clampOutputBudget({ max_completion_tokens: "lots" }, 512, "max_completion_tokens");
    assert.deepEqual(result, { max_completion_tokens: 512 });
  });

  it("bounds the transport's field but leaves an unreadable foreign field untouched", () => {
    const result = clampOutputBudget({ max_tokens: "lots" }, 512, "max_completion_tokens");
    assert.deepEqual(result, { max_tokens: "lots", max_completion_tokens: 512 });
  });

  it("does nothing for a non-positive or non-finite ceiling", () => {
    assert.equal(clampOutputBudget({ max_completion_tokens: 32_768 }, 0), undefined);
    assert.equal(clampOutputBudget({ max_completion_tokens: 32_768 }, Number.NaN), undefined);
  });

  it("does not mutate the input payload", () => {
    const payload = { max_completion_tokens: 32_768 };
    clampOutputBudget(payload, 512, "max_completion_tokens");
    assert.deepEqual(payload, { max_completion_tokens: 32_768 });
  });
});

describe("routeRequestPayload: output budget field", () => {
  it("clamps the transport's own field (max_completion_tokens) to the endpoint ceiling", async () => {
    const result = (await routeRequestPayload(wireBody(), deps())) as Record<string, unknown>;
    assert.equal(result.max_completion_tokens, ENDPOINT_CEILING);
  });

  it("never adds the legacy max_tokens field the transport does not read", async () => {
    const result = (await routeRequestPayload(wireBody(), deps())) as Record<string, unknown>;
    assert.equal("max_tokens" in result, false);
  });

  it("preserves a host budget SMALLER than the ceiling", async () => {
    const result = (await routeRequestPayload(
      wireBody({ max_completion_tokens: 64 }),
      deps(),
    )) as Record<string, unknown>;
    assert.equal(result.max_completion_tokens, 64);
    assert.equal("max_tokens" in result, false);
  });

  it("bounds a request that carries no budget at all", async () => {
    const result = (await routeRequestPayload(
      { model: MODEL_ID, messages: [{ role: "user", content: "hello" }] },
      deps(),
    )) as Record<string, unknown>;
    assert.equal(result.max_completion_tokens, ENDPOINT_CEILING);
    assert.equal("max_tokens" in result, false);
  });

  it("clamps the legacy field when the model's transport writes that field", async () => {
    const legacyDeps = deps({
      catalogModels: () => [catalogModel({ compat: { maxTokensField: "max_tokens" } })],
    });
    const result = (await routeRequestPayload(
      { model: MODEL_ID, messages: [], max_tokens: 65_536 },
      legacyDeps,
    )) as Record<string, unknown>;
    assert.equal(result.max_tokens, ENDPOINT_CEILING);
    assert.equal("max_completion_tokens" in result, false);
  });

  it("leaves the request alone when there is no selection to clamp against", async () => {
    const noSelection = deps({
      getSelection: () => undefined,
      getStaleSelection: () => undefined,
    });
    // No selection means no pick and no clamp, so the hook returns
    // undefined and the host keeps the original payload untouched.
    assert.equal(await routeRequestPayload(wireBody(), noSelection), undefined);
  });

  it("ignores a payload that is not an OpenRouter request", async () => {
    assert.equal(await routeRequestPayload({ model: "gpt-4o" }, deps()), undefined);
    assert.equal(await routeRequestPayload("nope", deps()), undefined);
    assert.equal(await routeRequestPayload(undefined, deps()), undefined);
  });
});

describe("routing hook + final guard", () => {
  // A body whose input nearly fills the endpoint's context window: reserving
  // the endpoint's 512-token ceiling does not fit, reserving the host's
  // smaller 64-token budget does.
  const CONTEXT_LENGTH = 8_192;
  const FILLING_INPUT_CHARS = 27_284;

  it("accepts an over-ceiling request after the hook clamps it", async () => {
    const { ok, error } = await hookThenGuard(wireBody());
    assert.equal(ok, true, error instanceof Error ? error.message : String(error));
  });

  it("rejects the same request when the hook does not clamp it", async () => {
    // The pre-fix shape: the transport's field still carries the over-ceiling
    // budget. The guard must fail closed on it.
    const { ok, error } = await hookThenGuard(
      wireBody(),
      deps({
        getSelection: () => undefined,
        getStaleSelection: () => undefined,
      }),
    );
    assert.equal(ok, false);
    assert.match((error as Error).message, /completion limit/);
  });

  it("reserves the host's SMALL budget, not a competing larger cap", async () => {
    // The host asked for 64 output tokens. The hook must not introduce a
    // larger competing budget — the guard then reserves 64 against the
    // endpoint's context window, and the request fits.
    const { ok, error } = await hookThenGuard(
      wireBody({ max_completion_tokens: 64 }, FILLING_INPUT_CHARS),
      deps(),
      CONTEXT_LENGTH,
    );
    assert.equal(ok, true, error instanceof Error ? error.message : String(error));
  });

  it("rejects a body that carries a competing larger budget in the other field", async () => {
    // The pre-fix shape for a small-budget request: the hook added
    // `max_tokens` next to the transport's smaller `max_completion_tokens`.
    // The guard measures the larger of the two and the request no longer
    // fits.
    const { ok, error } = await hookThenGuard(
      wireBody({ max_completion_tokens: 64, max_tokens: ENDPOINT_CEILING }, FILLING_INPUT_CHARS),
      deps({
        getSelection: () => undefined,
        getStaleSelection: () => undefined,
      }),
      CONTEXT_LENGTH,
    );
    assert.equal(ok, false);
    assert.match((error as Error).message, /context limit/);
  });
});
