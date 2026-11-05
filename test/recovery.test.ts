/**
 * Unit tests for OpenRouter thought signature failure-recovery logic (pi/recovery.ts and pi/stream.ts).
 *
 * Run: npm test   (or: node --test test/)
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  driveRecoveryLoop,
  isThoughtSignatureError,
  planErrorRecovery,
  stripReasoningReplaySignatures,
  type AttemptSpec,
  type RecoveryState,
} from "../pi/recovery.ts";
import {
  defaultDelegate,
  buildStandaloneProviderConfig,
  streamOpenRouterWithRecovery,
  OPENROUTER_RECOVERY_API,
  type StreamSimpleDelegate,
} from "../pi/stream.ts";
import { clearEndpointLimits, setEndpointLimits } from "../pi/selection-limits-state.ts";
import { setRuntimeOverride } from "../pi/platform.ts";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEvent,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai/compat";

// Verbatim from a pi session log: invalid thought signature 400 (Gemini after
// OpenRouter failed over from Google AI Studio's 429 to another Google
// provider). previous_errors was elided in the log excerpt.
const THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER = String.raw`400: {"message":"Provider returned error","code":400,"metadata":{"raw":"Gemini models require OpenRouter reasoning details to be preserved in each request. Please refer to our docs: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning-blocks. Upstream error: {\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"Invalid thought signature.\",\n    \"status\": \"INVALID_ARGUMENT\"\n  }\n}\n","provider_name":"Google","is_byok":false,"provider_error_code":"400","previous_errors":[{"message":"429 ...","provider_name":"Google AI Studio"}]}`;

// Observed Vertex AI error format when thought signatures fail validation.
const THOUGHT_SIGNATURE_ERROR_VERTEX =
  '400: {"error":{"code":400,"message":"Unable to submit request because Thought signature is not valid.. Learn more: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/gemini","status":"INVALID_ARGUMENT"}}';

function freshState(): RecoveryState {
  return { signatureRetried: false };
}

function fakeModel(api: "openai-completions" = "openai-completions"): Model<"openai-completions"> {
  return {
    id: "google/gemini-2.5-pro",
    name: "Gemini 2.5 Pro",
    provider: "openrouter",
    api,
    baseUrl: "https://openrouter.ai/api/v1",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1048576,
    maxTokens: 65536,
  };
}

describe("isThoughtSignatureError", () => {
  it("detects the verbatim Google AI Studio failover signature 400", () => {
    assert.equal(isThoughtSignatureError(THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER), true);
  });

  it("detects the Vertex AI 'Thought signature is not valid' variant", () => {
    assert.equal(isThoughtSignatureError(THOUGHT_SIGNATURE_ERROR_VERTEX), true);
  });

  it("detects case-insensitive variations", () => {
    assert.equal(isThoughtSignatureError("400: invalid thought signature"), true);
    assert.equal(isThoughtSignatureError("Thought signature is not valid"), true);
  });

  it("does not match unrelated 400 errors or token count errors", () => {
    assert.equal(
      isThoughtSignatureError(
        "Requested token count exceeds the model's maximum context length of 1048576 tokens.",
      ),
      false,
    );
    assert.equal(isThoughtSignatureError("400: Invalid parameter 'temperature'"), false);
    assert.equal(isThoughtSignatureError("500: Internal server error"), false);
    assert.equal(isThoughtSignatureError(undefined), false);
    assert.equal(isThoughtSignatureError(""), false);
  });
});

describe("stripReasoningReplaySignatures", () => {
  it("removes thinkingSignature, thoughtSignature, and textSignature from assistant messages", () => {
    const messages: (TranscriptContext["messages"][number])[] = [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      {
        role: "assistant",
        api: "openai-completions",
        provider: "openrouter",
        model: "google/gemini-2.5-pro",
        usage: { input: 10, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 30, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: "stop",
        timestamp: 1000,
        content: [
          { type: "thinking", thinking: "reasoning", thinkingSignature: '{"encrypted":true}' },
          { type: "text", text: "answer", textSignature: "txt-sig" },
          { type: "toolCall", id: "t1", name: "tool", arguments: {}, thoughtSignature: "tool-sig" },
        ],
      },
    ];

    const stripped = stripReasoningReplaySignatures(messages);
    const assistant = stripped[1] as AssistantMessage;

    assert.equal((assistant.content[0] as Record<string, unknown>).thinkingSignature, undefined);
    assert.equal("thinkingSignature" in assistant.content[0], false);
    assert.equal((assistant.content[0] as Record<string, unknown>).thinking, "reasoning");

    assert.equal((assistant.content[1] as Record<string, unknown>).textSignature, undefined);
    assert.equal("textSignature" in assistant.content[1], false);
    assert.equal((assistant.content[1] as Record<string, unknown>).text, "answer");

    assert.equal((assistant.content[2] as Record<string, unknown>).thoughtSignature, undefined);
    assert.equal("thoughtSignature" in assistant.content[2], false);
    assert.equal((assistant.content[2] as Record<string, unknown>).name, "tool");

    // The original message is not mutated
    const originalAssistant = messages[1] as AssistantMessage;
    assert.equal((originalAssistant.content[0] as Record<string, unknown>).thinkingSignature, '{"encrypted":true}');
    assert.equal((originalAssistant.content[1] as Record<string, unknown>).textSignature, "txt-sig");
    assert.equal((originalAssistant.content[2] as Record<string, unknown>).thoughtSignature, "tool-sig");
  });

  it("leaves user, system, and toolResult messages completely unchanged", () => {
    const messages = [
      { role: "system", content: "system prompt" },
      { role: "user", content: [{ type: "text", text: "query" }] },
      { role: "toolResult", toolCallId: "t1", content: [{ type: "text", text: "result" }] },
    ];
    const stripped = stripReasoningReplaySignatures(messages);
    assert.deepEqual(stripped, messages);
  });

  it("returns the exact same array reference when there are no signatures to strip", () => {
    const messages = [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      {
        role: "assistant",
        api: "openai-completions",
        provider: "openrouter",
        model: "google/gemini-2.5-pro",
        usage: { input: 10, output: 20, cacheRead: 0, cacheWrite: 0, totalTokens: 30, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
        stopReason: "stop",
        timestamp: 1000,
        content: [
          { type: "thinking", thinking: "just text" },
          { type: "text", text: "hello" },
        ],
      },
    ];
    const stripped = stripReasoningReplaySignatures(messages);
    assert.equal(stripped, messages);
  });
});

describe("planErrorRecovery", () => {
  it("plans a thought-signature retry and marks the state", () => {
    const state = freshState();
    const action = planErrorRecovery(THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER, state);
    assert.deepEqual(action, { kind: "thought-signature" });
    assert.equal(state.signatureRetried, true);
  });

  it("allows at most one signature retry per request", () => {
    const state = freshState();
    assert.ok(planErrorRecovery(THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER, state));
    assert.equal(planErrorRecovery(THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER, state), undefined);
  });

  it("returns undefined for unrelated errors and does not mark retried", () => {
    const state = freshState();
    assert.equal(planErrorRecovery("400: Unrelated bad request", state), undefined);
    assert.equal(state.signatureRetried, false);
    assert.equal(planErrorRecovery(undefined, state), undefined);
  });
});

// ---------- driveRecoveryLoop ----------

type FakeEvent = { type: string; reason?: string; error?: { errorMessage?: string } };

function fakeAttempt(
  attempts: FakeEvent[][],
  capturedSpecs?: AttemptSpec<string, string[]>[],
): (spec: AttemptSpec<string, string[]>) => AsyncIterable<FakeEvent> {
  let call = 0;
  return (spec) => {
    capturedSpecs?.push(spec);
    return (async function* () {
      const events = attempts[Math.min(call++, attempts.length - 1)];
      for (const e of events) yield e;
    })();
  };
}

async function drive(attempts: FakeEvent[][]) {
  const emitted: FakeEvent[] = [];
  const capturedSpecs: AttemptSpec<string, string[]>[] = [];
  const outcome = await driveRecoveryLoop(
    fakeAttempt(attempts, capturedSpecs),
    (_action, spec) => ({ options: spec.options, messages: [...spec.messages, "stripped"] }),
    { options: "initial", messages: ["m"] },
    freshState(),
    (e) => emitted.push(e),
  );
  return { outcome, emitted, capturedSpecs };
}

describe("driveRecoveryLoop", () => {
  it("retries once and suppresses the signature error when the retry succeeds, asserting retry spec", async () => {
    const { outcome, emitted, capturedSpecs } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "text" }, { type: "done" }],
    ]);
    assert.equal(outcome.succeeded, true);
    assert.equal(outcome.lastError, undefined);
    assert.deepEqual(
      emitted.map((e) => e.type),
      ["text", "done"],
    );
    // Verify attempt 2 was invoked with the retry spec (stripped messages)
    assert.equal(capturedSpecs.length, 2);
    assert.deepEqual(capturedSpecs[0].messages, ["m"]);
    assert.deepEqual(capturedSpecs[1].messages, ["m", "stripped"]);
  });

  it("surfaces the freshest error when a retry fails with a different error", async () => {
    const { outcome, emitted, capturedSpecs } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "error", error: { errorMessage: "429: rate limited" } }],
    ]);
    assert.equal(outcome.succeeded, false);
    assert.equal(outcome.lastError?.error?.errorMessage, "429: rate limited");
    assert.equal(emitted.length, 0);
    assert.equal(capturedSpecs.length, 2);
  });

  it("surfaces the second error when retried signature fails again (at most one retry)", async () => {
    const { outcome, emitted, capturedSpecs } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
    ]);
    assert.equal(outcome.succeeded, false);
    assert.equal(outcome.lastError?.error?.errorMessage, THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER);
    assert.equal(emitted.length, 0);
    assert.equal(capturedSpecs.length, 2);
  });

  it("never retries after content has been forwarded", async () => {
    const makeAttempt = fakeAttempt([
      [{ type: "text" }, { type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
    ]);
    let attempts = 0;
    const wrapped = (spec: AttemptSpec<string, string[]>) => {
      attempts++;
      return makeAttempt(spec);
    };
    const emitted: FakeEvent[] = [];
    const outcome = await driveRecoveryLoop(
      wrapped,
      (_a, s) => s,
      { options: "o", messages: ["m"] },
      freshState(),
      (e) => emitted.push(e),
    );
    assert.equal(attempts, 1);
    assert.equal(outcome.succeeded, false);
    assert.equal(outcome.lastError?.error?.errorMessage, THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER);
    assert.equal(emitted.length, 1);
  });

  it("lets an abort win by being the freshest error", async () => {
    const { outcome } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "error", reason: "aborted", error: { errorMessage: "aborted" } }],
    ]);
    assert.equal(outcome.succeeded, false);
    assert.equal(outcome.lastError?.reason, "aborted");
  });
});

// ---------- streamOpenRouterWithRecovery wrapper tests ----------

describe("streamOpenRouterWithRecovery", () => {
  it("retries on thought signature error with signatures stripped and completes cleanly", async () => {
    const model = fakeModel();
    const context: TranscriptContext = {
      messages: [
        { role: "user", content: [{ type: "text", text: "what is 2+2?" }] },
        {
          role: "assistant",
          api: "openai-completions",
          provider: "openrouter",
          model: model.id,
          usage: { input: 10, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 15, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: "stop",
          timestamp: 1000,
          content: [
            { type: "thinking", thinking: "evaluating...", thinkingSignature: '{"data":"sig1"}' },
            { type: "toolCall", id: "call_1", name: "calc", arguments: { expr: "2+2" }, thoughtSignature: "tool-sig1" },
          ],
        },
      ],
    };

    let attemptCount = 0;
    const capturedContexts: TranscriptContext[] = [];

    const mockDelegate: StreamSimpleDelegate = (_model, ctx) => {
      attemptCount++;
      capturedContexts.push(ctx);
      const stream = createAssistantMessageEventStream();

      queueMicrotask(() => {
        if (attemptCount === 1) {
          const errMessage: AssistantMessage = {
            role: "assistant",
            content: [],
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
            stopReason: "error",
            errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER,
            timestamp: Date.now(),
          };
          stream.push({ type: "error", reason: "error", error: errMessage });
          stream.end(errMessage);
        } else {
          const finalMsg: AssistantMessage = {
            role: "assistant",
            content: [{ type: "text", text: "4" }],
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: { input: 20, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 25, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
            stopReason: "stop",
            timestamp: Date.now(),
          };
          stream.push({ type: "start", partial: finalMsg });
          stream.push({ type: "text_delta", contentIndex: 0, delta: "4", partial: finalMsg });
          stream.push({ type: "text_end", contentIndex: 0, content: "4", partial: finalMsg });
          stream.push({ type: "done", reason: "stop", message: finalMsg });
          stream.end(finalMsg);
        }
      });

      return stream;
    };

    const stream = streamOpenRouterWithRecovery(model, context, {}, mockDelegate);
    const events: AssistantMessageEvent[] = [];
    for await (const event of stream) {
      events.push(event);
    }
    const finalResult = await stream.result();

    assert.equal(attemptCount, 2);
    // Attempt 1 had original signatures
    const att1Assistant = capturedContexts[0].messages[1] as AssistantMessage;
    assert.equal((att1Assistant.content[0] as Record<string, unknown>).thinkingSignature, '{"data":"sig1"}');
    assert.equal((att1Assistant.content[1] as Record<string, unknown>).thoughtSignature, "tool-sig1");

    // Attempt 2 had signatures stripped
    const att2Assistant = capturedContexts[1].messages[1] as AssistantMessage;
    assert.equal((att2Assistant.content[0] as Record<string, unknown>).thinkingSignature, undefined);
    assert.equal("thinkingSignature" in att2Assistant.content[0], false);
    assert.equal((att2Assistant.content[1] as Record<string, unknown>).thoughtSignature, undefined);
    assert.equal("thoughtSignature" in att2Assistant.content[1], false);

    // Initial 400 error was NOT emitted to consumer; clean start..done
    assert.equal(events.some((e) => e.type === "error"), false);
    assert.equal(events[0].type, "start");
    assert.equal(events[events.length - 1].type, "done");
    assert.equal(finalResult.stopReason, "stop");
  });

  it("rechecks context after the final payload hook changes the selected endpoint", async () => {
    const model = fakeModel();
    const context: TranscriptContext = {
      messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(300) }] }],
    };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 512, maxCompletionTokens: 128 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      const pending = options?.onPayload?.({ model: model.id, max_tokens: 128 }, model)
        .then(() => {
          requestSent = true;
        })
        .catch((error: unknown) => {
          guardError = error;
        });
      void pending?.then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(
        model,
        context,
        {
          onPayload: async (payload) => {
            setEndpointLimits(model.id, { contextLength: 300, maxCompletionTokens: 128 });
            return payload;
          },
        },
        delegate,
      );
      for await (const _event of stream) {
        // The callback guard is the tested boundary; no request should be sent.
      }
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /selected OpenRouter endpoint context limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("rejects final payload messages that exceed the endpoint context", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 512, maxCompletionTokens: 128 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 128 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({
          ...(payload as Record<string, unknown>),
          messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(4_000) }] }],
        }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /selected OpenRouter endpoint context limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("accepts a request whose requested maxTokens is smaller than the endpoint's completion ceiling", async () => {
    const model = fakeModel();
    // ~40k input tokens: this fits the endpoint context when the request
    // reserves its own small output budget, but not when the guard reserves
    // the endpoint's advertised completion ceiling.
    const context: TranscriptContext = {
      messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(140_000) }] }],
    };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 65_536, maxCompletionTokens: 32_768 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 4_096 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, { maxTokens: 4_096 }, delegate);
      for await (const _event of stream) {}
      assert.equal(guardError, undefined);
      assert.equal(requestSent, true);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("still reserves the endpoint completion ceiling when the request carries no explicit budget", async () => {
    const model = fakeModel();
    const context: TranscriptContext = {
      messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(140_000) }] }],
    };
    setRuntimeOverride("pi");
    // No explicit budget in the request: the guard falls back to the model's
    // registered maximum, clamped by the endpoint's completion ceiling, and
    // the request no longer fits the endpoint context window.
    setEndpointLimits(model.id, { contextLength: 65_536, maxCompletionTokens: 32_768 });
    let requestSent = false;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id }, model)
        .then(() => { requestSent = true; })
        .catch(() => {})
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {}, delegate);
      const events: AssistantMessageEvent[] = [];
      for await (const event of stream) {
        events.push(event);
      }
      assert.equal(requestSent, false);
      const errorEvent = events.find(
        (event): event is Extract<AssistantMessageEvent, { type: "error" }> => event.type === "error",
      );
      assert.ok(errorEvent, "expected a pre-dispatch context-limit error");
      assert.match(errorEvent.error.errorMessage, /selected OpenRouter endpoint context limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("rejects final payload output budgets above the endpoint ceiling", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 4_096, maxCompletionTokens: 512 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 512 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({ ...(payload as Record<string, unknown>), max_tokens: 8_000 }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /selected OpenRouter endpoint completion limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });


  it("accepts an ordinary wire-shaped final payload that fits the endpoint", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 4_096, maxCompletionTokens: 512 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 512 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({
          ...(payload as Record<string, unknown>),
          // The serialized OpenAI wire shape: plain role strings, string or
          // part-array content, tool_calls, tool results.
          messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: "What is 2+2?" },
            { role: "assistant", content: [{ type: "text", text: "Let me compute." }] },
            {
              role: "assistant",
              content: null,
              tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: '{"expr":"2+2"}' } }],
            },
            { role: "tool", tool_call_id: "call_1", content: "4" },
            { role: "user", content: [{ type: "text", text: "Thanks." }] },
          ],
          tools: [{ type: "function", function: { name: "calc", parameters: { type: "object" } } }],
        }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(guardError, undefined);
      assert.equal(requestSent, true);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("rejects a wire-shaped final payload that exceeds the endpoint context", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 512, maxCompletionTokens: 128 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 128 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({
          ...(payload as Record<string, unknown>),
          messages: [{ role: "user", content: "x".repeat(4_000) }],
        }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /selected OpenRouter endpoint context limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("rejects a final payload whose max_completion_tokens exceeds the endpoint ceiling", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 4_096, maxCompletionTokens: 512 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_completion_tokens: 512 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({ ...(payload as Record<string, unknown>), max_completion_tokens: 8_000 }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /selected OpenRouter endpoint completion limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("rejects a final payload whose max_tokens exceeds the ceiling even when max_completion_tokens is under it", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 4_096, maxCompletionTokens: 512 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 512 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({
          ...(payload as Record<string, unknown>),
          max_tokens: 8_000,
          max_completion_tokens: 512,
        }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /selected OpenRouter endpoint completion limit/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("rejects a final payload carrying a non-numeric output budget", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    setRuntimeOverride("pi");
    setEndpointLimits(model.id, { contextLength: 4_096, maxCompletionTokens: 512 });
    let requestSent = false;
    let guardError: unknown;
    const delegate: StreamSimpleDelegate = (_model, _context, options) => {
      const stream = createAssistantMessageEventStream();
      void options?.onPayload?.({ model: model.id, max_tokens: 512 }, model)
        .then(() => { requestSent = true; })
        .catch((error: unknown) => { guardError = error; })
        .then(() => stream.end());
      return stream;
    };
    try {
      const stream = streamOpenRouterWithRecovery(model, context, {
        onPayload: async (payload) => ({ ...(payload as Record<string, unknown>), max_completion_tokens: "many" }),
      }, delegate);
      for await (const _event of stream) {}
      assert.equal(requestSent, false);
      assert.ok(guardError instanceof Error);
      assert.match((guardError as Error).message, /non-numeric "max_completion_tokens" output budget/);
    } finally {
      clearEndpointLimits();
      setRuntimeOverride(undefined);
    }
  });

  it("surfaces error when retry also fails", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };

    const mockDelegate: StreamSimpleDelegate = () => {
      const stream = createAssistantMessageEventStream();
      queueMicrotask(() => {
        const errMessage: AssistantMessage = {
          role: "assistant",
          content: [],
          api: model.api,
          provider: model.provider,
          model: model.id,
          usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
          stopReason: "error",
          errorMessage: THOUGHT_SIGNATURE_ERROR_VERTEX,
          timestamp: Date.now(),
        };
        stream.push({ type: "error", reason: "error", error: errMessage });
        stream.end(errMessage);
      });
      return stream;
    };

    const stream = streamOpenRouterWithRecovery(model, context, {}, mockDelegate);
    const events: AssistantMessageEvent[] = [];
    for await (const event of stream) {
      events.push(event);
    }

    assert.equal(events.length, 1);
    assert.equal(events[0].type, "error");
    assert.equal(events[0].error.errorMessage, THOUGHT_SIGNATURE_ERROR_VERTEX);
  });

  it("surfaces a synthesized terminal error event if an attempt terminates with neither done nor error", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };

    const mockDelegate: StreamSimpleDelegate = () => {
      const stream = createAssistantMessageEventStream();
      queueMicrotask(() => {
        // Abrupt end without pushing done or error
        stream.end();
      });
      return stream;
    };

    const stream = streamOpenRouterWithRecovery(model, context, {}, mockDelegate);
    const events: AssistantMessageEvent[] = [];
    for await (const event of stream) {
      events.push(event);
    }

    assert.equal(events.length, 1);
    assert.equal(events[0].type, "error");
    assert.match(events[0].error.errorMessage, /without a terminal event/);
  });

  it("surfaces a synchronous delegate throw as an error event", async () => {
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };

    const mockDelegate: StreamSimpleDelegate = () => {
      throw new Error("Synchronous network failure");
    };

    const stream = streamOpenRouterWithRecovery(model, context, {}, mockDelegate);
    const events: AssistantMessageEvent[] = [];
    for await (const event of stream) {
      events.push(event);
    }

    assert.equal(events.length, 1);
    assert.equal(events[0].type, "error");
    assert.equal(events[0].error.errorMessage, "Synchronous network failure");
  });
});

describe("defaultDelegate", () => {
  it("rejects non-openai-completions models with an informative error", () => {
    setRuntimeOverride("pi");
    const model = {
      ...fakeModel(),
      api: "anthropic-messages" as any,
    };
    const context: TranscriptContext = { messages: [] };
    assert.throws(
      () => defaultDelegate(model, context),
      /OpenRouter recovery wrapper expects api "openai-completions", got "anthropic-messages"/,
    );
  });

  it("returns an AssistantMessageEventStream for completions models", () => {
    setRuntimeOverride("pi");
    const model = fakeModel();
    const context: TranscriptContext = { messages: [] };
    const stream = defaultDelegate(model, context);
    assert.ok(stream);
    assert.equal(typeof stream.push, "function");
    assert.equal(typeof stream.end, "function");
  });

  it("rejects models not carrying the omp recovery api id under omp", () => {
    setRuntimeOverride("omp");
    try {
      const model = { ...fakeModel(), api: "openai-completions" as any };
      assert.throws(
        () => defaultDelegate(model, { messages: [] }),
        new RegExp(`expects api "${OPENROUTER_RECOVERY_API}"`),
      );
    } finally {
      setRuntimeOverride(undefined);
    }
  });

  it("hands omp recovery-api models to the host dispatcher as openai-completions", () => {
    setRuntimeOverride("omp");
    try {
      const model = { ...fakeModel(), api: OPENROUTER_RECOVERY_API as any };
      const stream = defaultDelegate(model, { messages: [] });
      assert.ok(stream);
      assert.equal(typeof stream.push, "function");
      assert.equal(typeof stream.end, "function");
    } finally {
      setRuntimeOverride(undefined);
    }
  });
});

describe("buildStandaloneProviderConfig", () => {
  it("attaches api, streamSimple recovery wrapper, and X-Title header", () => {
    setRuntimeOverride("pi");
    const dummyModels = [fakeModel()];
    const config = buildStandaloneProviderConfig(dummyModels as any);
    assert.equal(config.api, "openai-completions");
    assert.equal(config.streamSimple, streamOpenRouterWithRecovery);
    assert.equal(config.headers["X-Title"], "openrouter-advanced-provider");
    assert.deepEqual(config.models, dummyModels);
  });

  it("registers the unique recovery api id under omp with dynamic fetch and no pinned key", () => {
    setRuntimeOverride("omp");
    try {
      const config = buildStandaloneProviderConfig(undefined, {
        fetchDynamicModels: async () => [{ id: "x", api: "openrouter" }, { id: "y" }],
      });
      assert.equal(config.api, OPENROUTER_RECOVERY_API);
      assert.equal(config.streamSimple, streamOpenRouterWithRecovery);
      assert.equal(config.models, undefined);
      assert.equal(config.apiKey, undefined);
      assert.equal(typeof config.fetchDynamicModels, "function");
      return config.fetchDynamicModels("k").then((rows) => {
        assert.equal(rows.length, 2);
        for (const row of rows) {
          assert.equal((row as Record<string, unknown>).api, OPENROUTER_RECOVERY_API);
        }
        assert.equal((rows[0] as Record<string, unknown>).id, "x");
        assert.equal((rows[1] as Record<string, unknown>).id, "y");
      });
    } finally {
      setRuntimeOverride(undefined);
    }
  });

  it("propagates fetch failures instead of masking them with stale rows", () => {
    setRuntimeOverride("omp");
    try {
      // Key A: successful discovery. Key B: the fetch fails (e.g. 5xx or a
      // rotated key the store no longer serves). The wrapped fetcher must
      // reject — falling back to A's rows would report A's account catalog
      // as a successful B discovery.
      let call = 0;
      const config = buildStandaloneProviderConfig(undefined, {
        fetchDynamicModels: async () => {
          call += 1;
          if (call === 1) return [{ id: "a", api: "openrouter" }];
          throw new Error("discovery failed for key B");
        },
      });
      const fetcher = config.fetchDynamicModels!;
      return fetcher("key-A").then((rowsA) => {
        assert.equal(rowsA.length, 1);
        return fetcher("key-B").then(
          () => assert.fail("second fetch must reject"),
          (err: unknown) => {
            assert.ok(err instanceof Error);
            assert.match((err as Error).message, /key B/);
          },
        );
      });
    } finally {
      setRuntimeOverride(undefined);
    }
  });
});
