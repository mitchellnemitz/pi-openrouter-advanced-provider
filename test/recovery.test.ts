/**
 * Unit tests for OpenRouter thought signature failure-recovery logic (pi/recovery.ts).
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
    const messages = [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "reasoning", thinkingSignature: '{"encrypted":true}' },
          { type: "text", text: "answer", textSignature: "txt-sig" },
          { type: "toolCall", id: "t1", name: "tool", arguments: {}, thoughtSignature: "tool-sig" },
        ],
      },
    ];

    const stripped = stripReasoningReplaySignatures(messages);
    const assistant = stripped[1] as (typeof messages)[1];

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
    const originalAssistant = messages[1] as (typeof messages)[1];
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

  it("returns the exact same array when there are no signatures to strip", () => {
    const messages = [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      {
        role: "assistant",
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

function fakeAttempt(attempts: FakeEvent[][]): (spec: AttemptSpec<string, string[]>) => AsyncIterable<FakeEvent> {
  let call = 0;
  return () =>
    (async function* () {
      const events = attempts[Math.min(call++, attempts.length - 1)];
      for (const e of events) yield e;
    })();
}

async function drive(attempts: FakeEvent[][]) {
  const emitted: FakeEvent[] = [];
  const outcome = await driveRecoveryLoop(
    fakeAttempt(attempts),
    (_action, spec) => ({ options: spec.options, messages: [...spec.messages, "stripped"] }),
    { options: "initial", messages: ["m"] },
    freshState(),
    (e) => emitted.push(e),
  );
  return { outcome, emitted };
}

describe("driveRecoveryLoop", () => {
  it("retries once and suppresses the signature error when the retry succeeds", async () => {
    const { outcome, emitted } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "text" }, { type: "done" }],
    ]);
    assert.equal(outcome.succeeded, true);
    assert.equal(outcome.lastError, undefined);
    assert.deepEqual(
      emitted.map((e) => e.type),
      ["text", "done"],
    );
  });

  it("surfaces the freshest error when a retry fails with a different error", async () => {
    const { outcome, emitted } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "error", error: { errorMessage: "429: rate limited" } }],
    ]);
    assert.equal(outcome.succeeded, false);
    assert.equal(outcome.lastError?.error?.errorMessage, "429: rate limited");
    assert.equal(emitted.length, 0);
  });

  it("surfaces the second error when retried signature fails again (at most one retry)", async () => {
    const { outcome, emitted } = await drive([
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
      [{ type: "error", error: { errorMessage: THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER } }],
    ]);
    assert.equal(outcome.succeeded, false);
    assert.equal(outcome.lastError?.error?.errorMessage, THOUGHT_SIGNATURE_ERROR_STUDIO_FAILOVER);
    assert.equal(emitted.length, 0);
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
