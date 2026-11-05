/**
 * Unit tests for omp single-permit delegation in the OpenRouter recovery
 * wrapper (pi/stream.ts + pi/platform.ts).
 *
 * omp v18.16.0's provider in-flight permit is keyed by PROVIDER NAME
 * (resolveProviderInFlightLimit reads `model.provider`) and is acquired by the
 * host dispatcher around EVERY dispatch of that provider: both the custom-API
 * dispatch that enters our wrapper (packages/ai/src/stream.ts:1401-1404) and
 * the nested native dispatch the wrapper performs
 * (packages/ai/src/stream.ts:825-826). The enclosing permit is held until the
 * wrapper's stream ends, so a nested dispatch that resolves a limit for the
 * same provider waits for a slot only the enclosing dispatch can release — a
 * self-deadlock, reproduced with the real omp 18.16.0 binary under
 * `providers.maxInFlightRequests.openrouter = 1` (see
 * .context/2026-11-04.6-dual-host-harness/research-omp-transport.md).
 *
 * These tests stand in for omp's dispatcher with a fake permit host that
 * mirrors acquire/release semantics, and mirror the dispatch path: the host
 * resolves the limit from the per-call `maxInFlightRequests` map
 * (`options?.maxInFlightRequests ?? configured global`; per-call wins,
 * packages/ai/src/stream.ts:137-146), acquires, then calls the wrapper with
 * `waitForTerminalDrain` = limited. The injected delegate mirrors the nested
 * native dispatch: it resolves the limit from the options the WRAPPER hands
 * it, and acquires iff a limit applies.
 *
 * Run: npm test   (or: node --test test/)
 */

import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import { setRuntimeOverride } from "../pi/platform.ts";
import { streamOpenRouterWithRecovery, type StreamSimpleDelegate } from "../pi/stream.ts";
import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEvent,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai/compat";

const THOUGHT_SIGNATURE_ERROR = "400: {\"error\":{\"code\":400,\"message\":\"Invalid thought signature.\",\"status\":\"INVALID_ARGUMENT\"}}";

/** omp-only stream options; pi-ai 1.5 has neither field (structural view). */
interface HostPermitOptions {
  maxInFlightRequests?: Record<string, number>;
  waitForTerminalDrain?: boolean;
}

/**
 * Bounded wait for the fake permit: omp's real acquire waits indefinitely for
 * a cross-process slot, so the pre-fix nested acquire deadlocks. The deadline
 * turns that regression into a deterministic assertion failure instead of a
 * hung suite. The one real wall-clock deadline here is deliberate: there is no
 * deterministic signal for "deadlocked" — the whole point of this test is that
 * this wait never completes while the enclosing permit is held (assignment
 * allows bounded deadlines: "no timers beyond bounded deadlines").
 */
const PERMIT_WAIT_DEADLINE_MS = 2_000;

/** Mirrors omp v18.16.0's provider permit: one slot per acquire, keyed implicitly by the caller's provider. */
class FakeProviderPermit {
  readonly limit: number;
  held = 0;
  acquisitions = 0;
  releases = 0;

  constructor(limit = 1) {
    this.limit = limit;
  }

  async acquire(): Promise<void> {
    this.acquisitions += 1;
    if (this.held < this.limit) {
      this.held += 1;
      return;
    }
    // Mirrors omp's acquireProviderInFlightSlot: wait for a slot the enclosing
    // dispatch can only release after this acquire returns — under the fix the
    // nested dispatch never gets here, so the deadline never fires.
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const timer = setTimeout(
      () => reject(new Error("permit-wait deadlock: nested acquire exceeded the bounded deadline")),
      PERMIT_WAIT_DEADLINE_MS,
    );
    try {
      await promise;
      this.held += 1;
    } finally {
      clearTimeout(timer);
    }
  }

  release(): void {
    this.held -= 1;
    this.releases += 1;
  }
}

/** omp's resolveProviderInFlightLimit scoped to the per-call map (stream.ts:137-146). */
function resolvePerCallLimit(provider: string, options: SimpleStreamOptions | undefined): number | undefined {
  const limits = (options as HostPermitOptions | undefined)?.maxInFlightRequests;
  const value = limits?.[provider];
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? Math.max(1, Math.floor(value)) : undefined;
}

function fakeModel(): Model<"openai-completions"> {
  return {
    id: "google/gemini-2.5-pro",
    name: "Gemini 2.5 Pro",
    provider: "openrouter",
    api: "openai-completions",
    baseUrl: "https://openrouter.ai/api/v1",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 1048576,
    maxTokens: 65536,
  };
}

function usage(): AssistantMessage["usage"] {
  return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
}

function fakeContext(): TranscriptContext {
  return {
    messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
  } as TranscriptContext;
}

function signedContext(): TranscriptContext {
  const model = fakeModel();
  return {
    messages: [
      { role: "user", content: [{ type: "text", text: "what is 2+2?" }] },
      {
        role: "assistant",
        api: model.api,
        provider: model.provider,
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
  } as TranscriptContext;
}

function errorMessage(message: string, aborted = false): AssistantMessage {
  const model = fakeModel();
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: usage(),
    stopReason: aborted ? "aborted" : "error",
    errorMessage: message,
    timestamp: 1000,
  };
}

function doneEvent(text: string): AssistantMessageEvent {
  const model = fakeModel();
  const message: AssistantMessage = {
    role: "assistant",
    content: [{ type: "text", text }],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: usage(),
    stopReason: "stop",
    timestamp: 1000,
  };
  return { type: "done", reason: "stop", message };
}

function errorEvent(message: string, aborted = false): AssistantMessageEvent {
  return { type: "error", reason: aborted ? "aborted" : "error", error: errorMessage(message, aborted) };
}

function abortedOnce(signal: AbortSignal | undefined): Promise<void> {
  if (!signal || signal.aborted) return Promise.resolve();
  const { promise, resolve } = Promise.withResolvers<void>();
  signal.addEventListener("abort", () => resolve(), { once: true });
  return promise;
}

type NestedScript = (
  attempt: number,
  options: SimpleStreamOptions | undefined,
) => AssistantMessageEvent[] | Promise<AssistantMessageEvent[]>;

interface AttemptObservation {
  attempt: number;
  limit: number | undefined;
  limits: Record<string, number> | undefined;
  messages: unknown[];
}

/**
 * The nested dispatch, mirroring omp's native path: it resolves the limit for
 * the model's provider from the options the WRAPPER handed it, acquires iff a
 * limit applies, runs the scripted events, and releases on the attempt's end.
 */
function scriptedNestedDispatch(
  host: FakeProviderPermit,
  script: NestedScript,
  seen: AttemptObservation[],
): StreamSimpleDelegate {
  let attempt = 0;
  return (model, context, options) => {
    const thisAttempt = attempt;
    attempt += 1;
    const limit = resolvePerCallLimit(model.provider, options);
    seen.push({
      attempt: thisAttempt,
      limit,
      limits: structuredClone((options as HostPermitOptions | undefined)?.maxInFlightRequests),
      messages: structuredClone(context.messages),
    });
    const stream = createAssistantMessageEventStream();
    void (async () => {
      try {
        if (limit !== undefined) await host.acquire();
        try {
          for (const event of await script(thisAttempt, options)) stream.push(event);
        } finally {
          if (limit !== undefined) host.release();
        }
      } catch (error) {
        // A bounded-deadline rejection means the wrapper let the nested
        // dispatch re-acquire while the enclosing permit was held: surface it
        // as the attempt error so the regression fails assertions instead of
        // hanging the suite.
        stream.push({
          type: "error",
          reason: "error",
          error: errorMessage(error instanceof Error ? error.message : String(error)),
        });
      }
      stream.end();
    })();
    return stream;
  };
}

interface HostOutcome {
  terminal: AssistantMessageEvent | undefined;
  events: AssistantMessageEvent[];
}

/** Narrows the consumed terminal event to the error variant, asserting on the way in. */
function terminalError(outcome: HostOutcome): AssistantMessage {
  const terminal = outcome.terminal;
  assert.ok(terminal && terminal.type === "error", `expected a terminal error, got ${terminal?.type ?? "nothing"}`);
  return terminal.error;
}

/**
 * omp's custom-API dispatch (packages/ai/src/stream.ts:1401-1404): resolve the
 * limit for the provider, acquire, then call the wrapper with
 * `waitForTerminalDrain` = limited; release after the wrapper's stream ends.
 */
async function ompHostDispatch(
  host: FakeProviderPermit,
  model: Model<"openai-completions">,
  context: TranscriptContext,
  options: (SimpleStreamOptions & HostPermitOptions) | undefined,
  resolvedLimit: number | undefined,
  run: (model: Model<"openai-completions">, context: TranscriptContext, options: SimpleStreamOptions | undefined) => AssistantMessageEventStream,
): Promise<HostOutcome> {
  if (resolvedLimit !== undefined) await host.acquire();
  try {
    const hostCall: SimpleStreamOptions & HostPermitOptions = {
      ...options,
      waitForTerminalDrain: resolvedLimit !== undefined,
    };
    const stream = run(model, context, hostCall);
    const events: AssistantMessageEvent[] = [];
    let terminal: AssistantMessageEvent | undefined;
    for await (const event of stream) {
      if (event.type === "done" || event.type === "error") {
        terminal = event;
        break;
      }
      events.push(event);
    }
    return { terminal, events };
  } finally {
    if (resolvedLimit !== undefined) host.release();
  }
}

/** pi's dispatcher: no permit exists, options pass through untouched. */
async function piHostDispatch(
  host: FakeProviderPermit,
  model: Model<"openai-completions">,
  context: TranscriptContext,
  options: (SimpleStreamOptions & HostPermitOptions) | undefined,
  run: (model: Model<"openai-completions">, context: TranscriptContext, options: SimpleStreamOptions | undefined) => AssistantMessageEventStream,
): Promise<HostOutcome> {
  const stream = run(model, context, options);
  const events: AssistantMessageEvent[] = [];
  let terminal: AssistantMessageEvent | undefined;
  for await (const event of stream) {
    if (event.type === "done" || event.type === "error") {
      terminal = event;
      break;
    }
    events.push(event);
  }
  void host;
  return { terminal, events };
}

describe("omp single-permit delegation (recovery wrapper)", () => {
  afterEach(() => setRuntimeOverride(undefined));

  it("under a held permit the nested dispatch takes no second permit and completes", async () => {
    setRuntimeOverride("omp");
    const host = new FakeProviderPermit(1);
    const seen: AttemptObservation[] = [];
    const options = {
      maxInFlightRequests: { openrouter: 1, anthropic: 2 },
    } as SimpleStreamOptions & HostPermitOptions;

    const outcome = await ompHostDispatch(host, fakeModel(), fakeContext(), options, 1, (model, context, o) =>
      streamOpenRouterWithRecovery(model, context, o, scriptedNestedDispatch(host, () => [doneEvent("ok")], seen)),
    );

    assert.equal(outcome.terminal?.type, "done");
    assert.equal(host.acquisitions, 1, "only the enclosing dispatch may acquire");
    assert.equal(host.releases, 1);
    assert.equal(host.held, 0);
    assert.equal(seen.length, 1);
    // Our provider's entry is dropped from the nested per-call map…
    assert.equal(seen[0].limits?.openrouter, undefined);
    // …while every foreign provider's entry is preserved.
    assert.deepEqual(seen[0].limits, { anthropic: 2 });
  });

  it("a non-recovery error surfaces and still releases the single permit", async () => {
    setRuntimeOverride("omp");
    const host = new FakeProviderPermit(1);
    const seen: AttemptObservation[] = [];
    const options = { maxInFlightRequests: { openrouter: 1 } } as SimpleStreamOptions & HostPermitOptions;

    const outcome = await ompHostDispatch(host, fakeModel(), fakeContext(), options, 1, (model, context, o) =>
      streamOpenRouterWithRecovery(model, context, o, scriptedNestedDispatch(host, () => [errorEvent("500: internal error")], seen)),
    );

    const error = terminalError(outcome);
    assert.equal(error.errorMessage, "500: internal error");
    assert.equal(host.acquisitions, 1);
    assert.equal(host.releases, 1);
    assert.equal(host.held, 0);
    assert.equal(seen.length, 1, "no retry for a non-signature error");
  });

  it("a signature-error retry keeps delegation single-permit", async () => {
    setRuntimeOverride("omp");
    const host = new FakeProviderPermit(1);
    const seen: AttemptObservation[] = [];
    const options = { maxInFlightRequests: { openrouter: 1, anthropic: 2 } } as SimpleStreamOptions & HostPermitOptions;

    const outcome = await ompHostDispatch(host, fakeModel(), signedContext(), options, 1, (model, context, o) =>
      streamOpenRouterWithRecovery(
        model,
        context,
        o,
        scriptedNestedDispatch(
          host,
          (attempt) => (attempt === 0 ? [errorEvent(THOUGHT_SIGNATURE_ERROR)] : [doneEvent("recovered")]),
          seen,
        ),
      ),
    );

    assert.equal(outcome.terminal?.type, "done");
    assert.equal(host.acquisitions, 1, "the retry runs under the same enclosing permit");
    assert.equal(host.releases, 1);
    assert.equal(host.held, 0);
    assert.equal(seen.length, 2);
    // The retry ALSO receives scoped options: no second permit for our provider.
    assert.deepEqual(seen[1].limits, { anthropic: 2 });
    // And the retry's transcript has the replay signatures stripped.
    const retryAssistant = seen[1].messages[1] as AssistantMessage;
    assert.equal("thinkingSignature" in retryAssistant.content[0], false);
    assert.equal("thoughtSignature" in retryAssistant.content[1], false);
  });

  it("cancellation (AbortSignal) surfaces aborted and releases the permit", async () => {
    setRuntimeOverride("omp");
    const host = new FakeProviderPermit(1);
    const seen: AttemptObservation[] = [];
    const controller = new AbortController();
    const options = {
      maxInFlightRequests: { openrouter: 1 },
      signal: controller.signal,
    } as SimpleStreamOptions & HostPermitOptions;

    const pending = ompHostDispatch(host, fakeModel(), fakeContext(), options, 1, (model, context, o) =>
      streamOpenRouterWithRecovery(
        model,
        context,
        o,
        scriptedNestedDispatch(
          host,
          async (_attempt, scriptOptions) => {
            await abortedOnce(scriptOptions?.signal);
            return [errorEvent("Request was aborted", true)];
          },
          seen,
        ),
      ),
    );
    controller.abort();
    const outcome = await pending;

    const error = terminalError(outcome);
    assert.equal(error.stopReason, "aborted");
    assert.equal(error.errorMessage, "Request was aborted");
    assert.equal(host.acquisitions, 1);
    assert.equal(host.releases, 1);
    assert.equal(host.held, 0);
    assert.deepEqual(seen[0].limits, {});
  });

  it("without a held permit the per-call map passes through and the nested dispatch still enforces", async () => {
    setRuntimeOverride("omp");
    const host = new FakeProviderPermit(1);
    const seen: AttemptObservation[] = [];
    const map = { openrouter: 1, anthropic: 2 };
    const options = { maxInFlightRequests: { ...map } } as SimpleStreamOptions & HostPermitOptions;

    // The host resolved NO limit for our provider (waitForTerminalDrain false),
    // so it holds no permit: the nested dispatch is the only holder and must
    // keep enforcing the map's own entry — the wrapper must not touch it.
    const outcome = await ompHostDispatch(host, fakeModel(), fakeContext(), options, undefined, (model, context, o) =>
      streamOpenRouterWithRecovery(model, context, o, scriptedNestedDispatch(host, () => [doneEvent("ok")], seen)),
    );

    assert.equal(outcome.terminal?.type, "done");
    assert.deepEqual(seen[0].limits, map, "untouched when the host holds no permit");
    assert.equal(host.acquisitions, 1, "the nested dispatch enforces the limit itself");
    assert.equal(host.releases, 1);
    assert.equal(host.held, 0);
  });

  it("under pi the options pass through unchanged (no permit machinery)", async () => {
    setRuntimeOverride("pi");
    const host = new FakeProviderPermit(1);
    const seen: AttemptObservation[] = [];
    const map = { openrouter: 1 };
    const options = { maxInFlightRequests: { ...map } } as SimpleStreamOptions & HostPermitOptions;

    const outcome = await piHostDispatch(host, fakeModel(), fakeContext(), options, (model, context, o) =>
      streamOpenRouterWithRecovery(model, context, o, scriptedNestedDispatch(host, () => [doneEvent("ok")], seen)),
    );

    assert.equal(outcome.terminal?.type, "done");
    assert.deepEqual(seen[0].limits, map, "pi has no permit: nothing to scope");
    assert.equal(host.acquisitions, 1);
    assert.equal(host.releases, 1);
    assert.equal(host.held, 0);
  });
});
