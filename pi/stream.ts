/**
 * streamSimple wrapper for the OpenRouter provider registration. Delegates to
 * pi-ai's built-in stream implementation and adds automatic recovery for
 * invalid thought signature errors when OpenRouter fails over between backing
 * providers.
 *
 * At most one retry per request, and never after any content event has been
 * forwarded. When the retry fails, the freshest error event is surfaced.
 */

import {
  createAssistantMessageEventStream,
  streamSimple,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Api,
  type Message,
  type Model,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai/compat";
import {
  driveRecoveryLoop,
  stripReasoningReplaySignatures,
  type AttemptSpec,
  type RecoveryAction,
  type RecoveryState,
} from "./recovery.ts";

// Both hosts pass a transcript-shaped context to stream functions, but only
// stock pi names the type TranscriptContext (omp's pi-ai uses Context), and
// the delegate treats it as opaque. Derive it from the shared streamSimple
// import so one signature typechecks against either host's declarations.
type StreamContext = Parameters<typeof streamSimple>[1];

export type StreamSimpleDelegate = (
  model: Model<Api>,
  context: StreamContext,
  options?: SimpleStreamOptions,
) => AssistantMessageEventStream;

export function defaultDelegate(
  model: Model<Api>,
  context: StreamContext,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  if (model.api !== "openai-completions") {
    throw new Error(`OpenRouter recovery wrapper expects api "openai-completions", got "${model.api}"`);
  }
  // Generic api-dispatched streamSimple, not openAICompletionsApi(): the
  // latter does not exist under oh-my-pi's pi-ai, while both hosts export
  // the dispatcher from /compat (stock) or the package root (omp, remapped).
  // For openai-completions models it resolves to the same implementation.
  return streamSimple(model as Model<"openai-completions">, context, options);
}

function syncThrowErrorEvent(
  model: Model<Api>,
  error: unknown,
  options?: SimpleStreamOptions,
): AssistantMessageEvent {
  const isAborted = options?.signal?.aborted || (error instanceof Error && error.name === "AbortError");
  const stopReason = isAborted ? "aborted" : "error";
  const message: AssistantMessage = {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    errorMessage: isAborted ? "Request was aborted" : error instanceof Error ? error.message : String(error),
    timestamp: Date.now(),
  };
  return { type: "error", reason: stopReason, error: message };
}

export const APP_TITLE = "openrouter-advanced-provider";

export function buildStandaloneProviderConfig(models: unknown[] | undefined) {
  return {
    models: models ?? [],
    api: "openai-completions",
    streamSimple: streamOpenRouterWithRecovery,
    headers: {
      // No HTTP-Referer: this extension sends no public site URL.
      // X-Title identifies the app in OpenRouter's stats.
      "X-Title": APP_TITLE,
    },
  };
}

export function streamOpenRouterWithRecovery(
  model: Model<Api>,
  context: StreamContext,
  options?: SimpleStreamOptions,
  delegate: StreamSimpleDelegate = defaultDelegate,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  void (async () => {
    try {
      const state: RecoveryState = { signatureRetried: false };
      const initial: AttemptSpec<SimpleStreamOptions | undefined, Message[]> = {
        options,
        messages: context.messages,
      };
      const applyRetry = (
        _action: RecoveryAction,
        spec: AttemptSpec<SimpleStreamOptions | undefined, Message[]>,
      ): AttemptSpec<SimpleStreamOptions | undefined, Message[]> => ({
        options: spec.options,
        messages: stripReasoningReplaySignatures(spec.messages),
      });

      const outcome = await driveRecoveryLoop(
        (spec) =>
          delegate(model, { ...context, messages: spec.messages } as StreamContext, spec.options),
        applyRetry,
        initial,
        state,
        (event) => stream.push(event),
      );

      if (!outcome.succeeded) {
        if (outcome.lastError) {
          stream.push(outcome.lastError);
        } else {
          stream.push(
            syncThrowErrorEvent(model, new Error("Stream completed without a terminal event"), options),
          );
        }
      }
      stream.end();
    } catch (error) {
      stream.push(syncThrowErrorEvent(model, error, options));
      stream.end();
    }
  })();

  return stream;
}
