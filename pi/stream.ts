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

/**
 * Build the standalone provider registration. `apiKey` is only set when a
 * key was actually resolved: stock pi merges the registration with its builtin
 * OpenRouter entry (undefined preserves the builtin's credential handling),
 * while omp's validator requires baseUrl and apiKey whenever models are
 * defined - so under omp the caller always passes both or skips registration.
 * `includeApiStream` is false under omp, which reserves built-in API names
 * and rejects registering a custom streamSimple for "openai-completions";
 * there the models' own api field dispatches to omp's built-in transport
 * (the thought-signature recovery wrapper is stock-pi-only).
 */
export function buildStandaloneProviderConfig(
  models: unknown[] | undefined,
  apiKey?: string,
  includeApiStream = true,
) {
  return {
    models: models ?? [],
    // Stock pi merges a registration that omits baseUrl with its builtin
    // OpenRouter entry; omp instead requires baseUrl whenever models are
    // defined, and its validator rejects the registration without it.
    baseUrl: "https://openrouter.ai/api/v1",
    ...(apiKey ? { apiKey } : {}),
    // omp also validates that every model carries an api, so the provider-
    // level api stays; under omp only the streamSimple override is dropped
    // (omp reserves built-in API names for custom stream functions, so the
    // thought-signature recovery wrapper is stock-pi-only).
    api: "openai-completions",
    // omp only attaches the Authorization header when authHeader is set;
    // stock pi's builtin OpenRouter registration already does this, and its
    // merge keeps the explicit true (same value) harmless there.
    authHeader: true,
    ...(includeApiStream
      ? {
          streamSimple: streamOpenRouterWithRecovery,
        }
      : {}),
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
