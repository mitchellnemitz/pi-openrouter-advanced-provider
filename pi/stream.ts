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
  openAICompletionsApi,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Api,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai";
import {
  driveRecoveryLoop,
  stripReasoningReplaySignatures,
  type AttemptSpec,
  type RecoveryAction,
  type RecoveryState,
} from "./recovery.js";

function delegateStream(
  model: Model<Api>,
  context: TranscriptContext,
  options: SimpleStreamOptions | undefined,
): AssistantMessageEventStream {
  return openAICompletionsApi().streamSimple(model as Model<"openai-completions">, context, options);
}

function syncThrowErrorEvent(model: Model<Api>, error: unknown): AssistantMessageEvent {
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
    stopReason: "error",
    errorMessage: error instanceof Error ? error.message : String(error),
    timestamp: Date.now(),
  };
  return { type: "error", reason: "error", error: message };
}

export function streamOpenRouterWithRecovery(
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
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
          delegateStream(model, { ...context, messages: spec.messages } as TranscriptContext, spec.options),
        applyRetry,
        initial,
        state,
        (event) => stream.push(event),
      );

      if (!outcome.succeeded && outcome.lastError) stream.push(outcome.lastError);
      stream.end();
    } catch (error) {
      stream.push(syncThrowErrorEvent(model, error));
      stream.end();
    }
  })();

  return stream;
}
