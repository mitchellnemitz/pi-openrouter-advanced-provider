/**
 * streamSimple wrapper for the OpenRouter provider registration. Delegates to
 * the host's stream implementation and adds automatic recovery for invalid
 * thought signature errors when OpenRouter fails over between backing
 * providers.
 *
 * Handoff design: under omp, the provider is registered with the UNIQUE api
 * id "openrouter-recovery-completions", so omp's dispatcher routes ONLY our
 * models to this wrapper (foreign openai-completions providers never reach
 * it). The wrapper then clones the model with api "openai-completions" and
 * calls the host dispatcher via platform.ts's completionsStreamSimple() —
 * which under omp is the pi-ai root dispatcher running built-in dispatch WITH
 * mapOptionsForApi, and under pi is the built-in openai-completions simple
 * transport directly.
 *
 * At most one retry per request, and never after any content event has been
 * forwarded. When the retry fails, the freshest error event is surfaced.
 */

import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEvent,
  type AssistantMessageEventStream,
  type Api,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type TranscriptContext,
} from "@earendil-works/pi-ai/compat";
import { completionsStreamSimple, isOmp, scopedNestedPermitOptions } from "./platform.ts";
import { getEndpointLimits } from "./selection-limits-state.js";
import { estimateWireContextTokens, fitsEndpointContext, fitsWithinContext } from "./limits.js";
import { OPENROUTER_BASE_URL } from "./types.ts";
import {
  driveRecoveryLoop,
  stripReasoningReplaySignatures,
  type AttemptSpec,
  type RecoveryAction,
  type RecoveryState,
} from "./recovery.ts";

/**
 * Unique api id used to register our wrapper under omp, so omp's dispatcher
 * routes only our models here instead of every openai-completions provider.
 */
export const OPENROUTER_RECOVERY_API = "openrouter-recovery-completions";

export type StreamSimpleDelegate = (
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
) => AssistantMessageEventStream;

export function defaultDelegate(
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  if (isOmp()) {
    // The host dispatcher keys both registry lookup and built-in selection off
    // `api`; clone to the built-in id, the clone is otherwise identical.
    if (model.api !== OPENROUTER_RECOVERY_API) {
      throw new Error(
        `OpenRouter recovery wrapper expects api "${OPENROUTER_RECOVERY_API}", got "${model.api}"`,
      );
    }
    const native = { ...model, api: "openai-completions" } as unknown as Model<"openai-completions">;
    return completionsStreamSimple()(native, context, options);
  }
  if (model.api !== "openai-completions") {
    throw new Error(`OpenRouter recovery wrapper expects api "openai-completions", got "${model.api}"`);
  }
  return completionsStreamSimple()(model, context, options);
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

/**
 * Output-budget fields the OpenAI wire protocol may carry. OpenRouter
 * endpoints accept either `max_tokens` (legacy) or `max_completion_tokens`;
 * reading only one of them let a request slip past the endpoint ceiling.
 */
const OUTPUT_BUDGET_FIELDS = ["max_tokens", "max_completion_tokens"] as const;

/**
 * Largest numeric output budget present in the outgoing payload, or the
 * fallback when the payload carries none. Every numeric budget field counts:
 * a payload that sets one field under the ceiling while another exceeds it
 * still asks the endpoint for the larger budget, so the guard fails closed
 * on the largest one.
 *
 * A budget field that is present but not a finite number is rejected: an
 * unreadable budget cannot be proven to respect the endpoint ceiling.
 */
function requestMaxTokens(payload: unknown, fallback: number): number {
  if (typeof payload !== "object" || payload === null) return fallback;
  const record = payload as Record<string, unknown>;
  let requested: number | undefined;
  for (const field of OUTPUT_BUDGET_FIELDS) {
    const value = record[field];
    if (value === undefined || value === null) continue;
    // A budget may arrive as a numeric string; it is still a budget, so it is
    // compared against the ceiling rather than ignored.
    const numeric = typeof value === "number"
      ? value
      : (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))
        ? Number(value)
        : null);
    if (numeric === null || !Number.isFinite(numeric)) {
      throw new Error(`Request carries a non-numeric "${field}" output budget`);
    }
    if (requested === undefined || numeric > requested) requested = numeric;
  }
  return requested === undefined ? fallback : requested;
}

/**
 * The output allowance a request actually reserves against an endpoint's
 * context window.
 *
 * The host's `options.maxTokens` is the budget the request carries, and it
 * is what the endpoint will be asked to produce; when the host sets none,
 * the model's registered maximum is the fallback. A request cannot ask for
 * more output than the model allows, so the model's maximum caps the
 * allowance; the endpoint's advertised completion ceiling caps it too (see
 * `fitsWithinContext`).
 *
 * Reserving the model's maximum for a request that asked for less rejects
 * requests that fit the endpoint's context window.
 */
function effectiveOutputAllowance(
  model: Model<Api>,
  options: SimpleStreamOptions | undefined,
): number {
  const requested = options?.maxTokens;
  if (typeof requested === "number" && Number.isFinite(requested) && requested > 0) {
    return Math.min(requested, model.maxTokens);
  }
  return model.maxTokens;
}

export const APP_TITLE = "openrouter-advanced-provider";

export function buildStandaloneProviderConfig(
  models: unknown[] | undefined,
  opts?: {
    apiKey?: string;
    fetchDynamicModels?: (apiKey: string | undefined) => Promise<unknown[]>;
  },
) {
  // Pi keeps builtin defaults unless an endpoint override is configured.
  // Omp always needs a baseUrl for dynamic model registration.
  const baseUrlKeys = isOmp() || process.env.OPENROUTER_BASE_URL
    ? { baseUrl: OPENROUTER_BASE_URL }
    : {};
  if (isOmp()) {
    // Dynamic registration under omp: no static `models` list and NO
    // `apiKey`. Omp's validateProviderConfiguration only requires
    // baseUrl/apiKey when static models are present; the dynamic-models
    // fetcher receives the HOST-RESOLVED key on every call, so the key
    // cascade (and /logout / key rotation) stays authoritative instead of a
    // pinned config key shadowing it.
    //
    // The fetcher's rows must be remapped: omp builds model rows with a
    // modelDef-first `api` preference, so any row carrying an explicit api
    // (e.g. "openrouter") would bypass this wrapper's dispatcher entirely.
    const wrappedFetch = opts?.fetchDynamicModels
      ? (key: string | undefined) =>
          opts.fetchDynamicModels!(key).then((rows) =>
            rows.map((row) => ({ ...(row as Record<string, unknown>), api: OPENROUTER_RECOVERY_API })),
          )
      : undefined;
    return {
      ...baseUrlKeys,
      api: OPENROUTER_RECOVERY_API,
      streamSimple: streamOpenRouterWithRecovery,
      headers: {
        // No HTTP-Referer: this extension sends no public site URL.
        // X-Title identifies the app in OpenRouter's stats.
        "X-Title": APP_TITLE,
      },
      fetchDynamicModels: wrappedFetch,
    };
  }
  return {
    ...baseUrlKeys,
    models: models ?? [],
    // Under pi registration is per-provider with the built-in id.
    api: "openai-completions",
    streamSimple: streamOpenRouterWithRecovery,
    headers: {
      // No HTTP-Referer: this extension sends no public site URL.
      // X-Title identifies the app in OpenRouter's stats.
      "X-Title": APP_TITLE,
    },
  };
}

type ExtendedTranscriptContext = TranscriptContext & {
  systemPrompt?: string;
  tools?: unknown[];
};


export function streamOpenRouterWithRecovery(
  model: Model<Api>,
  context: TranscriptContext,
  options?: SimpleStreamOptions,
  delegate: StreamSimpleDelegate = defaultDelegate,
): AssistantMessageEventStream {
  // No provider guard: foreign openai-completions providers cannot reach this
  // wrapper — under omp the provider is registered with the unique recovery
  // api id, under pi registration is per-provider. An injected delegate
  // (argument !== defaultDelegate) is used verbatim, as before — it stands in
  // for the host dispatcher, so it receives the same scoped options.
  const isInjected = delegate !== defaultDelegate;

  // Under omp the enclosing dispatch already holds our provider's permit for
  // this whole request; the nested dispatch must not ask for it again (see
  // scopedNestedPermitOptions). Applied to every delegate invocation, default
  // or injected; under pi this is an identity pass-through.
  const hostOptions = scopedNestedPermitOptions(options, model.provider);
  const originalOnPayload = hostOptions?.onPayload;
  const limitContext = context as ExtendedTranscriptContext;
  const guardedOptions = {
    ...(hostOptions ?? {}),
    onPayload: async (payload: unknown, requestModel: Model<Api>) => {
      const replacement = await originalOnPayload?.(payload, requestModel);
      const outgoing = replacement ?? payload;
      const limits = getEndpointLimits(model.id);
      if (limits) {
        const body = outgoing && typeof outgoing === "object" ? outgoing as Record<string, unknown> : {};
        const hasFinalMessages = Array.isArray(body.messages);
        const outputCeiling = Math.min(
          requestModel.maxTokens,
          limits.maxCompletionTokens && limits.maxCompletionTokens > 0
            ? limits.maxCompletionTokens
            : requestModel.maxTokens,
        );
        const requested = requestMaxTokens(outgoing, outputCeiling);
        if (requested > outputCeiling) {
          throw new Error(`Request exceeds the selected OpenRouter endpoint completion limit (${outputCeiling} tokens)`);
        }
        let fits: boolean;
        if (hasFinalMessages) {
          // The FINAL payload is the serialized OpenAI wire body: role
          // strings, string-or-part content, `tool_calls`. pi's estimator
          // expects pi Message blocks, so measuring the wire body with it
          // mis-estimated (or threw on) ordinary requests. Estimate the wire
          // shape directly instead.
          const wireTokens = estimateWireContextTokens({
            messages: body.messages,
            tools: Array.isArray(body.tools) ? body.tools : undefined,
          });
          if (wireTokens === null) {
            throw new Error(
              `Request could not be measured against the selected OpenRouter endpoint context limit (${limits.contextLength} tokens)`,
            );
          }
          let inputTokens = wireTokens;
          // Hosts normally fold the system prompt into the wire messages. When
          // they do not, the prompt still occupies context on the wire, so
          // count it (only when no system/developer message is present, to
          // avoid counting it twice).
          const hasSystemMessage = (body.messages as unknown[]).some((message) => {
            if (!message || typeof message !== "object") return false;
            const role = (message as Record<string, unknown>).role;
            return role === "system" || role === "developer";
          });
          if (!hasSystemMessage && limitContext.systemPrompt) {
            const promptTokens = estimateWireContextTokens({
              messages: [{ role: "system", content: limitContext.systemPrompt }],
            });
            if (promptTokens === null) {
              throw new Error(
                `Request could not be measured against the selected OpenRouter endpoint context limit (${limits.contextLength} tokens)`,
              );
            }
            inputTokens += promptTokens;
          }
          // Tool schemas occupy context too; the wire body carries them, but
          // fall back to the transcript's declarations when it does not.
          if (!Array.isArray(body.tools) && limitContext.tools?.length) {
            const toolTokens = estimateWireContextTokens({ tools: limitContext.tools });
            if (toolTokens === null) {
              throw new Error(
                `Request could not be measured against the selected OpenRouter endpoint context limit (${limits.contextLength} tokens)`,
              );
            }
            inputTokens += toolTokens;
          }
          fits = fitsWithinContext(inputTokens, limits, requested);
        } else {
          // No serialized messages yet: keep estimating the pre-dispatch
          // transcript and system prompt.
          const finalContext = {
            messages: limitContext.messages,
            systemPrompt: limitContext.systemPrompt,
            tools: Array.isArray(body.tools) ? body.tools : undefined,
          } as ExtendedTranscriptContext;
          fits = fitsEndpointContext(finalContext, limits, requested);
        }
        if (!fits) {
          throw new Error(`Request exceeds the selected OpenRouter endpoint context limit (${limits.contextLength} tokens)`);
        }
      }
      return replacement;
    },
  };

  const stream = createAssistantMessageEventStream();
  const endpointLimits = getEndpointLimits(model.id);
  // The pre-dispatch guard must reserve the output the request actually
  // asks for (`options.maxTokens`), not the model's registered maximum:
  // reserving the maximum rejects requests that fit the endpoint's context
  // window with the budget the request carries.
  const outputAllowance = effectiveOutputAllowance(model, options);
  if (endpointLimits && !fitsEndpointContext({ ...limitContext, tools: undefined }, endpointLimits, outputAllowance)) {
    stream.push(
      syncThrowErrorEvent(
        model,
        new Error(`Request exceeds the selected OpenRouter endpoint context limit (${endpointLimits.contextLength} tokens)`),
        options,
      ),
    );
    stream.end();
    return stream;
  }

  void (async () => {
    try {
      const state: RecoveryState = { signatureRetried: false };
      const initial: AttemptSpec<SimpleStreamOptions | undefined, Message[]> = {
        options: guardedOptions,
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
        isInjected
          ? (spec) =>
              delegate(model, { ...context, messages: spec.messages } as TranscriptContext, spec.options)
          : (spec) =>
              defaultDelegate(model, { ...context, messages: spec.messages } as TranscriptContext, spec.options),
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
