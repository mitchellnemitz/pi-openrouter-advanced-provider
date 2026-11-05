import { estimateContextTokens, estimateTextTokens } from "@earendil-works/pi-ai/utils/estimate";
import type { Message } from "@earendil-works/pi-ai";
import type { SelectionResult } from "./types.js";

/**
 * Merge a selection's advertised context/completion limits over the
 * registered model base, lowering only. OpenRouter endpoints advertise
 * generous context_length and max_completion_tokens values that upstream
 * providers do not actually honor — request budgeting that trusts those
 * numbers sends max_tokens the serving provider rejects. The registered
 * base is the verified ceiling; a selection may only narrow it.
 */
export function clampedSelectionLimits(
  base: { contextWindow: number; maxTokens: number },
  result: Pick<SelectionResult, "contextLength" | "maxCompletionTokens">,
): { contextWindow: number; maxTokens: number } {
  const { contextLength, maxCompletionTokens } = result;
  return {
    contextWindow:
      contextLength !== undefined && contextLength > 0
        ? Math.min(contextLength, base.contextWindow)
        : base.contextWindow,
    maxTokens:
      maxCompletionTokens !== undefined && maxCompletionTokens > 0
        ? Math.min(maxCompletionTokens, base.maxTokens)
        : base.maxTokens,
  };
}

/** Apply selected endpoint ceilings to dynamic registry rows without raising catalog limits. */
export function clampCatalogModelLimits<T extends { id: string; contextWindow: number; maxTokens: number }>(
  models: T[],
  selections: ReadonlyMap<string, Pick<SelectionResult, "contextLength" | "maxCompletionTokens">>,
): T[] {
  return models.map((model) => {
    const selection = selections.get(model.id);
    if (!selection) return model;
    return { ...model, ...clampedSelectionLimits(model, selection) };
  });
}
/**
 * Wire-shape estimation constants. pi's estimator works on pi Message
 * blocks; the SERIALIZED request body uses the OpenAI wire shape (role
 * strings, plain string or part-array content, `tool_calls`), which pi's
 * estimator cannot consume — casting it to Message[] made the estimator
 * throw (and the guard reject ordinary requests).
 */
const WIRE_CHARS_PER_TOKEN = 3.5;
const WIRE_IMAGE_CHARS = 4800;
/** Per-message wire framing overhead (role name, brackets, separators). */
const WIRE_MESSAGE_OVERHEAD_CHARS = 16;

function wireJsonLength(value: unknown): number {
  try {
    const serialized = JSON.stringify(value);
    return serialized === undefined ? 0 : serialized.length;
  } catch {
    return -1;
  }
}

/**
 * Characters contributed by one wire message's `content` field: a plain
 * string, or an array of parts (`text`, `image_url`, ...).
 */
function wireContentChars(content: unknown): number {
  if (typeof content === "string") return content.length;
  if (content === null || content === undefined) return 0;
  if (!Array.isArray(content)) {
    const length = wireJsonLength(content);
    return length < 0 ? -1 : length;
  }
  let chars = 0;
  for (const part of content) {
    if (typeof part === "string") {
      chars += part.length;
      continue;
    }
    if (!part || typeof part !== "object") {
      const length = wireJsonLength(part);
      if (length < 0) return -1;
      chars += length;
      continue;
    }
    const record = part as Record<string, unknown>;
    const type = typeof record.type === "string" ? record.type : "";
    if (type === "text" || type === "input_text" || type === "output_text") {
      chars += typeof record.text === "string" ? record.text.length : 0;
    } else if (type === "image_url" || type === "input_image") {
      chars += WIRE_IMAGE_CHARS;
    } else {
      const length = wireJsonLength(record);
      if (length < 0) return -1;
      chars += length;
    }
  }
  return chars;
}

/**
 * Characters contributed by one structured `reasoning_details` entry.
 *
 * `reasoning.text` entries carry the replayed reasoning prose in `text` and
 * `reasoning.summary` entries a summary in `summary`. That prose is
 * replayed into the model's context, so it is charged in full.
 *
 * The opaque payloads — a detail's `signature` and the `data` of an
 * encrypted entry — are provider replay metadata, not model-visible
 * context. They are validated (a present payload must be a string) but not
 * charged: charging ciphertext at the prose ratio would inflate the
 * estimate by an order of magnitude and wrongly reject requests that fit.
 *
 * An entry whose shape cannot be measured returns -1, so the caller fails
 * closed instead of undercounting the final wire input: an entry that is
 * not an object, a known detail type missing its own payload, or a payload
 * field that is present but not a string. An entry of an unrecognized type
 * carries an unknown payload and is charged its serialized length: never
 * free, but not assumed to be prose.
 *
 * Note the asymmetry with the plain-text fields: an empty `text` is
 * charged zero (it really is empty), while an unknown-typed entry is
 * charged its serialized length whatever it contains.
 */
function wireReasoningDetailChars(detail: unknown): number {
  if (!detail || typeof detail !== "object" || Array.isArray(detail)) return -1;
  const record = detail as Record<string, unknown>;
  const type = typeof record.type === "string" ? record.type : "";
  // Prose fields carry the replayed reasoning: charged in full.
  let chars = 0;
  for (const field of ["text", "summary"]) {
    const value = record[field];
    if (value === undefined || value === null) continue;
    if (typeof value !== "string") return -1;
    chars += value.length;
  }
  // Opaque replay payloads: validated, never charged as prompt tokens.
  for (const field of ["signature", "data"]) {
    const value = record[field];
    if (value === undefined || value === null) continue;
    if (typeof value !== "string") return -1;
  }
  switch (type) {
    case "reasoning.text":
      // A known detail type must carry its own payload.
      if (typeof record.text !== "string") return -1;
      return chars;
    case "reasoning.summary":
      if (typeof record.summary !== "string") return -1;
      return chars;
    case "reasoning.encrypted":
      if (typeof record.data !== "string") return -1;
      return chars;
    default: {
      // An unrecognized type carries an unknown payload: charge its
      // serialized length, so it is never free.
      const length = wireJsonLength(record);
      return length;
    }
  }
}

/**
 * Characters contributed by an assistant message's replayed reasoning:
 * the plain-text reasoning fields and the structured `reasoning_details`
 * entries.
 *
 * `reasoning` and `reasoning_content` keep their established handling (a
 * string is counted, any other shape is ignored). The newly measured
 * fields fail closed: a present `reasoning_text` that is not a string, a
 * `reasoning_details` value that is not an array, and an entry that cannot
 * be measured all return -1, so the caller rejects the request instead of
 * undercounting the final wire input.
 */
function wireReasoningChars(record: Record<string, unknown>): number {
  let chars = 0;
  if (typeof record.reasoning === "string") chars += record.reasoning.length;
  if (typeof record.reasoning_content === "string") chars += record.reasoning_content.length;
  const text = record.reasoning_text;
  if (text !== undefined && text !== null) {
    if (typeof text !== "string") return -1;
    chars += text.length;
  }
  const details = record.reasoning_details;
  if (details === undefined || details === null) return chars;
  if (!Array.isArray(details)) return -1;
  for (const detail of details) {
    const detailChars = wireReasoningDetailChars(detail);
    if (detailChars < 0) return -1;
    chars += detailChars;
  }
  return chars;
}

/**
 * Estimate the input tokens of a SERIALIZED OpenAI-wire request body
 * (messages + tools). Returns null when the body cannot be measured
 * (unserializable parts) — callers must fail closed on null.
 *
 * The wire shape differs from pi's Message blocks in every relevant way:
 * roles are plain strings ("user", "assistant", "system", "developer",
 * "tool"), content is a string or an array of parts, and assistant tool
 * calls live in `tool_calls` rather than in content blocks.
 */
export function estimateWireContextTokens(body: {
  messages?: unknown;
  tools?: unknown;
}): number | null {
  let chars = 0;
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!message || typeof message !== "object") {
      const length = wireJsonLength(message);
      if (length < 0) return null;
      chars += length;
      continue;
    }
    const record = message as Record<string, unknown>;
    const contentChars = wireContentChars(record.content);
    if (contentChars < 0) return null;
    chars += contentChars + WIRE_MESSAGE_OVERHEAD_CHARS;
    if (record.tool_calls !== undefined) {
      const callsLength = wireJsonLength(record.tool_calls);
      if (callsLength < 0) return null;
      chars += callsLength;
    }
    // OpenRouter/pi replay prior reasoning on the assistant message:
    // as plain text (`reasoning`, `reasoning_content`, `reasoning_text`)
    // or as structured `reasoning_details` entries. Both occupy context
    // on the next request, so both are charged.
    const reasoningChars = wireReasoningChars(record);
    if (reasoningChars < 0) return null;
    chars += reasoningChars;
    if (record.name !== undefined && typeof record.name === "string") {
      chars += record.name.length;
    }
  }
  const tools = Array.isArray(body.tools) ? body.tools : [];
  if (tools.length > 0) {
    const toolsLength = wireJsonLength(tools);
    if (toolsLength < 0) return null;
    chars += toolsLength;
  }
  return Math.ceil(chars / WIRE_CHARS_PER_TOKEN);
}

/**
 * Shared context-window check: input tokens + output allowance + protocol
 * headroom must fit the endpoint's context length.
 */
export function fitsWithinContext(
  inputTokens: number,
  limits: Pick<SelectionResult, "contextLength" | "maxCompletionTokens">,
  maxTokens: number,
  protocolHeadroom = 256,
): boolean {
  if (!Number.isFinite(limits.contextLength) || limits.contextLength <= 0) return false;
  if (!Number.isFinite(inputTokens) || inputTokens < 0) return false;
  const outputAllowance = Math.min(
    maxTokens,
    limits.maxCompletionTokens && limits.maxCompletionTokens > 0 ? limits.maxCompletionTokens : maxTokens,
  );
  return inputTokens + outputAllowance + protocolHeadroom <= limits.contextLength;
}

/**
 * Estimate transcript, separately supplied system prompt, and tool schema
 * tokens against a selected endpoint. Omp carries developer strings in
 * `messages`; normalize those into pi's estimator's user-text representation.
 */
export function fitsEndpointContext(
  context: {
    messages: readonly Message[];
    systemPrompt?: string;
    tools?: readonly unknown[];
  },
  limits: Pick<SelectionResult, "contextLength" | "maxCompletionTokens">,
  maxTokens: number,
  protocolHeadroom = 256,
): boolean {
  if (!Number.isFinite(limits.contextLength) || limits.contextLength <= 0) return false;
  try {
    const messages = context.messages.map((message) => {
      const candidate = message as unknown as { role: string; content: unknown };
      if (candidate.role === "developer" && typeof candidate.content === "string") {
        return {
          role: "user",
          content: [{ type: "text", text: candidate.content }],
        } as Message;
      }
      return message;
    });
    let inputTokens = estimateContextTokens(messages).tokens;
    if (context.systemPrompt) inputTokens += estimateTextTokens(context.systemPrompt);
    if (context.tools?.length) {
      const serializedTools = JSON.stringify(context.tools);
      if (serializedTools === undefined) return false;
      inputTokens += estimateTextTokens(serializedTools);
    }
    return fitsWithinContext(inputTokens, limits, maxTokens, protocolHeadroom);
  } catch {
    return false;
  }
}
