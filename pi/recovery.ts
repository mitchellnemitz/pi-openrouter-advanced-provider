/**
 * Automatic recovery for invalid thought signature errors on OpenRouter.
 *
 * OpenRouter routes multi-provider models (like Gemini) across different backing
 * providers (Google AI Studio, Google Vertex AI, etc.). When OpenRouter fails over
 * between providers mid-session, the new provider cannot validate the prior
 * provider's encrypted reasoning details or thought signatures and returns HTTP 400
 * ("Invalid thought signature" or "Thought signature is not valid").
 *
 * This module provides the pure logic (error detection, signature stripping, and
 * retry loop) to recover by stripping invalid signatures and retrying the request once.
 */

// ---------- Error detection ----------

const THOUGHT_SIGNATURE_PATTERN = /invalid thought signature|thought signature.*not valid/i;

/**
 * Detect failover signature errors (e.g. "Invalid thought signature" or
 * "Unable to submit request because Thought signature is not valid").
 */
export function isThoughtSignatureError(text: string | undefined): boolean {
  return !!text && THOUGHT_SIGNATURE_PATTERN.test(text);
}

// ---------- Signature strip ----------

/**
 * Remove reasoning replay data that OpenRouter cannot validate after a provider
 * failover: thinking blocks' thinkingSignature (the encrypted reasoning_details payload),
 * tool calls' thoughtSignature, and text blocks' textSignature.
 *
 * Thinking text, tool calls, and text content themselves stay intact.
 * Returns the original array when there is nothing to strip.
 */
export function stripReasoningReplaySignatures<T extends object>(messages: readonly T[]): T[] {
  let stripped = false;
  const next = messages.map((message) => {
    const msg = message as Record<string, unknown>;
    if (msg?.role !== "assistant" || !Array.isArray(msg.content)) return message;
    let messageChanged = false;
    const content = msg.content.map((block) => {
      if (!block || typeof block !== "object") return block;
      const b = block as Record<string, unknown>;
      if (b.type === "thinking" && b.thinkingSignature !== undefined) {
        stripped = true;
        messageChanged = true;
        const { thinkingSignature, ...rest } = b;
        return rest;
      }
      if (b.type === "toolCall" && b.thoughtSignature !== undefined) {
        stripped = true;
        messageChanged = true;
        const { thoughtSignature, ...rest } = b;
        return rest;
      }
      if (b.type === "text" && b.textSignature !== undefined) {
        stripped = true;
        messageChanged = true;
        const { textSignature, ...rest } = b;
        return rest;
      }
      return block;
    });
    return messageChanged ? ({ ...msg, content } as unknown as T) : message;
  });
  return stripped ? next : (messages as T[]);
}

// ---------- Retry decisions ----------

/** Per-request retry bookkeeping: at most one signature retry per request. */
export interface RecoveryState {
  signatureRetried: boolean;
}

export type RecoveryAction = { kind: "thought-signature" };

/**
 * Decide the retry for a stream error, consuming the one signature retry attempt
 * per request. Returns undefined when the error is not a thought signature error
 * or when the retry has already been used.
 */
export function planErrorRecovery(
  errorMessage: string | undefined,
  state: RecoveryState,
): RecoveryAction | undefined {
  if (!errorMessage) return undefined;
  if (!state.signatureRetried && isThoughtSignatureError(errorMessage)) {
    state.signatureRetried = true;
    return { kind: "thought-signature" };
  }
  return undefined;
}

// ---------- Attempt loop ----------

export interface AttemptSpec<TOptions, TMessages> {
  options: TOptions;
  messages: TMessages;
}

export interface RecoveryLoopOutcome<TEvent> {
  succeeded: boolean;
  /** The freshest error observed; always undefined when succeeded. */
  lastError: TEvent | undefined;
}

/**
 * Generic attempt loop for the failure-recovery wrapper: run an attempt,
 * forward every non-error event to `emit`, and on an error either plan a
 * retry (once per error class, never after content has been forwarded) or
 * stop. The FRESHEST error wins: when a retry fails with a different error,
 * that error is surfaced, not the original. (An abort naturally wins by being last:
 * no error class matches a cancelled request, so the loop stops there.)
 */
export async function driveRecoveryLoop<TEvent extends { type: string }, TOptions, TMessages>(
  makeAttempt: (spec: AttemptSpec<TOptions, TMessages>) => AsyncIterable<TEvent>,
  applyRetry: (action: RecoveryAction, spec: AttemptSpec<TOptions, TMessages>) => AttemptSpec<TOptions, TMessages>,
  initial: AttemptSpec<TOptions, TMessages>,
  state: RecoveryState,
  emit: (event: TEvent) => void,
): Promise<RecoveryLoopOutcome<TEvent>> {
  let attempt: AttemptSpec<TOptions, TMessages> | undefined = initial;
  let lastError: TEvent | undefined;
  let succeeded = false;
  while (attempt) {
    const current: AttemptSpec<TOptions, TMessages> = attempt;
    attempt = undefined;
    let forwarded = false;
    for await (const event of makeAttempt(current)) {
      if (event.type === "error") {
        lastError = event;
        // Retrying after forwarded content would duplicate the partial response;
        // signature failures reject the request before any content, but this
        // guard keeps it true by construction.
        if (forwarded) break;
        const action = planErrorRecovery((event as { error?: { errorMessage?: string } })?.error?.errorMessage, state);
        if (action) attempt = applyRetry(action, current);
        break;
      }
      if (event.type === "done") succeeded = true;
      forwarded = true;
      emit(event);
    }
  }
  return { succeeded, lastError: succeeded ? undefined : lastError };
}
