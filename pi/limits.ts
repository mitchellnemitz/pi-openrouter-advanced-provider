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
