/**
 * Unit tests for selection-limit clamping (prewarm merging a selection's
 * advertised context/completion limits over the registered model base).
 *
 * OpenRouter endpoints advertise generous context_length and
 * max_completion_tokens values that upstream providers do not actually
 * honor — pi's request budgeting then sends max_tokens the serving
 * provider rejects. A selection may only ever LOWER the registered
 * limits, never raise them.
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { clampedSelectionLimits } from "../pi/limits.ts";

describe("clampedSelectionLimits", () => {
  const base = { contextWindow: 131_072, maxTokens: 32_768 };

  it("lowers the context window when the selection advertises less", () => {
    assert.deepEqual(clampedSelectionLimits(base, { contextLength: 65_536 }), {
      contextWindow: 65_536,
      maxTokens: base.maxTokens,
    });
  });

  it("lowers max tokens when the selection advertises less", () => {
    assert.deepEqual(clampedSelectionLimits(base, { maxCompletionTokens: 8_192 }), {
      contextWindow: base.contextWindow,
      maxTokens: 8_192,
    });
  });

  it("never raises the context window above the registered base", () => {
    assert.deepEqual(clampedSelectionLimits(base, { contextLength: 1_048_576 }), {
      contextWindow: base.contextWindow,
      maxTokens: base.maxTokens,
    });
  });

  it("never raises max tokens above the registered base", () => {
    // The bug this guards: an endpoint advertising max_completion_tokens
    // near the full context window, where the upstream enforces far less.
    assert.deepEqual(clampedSelectionLimits(base, { maxCompletionTokens: 943_718 }), {
      contextWindow: base.contextWindow,
      maxTokens: base.maxTokens,
    });
  });

  it("keeps the base limits when the selection omits them", () => {
    assert.deepEqual(clampedSelectionLimits(base, {}), base);
  });

  it("ignores a non-positive context length from the selection", () => {
    assert.deepEqual(clampedSelectionLimits(base, { contextLength: 0 }), base);
  });

  it("lowers both limits together when both advertise less", () => {
    assert.deepEqual(
      clampedSelectionLimits(base, { contextLength: 100_000, maxCompletionTokens: 16_384 }),
      {
        contextWindow: 100_000,
        maxTokens: 16_384,
      },
    );
  });
});
