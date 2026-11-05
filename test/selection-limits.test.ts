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
import {
  clampCatalogModelLimits,
  clampedSelectionLimits,
  estimateWireContextTokens,
  fitsEndpointContext,
  fitsWithinContext,
} from "../pi/limits.ts";

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

describe("clampCatalogModelLimits", () => {
  it("clamps only selected catalog rows and never raises their limits", () => {
    const rows = [
      { id: "selected", contextWindow: 131_072, maxTokens: 32_768 },
      { id: "other", contextWindow: 64_000, maxTokens: 4_096 },
    ];
    const result = clampCatalogModelLimits(
      rows,
      new Map([["selected", { contextLength: 4_096, maxCompletionTokens: 512 }]]),
    );
    assert.deepEqual(result, [
      { id: "selected", contextWindow: 4_096, maxTokens: 512 },
      rows[1],
    ]);
  });
});

describe("fitsEndpointContext", () => {
  const limits = { contextLength: 2_048, maxCompletionTokens: 512 };

  it("accepts a transcript with system prompt, tools, and output allowance inside the endpoint context", () => {
    assert.equal(
      fitsEndpointContext(
        {
          messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(1_000) }] }],
          systemPrompt: "system instructions",
          tools: [{ name: "lookup", parameters: { type: "object" } }],
        },
        limits,
        32_768,
      ),
      true,
    );
  });

  it("rejects separately supplied system prompt tokens that exceed the endpoint context", () => {
    assert.equal(
      fitsEndpointContext(
        {
          messages: [{ role: "user", content: [{ type: "text", text: "short" }] }],
          systemPrompt: "x".repeat(8_000),
        },
        limits,
        32_768,
      ),
      false,
    );
  });

  it("rejects tool schemas that exceed the endpoint context", () => {
    assert.equal(
      fitsEndpointContext(
        {
          messages: [{ role: "user", content: [{ type: "text", text: "short" }] }],
          tools: [{ name: "lookup", description: "x".repeat(8_000) }],
        },
        limits,
        32_768,
      ),
      false,
    );
  });

  it("normalizes omp developer string messages for the pi host estimator", () => {
    assert.equal(
      fitsEndpointContext(
        {
          messages: [{ role: "developer", content: "Date and working directory changed." } as never],
        },
        limits,
        512,
      ),
      true,
    );
  });

  it("rejects a transcript whose estimated input exceeds the endpoint context", () => {
    assert.equal(
      fitsEndpointContext(
        { messages: [{ role: "user", content: [{ type: "text", text: "x".repeat(10_000) }] }] },
        limits,
        32_768,
      ),
      false,
    );
  });
});

describe("estimateWireContextTokens", () => {
  it("measures ordinary wire messages with string content", () => {
    const tokens = estimateWireContextTokens({
      messages: [
        { role: "system", content: "you are helpful" },
        { role: "user", content: "hello" },
        { role: "assistant", content: "hi" },
      ],
    });
    assert.ok(tokens !== null);
    assert.ok(tokens > 0 && tokens < 100, `unexpected token estimate: ${tokens}`);
  });

  it("measures wire messages with content-part arrays, tool_calls, and tool results", () => {
    const tokens = estimateWireContextTokens({
      messages: [
        { role: "user", content: [{ type: "text", text: "hello" }] },
        {
          role: "assistant",
          content: null,
          tool_calls: [{ id: "call_1", type: "function", function: { name: "calc", arguments: '{"expr":"2+2"}' } }],
        },
        { role: "tool", tool_call_id: "call_1", content: "4" },
      ],
      tools: [{ type: "function", function: { name: "calc", parameters: { type: "object" } } }],
    });
    assert.ok(tokens !== null);
    assert.ok(tokens > 0 && tokens < 200, `unexpected token estimate: ${tokens}`);
  });

  it("counts image parts as image-sized rather than as text", () => {
    const withImage = estimateWireContextTokens({
      messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "https://example/x.png" } }] }],
    });
    const withoutImage = estimateWireContextTokens({
      messages: [{ role: "user", content: [{ type: "text", text: "" }] }],
    });
    assert.ok(withImage !== null && withoutImage !== null);
    assert.ok(withImage > withoutImage + 1_000, `image not charged: ${withImage} vs ${withoutImage}`);
  });

  it("returns null when a wire part cannot be serialized", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    assert.equal(estimateWireContextTokens({ messages: [{ role: "user", content: [circular] }] }), null);
  });

  it("counts a plain reasoning_text field replayed on the assistant message", () => {
    const withoutReasoning = estimateWireContextTokens({
      messages: [{ role: "assistant", content: "answer" }],
    });
    const withReasoning = estimateWireContextTokens({
      messages: [{ role: "assistant", content: "answer", reasoning_text: "x".repeat(1_400) }],
    });
    assert.ok(withoutReasoning !== null && withReasoning !== null);
    assert.ok(
      withReasoning > withoutReasoning + 300,
      `reasoning_text not charged: ${withReasoning} vs ${withoutReasoning}`,
    );
  });

  it("counts structured reasoning_details text replayed on the assistant message", () => {
    const withoutReasoning = estimateWireContextTokens({
      messages: [{ role: "assistant", content: "answer" }],
    });
    const withReasoning = estimateWireContextTokens({
      messages: [
        {
          role: "assistant",
          content: "answer",
          reasoning_details: [{ type: "reasoning.text", text: "x".repeat(1_400), signature: "sig" }],
        },
      ],
    });
    assert.ok(withoutReasoning !== null && withReasoning !== null);
    assert.ok(
      withReasoning > withoutReasoning + 300,
      `reasoning_details text not charged: ${withReasoning} vs ${withoutReasoning}`,
    );
  });

  it("counts every reasoning_details entry, not just the first", () => {
    const single = estimateWireContextTokens({
      messages: [
        {
          role: "assistant",
          content: "answer",
          reasoning_details: [{ type: "reasoning.text", text: "x".repeat(700) }],
        },
      ],
    });
    const doubled = estimateWireContextTokens({
      messages: [
        {
          role: "assistant",
          content: "answer",
          reasoning_details: [
            { type: "reasoning.text", text: "x".repeat(700) },
            { type: "reasoning.text", text: "x".repeat(700) },
          ],
        },
      ],
    });
    assert.ok(single !== null && doubled !== null);
    assert.ok(doubled > single + 150, `second detail not charged: ${doubled} vs ${single}`);
  });

  it("fails closed when a reasoning field cannot be measured", () => {
    assert.equal(
      estimateWireContextTokens({
        messages: [{ role: "assistant", content: "answer", reasoning_text: { parts: [] } }],
      }),
      null,
    );
    assert.equal(
      estimateWireContextTokens({
        messages: [{ role: "assistant", content: "answer", reasoning_details: "not-an-array" }],
      }),
      null,
    );
    assert.equal(
      estimateWireContextTokens({
        messages: [
          { role: "assistant", content: "answer", reasoning_details: [{ type: "reasoning.text", text: 42 }] },
        ],
      }),
      null,
    );
    assert.equal(
      estimateWireContextTokens({
        messages: [
          { role: "assistant", content: "answer", reasoning_details: [{ type: "reasoning.text" }] },
        ],
      }),
      null,
    );
  });

  it("treats an empty body as zero tokens", () => {
    assert.equal(estimateWireContextTokens({}), 0);
  });
});

describe("fitsWithinContext", () => {
  const limits = { contextLength: 2_048, maxCompletionTokens: 512 };

  it("accepts input that fits with the output allowance and headroom", () => {
    assert.equal(fitsWithinContext(1_000, limits, 32_768), true);
  });

  it("rejects input that leaves no room for the output allowance", () => {
    assert.equal(fitsWithinContext(2_000, limits, 32_768), false);
  });

  it("rejects a non-positive context length", () => {
    assert.equal(fitsWithinContext(10, { contextLength: 0, maxCompletionTokens: 512 }, 32_768), false);
  });

  it("rejects a negative or non-finite input estimate", () => {
    assert.equal(fitsWithinContext(-1, limits, 32_768), false);
    assert.equal(fitsWithinContext(Number.NaN, limits, 32_768), false);
  });
});
