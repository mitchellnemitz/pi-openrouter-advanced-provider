/**
 * Unit tests for provider-selection tuning (scoring.ts honoring
 * SelectionTuning from the config file's selection section).
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DEFAULT_SELECTION, parseSelectionSection } from "../pi/config.ts";
import { scoreProviders, tiebreakAxes } from "../pi/scoring.ts";
import type { SelectionTuning, StatEndpoint } from "../pi/types.ts";

const tuned = (raw: unknown): SelectionTuning =>
  parseSelectionSection(raw, []);

// Two healthy sampled endpoints from the same model: A fast and pricey,
// B slow and cheap, with identical latency so throughput and price are the
// only discriminating axes.
const endpoint = (over: Partial<StatEndpoint>): StatEndpoint => ({
  id: "prov/region",
  provider_slug: "prov/region",
  status: 0,
  supported_parameters: ["tools"],
  quantization: "bf16",
  context_length: 100_000,
  pricing: { prompt: "0.0000005", completion: "0.000002" },
  stats: {
    p50_latency: 500,
    p99_latency: 600,
    p50_throughput: 100,
    p99_throughput: 120,
    request_count: 500,
  },
  ...over,
});

const ENDPOINTS = [
  endpoint({
    id: "fast/region",
    provider_slug: "fast/region",
    p50_latency: undefined as never,
    pricing: { prompt: "0.0000015", completion: "0.000006" },
    stats: {
      p50_latency: 500, p99_latency: 600, p50_throughput: 200,
      p99_throughput: 240, request_count: 500,
    },
  }),
  endpoint({
    id: "slow/region",
    provider_slug: "slow/region",
    pricing: { prompt: "0.000000125", completion: "0.0000005" },
    stats: {
      p50_latency: 500, p99_latency: 600, p50_throughput: 50,
      p99_throughput: 60, request_count: 500,
    },
  }),
];

const NO_TOOL_RATES = new Map<string, number>();

describe("scoreProviders tuning", () => {
  it("picks the fast provider under the default tuning (budget anchor, throughput-heavy)", () => {
    const result = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, DEFAULT_SELECTION);
    assert.equal(result?.pick, "fast/region");
  });

  it("flips to the cheap provider when weights favor price", () => {
    const tuning = tuned({ weights: { budget: { throughput: 0.2, price: 0.8 }, flagship: { throughput: 0.2, price: 0.8 } } });
    const result = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, tuning);
    assert.equal(result?.pick, "slow/region");
  });

  it("priceAnchor moves the budget/flagship split", () => {
    const cheap = tuned({ priceAnchor: 0.5 });
    const pricey = tuned({ priceAnchor: 9 });
    const a = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, cheap);
    const b = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, pricey);
    assert.ok(a && b);
    // fast/region blended price ~$4.69/M, slow/region ~$0.22/M, median ~$2.45/M:
    // anchor 0.5 -> flagship weights; anchor 9 -> budget weights. Same pick,
    // measurably different scores.
    assert.notEqual(a.table[0].score, b.table[0].score);
  });

  it("single-axis tuning picks strictly by that axis (throughput max)", () => {
    const tuning = tuned({
      weights: {
        budget: { throughput: 1, latency: 0, toolCall: 0, price: 0 },
        flagship: { throughput: 1, latency: 0, toolCall: 0, price: 0 },
      },
    });
    const result = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, tuning);
    assert.equal(result?.pick, "fast/region");
    // The disabled axes contribute nothing: the winner's score equals the
    // normalized throughput score alone (1.0 for the fastest row).
    assert.equal(result?.table[0].score, 1);
  });

  it("still gates stunted-context and heavy-quantization rows under custom weights", () => {
    const tuning = tuned({ weights: { budget: { throughput: 1, latency: 0, toolCall: 0, price: 0 }, flagship: { throughput: 1, latency: 0, toolCall: 0, price: 0 } } });
    const endpoints = [
      ...ENDPOINTS,
      endpoint({
        id: "stunted/region",
        provider_slug: "stunted/region",
        context_length: 20_000, // < 50% of the 100k median
        stats: { p50_latency: 10, p99_latency: 11, p50_throughput: 9_999, p99_throughput: 10_000, request_count: 500 },
      }),
      endpoint({
        id: "quant/region",
        provider_slug: "quant/region",
        quantization: "fp4",
        stats: { p50_latency: 10, p99_latency: 11, p50_throughput: 9_999, p99_throughput: 10_000, request_count: 500 },
      }),
    ];
    const result = scoreProviders("m", endpoints, NO_TOOL_RATES, tuning);
    assert.equal(result?.pick, "fast/region");
    assert.ok(result?.ignore.includes("stunted"));
    assert.ok(result!.table.every((row) => row.tag !== "quant/region"));
  });
});

describe("tiebreakAxes", () => {
  it("orders axes by descending weight", () => {
    assert.deepEqual(
      tiebreakAxes(DEFAULT_SELECTION.budgetWeights),
      ["throughput", "latency", "toolCall", "price"],
    );
    assert.deepEqual(
      tiebreakAxes(DEFAULT_SELECTION.flagshipWeights),
      ["throughput", "latency", "price", "toolCall"],
    );
  });

  it("excludes zero-weight axes from the tiebreak", () => {
    assert.deepEqual(
      tiebreakAxes({ throughput: 1, latency: 0, toolCall: 0, price: 0 }),
      ["throughput"],
    );
    assert.deepEqual(
      tiebreakAxes({ throughput: 0, latency: 0, toolCall: 0, price: 2 }),
      ["price"],
    );
  });

  it("puts the user's dominant axis first", () => {
    assert.deepEqual(
      tiebreakAxes({ throughput: 0.2, latency: 0.3, toolCall: 0.1, price: 0.4 }),
      ["price", "latency", "throughput", "toolCall"],
    );
  });
});
