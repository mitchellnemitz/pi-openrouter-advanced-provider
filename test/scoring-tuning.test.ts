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

  it("priceAnchor moves the budget/flagship split (hand-computed per-tier scores)", () => {
    // fast/region blended $2.625/M, slow/region $0.219/M, median $1.42/M:
    // anchor 0.5 -> flagship weights, anchor 9 -> budget weights. The two rows
    // have equal latency (score 1 each), unmeasured tool-call quality (0.85
    // each), and split throughput/price min-max 1/0, so the per-tier weighted
    // totals are exactly:
    //   budget   (0.55/0.20/0.15/0.10): fast 0.8775, slow 0.4275
    //   flagship (0.30/0.25/0.20/0.25): fast 0.72,   slow 0.67
    const flagship = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, tuned({ priceAnchor: 0.5 }));
    const budget = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, tuned({ priceAnchor: 9 }));
    assert.ok(flagship && budget);
    const near = (a: number | undefined, b: number) => assert.ok(Math.abs((a ?? NaN) - b) < 1e-9);
    const byTag = (r: NonNullable<ReturnType<typeof scoreProviders>>) =>
      Object.fromEntries(r.table.map((row) => [row.tag, row.score]));
    near(byTag(flagship)["fast/region"], 0.72);
    near(byTag(flagship)["slow/region"], 0.67);
    near(byTag(budget)["fast/region"], 0.8775);
    near(byTag(budget)["slow/region"], 0.4275);
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

  it("single-axis latency tuning picks the lower-latency provider", () => {
    const tuning = tuned({
      weights: {
        budget: { throughput: 0, latency: 1, toolCall: 0, price: 0 },
        flagship: { throughput: 0, latency: 1, toolCall: 0, price: 0 },
      },
    });
    const endpoints = [
      endpoint({ id: "laggy/region", provider_slug: "laggy/region" }),
      endpoint({
        id: "snappy/region",
        provider_slug: "snappy/region",
        stats: { p50_latency: 250, p99_latency: 300, p50_throughput: 5, p99_throughput: 6, request_count: 500 },
      }),
    ];
    const result = scoreProviders("m", endpoints, NO_TOOL_RATES, tuning);
    assert.equal(result?.pick, "snappy/region");
  });

  it("single-axis tool-call tuning picks the lower error rate regardless of speed", () => {
    const tuning = tuned({
      weights: {
        budget: { throughput: 0, latency: 0, toolCall: 1, price: 0 },
        flagship: { throughput: 0, latency: 0, toolCall: 1, price: 0 },
      },
    });
    const rates = new Map([
      ["fast/region", 0.1],
      ["slow/region", 0.01],
    ]);
    const result = scoreProviders("m", ENDPOINTS, rates, tuning);
    assert.equal(result?.pick, "slow/region");
  });

  it("breaks score ties along the highest-weighted axis first", () => {
    // Symmetric pair under {throughput: 0.5, price: 0.5}: both rows score
    // exactly 0.5 (each owns one min-max extreme), so the tie band decides —
    // throughput and price carry equal weight, and declaration order puts
    // throughput first, so the faster row wins.
    const tuning = tuned({
      weights: {
        budget: { throughput: 0.5, latency: 0, toolCall: 0, price: 0.5 },
        flagship: { throughput: 0.5, latency: 0, toolCall: 0, price: 0.5 },
      },
    });
    const result = scoreProviders("m", ENDPOINTS, NO_TOOL_RATES, tuning);
    assert.equal(result?.pick, "fast/region");
    const scores = result!.table.map((row) => row.score);
    assert.ok(Math.abs((scores[0] ?? 0) - 0.5) < 1e-9);
    assert.ok(Math.abs((scores[1] ?? 0) - 0.5) < 1e-9);
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
