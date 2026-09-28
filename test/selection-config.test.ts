/**
 * Unit tests for the user-tunable selection config (config.ts selection
 * section parsing).
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { DEFAULT_SELECTION, parseSelectionSection } from "../pi/config.ts";

const warningsTo = (fn: (w: string[]) => unknown): { result: unknown; warnings: string[] } => {
  const warnings: string[] = [];
  const result = fn(warnings);
  return { result, warnings };
};

describe("parseSelectionSection", () => {
  it("returns today's built-in defaults when no section is present", () => {
    const { result, warnings } = warningsTo((w) => parseSelectionSection(undefined, w));
    assert.deepEqual(result, {
      enabled: true,
      priceAnchor: 3.0,
      budgetWeights: { throughput: 0.55, latency: 0.2, toolCall: 0.15, price: 0.1 },
      flagshipWeights: { throughput: 0.3, latency: 0.25, toolCall: 0.2, price: 0.25 },
    });
    assert.deepEqual(warnings, []);
    assert.equal(DEFAULT_SELECTION.priceAnchor, 3.0);
  });

  it("applies user overrides per field, leaving every other field at its default", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection({ weights: { budget: { price: 0.4 } } }, w),
    );
    const tuned = result as any;
    assert.equal(tuned.budgetWeights.price, 0.4);
    assert.equal(tuned.budgetWeights.throughput, 0.55);
    assert.equal(tuned.budgetWeights.latency, 0.2);
    assert.equal(tuned.budgetWeights.toolCall, 0.15);
    assert.equal(tuned.flagshipWeights.price, 0.25);
    assert.equal(tuned.priceAnchor, 3.0);
    assert.equal(tuned.enabled, true);
    assert.deepEqual(warnings, []);
  });

  it("keeps ratio semantics: weight values are stored raw, not normalized", () => {
    const { result } = warningsTo((w) =>
      parseSelectionSection({ weights: { flagship: { throughput: 2, price: 1 } } }, w),
    );
    const tuned = result as any;
    assert.equal(tuned.flagshipWeights.throughput, 2);
    assert.equal(tuned.flagshipWeights.price, 1);
    assert.equal(tuned.flagshipWeights.latency, 0.25);
  });

  it("accepts disabling the algorithm outright", () => {
    const { result, warnings } = warningsTo((w) => parseSelectionSection({ enabled: false }, w));
    assert.equal((result as any).enabled, false);
    assert.deepEqual(warnings, []);
  });

  it("drops invalid fields with a warning, keeping the default for that field", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection(
        { enabled: "yes", priceAnchor: -1, weights: { budget: { throughput: "fast" } } },
        w,
      ),
    );
    const tuned = result as any;
    assert.equal(tuned.enabled, true);
    assert.equal(tuned.priceAnchor, 3.0);
    assert.equal(tuned.budgetWeights.throughput, 0.55);
    assert.equal(warnings.length, 3);
  });

  it("rejects non-positive priceAnchor values", () => {
    for (const bad of [0, -3]) {
      const { result, warnings } = warningsTo((w) => parseSelectionSection({ priceAnchor: bad }, w));
      assert.equal((result as any).priceAnchor, 3.0);
      assert.equal(warnings.length, 1);
    }
  });

  it("warns on unknown weight keys instead of silently accepting typos", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection({ weights: { budget: { througput: 0.9 } } }, w),
    );
    assert.equal((result as any).budgetWeights.throughput, 0.55);
    assert.equal(warnings.length, 1);
  });

  it("keeps the default set when every weight in a set is zero", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection(
        { weights: { flagship: { throughput: 0, latency: 0, toolCall: 0, price: 0 } } },
        w,
      ),
    );
    assert.deepEqual((result as any).flagshipWeights, DEFAULT_SELECTION.flagshipWeights);
    assert.equal(warnings.length, 1);
  });

  it("keeps the default set when a weights tier is not an object", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection({ weights: { budget: "fast" } }, w),
    );
    assert.deepEqual((result as any).budgetWeights, DEFAULT_SELECTION.budgetWeights);
    assert.equal(warnings.length, 1);
  });

  it("does not mutate the built-in defaults", () => {
    warningsTo((w) => parseSelectionSection({ priceAnchor: 9, weights: { budget: { price: 1 } } }, w));
    assert.equal(DEFAULT_SELECTION.priceAnchor, 3.0);
    assert.equal(DEFAULT_SELECTION.budgetWeights.price, 0.1);
  });
});
