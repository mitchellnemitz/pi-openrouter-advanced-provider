/**
 * Unit tests for the user-tunable selection config (config.ts selection
 * section parsing).
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import {
  DEFAULT_SELECTION,
  loadRequestConfig,
  parseSelectionSection,
} from "../pi/config.ts";

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

  it("rejects NaN and Infinity price anchors", () => {
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
      const { result, warnings } = warningsTo((w) => parseSelectionSection({ priceAnchor: bad }, w));
      assert.equal((result as any).priceAnchor, 3.0);
      assert.equal(warnings.length, 1);
    }
  });

  it("rejects negative weight values", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection({ weights: { budget: { price: -0.5 } } }, w),
    );
    assert.equal((result as any).budgetWeights.price, 0.1);
    assert.equal(warnings.length, 1);
  });

  it("keeps the default weights when the weights section is not an object", () => {
    const { result, warnings } = warningsTo((w) => parseSelectionSection({ weights: "fast" }, w));
    assert.deepEqual((result as any).budgetWeights, DEFAULT_SELECTION.budgetWeights);
    assert.equal(warnings.length, 1);
  });

  it("keeps the default budget weights when the budget tier is all zero", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection({ weights: { budget: { throughput: 0, latency: 0, toolCall: 0, price: 0 } } }, w),
    );
    assert.deepEqual((result as any).budgetWeights, DEFAULT_SELECTION.budgetWeights);
    assert.equal((result as any).flagshipWeights.throughput, 0.3);
    assert.equal(warnings.length, 1);
  });

  it("warns on unknown weight tiers instead of silently dropping them", () => {
    const { result, warnings } = warningsTo((w) =>
      parseSelectionSection({ weights: { midrange: { throughput: 1 } } }, w),
    );
    assert.deepEqual((result as any).budgetWeights, DEFAULT_SELECTION.budgetWeights);
    assert.equal(warnings.length, 1);
  });

  it("pinned: the shipped config file carries the built-in defaults", () => {
    const shipped = JSON.parse(
      fs.readFileSync(path.join(import.meta.dirname, "..", "pi", "openrouter-advanced-provider.json"), "utf-8"),
    );
    assert.deepEqual(shipped.selection, {
      enabled: DEFAULT_SELECTION.enabled,
      priceAnchor: DEFAULT_SELECTION.priceAnchor,
      weights: { budget: DEFAULT_SELECTION.budgetWeights, flagship: DEFAULT_SELECTION.flagshipWeights },
    });
  });
});

describe("loadRequestConfig selection seam", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "seltune-cfg-"));
  const write = (name: string, content: unknown) => {
    const file = path.join(tmp, name);
    fs.writeFileSync(file, JSON.stringify(content));
    return file;
  };

  it("merges shipped and user selection per field, surviving false and zero", () => {
    const shipped = write("shipped.json", {
      selection: { priceAnchor: 4.0, weights: { budget: { price: 0.2 } } },
    });
    const user = write("user.json", {
      selection: { enabled: false, weights: { budget: { price: 0 }, flagship: { throughput: 0.9 } } },
    });
    const loaded = loadRequestConfig(shipped, user);
    assert.equal(loaded.selection.enabled, false);
    assert.equal(loaded.selection.priceAnchor, 4.0);
    assert.equal(loaded.selection.budgetWeights.price, 0);
    assert.equal(loaded.selection.budgetWeights.throughput, 0.55);
    assert.equal(loaded.selection.flagshipWeights.throughput, 0.9);
    assert.equal(loaded.selection.flagshipWeights.price, 0.25);
    assert.deepEqual(loaded.warnings, []);
  });

  it("warns and keeps defaults when the user selection section is not an object", () => {
    const shipped = write("shipped2.json", {});
    const user = write("user2.json", { selection: 5 });
    const loaded = loadRequestConfig(shipped, user);
    assert.deepEqual(loaded.selection, DEFAULT_SELECTION);
    assert.ok(loaded.warnings.some((warning) => warning.includes("selection")));
  });

  it("warns on an unknown weight tier coming from a real file", () => {
    const shipped = write("shipped3.json", {});
    const user = write("user3.json", { selection: { weights: { midrange: { throughput: 1 } } } });
    const loaded = loadRequestConfig(shipped, user);
    assert.deepEqual(loaded.selection.budgetWeights, DEFAULT_SELECTION.budgetWeights);
    assert.ok(loaded.warnings.some((warning) => warning.includes("midrange")));
  });
});
