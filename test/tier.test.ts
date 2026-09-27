/**
 * Unit tests for the cost-tier selector helpers (pi/tier.ts).
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 *
 * Covers slug-to-plugin-id mapping, argument parsing, and the plugins
 * merge that carries the selected cost_tier onto the wire.
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
	COST_TIERS,
	applyCostTier,
	parseTierArg,
	routerPluginId,
	tierPickerOptions,
} from "../pi/tier.ts";

describe("routerPluginId", () => {
	it("maps the auto slug to the auto-router plugin id", () => {
		assert.equal(routerPluginId("openrouter/auto"), "auto-router");
	});

	it("maps the auto-beta slug to the auto-beta-router plugin id", () => {
		assert.equal(routerPluginId("openrouter/auto-beta"), "auto-beta-router");
	});

	it("maps routing-suffixed router slugs to their plugin id", () => {
		assert.equal(routerPluginId("openrouter/auto:nitro"), "auto-router");
		assert.equal(routerPluginId("openrouter/auto-beta:floor"), "auto-beta-router");
	});

	it("returns undefined for concrete models and unrelated slugs", () => {
		assert.equal(routerPluginId("z-ai/glm-5.3"), undefined);
		assert.equal(routerPluginId("openrouter/other"), undefined);
		assert.equal(routerPluginId("openrouter/auto-beta2"), undefined);
		assert.equal(routerPluginId(""), undefined);
	});
});

describe("parseTierArg", () => {
	it("parses every tier name", () => {
		for (const tier of COST_TIERS) {
			assert.deepEqual(parseTierArg(tier), { kind: "tier", tier });
		}
	});

	it("parses the clear aliases", () => {
		for (const arg of ["off", "default", "reset", "clear"]) {
			assert.deepEqual(parseTierArg(arg), { kind: "clear" });
		}
	});

	it("rejects unknown input", () => {
		assert.deepEqual(parseTierArg("cheap"), { kind: "invalid", input: "cheap" });
		assert.deepEqual(parseTierArg("max "), { kind: "invalid", input: "max " });
	});

	it("returns undefined for the no-arg invocation (opens the picker)", () => {
		assert.equal(parseTierArg(undefined), undefined);
		assert.equal(parseTierArg(""), undefined);
	});
});

describe("tierPickerOptions", () => {
	it("lists the workspace default plus every named tier", () => {
		const options = tierPickerOptions(null);
		assert.equal(options.length, COST_TIERS.length + 1);
		assert.deepEqual(options.map((o) => o.tier), [null, ...COST_TIERS]);
	});

	it("annotates each tier with its cost band", () => {
		const options = tierPickerOptions(null);
		assert.match(options[1].label, /low.*0-20%/);
		assert.match(options[5].label, /max.*80-100%/);
	});

	it("marks the current selection and leaves others unmarked", () => {
		const options = tierPickerOptions("high");
		assert.match(options[0].label, /default/);
		assert.doesNotMatch(options[0].label, /current/);
		assert.match(options[3].label, /high.*\(current\)/);
		assert.doesNotMatch(options[4].label, /current/);
	});
});

describe("applyCostTier", () => {
	it("leaves the payload untouched when no tier is selected", () => {
		const payload = { model: "openrouter/auto", messages: [] };
		assert.equal(applyCostTier(payload, "openrouter/auto", null), payload);
	});

	it("leaves non-object payloads untouched", () => {
		assert.deepEqual(applyCostTier("x", "openrouter/auto", "max"), "x");
		assert.deepEqual(applyCostTier([1], "openrouter/auto", "max"), [1]);
	});

	it("adds the auto-router plugin to a router-slug request", () => {
		const out = applyCostTier(
			{ model: "openrouter/auto", messages: [] } as any,
			"openrouter/auto",
			"max",
		) as any;
		assert.deepEqual(out.plugins, [{ id: "auto-router", cost_tier: "max" }]);
		// The original payload object must not be mutated.
		assert.equal((out as any).plugins !== undefined, true);
	});

	it("adds the auto-beta-router plugin for the auto-beta slug", () => {
		const out = applyCostTier(
			{ model: "openrouter/auto-beta", messages: [] } as any,
			"openrouter/auto-beta",
			"low",
		) as any;
		assert.deepEqual(out.plugins, [{ id: "auto-beta-router", cost_tier: "low" }]);
	});

	it("does nothing on concrete models — the tier is inert there", () => {
		const payload = { model: "z-ai/glm-5.3", messages: [] };
		const out = applyCostTier(payload, "z-ai/glm-5.3", "max");
		assert.equal(out, payload);
		assert.equal((out as any).plugins, undefined);
	});

	it("preserves unrelated plugins while replacing a matching one in place", () => {
		const payload = {
			model: "openrouter/auto",
			plugins: [
				{ id: "web-plugin", flag: true },
				{ id: "auto-router", cost_tier: "low" },
			],
		};
		const out = applyCostTier(payload as any, "openrouter/auto", "high") as any;
		assert.equal(out.plugins.length, 2);
		assert.deepEqual(out.plugins[0], { id: "web-plugin", flag: true });
		assert.deepEqual(out.plugins[1], { id: "auto-router", cost_tier: "high" });
	});

	it("appends when no matching plugin exists alongside others", () => {
		const payload = { model: "openrouter/auto", plugins: [{ id: "web-plugin" }] };
		const out = applyCostTier(payload as any, "openrouter/auto", "medium") as any;
		assert.equal(out.plugins.length, 2);
		assert.deepEqual(out.plugins[1], { id: "auto-router", cost_tier: "medium" });
	});

	it("treats a non-array plugins field as absent", () => {
		const out = applyCostTier(
			{ model: "openrouter/auto", plugins: "garbage" } as any,
			"openrouter/auto",
			"xhigh",
		) as any;
		assert.deepEqual(out.plugins, [{ id: "auto-router", cost_tier: "xhigh" }]);
	});
});
