/**
 * Cost-tier selector helpers for the OpenRouter auto-router plugins.
 *
 * The OpenRouter auto-router accepts a named cost_tier that selects a
 * cost-percentile band of average generation cost for the classified task
 * (low [0,20), medium [20,40), high [40,60), xhigh [60,80), max [80,100]).
 * The tier applies only to the router slugs (openrouter/auto, openrouter/
 * auto-beta); on concrete models it is inert. When no tier is set the
 * extension adds nothing to the payload and the workspace default applies —
 * including when workspace-level "prevent overrides" discards request-level
 * settings (the value is ignored, not an error).
 *
 * The plugin id must match the request's router slug: "auto-router" for
 * openrouter/auto and "auto-beta-router" for openrouter/auto-beta. A
 * mismatched id is silently ignored by the API.
 */

export const COST_TIERS = ["low", "medium", "high", "xhigh", "max"] as const;

export type CostTier = (typeof COST_TIERS)[number];

/** Average-generation-cost percentile band each named tier selects. */
export const COST_TIER_BANDS: Record<CostTier, string> = {
  low: "0-20%",
  medium: "20-40%",
  high: "40-60%",
  xhigh: "60-80%",
  max: "80-100%",
};

const ROUTER_PLUGIN_IDS = ["auto-router", "auto-beta-router"] as const;

/** Plugin id for a router slug: "auto-router" for openrouter/auto (with an
 * optional routing suffix), "auto-beta-router" for openrouter/auto-beta,
 * undefined for anything else. */
export function routerPluginId(model: string): (typeof ROUTER_PLUGIN_IDS)[number] | undefined {
  if (/^openrouter\/auto-beta(:[a-z0-9._-]+)?$/i.test(model)) return "auto-beta-router";
  if (/^openrouter\/auto(:[a-z0-9._-]+)?$/i.test(model)) return "auto-router";
  return undefined;
}

export type ParsedTierArg =
  | { kind: "tier"; tier: CostTier }
  | { kind: "clear" }
  | { kind: "invalid"; input: string };

/**
 * Parse a /openrouter-tier argument. Undefined/empty means "no argument" —
 * the command should open the interactive picker instead. "off" and its
 * aliases clear back to the workspace default.
 */
export function parseTierArg(arg: string | undefined): ParsedTierArg | undefined {
  if (arg === undefined || arg === "") return undefined;
  if ((COST_TIERS as readonly string[]).includes(arg)) {
    return { kind: "tier", tier: arg as CostTier };
  }
  if (["off", "default", "reset", "clear"].includes(arg)) return { kind: "clear" };
  return { kind: "invalid", input: arg };
}

/**
 * Option list for the /openrouter-tier picker: "default" plus the five named
 * tiers, each annotated with its cost-percentile band, the current selection
 * marked. Pure so the labels stay testable.
 */
export function tierPickerOptions(current: CostTier | null): Array<{ label: string; tier: CostTier | null }> {
  const entries: Array<{ label: string; tier: CostTier | null }> = [
    { label: "default — workspace setting", tier: null },
    ...COST_TIERS.map((tier) => ({
      label: `${tier} — cost band ${COST_TIER_BANDS[tier]}`,
      tier: tier as CostTier,
    })),
  ];
  return entries.map((entry) =>
    entry.tier === current ? { ...entry, label: `${entry.label} (current)` } : entry,
  );
}

/**
 * Resolve the active session id from a pi context. Both the command and
 * session_start call this, so the hook's key and the command's key can
 * never drift apart (the hook's own event carries no context).
 */
export function resolveSessionId(ctx: unknown): string | undefined {
  const sm = (ctx as { sessionManager?: { getSessionId?: () => unknown } } | null | undefined)
    ?.sessionManager;
  const id = sm?.getSessionId?.();
  return typeof id === "string" && id ? id : undefined;
}

/**
 * Attach the session's tier to the hook's merged result. Returns the result
 * unchanged when no tier is selected (preserving the hook's undefined =
 * "no change" contract); when a tier is set and the hook produced no
 * result, the tier is applied to the untouched payload instead.
 */
export function withSessionTier(
  result: unknown,
  payload: unknown,
  model: string,
  tier: CostTier | null,
): unknown {
  if (tier === null) return result;
  return applyCostTier(result ?? payload, model, tier);
}

/**
 * Carry the selected cost_tier onto the wire as a plugins entry. Returns the
 * payload unchanged (same reference) when there is nothing to do: no tier
 * selected, a non-router model, or a non-object payload. A tier selection
 * replaces any existing same-id plugin entry in place and leaves other
 * plugins (e.g. from requests.defaults) untouched.
 */
export function applyCostTier(
  payload: unknown,
  model: string,
  tier: CostTier | null,
): unknown {
  if (tier === null) return payload;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
  const pluginId = routerPluginId(model);
  if (!pluginId) return payload;

  const record = payload as Record<string, unknown>;
  const entry = { id: pluginId, cost_tier: tier };
  const existing = Array.isArray(record.plugins) ? [...record.plugins] : [];
  const index = existing.findIndex(
    (p) => p && typeof p === "object" && !Array.isArray(p) && (p as any).id === pluginId,
  );
  if (index >= 0) existing[index] = entry;
  else existing.push(entry);
  return { ...record, plugins: existing };
}
