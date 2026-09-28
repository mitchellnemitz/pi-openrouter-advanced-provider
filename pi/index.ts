import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  PROVIDER_NAME,
  type ProviderRegistration,
  type RequestConfig,
  type SelectionResult,
} from "./types.js";
import {
  invalidateAllCaches,
  invalidatePermaslugCache,
  permaslugCacheAge,
  fetchEndpointStats,
  fetchToolCallErrorRates,
  fetchPermaslugMap,
  fetchKeyInfo,
  fetchCredits,
} from "./api.js";
import { scoreProviders } from "./scoring.js";
import { restoreRouterCost, wrapRouterCostFetch } from "./auto-cost.js";
import { loadRequestConfig, mergeRequestConfig, isVariantModelId } from "./config.js";
import {
  COST_TIERS,
  COST_TIER_BANDS,
  parseTierArg,
  resolveSessionId,
  tierPickerOptions,
  withSessionTier,
  type CostTier,
} from "./tier.js";
import {
  getSnapshot,
  nextGeneration,
  isStale,
  buildPlainSync,
  commitSnapshot,
  getSelection,
  setSelection,
  getStaleSelection,
  clearSelections,
} from "./state.js";

const APP_TITLE = "openrouter-advanced-provider";
const OPENROUTER_INFO_MESSAGE_TYPE = "openrouter-info";

function emitMessage(pi: ExtensionAPI, text: string) {
  pi.sendMessage({
    customType: OPENROUTER_INFO_MESSAGE_TYPE,
    content: text,
    display: true,
  });
}

export default async function openrouterModelsExtension(pi: ExtensionAPI) {
  // Pi prices router slugs from their zero-cost catalog entry rather than the
  // model OpenRouter served. Capture its billed SSE cost once per response and
  // replace the finalized message so session totals see the real amount.
  const key = Symbol.for("pi-openrouter-advanced-provider:router-cost");
  const globals = globalThis as any;
  const costState: {
    wrapped: boolean;
    captures: Map<string, Promise<number | undefined>>;
  } = globals[key] ??= { wrapped: false, captures: new Map() };
  if (!costState.wrapped) {
    // Reload intentionally keeps this first wrapper: it closes over the
    // original module's parsing code, so wrapper changes need a process
    // restart. Re-wrapping on every load would stack wrappers instead.
    globalThis.fetch = wrapRouterCostFetch(globalThis.fetch, costState.captures);
    costState.wrapped = true;
  }
  pi.on("message_end", async (event) => {
    if (event.message.role !== "assistant") return;
    const message = await restoreRouterCost(event.message, costState.captures);
    if (message) return { message };
  });

  // ---------- Per-model config overrides ----------

  // Optional escape hatch: openrouter-advanced-provider.json (shipped empty) +
  // user overrides at ~/.pi/agent/openrouter-advanced-provider.json. An entry
  // for a model bypasses the selection algorithm and pins exactly what the
  // file says. Re-loaded on every sync.
  let requestConfig: RequestConfig = {};
  let configWarnings: string[] = [];

  function refreshRequestConfig(): void {
    const loaded = loadRequestConfig();
    requestConfig = loaded.config;
    configWarnings = loaded.warnings;
  }

  function notifyConfigWarnings(ctx: any) {
    for (const warning of configWarnings.slice(0, 5)) {
      ctx.ui.notify(`OpenRouter config: ${warning}`, "warning");
    }
  }

  // Keep extension info messages (/openrouter-status, /openrouter-balance) out
  // of the LLM context, or account/status text pollutes prompts.
  pi.on("context", async (event) => {
    return {
      messages: event.messages.filter(
        (message: any) =>
          !(message.role === "custom" && message.customType === OPENROUTER_INFO_MESSAGE_TYPE),
      ),
    };
  });

  // ---------- Just-in-time provider selection ----------

  const selectionInFlight = new Map<string, Promise<SelectionResult | undefined>>();

  async function selectProvider(modelId: string, apiKey: string | undefined, force = false): Promise<SelectionResult | undefined> {
    if (!modelId || modelId.startsWith("~") || modelId.startsWith("openrouter/")) return undefined;
    if (isVariantModelId(modelId)) return undefined;
    if (requestConfig.models?.has(modelId)) return undefined; // config override bypasses the algorithm

    const cached = force ? undefined : getSelection(modelId);
    if (cached) return cached.result;

    const inFlight = selectionInFlight.get(modelId);
    if (inFlight) return inFlight;

    const promise = (async () => {
      let permaslug = (await fetchPermaslugMap(force)).get(modelId);
      // The map caches for 24h but the account catalog refreshes per session:
      // a model added after the map was loaded would never select. On a miss,
      // refetch the map once if it is older than 10 minutes.
      if (!permaslug && permaslugCacheAge() > 10 * 60 * 1000) {
        invalidatePermaslugCache();
        permaslug = (await fetchPermaslugMap()).get(modelId);
      }
      if (!permaslug) return undefined;
      const [endpoints, toolRates] = await Promise.all([
        fetchEndpointStats(permaslug),
        fetchToolCallErrorRates(permaslug).catch(() => new Map<string, number>()),
      ]);
      return scoreProviders(modelId, endpoints, toolRates);
    })()
      .then((result) => {
        if (result) setSelection(modelId, result);
        return result;
      })
      .catch(() => undefined)
      .finally(() => selectionInFlight.delete(modelId));

    selectionInFlight.set(modelId, promise);
    return promise;
  }

  // Original registered values per model, captured before the first selection
  // mutates them — a later refresh without a limit must restore the base, not
  // keep the previous selection's stale cap.
  const baseRegistration = new Map<string, { contextWindow: number; maxTokens: number }>();

  /** Pre-warm: score the model now and adjust its registered context window. */
  async function prewarm(modelId: string, apiKey: string | undefined, ctx: any, force = false): Promise<void> {
    const result = await selectProvider(modelId, apiKey, force);
    if (!result) return;

    // Adjust the model's registered context window to the picked provider's,
    // so budgeting/autocompact reflect the provider that will actually serve.
    const registered = ctx.modelRegistry?.getRegisteredProviderConfig?.(PROVIDER_NAME);
    const models = registered?.models as any[] | undefined;
    const entry = models?.find((m) => m.id === modelId);
    if (!entry) return;
    if (!baseRegistration.has(modelId)) {
      baseRegistration.set(modelId, { contextWindow: entry.contextWindow, maxTokens: entry.maxTokens });
    }
    const base = baseRegistration.get(modelId)!;
    const nextContext = result.contextLength > 0 ? result.contextLength : base.contextWindow;
    const nextMaxTokens = result.maxCompletionTokens ?? base.maxTokens;
    if (entry.contextWindow !== nextContext || entry.maxTokens !== nextMaxTokens) {
      entry.contextWindow = nextContext;
      entry.maxTokens = nextMaxTokens;
      // Re-registration merges defined values over the previous registration
      // and preserves undefined ones (api/auth/refreshModels all survive).
      pi.registerProvider(PROVIDER_NAME, { models } as any);
    }

    if (result.advisory) {
      try {
        ctx.ui.notify(`OpenRouter advisory: ${result.advisory}`, "info");
      } catch {
        // UI not available in all modes
      }
    }
  }

  // ---------- Session cost tier (auto-router) ----------

  // /openrouter-tier selects a cost_tier for this session's auto-router
  // requests; the absence of an entry means "workspace default" — no plugins
  // entry is added and OpenRouter's saved routing settings apply. Reset on
  // session_start (mirrors /thinking's session scope).
  //
  // The request hook's event carries no context, so the session id is
  // tracked here: session_start and the command both record it via
  // resolveSessionId, and the hook reads the same variable — the two keys
  // cannot drift apart. One process serves one active pi session at a
  // time; a newly started session re-points the id (its tier starts
  // empty).
  const tierBySession = new Map<string, CostTier>();
  let activeSessionId: string | undefined;

  // ---------- Per-request routing ----------

  // Routing sources, layered in order:
  //   1. requests.defaults — applied to every request (e.g. a workspace-wide
  //      provider.data_collection policy)
  //   2. config override entry (user file / shipped defaults) — pins exactly
  //      what the entry says, bypassing the algorithm
  //   3. the just-in-time selection for the payload's model — order is our
  //      single pick, allow_fallbacks lets OpenRouter balance the rest, and
  //      stunted-context providers are ignored so fallbacks cannot land there
  //   4. nothing — the request goes out with OpenRouter's default routing
  pi.on("before_provider_request", (event) => {
    const payload = event.payload;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return;
    const model = (payload as Record<string, unknown>).model;
    if (typeof model !== "string") return;

    // The event fires for every provider's requests and carries no provider
    // field. OpenRouter settings must only touch OpenRouter requests: gate on
    // the model being one this extension knows — in the registered catalog,
    // carrying a config entry, or holding a cached selection — or being
    // OpenRouter-shaped (author/slug with optional :variant), so
    // workspace-wide routing defaults still reach models the catalog sync
    // has not seen yet. Other providers' wire model ids are neither.
    const openrouterShaped = /^[a-z0-9][a-z0-9._-]*\/[a-z0-9][a-z0-9._:-]*$/i.test(model);
    const inCatalog = getSnapshot().models.some((m) => m.id === model);
    if (
      !inCatalog &&
      !openrouterShaped &&
      !requestConfig.models?.has(model) &&
      !getStaleSelection(model)
    )
      return;

    // Defaults (requests.defaults) apply to every request; the per-model
    // entry and the JIT pick then layer on top — their provider objects
    // merge over the defaults', so e.g. defaults' data_collection survives
    // alongside a selection's order. The session cost tier is applied last,
    // so an explicit /openrouter-tier selection replaces any auto-router
    // plugin entry the static config layers put on the payload.
    let base: Record<string, unknown> = payload as Record<string, unknown>;
    if (requestConfig.defaults) {
      base = mergeRequestConfig(payload, requestConfig.defaults) as Record<string, unknown>;
    }

    let result: Record<string, unknown> | undefined;
    const override = requestConfig.models?.get(model);
    if (override) {
      result = mergeRequestConfig(base, override) as Record<string, unknown>;
    } else {
      const selection = getSelection(model) ?? getStaleSelection(model);
      const pick = selection?.result.pick;
      if (pick) {
        const ignore = selection.result.ignore;
        const provider: Record<string, unknown> = {
          order: [pick],
          allow_fallbacks: true,
        };
        // No provider.quantizations pin: it is a request-wide filter, and when
        // no endpoint matches it the request hard-fails ("No endpoints found
        // for the request with quantization: ...") even with allow_fallbacks —
        // a stale score would break every request. A fallback serving a
        // different quantization than the scored row is a soft mismatch; the
        // registered context window may overestimate until the next selection
        // refresh. Merge the computed stunted-provider list with any ignore
        // list the defaults or a prior layer already set, instead of replacing
        // it.
        const baseIgnore = Array.isArray((base.provider as Record<string, unknown> | undefined)?.ignore)
          ? ((base.provider as Record<string, unknown>).ignore as unknown[])
          : [];
        const mergedIgnore = [...new Set([...baseIgnore, ...ignore])];
        if (mergedIgnore.length > 0) provider.ignore = mergedIgnore;
        result = { ...base, provider: { ...(base.provider as Record<string, unknown> ?? {}), ...provider } };
      } else {
        // No selection: leave the payload as the config layers left it.
        result = base === payload ? undefined : base;
      }
    }

    const tier = activeSessionId !== undefined ? (tierBySession.get(activeSessionId) ?? null) : null;
    return withSessionTier(result, payload, model, tier);
  });

  // ---------- Provider registration (standalone catalog) ----------

  /**
   * Models-only registration. Pi merges re-registrations key by key — defined
   * values win, undefined values preserve what is beneath — so leaving
   * api/apiKey/baseUrl undefined keeps pi's builtin OpenRouter serving
   * defaults (transport, auth handling) active underneath. This extension
   * owns the catalog; serving stays with the builtin transport.
   */
  function registerStandalone(models: ProviderRegistration["models"]) {
    pi.registerProvider(PROVIDER_NAME, {
      models: models!,
      headers: {
        // No HTTP-Referer: this extension sends no public site URL.
        // X-Title identifies the app in OpenRouter's stats.
        "X-Title": APP_TITLE,
      },
    } as any);
  }

  /**
   * Load-time key resolution: env first, then pi's auth store (~/.pi/agent/
   * auth.json, where /login stores keys — the registry is not reachable this
   * early). Without a key the account catalog is unreachable, and falling
   * back to the public catalog would silently swap account-scoped data (BYOK
   * models, fallback aliases) for the generic list — so no key means no
   * bootstrap registration.
   */
  function bootstrapApiKey(): string | undefined {
    if (process.env.OPENROUTER_API_KEY) return process.env.OPENROUTER_API_KEY;
    try {
      const authPath = path.join(os.homedir(), ".pi", "agent", "auth.json");
      if (!fs.existsSync(authPath)) return undefined;
      const auth = JSON.parse(fs.readFileSync(authPath, "utf8"));
      const entry = auth?.openrouter;
      if (typeof entry === "string" && entry.startsWith("sk-or-")) return entry;
      if (typeof entry?.key === "string" && entry.key.startsWith("sk-or-")) return entry.key;
    } catch {
      // unreadable auth store — fall through to undefined
    }
    return undefined;
  }

  async function bootstrapPlainSync() {
    const generation = nextGeneration();
    const apiKey = bootstrapApiKey();
    if (!apiKey) return; // session_start syncs with the registry key

    try {
      refreshRequestConfig();
      // Register the live account catalog during extension load so Pi can
      // resolve saved scoped-model patterns before session_start fires.
      const result = await buildPlainSync(apiKey, true, requestConfig);

      if (isStale(generation)) return;
      commitSnapshot(generation, result.models);
      registerStandalone(result.models);
      // The registered snapshot changed: selections (and their recorded base
      // limits) from the previous catalog are stale.
      clearSelections();
      baseRegistration.clear();
    } catch {
      // Keep startup resilient. If OpenRouter is temporarily unavailable, Pi's
      // built-in OpenRouter list remains registered and manual /openrouter-sync
      // can recover later.
    }
  }

  await bootstrapPlainSync();

  async function syncPlain(ctx: any, silent = false, force = false) {
    const generation = nextGeneration();

    try {
      const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER_NAME);
      if (!silent) ctx.ui.notify("Fetching OpenRouter models...", "info");

      refreshRequestConfig();
      clearSelections();
      const result = await buildPlainSync(apiKey, force, requestConfig);

      if (isStale(generation)) return;
      commitSnapshot(generation, result.models);
      registerStandalone(result.models);
      // The registered snapshot changed: stale selections (and their recorded
      // base limits) from the previous catalog must not survive the sync.
      clearSelections();
      baseRegistration.clear();

      if (!silent) {
        ctx.ui.notify(`OpenRouter: ${result.modelCount} models synced`, "info");
      }
      notifyConfigWarnings(ctx);
    } catch (err: any) {
      if (!silent) ctx.ui.notify(`OpenRouter sync failed: ${err?.message}`, "error");
    }
  }

  // ---------- Pre-warm hooks ----------

  pi.on("session_start", async (_event, ctx) => {
    activeSessionId = resolveSessionId(ctx) ?? activeSessionId;
    try {
      await syncPlain(ctx, true, true);
      updateStatusBar(ctx);
    } catch {
      // No auth configured — skip silently
    }
  });

  // Pre-warm on prompt submit: before_agent_start is the earliest point where
  // the selected model is known and a prompt is committed — the pre-warm fetch
  // lands here, before the first provider request.
  pi.on("before_agent_start", async (_event, ctx) => {
    try {
      const model = ctx.model ?? (ctx as any).getModel?.();
      if (!model) return;
      const id = String((model as any).id ?? model);
      if (getSelection(id)) return;
      const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER_NAME);
      await prewarm(id, apiKey, ctx);
    } catch {
      // never block the turn on pre-warm
    }
  });

  // ---------- Commands ----------

  pi.registerCommand("openrouter-sync", {
    description: "Refresh the OpenRouter catalog and re-run provider selection",
    handler: async (_args, ctx) => {
      invalidateAllCaches();
      invalidatePermaslugCache();
      await syncPlain(ctx, false, true);
      updateStatusBar(ctx);
    },
  });

  pi.registerCommand("openrouter-balance", {
    description: "Show your OpenRouter credit balance and usage",
    handler: async (_args, ctx) => {
      try {
        const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER_NAME);
        if (!apiKey) {
          ctx.ui.notify("No OpenRouter API key configured", "warning");
          return;
        }

        const [info, credits] = await Promise.all([
          fetchKeyInfo(apiKey),
          fetchCredits(apiKey),
        ]);

        const lines: string[] = ["**OpenRouter Account**", ""];

        // OpenRouter exposes two different concepts here:
        // - /credits => account-level purchased credits and total account usage (management key only)
        // - /key     => current API key limits and usage counters
        if (credits && credits.total_credits !== undefined && credits.total_usage !== undefined) {
          const balance = credits.total_credits - credits.total_usage;
          lines.push(`💰 **Balance: $${balance.toFixed(4)}**`);
          lines.push(`   Account credits: $${credits.total_credits.toFixed(4)} — Account used: $${credits.total_usage.toFixed(4)}`);
        } else if (info.limit_remaining !== null && info.limit_remaining !== undefined) {
          lines.push(`💰 **Remaining key limit: $${info.limit_remaining.toFixed(4)}**`);
        }

        lines.push("");

        if (info.is_free_tier) lines.push("Tier: Free");

        if (info.limit !== null && info.limit !== undefined) {
          const limitStr = `$${info.limit.toFixed(2)}`;
          const resetStr = info.limit_reset ? ` (resets ${info.limit_reset})` : "";
          lines.push(`Key spend limit: ${limitStr}${resetStr}`);
        }

        if (
          info.usage !== undefined
          || info.usage_daily !== undefined
          || info.usage_monthly !== undefined
        ) {
          lines.push("");
          lines.push("**Current API key usage**");
          if (info.usage_daily !== undefined) {
            lines.push(`  Today: $${info.usage_daily.toFixed(4)}`);
          }
          if (info.usage_monthly !== undefined) {
            lines.push(`  This month: $${info.usage_monthly.toFixed(4)}`);
          }
          if (info.usage !== undefined) {
            lines.push(`  All-time for this key: $${info.usage.toFixed(4)}`);
          }
        }

        emitMessage(pi, lines.join("\n"));
      } catch (err: any) {
        ctx.ui.notify(`Balance check failed: ${err?.message}`, "error");
      }
    },
  });

  pi.registerCommand("openrouter-status", {
    description: "Show current extension state: registered models, provider selection for the active model",
    handler: async (_args, ctx) => {
      const snapshot = getSnapshot();
      const lines: string[] = ["**OpenRouter Extension Status**", ""];

      lines.push(`Models registered: ${snapshot.models.length}`);
      lines.push(`Config overrides: ${requestConfig.models?.size ?? 0}`);
      const tier = tierBySession.get(resolveSessionId(ctx) ?? "") ?? null;
      lines.push(`Cost tier: ${tier ? `${tier} (cost band ${COST_TIER_BANDS[tier]})` : "workspace default"}`);

      if (snapshot.timestamp > 0) {
        const ageMin = Math.round((Date.now() - snapshot.timestamp) / 60000);
        lines.push(`Last sync: ${ageMin} minute(s) ago`);
      } else {
        lines.push("Last sync: never");
      }

      // Selection detail for the session's current model
      try {
        const model = ctx.model ?? (ctx as any).getModel?.();
        const id = model ? String((model as any).id ?? model) : undefined;
        const selection = id ? getSelection(id) ?? getStaleSelection(id) : undefined;
        if (selection) {
          const r = selection.result;
          lines.push("");
          lines.push(`**Selection for ${id}** (${Math.round((Date.now() - selection.timestamp) / 60000)} min ago)`);
          lines.push(`Pinned provider: ${r.pick} — context ${r.contextLength?.toLocaleString()}`);
          if (r.ignore.length > 0) lines.push(`Ignored (stunted context): ${r.ignore.join(", ")}`);
          if (r.advisory) lines.push(`⚠ ${r.advisory}`);
          const top = r.table.slice(0, 5);
          if (top.length > 0) {
            lines.push("");
            lines.push("| Provider | Score | tok/s | p50 ms | tail × | $/M |");
            lines.push("|---|---|---|---|---|---|");
            for (const row of top) {
              lines.push(
                `| ${row.tag} | ${row.score !== undefined ? row.score.toFixed(3) : "—"} | ${row.tpsP50 ?? "—"} | ${row.latP50 !== undefined ? Math.round(row.latP50) : "—"} | ${row.tailRatio !== undefined ? row.tailRatio.toFixed(1) : "—"} | ${row.blendedPrice !== undefined ? `$${(row.blendedPrice * 1e6).toFixed(2)}` : "—"} |`,
              );
            }
          }
        } else if (id) {
          lines.push("");
          lines.push(`No provider selection yet for ${id} (pre-warms on first prompt).`);
        }
      } catch {
        // status command must never throw
      }

      emitMessage(pi, lines.join("\n"));
    },
  });

  pi.registerCommand("openrouter-tier", {
    description: "Set the OpenRouter auto-router cost tier for this session (default clears back to the workspace setting)",
    handler: async (args, ctx) => {
      const sessionId = resolveSessionId(ctx);
      if (!sessionId) {
        // Without a session id nothing would read the stored tier — say so
        // instead of storing under a key the request hook never looks up.
        ctx.ui.notify("OpenRouter: no active session; cost tier not set", "warning");
        return;
      }
      activeSessionId = sessionId;
      const current = tierBySession.get(sessionId) ?? null;
      const describe = (tier: CostTier | null) =>
        tier ? `${tier} (cost band ${COST_TIER_BANDS[tier]})` : "workspace default";

      const parsed = parseTierArg(args);
      if (parsed === undefined) {
        // Interactive picker. Not available in every UI mode; fall back to
        // printing the CLI form.
        try {
          const options = tierPickerOptions(current);
          const choice = await ctx.ui.select(
            `OpenRouter cost tier (current: ${describe(current)})`,
            options.map((o) => o.label),
          );
          if (choice === undefined) return; // cancelled
          const picked = options[options.map((o) => o.label).indexOf(choice)];
          if (picked) {
            if (picked.tier === null) tierBySession.delete(sessionId);
            else tierBySession.set(sessionId, picked.tier);
            ctx.ui.notify(`OpenRouter cost tier: ${describe(picked.tier)}`, "info");
          }
        } catch {
          ctx.ui.notify("Usage: /openrouter-tier [low|medium|high|xhigh|max|off]", "info");
        }
        return;
      }

      if (parsed.kind === "invalid") {
        ctx.ui.notify(
          `OpenRouter: unknown tier "${parsed.input}" — usage: /openrouter-tier [low|medium|high|xhigh|max|off]`,
          "warning",
        );
        return;
      }

      if (parsed.kind === "clear") tierBySession.delete(sessionId);
      else tierBySession.set(sessionId, parsed.tier);
      ctx.ui.notify(`OpenRouter cost tier: ${describe(parsed.kind === "clear" ? null : parsed.tier)}`, "info");
    },
  });

  // ---------- Status bar ----------

  function updateStatusBar(ctx: any) {
    const snapshot = getSnapshot();
    if (snapshot.models.length > 0) {
      try {
        ctx.ui.setStatus("openrouter", `OR: ${snapshot.models.length} models`);
      } catch {
        // setStatus may not be available in all UI modes
      }
    }
  }
}