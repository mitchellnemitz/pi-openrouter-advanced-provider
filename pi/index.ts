import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { ExtensionAPI, ProviderConfig } from "@earendil-works/pi-coding-agent";
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
import { clampCatalogModelLimits, clampedSelectionLimits } from "./limits.js";
import { buildStandaloneProviderConfig, streamOpenRouterWithRecovery } from "./stream.js";
import { restoreRouterCost, wrapRouterCostFetch } from "./auto-cost.js";
import { bootstrapApiKey, agentRoot, initHost, isOmp } from "./platform.ts";
import {
  DEFAULT_SELECTION,
  isVariantModelId,
  loadRequestConfig,
  mergeRequestConfig,
} from "./config.js";
import type { ProviderModelConfig, SelectionTuning } from "./types.js";
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
  beginDiscoveryAttempt,
  completeDiscoveryAttempt,
  publishDiscoveryRows,
  getNewerDiscoveryAttempt,
  getDiscoveryAttemptCount,
  getDiscoveryOutcomesSince,
  keyFingerprint,
  summarizeDiscovery,
  DiscoveryFailedError,
  type DiscoveryAttempt,
} from "./state.js";
// The last-good store is configured and consumed here (host wiring), so it
// comes straight from the ledger rather than through state's re-exports.
import {
  configureLastGoodPersistence,
  lastGoodRowsForFailedAttempt,
} from "./discovery.js";

const APP_TITLE = "openrouter-advanced-provider";
const OPENROUTER_INFO_MESSAGE_TYPE = "openrouter-info";

function emitMessage(pi: ExtensionAPI, text: string) {
  pi.sendMessage({
    customType: OPENROUTER_INFO_MESSAGE_TYPE,
    content: text,
    display: true,
  });
}

// ---------------------------------------------------------------------------
// Output-budget field selection
//
// An OpenAI-compatible request body may carry its output budget in either
// `max_tokens` (legacy) or `max_completion_tokens` (current). pi's OpenAI
// Completions transport writes the budget into exactly ONE of them: the
// field its compat selects — `max_completion_tokens` unless the model pins
// `maxTokensField: "max_tokens"` (only the providers that reject the
// current field do that; OpenRouter's own models use the current field).
//
// Writing the budget into the field the transport does NOT read creates a
// COMPETING cap: the body then carries two budgets, and the final guard —
// which measures the largest budget in the body — reserves the larger one
// against the endpoint's context window. A request whose host budget was
// smaller than the endpoint's ceiling was therefore measured as if it had
// asked for the ceiling, and rejected even though it fits.
// ---------------------------------------------------------------------------

/** Every field an OpenAI-compatible body may use for its output budget. */
export const OUTPUT_BUDGET_FIELDS = ["max_tokens", "max_completion_tokens"] as const;

export type OutputBudgetField = (typeof OUTPUT_BUDGET_FIELDS)[number];

/**
 * The field the transport writes by default: `max_completion_tokens`.
 */
export const DEFAULT_OUTPUT_BUDGET_FIELD: OutputBudgetField = "max_completion_tokens";

/**
 * The numeric budget a field carries, or undefined when the field is
 * absent or unreadable. A numeric string is a budget (it reaches the wire
 * as one), so it is read as a number; anything else is not a budget that can
 * be compared against a ceiling.
 */
function numericBudget(value: unknown): number | undefined {
  if (typeof value === "number") return Number.isFinite(value) ? value : undefined;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

/** Does the body carry a readable budget in this field? */
function hasBudget(payload: Record<string, unknown>, field: OutputBudgetField): boolean {
  const value = payload[field];
  if (value === undefined || value === null) return false;
  return numericBudget(value) !== undefined;
}

/**
 * The output-budget field the transport writes for a model.
 *
 * The model's own compat is authoritative — it is what the transport reads.
 * Without a compat pin, the payload itself is the evidence: the transport
 * already built it and wrote its budget into exactly one field.
 */
export function resolveOutputBudgetField(
  compat: unknown,
  payload: Record<string, unknown>,
): OutputBudgetField {
  if (compat && typeof compat === "object" && "maxTokensField" in compat) {
    const pinned = compat.maxTokensField;
    if (pinned === "max_tokens" || pinned === "max_completion_tokens") return pinned;
  }
  const legacy = hasBudget(payload, "max_tokens");
  const current = hasBudget(payload, "max_completion_tokens");
  if (legacy && !current) return "max_tokens";
  if (current && !legacy) return "max_completion_tokens";
  return DEFAULT_OUTPUT_BUDGET_FIELD;
}

/**
 * Clamp the body's output budget to `ceiling`, in the field the transport
 * actually writes.
 *
 * - A budget ABOVE the ceiling is lowered to it (the endpoint cannot serve
 *   more than its own completion limit).
 * - A budget at or below the ceiling is PRESERVED: a caller that asked for
 *   less than the endpoint allows keeps its smaller request, and the guard
 *   measures that smaller allowance.
 * - An UNREADABLE budget in the transport's own field is replaced by the
 *   ceiling: such a budget cannot be honored, and the ceiling is the known
 *   safe bound. A budget in another field is left alone — the final guard
 *   fails closed on it with a precise error.
 * - The transport's own field is set to the ceiling when the body carries
 *   no budget at all, so an unbounded request stays bounded.
 * - No other field is ever added: a body that carries the legacy field keeps
 *   exactly that field, so the transport never sees two competing budgets.
 *
 * Returns the new body, or undefined when nothing had to change.
 */
export function clampOutputBudget(
  payload: Record<string, unknown>,
  ceiling: number,
  transportField: OutputBudgetField = DEFAULT_OUTPUT_BUDGET_FIELD,
): Record<string, unknown> | undefined {
  if (!Number.isFinite(ceiling) || ceiling <= 0) return undefined;

  let changed = false;
  const next: Record<string, unknown> = { ...payload };

  for (const field of OUTPUT_BUDGET_FIELDS) {
    const numeric = numericBudget(next[field]);
    if (numeric === undefined) continue;
    if (numeric > ceiling) {
      next[field] = ceiling;
      changed = true;
    }
  }

  const transportValue = next[transportField];
  if (transportValue === undefined || transportValue === null) {
    // The transport reads THIS field; without it the request carries no
    // budget at all. Bound it by the endpoint's ceiling.
    next[transportField] = ceiling;
    changed = true;
  } else if (numericBudget(transportValue) === undefined) {
    // Unreadable budget in the transport's own field: replace it with the
    // known-safe bound rather than let it reach the wire.
    next[transportField] = ceiling;
    changed = true;
  }

  return changed ? next : undefined;
}

// ---------------------------------------------------------------------------
// Per-request routing transform
//
// The `before_provider_request` hook fires for EVERY provider's requests, so
// the transform first decides whether the request is ours, then layers the
// routing configuration, the just-in-time provider pick, the output-budget
// clamp and the session cost tier onto the payload.
//
// The transform lives here as a standalone function (with the extension's
// live state injected) so the routing hook AND the final guard can be
// exercised together in a test.
// ---------------------------------------------------------------------------

/** A cached selection, as pi/state.ts stores it. */
export interface CachedSelection {
  result: SelectionResult;
  timestamp: number;
}

/** Everything the routing transform needs from the extension's live state. */
export interface RequestRoutingDeps {
  /** Per-request config layers (workspace defaults + per-model entries). */
  requestConfig: RequestConfig;
  /** Selection tuning (disabled selection short-circuits the JIT pick). */
  selectionTuning: SelectionTuning;
  /** The registered catalog rows (the extension's snapshot). */
  catalogModels: () => ProviderModelConfig[];
  /**
   * The catalog `maxTokens` registered for a model BEFORE the selected
   * endpoint narrowed it — the ceiling a request may never exceed.
   */
  baseMaxTokens: (modelId: string) => number | undefined;
  /** The fresh (non-stale) selection for a model, if any. */
  getSelection: (modelId: string) => CachedSelection | undefined;
  /** The last-known selection for a model, even past its TTL. */
  getStaleSelection: (modelId: string) => CachedSelection | undefined;
  /** The just-in-time provider pick (network-backed, may be undefined). */
  selectProvider: (
    modelId: string,
    apiKey: string | undefined,
    force: boolean,
  ) => Promise<SelectionResult | undefined>;
  /** The session's cost tier, or null for the workspace default. */
  tier: () => CostTier | null;
}

/**
 * The `before_provider_request` payload transform.
 *
 * Returns the replacement payload, or undefined when the request is not ours
 * (the host then keeps the original payload).
 */
export async function routeRequestPayload(
  payload: unknown,
  deps: RequestRoutingDeps,
): Promise<unknown> {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return undefined;
  const model = (payload as Record<string, unknown>).model;
  if (typeof model !== "string") return undefined;

  // The event fires for every provider's requests and carries no provider
  // field. OpenRouter settings must only touch OpenRouter requests: gate on
  // the model being one this extension knows — in the registered catalog,
  // carrying a config entry, or holding a cached selection — or being
  // OpenRouter-shaped (author/slug with optional :variant), so
  // workspace-wide routing defaults still reach models the catalog sync
  // has not seen yet. Other providers' wire model ids are neither.
  const openrouterShaped = /^[a-z0-9][a-z0-9._-]*\/[a-z0-9][a-z0-9._:-]*$/i.test(model);
  const catalog = deps.catalogModels();
  const inCatalog = catalog.some((entry) => entry.id === model);
  if (
    !inCatalog &&
    !openrouterShaped &&
    !deps.requestConfig.models?.has(model) &&
    !deps.getStaleSelection(model)
  )
    return undefined;

  // Defaults (requests.defaults) apply to every request; the per-model
  // entry and the JIT pick then layer on top — their provider objects
  // merge over the defaults', so e.g. defaults' data_collection survives
  // alongside a selection's order. The session cost tier is applied last,
  // so an explicit /openrouter-tier selection replaces any auto-router
  // plugin entry the static config layers put on the payload.
  let base: Record<string, unknown> = payload as Record<string, unknown>;
  if (deps.requestConfig.defaults) {
    base = mergeRequestConfig(payload, deps.requestConfig.defaults) as Record<string, unknown>;
  }

  let result: Record<string, unknown> | undefined;
  const override = deps.requestConfig.models?.get(model);
  // With selection disabled the JIT pick is skipped entirely — the config
  // layers above still apply, and OpenRouter's default routing serves.
  const selected =
    deps.selectionTuning.enabled && !deps.getSelection(model)
      ? await deps.selectProvider(model, undefined, false)
      : undefined;
  const selection = selected
    ? { result: selected }
    : deps.selectionTuning.enabled
      ? (deps.getSelection(model) ?? deps.getStaleSelection(model))
      : undefined;
  if (override) {
    result = mergeRequestConfig(base, override) as Record<string, unknown>;
  } else {
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

  const endpointMaxTokens = selection?.result.maxCompletionTokens;
  if (!override && endpointMaxTokens && endpointMaxTokens > 0) {
    const outgoing = result ?? base;
    const catalogMaxTokens =
      deps.baseMaxTokens(model) ?? catalog.find((entry) => entry.id === model)?.maxTokens;
    const outputCeiling =
      catalogMaxTokens && catalogMaxTokens > 0
        ? Math.min(endpointMaxTokens, catalogMaxTokens)
        : endpointMaxTokens;
    // Clamp the budget in the field the transport actually writes. Writing
    // the OTHER field would add a second, competing budget the transport
    // ignores — and the final guard, which measures the largest budget in
    // the body, would then reserve that larger allowance against the
    // endpoint's context window and reject a request that fits.
    const transportField = resolveOutputBudgetField(
      catalog.find((entry) => entry.id === model)?.compat,
      outgoing,
    );
    const clamped = clampOutputBudget(outgoing, outputCeiling, transportField);
    if (clamped) result = clamped;
  }

  return withSessionTier(result, payload, model, deps.tier());
}

export default async function openrouterModelsExtension(pi: ExtensionAPI) {
  initHost(pi);

  // Pi prices router slugs from their zero-cost catalog entry rather than the
  // model OpenRouter served. Capture its billed SSE cost once per response and
  // replace the finalized message so session totals see the real amount.
  //
  // Under omp this is skipped entirely: omp natively stamps assistant
  // usage.cost.total from the wire for the "openrouter" provider, so no
  // fetch wrapping or message_end restore is needed.
  if (!isOmp()) {
    const key = Symbol.for("pi-openrouter-advanced-provider:router-cost");
    type RouterCostState = {
      wrapped: boolean;
      captures: Map<string, Promise<number | undefined>>;
    };
    const globals = globalThis as unknown as Record<symbol, RouterCostState | undefined>;
    const costState: RouterCostState = (globals[key] ??= {
      wrapped: false,
      captures: new Map(),
    });
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
  }

  // ---------- Per-model config overrides ----------

  // Optional escape hatch: openrouter-advanced-provider.json (shipped empty) +
  // user overrides at ~/.pi/agent/openrouter-advanced-provider.json. An entry
  // for a model bypasses the selection algorithm and pins exactly what the
  // file says. Re-loaded on every sync.
  let requestConfig: RequestConfig = {};
  let configWarnings: string[] = [];
  let selectionTuning: SelectionTuning = DEFAULT_SELECTION;

  function refreshRequestConfig(): void {
    const loaded = loadRequestConfig();
    requestConfig = loaded.config;
    configWarnings = loaded.warnings;
    selectionTuning = loaded.selection;
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
    if (!selectionTuning.enabled) return undefined;
    if (!modelId || modelId.startsWith("~") || modelId.startsWith("openrouter/")) return undefined;
    if (isVariantModelId(modelId)) return undefined;
    if (requestConfig.models?.has(modelId)) return undefined;

    const cached = force ? undefined : getSelection(modelId);
    if (cached) return cached.result;

    const inFlight = selectionInFlight.get(modelId);
    if (inFlight) return inFlight;

    const promise = (async () => {
      let permaslug = (await fetchPermaslugMap(force)).get(modelId);
      if (!permaslug && permaslugCacheAge() > 10 * 60 * 1000) {
        invalidatePermaslugCache();
        permaslug = (await fetchPermaslugMap()).get(modelId);
      }

      if (!permaslug) return undefined;
      const [endpoints, toolRates] = await Promise.all([
        fetchEndpointStats(permaslug),
        fetchToolCallErrorRates(permaslug).catch(() => new Map<string, number>()),
      ]);

      return scoreProviders(modelId, endpoints, toolRates, selectionTuning);
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
  // Omp's dynamic registry rows are refreshed after selection so the active
  // model budget follows the chosen endpoint without registering a static key.
  const ompSelectionLimits = new Map<string, SelectionResult>();
  const ompAppliedSelectionLimits = new Map<string, { contextWindow: number; maxTokens: number }>();
  let ompLimitCredentialFingerprint: string | undefined;

  function applyOmpSelectionLimits(models: ProviderRegistration["models"], apiKey: string | undefined) {
    const fingerprint = keyFingerprint(apiKey);
    if (fingerprint !== ompLimitCredentialFingerprint) {
      ompSelectionLimits.clear();
      ompAppliedSelectionLimits.clear();
      ompLimitCredentialFingerprint = fingerprint;
    }
    if (!isOmp() || !models) return models;
    for (const model of models) {
      if (!selectionTuning.enabled || requestConfig.models?.has(model.id)) {
        ompSelectionLimits.delete(model.id);
      }
    }
    ompAppliedSelectionLimits.clear();
    if (ompSelectionLimits.size === 0) return models;
    const clamped = clampCatalogModelLimits(models, ompSelectionLimits);
    for (const model of clamped) {
      if (ompSelectionLimits.has(model.id)) {
        ompAppliedSelectionLimits.set(model.id, {
          contextWindow: model.contextWindow,
          maxTokens: model.maxTokens,
        });
      }
    }
    return clamped;
  }

  type ModelLimitRefreshContext = {
    modelRegistry: { refreshProvider(provider: string): Promise<void> };
  };

  async function refreshOmpModelLimits(
    ctx: ModelLimitRefreshContext,
    modelId: string,
    apiKey: string | undefined,
    result: SelectionResult,
  ) {
    const fingerprint = keyFingerprint(apiKey);
    if (fingerprint !== ompLimitCredentialFingerprint) {
      ompSelectionLimits.clear();
      ompAppliedSelectionLimits.clear();
      ompLimitCredentialFingerprint = fingerprint;
    }
    ompSelectionLimits.set(modelId, result);
    await ctx.modelRegistry.refreshProvider(PROVIDER_NAME);
    // refreshProvider re-runs dynamic discovery, which clears the selection
    // cache. Keep this prompt's scored choice stable for the request hook.
    setSelection(modelId, result);
  }

  /** Pre-warm: score the model now and adjust its registered context window. */
  async function prewarm(modelId: string, apiKey: string | undefined, ctx: any, force = false): Promise<void> {
    const result = await selectProvider(modelId, apiKey, force);
    if (!result) return;

    if (isOmp()) {
      const models = getSnapshot().models;
      const entry = models.find((m) => m.id === modelId);
      if (!entry) return;
      const next = clampedSelectionLimits(entry, result);
      const previous = ompAppliedSelectionLimits.get(modelId);
      const registered = previous ?? entry;
      if (registered.contextWindow !== next.contextWindow || registered.maxTokens !== next.maxTokens) {
        await refreshOmpModelLimits(ctx, modelId, apiKey, result);
      }
    } else {
      // Adjust the model's registered context window to the picked provider's,
      // so budgeting/autocompact reflect the provider that will actually serve.
      const models = getSnapshot().models.slice();
      if (models.length === 0) return;
      const entry = models.find((m) => m.id === modelId);
      if (!entry) return;
      if (!baseRegistration.has(modelId)) {
        baseRegistration.set(modelId, { contextWindow: entry.contextWindow, maxTokens: entry.maxTokens });
      }
      const base = baseRegistration.get(modelId)!;
      const next = clampedSelectionLimits(base, result);
      if (entry.contextWindow !== next.contextWindow || entry.maxTokens !== next.maxTokens) {
        entry.contextWindow = next.contextWindow;
        entry.maxTokens = next.maxTokens;
        registerStandalone(models, apiKey);
      }
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
  pi.on("before_provider_request", async (event) => {
    return routeRequestPayload(event.payload, {
      requestConfig,
      selectionTuning,
      catalogModels: () => getSnapshot().models,
      baseMaxTokens: (modelId) => baseRegistration.get(modelId)?.maxTokens,
      getSelection,
      getStaleSelection,
      selectProvider,
      tier: () => (activeSessionId !== undefined ? (tierBySession.get(activeSessionId) ?? null) : null),
    });
  });

  // ---------- Provider registration (standalone catalog) ----------

  /**
   * Pi-only static models registration plus the failure-recovery streamSimple
   * wrapper for thought signatures. Pi merges re-registrations key by key —
   * defined values win, undefined values preserve what is beneath — so
   * leaving apiKey/baseUrl undefined keeps the builtin defaults in standalone
   * mode. Attaching api and streamSimple routes completions requests through
   * the recovery wrapper while leaving non-completions APIs to base dispatch.
   * Under omp registration is dynamic (see bootstrapPlainSync); this function
   * must not be called there.
   */
  function registerStandalone(models: ProviderRegistration["models"], apiKey: string | undefined) {
    // Host-normalized registration shape: the builder's structural mirror is
    // a subset of each host's ProviderConfig; this is the one boundary cast.
    pi.registerProvider(
      PROVIDER_NAME,
      buildStandaloneProviderConfig(models, { apiKey }) as ProviderConfig,
    );
  }

  /**
   * Dynamic-models fetcher handed to omp via the provider registration. Does
   * what syncPlain's body does minus registerProvider: refreshes the request
   * config, clears stale selections, rebuilds the catalog, and commits the
   * snapshot. Omp calls this with the host-resolved key each time, so
   * rotation/logout flow naturally.
   *
   * The host consumes the rows this returns as DISCOVERY DATA (omp merges
   * them into the provider's model set), so the rows this publishes must
   * always be the current account's catalog. Every invocation is therefore
   * recorded as a discovery attempt:
   *
   *   - On failure the attempt records the error as FAILED. omp's discovery
   *     machinery does NOT keep the previous catalog usable: omp 18.16
   *     evicts both the served catalog and its persisted agent/models.db
   *     cache when a discovery fails, so propagating the error would wipe
   *     the account's usable catalog. When THIS credential previously
   *     committed a good catalog — this process's ledger, or the agent-dir
   *     mirror a previous process wrote — the fetcher therefore SERVES
   *     those last-known-good rows instead of throwing: the host keeps a
   *     usable prior catalog, while the attempt's recorded outcome stays
   *     "failed" so syncPlain still reports the failure from the ledger
   *     (never a success announcement). A first-ever failure has nothing to
   *     preserve and propagates the error; a retired account's rows are
   *     never served — the supersession veto below covers served rows too.
   *   - On success, when a NEWER attempt has already started, this attempt
   *     was superseded: a host refresh that overlaps a credential change
   *     would otherwise hand the host the retired account's rows. The newer
   *     attempt's rows are published instead (the same rows for every attempt
   *     of one refresh), so the host registry only ever sees the newest
   *     catalog. A newer attempt with a DIFFERENT credential that failed
   *     leaves nothing publishable, so this attempt publishes no rows
   *     rather than the retired account's.
   */
  async function syncFetcher(
    apiKey: string | undefined,
  ): Promise<NonNullable<ProviderRegistration["models"]>> {
    const attempt = beginDiscoveryAttempt(keyFingerprint(apiKey));
    if (!apiKey) {
      // Omp invokes the fetcher even unauthenticated. Returning the public
      // catalog would present generic models as this provider's account
      // rows; an empty set honestly reports "nothing available" and leaves
      // omp's builtin openrouter rows in place.
      ompSelectionLimits.clear();
      ompLimitCredentialFingerprint = undefined;
      completeDiscoveryAttempt(attempt, { status: "no-key", modelCount: 0, rows: [] });
      return [];
    }
    const generation = nextGeneration();
    refreshRequestConfig();
    clearSelections();
    baseRegistration.clear();
    try {
      const result = await buildPlainSync(apiKey, false, requestConfig);
      const models = applyOmpSelectionLimits(result.models, apiKey) ?? [];
      const published = await publishDiscoveryRows(attempt, models);
      if (published.status === "superseded") {
        // A newer attempt owns the catalog now: publish ITS rows, not ours.
        completeDiscoveryAttempt(attempt, { status: "superseded", modelCount: published.rows.length, rows: published.rows });
        return published.rows;
      }
      commitSnapshot(generation, result.models);
      completeDiscoveryAttempt(attempt, {
        status: "ok",
        modelCount: models.length,
        rows: models,
      });
      return models;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      completeDiscoveryAttempt(attempt, { status: "failed", error: message });
      // omp evicts the served catalog (and its persisted cache) when
      // discovery fails. Serving this credential's last-known-good rows
      // keeps the host catalog usable across the failure; the FAILED
      // outcome recorded above is what syncPlain reports, so the failure is
      // still surfaced — the served rows are never announced as a success.
      // Nothing to preserve (first-ever failure), or a newer
      // different-credential attempt retired this account ([]-publishing is
      // the veto): propagate.
      const lastGood = lastGoodRowsForFailedAttempt(attempt);
      if (lastGood) return lastGood;
      throw err;
    }
  }

  async function bootstrapPlainSync() {
    if (isOmp()) {
      // Dynamic registration under omp: register ONCE at load with no
      // network fetch and no key check — the registration carries
      // fetchDynamicModels instead of a static list, and omp invokes it with
      // the HOST-RESOLVED key during its own startup refresh. No key is
      // pinned here, so /logout and key rotation stay authoritative.
      //
      // Mirror the last-known-good catalogs into the agent dir: omp runs are
      // one-shot processes, and without the mirror a fresh process has
      // nothing to serve when its first discovery fails, so it would evict
      // the persisted catalog the previous process left behind (see
      // syncFetcher). Fingerprints only — raw keys are never stored.
      configureLastGoodPersistence(path.join(agentRoot(), "openrouter-last-good.json"));
      pi.registerProvider(
        PROVIDER_NAME,
        buildStandaloneProviderConfig(undefined, { fetchDynamicModels: syncFetcher }) as ProviderConfig,
      );
      return;
    }
    const generation = nextGeneration();
    const apiKey = await bootstrapApiKey();
    if (!apiKey) return; // session_start syncs with the registry key

    try {
      refreshRequestConfig();
      // Register the live account catalog during extension load so Pi can
      // resolve saved scoped-model patterns before session_start fires.
      const result = await buildPlainSync(apiKey, true, requestConfig);

      if (isStale(generation)) return;
      commitSnapshot(generation, result.models);
      registerStandalone(result.models, apiKey);
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
    if (isOmp()) {
      // Dynamic registration: no re-registration with a pinned key. Ask the
      // host registry to re-run fetchDynamicModels — strategy "online" forces a
      // refresh past the 24h dynamic-models cache.
      //
      // The host RESOLVES that refresh even when the fetch failed: omp's
      // model manager catches a discovery error and falls back to its cached
      // rows, so the promise says nothing about the outcome. Reporting the
      // snapshot size here would announce the PREVIOUS catalog as a fresh
      // success (or "0 models synced" on a first-ever failure). The report
      // therefore comes from the discovery attempts this refresh produced.
      if (!silent) ctx.ui.notify("Fetching OpenRouter models...", "info");
      const mark = getDiscoveryAttemptCount();
      try {
        await ctx.modelRegistry.refreshProvider(PROVIDER_NAME, "online");
        const report = summarizeDiscovery(getDiscoveryOutcomesSince(mark));
        if (report.ok) {
          if (!silent) ctx.ui.notify(`OpenRouter: ${report.modelCount} models synced`, "info");
        } else if (report.noKey) {
          if (!silent) ctx.ui.notify("No OpenRouter API key configured — run /login openrouter", "warning");
        } else {
          if (!silent) ctx.ui.notify(`OpenRouter sync failed: ${report.error}`, "error");
        }
        notifyConfigWarnings(ctx);
      } catch (err: any) {
        if (!silent) ctx.ui.notify(`OpenRouter sync failed: ${err?.message}`, "error");
      }
      return;
    }

    let attempt: DiscoveryAttempt | undefined;
    try {
      const apiKey = await ctx.modelRegistry.getApiKeyForProvider(PROVIDER_NAME);
      if (!apiKey) {
        // No key: registering the public catalog would masquerade generic
        // models as the account-scoped list this extension exists to provide.
        // Pi's builtin OpenRouter list stays untouched.
        if (!silent) ctx.ui.notify("No OpenRouter API key configured — run /login openrouter", "warning");
        return;
      }
      if (!silent) ctx.ui.notify("Fetching OpenRouter models...", "info");

      // Pi's registry consumes the rows this sync registers, so the attempt
      // ledger guards them exactly as it guards omp's discovery data. The
      // attempt id doubles as the snapshot generation.
      attempt = beginDiscoveryAttempt(keyFingerprint(apiKey));
      const generation = attempt.id;
      refreshRequestConfig();
      clearSelections();
      const result = await buildPlainSync(apiKey, force, requestConfig);

      const published = await publishDiscoveryRows(attempt, result.models);
      if (published.status === "superseded") {
        // A newer sync owns the catalog; it registers its own rows.
        completeDiscoveryAttempt(attempt, {
          status: "superseded",
          modelCount: published.rows.length,
          rows: published.rows,
        });
        return;
      }
      if (isStale(generation)) {
        completeDiscoveryAttempt(attempt, { status: "superseded", modelCount: 0, rows: [] });
        return;
      }
      commitSnapshot(generation, result.models);
      registerStandalone(result.models, apiKey);
      // The registered snapshot changed: stale selections (and their recorded
      // base limits) from the previous catalog must not survive the sync.
      clearSelections();
      baseRegistration.clear();
      completeDiscoveryAttempt(attempt, {
        status: "ok",
        modelCount: result.modelCount,
        rows: result.models,
      });

      if (!silent) {
        ctx.ui.notify(`OpenRouter: ${result.modelCount} models synced`, "info");
      }
      notifyConfigWarnings(ctx);
    } catch (err: any) {
      if (attempt) {
        completeDiscoveryAttempt(attempt, {
          status: "failed",
          error: err instanceof Error ? err.message : String(err),
        });
      }
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
      if (!selectionTuning.enabled) return; // no stats fetches when selection is off
      const model = ctx.model ?? (ctx as any).getModel?.();
      if (!model) return;
      const modelRecord = model as { id?: unknown };
      const id = typeof modelRecord.id === "string" ? modelRecord.id : String(model);
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
      lines.push(
        selectionTuning.enabled
          ? `Selection: enabled — price anchor $${selectionTuning.priceAnchor}/M blended`
          : "Selection: disabled — OpenRouter default routing",
      );
      if (selectionTuning.enabled) {
        const fmt = (w: typeof selectionTuning.budgetWeights) =>
          `throughput ${w.throughput}, latency ${w.latency}, tool-call ${w.toolCall}, price ${w.price}`;
        lines.push(`  Budget weights: ${fmt(selectionTuning.budgetWeights)}`);
        lines.push(`  Flagship weights: ${fmt(selectionTuning.flagshipWeights)}`);
      }
      const tier = tierBySession.get(resolveSessionId(ctx) ?? "") ?? null;
      lines.push(`Cost tier: ${tier ? `${tier} (cost band ${COST_TIER_BANDS[tier]})` : "workspace default"}`);

      if (isOmp()) {
        // Wrapper-engagement proof: the registered rows must carry the
        // recovery api id, otherwise requests bypass the recovery wrapper.
        // Feature-detected — pi's registry lacks getProviderModels.
        try {
          const registry = ctx.modelRegistry as {
            getProviderModels?: (providerId: string) => unknown;
          };
          const rows = registry.getProviderModels?.(PROVIDER_NAME);
          if (Array.isArray(rows)) {
            const apis = new Set(
              rows
                .filter((r): r is Record<string, unknown> => r !== null && typeof r === "object" && "api" in r)
                .map((r) => String(r.api)),
            );
            lines.push(`Registered rows: ${rows.length} — apis: ${[...apis].join(", ") || "none"}`);
            const active = (ctx as { model?: { id?: unknown } }).model;
            const activeId = typeof active?.id === "string" ? active.id : undefined;
            const activeRow = activeId ? rows.find((r) => (r as { id?: unknown })?.id === activeId) : undefined;
            if (activeRow) {
              lines.push(`Active model row api: ${String((activeRow as Record<string, unknown>).api)}`);
            }
          }
        } catch {
          // status command must never throw
        }
      }

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