import type {
  ProviderModelConfig,
  RequestConfig,
  SelectionResult,
  SyncSnapshot,
} from "./types.js";
import { toProviderModel, applyModelConfigOverrides, isReasoningModel } from "./models.js";
import { fetchCatalogModels, fetchModelEndpoints, getCachedModels } from "./api.js";
import { isVariantModelId } from "./config.js";
import { beginDiscoveryAttempt as beginAttempt, type DiscoveryAttempt } from "./discovery.js";
import type { OpenRouterModel } from "./types.js";
import { clearEndpointLimits, setEndpointLimits } from "./selection-limits-state.js";

// ---------------------------------------------------------------------------
// Snapshot
//
// The registered model set is versioned by a monotonically increasing
// generation. Async syncs capture a generation up front and commit at the
// end only if it is still current, so a slow sync can never overwrite a
// newer one's result.
// ---------------------------------------------------------------------------

let currentSnapshot: SyncSnapshot = {
  generation: 0,
  models: [],
  timestamp: 0,
};

let syncGeneration = 0;

export function getSnapshot(): SyncSnapshot {
  return currentSnapshot;
}

export function getGeneration(): number {
  return syncGeneration;
}

export function nextGeneration(): number {
  return ++syncGeneration;
}

export function isStale(generation: number): boolean {
  return generation !== syncGeneration;
}

export function commitSnapshot(generation: number, models: ProviderModelConfig[]): boolean {
  if (isStale(generation)) return false;
  currentSnapshot = {
    generation,
    models,
    timestamp: Date.now(),
  };
  return true;
}

// ---------------------------------------------------------------------------
// Discovery attempts
//
// The attempt ledger lives in pi/discovery.ts: it is the unit under test, so
// it must not drag the host-facing imports (pi/models.ts reaches
// @earendil-works/pi-ai) into a plain-Node test run. The attempt id shares
// the snapshot generation counter below, so "a newer attempt started" and
// "this snapshot is stale" can never disagree about which is newer.
// ---------------------------------------------------------------------------

export {
  awaitNewerDiscoveryRows,
  completeDiscoveryAttempt,
  DiscoveryFailedError,
  getDiscoveryOutcomesSince,
  getNewerDiscoveryAttempt,
  keyFingerprint,
  publishDiscoveryRows,
  recordDiscoveryOutcome,
  resetDiscoveryLedger,
  summarizeDiscovery,
} from "./discovery.js";
export type {
  DiscoveryAttempt,
  DiscoveryOutcome,
  DiscoveryReport,
  DiscoveryStatus,
} from "./discovery.js";

/**
 * Start a discovery attempt whose id is the NEXT snapshot generation.
 *
 * The attempt id and the snapshot generation are one ordering: an attempt is
 * stale exactly when a newer attempt has started, and the committed catalog
 * belongs to the newest attempt. Sharing the counter keeps those two
 * statements from ever disagreeing.
 */
export function beginDiscoveryAttempt(keyFingerprintValue: string): DiscoveryAttempt {
  return beginAttempt(keyFingerprintValue, nextGeneration());
}

/** Number of attempts started so far; use it to mark the start of a refresh. */
export function getDiscoveryAttemptCount(): number {
  return getGeneration();
}

// ---------------------------------------------------------------------------
// Sync
// ---------------------------------------------------------------------------

export interface SyncPlainResult {
  models: ProviderModelConfig[];
  modelCount: number;
}

/**
 * Agentic-usable capability gate: the model must support tool calling AND
 * reasoning — the set an agent loop can actually drive. Routing constructs
 * (`openrouter/auto`, `~` fallback aliases) are kept regardless, so pi's
 * router and fallback aliases survive the filter.
 */
export function isAgenticUsableModel(m: OpenRouterModel): boolean {
  if (m.id.startsWith("~") || m.id.startsWith("openrouter/")) return true;
  return (m.supported_parameters ?? []).includes("tools") && isReasoningModel(m);
}

/**
 * Build the standalone catalog: account models (or public, per the catalog
 * source selection in api.ts), filtered to agentic-usable entries, converted
 * to pi model configs, then narrowed per config-pinned model to the
 * endpoints its entry approves.
 */
export async function buildPlainSync(
  apiKey?: string,
  force?: boolean,
  config?: RequestConfig,
): Promise<SyncPlainResult> {
  const rawModels = await fetchCatalogModels(apiKey, force);
  let models = rawModels
    .filter((m) => !isVariantModelId(m.id) && isAgenticUsableModel(m))
    .map(toProviderModel);

  // Config-pinned models bypass the JIT algorithm, so their documented
  // context_window / max_completion_tokens overrides are applied at
  // registration. The shipped config is empty, so this costs nothing unless
  // the user pins models.
  const pinned = config?.models;
  if (pinned && pinned.size > 0) {
    models = await Promise.all(
      models.map(async (model) => {
        const entry = pinned.get(model.id);
        if (!entry) return model;
        try {
          const endpoints = await fetchModelEndpoints(model.id, apiKey, force);
          return applyModelConfigOverrides(model, entry, endpoints);
        } catch {
          // Endpoint catalog unavailable: explicit entry overrides still
          // apply; the approved-endpoint minimum falls back to the catalog
          // value.
          return applyModelConfigOverrides(model, entry, []);
        }
      }),
    );
  }

  return { models, modelCount: models.length };
}

// ---------------------------------------------------------------------------
// Just-in-time selection cache
//
// One cached selection per model, refreshed at most every 10 minutes.
// getStaleSelection deliberately outlives the TTL: routing against a
// last-known pick beats routing blind while a refresh is in flight.
// ---------------------------------------------------------------------------

const SELECTION_TTL_MS = 10 * 60 * 1000;

const selections = new Map<string, { result: SelectionResult; timestamp: number }>();

export function getSelection(modelId: string): { result: SelectionResult; timestamp: number } | undefined {
  const cached = selections.get(modelId);
  if (!cached) return undefined;
  if (Date.now() - cached.timestamp > SELECTION_TTL_MS) return undefined;
  return cached;
}

export function getStaleSelection(modelId: string): { result: SelectionResult; timestamp: number } | undefined {
  return selections.get(modelId);
}

export function setSelection(modelId: string, result: SelectionResult): void {
  selections.set(modelId, { result, timestamp: Date.now() });
  setEndpointLimits(modelId, result);
}

export function clearSelections(): void {
  selections.clear();
  clearEndpointLimits();
}

export function getCachedModelList(): OpenRouterModel[] | null {
  return getCachedModels();
}
