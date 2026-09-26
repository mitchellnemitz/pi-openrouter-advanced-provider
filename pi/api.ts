import {
  OPENROUTER_MODELS_URL,
  OPENROUTER_BASE_URL,
  CACHE_TTL_MS,
  FETCH_TIMEOUT_MS,
  type OpenRouterModel,
  type OpenRouterEndpoint,
  type OpenRouterEndpointsResponse,
  type OpenRouterKeyInfo,
  type OpenRouterCreditsInfo,
  type EndpointCacheEntry,
  type StatEndpoint,
} from "./types.js";

/**
 * HTTP failure from an OpenRouter endpoint. Carries the status so callers
 * can branch on it (auth fallback, silence-on-missing) without parsing
 * error text.
 */
export class OpenRouterApiError extends Error {
  readonly status: number;

  constructor(status: number, statusText: string, context: string) {
    let hint = "";
    if (status === 401 || status === 403) hint = " — check your OpenRouter API key";
    else if (status === 429) hint = " — rate limited, try again shortly";
    else if (status >= 500) hint = " — OpenRouter is having issues, try again later";
    super(`${context}: ${status} ${statusText}${hint}`);
    this.name = "OpenRouterApiError";
    this.status = status;
  }
}

function hashKey(key?: string): string {
  if (!key) return "";
  return key.slice(0, 8) + key.slice(-4);
}

function authHeaders(apiKey?: string): Record<string, string> {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

async function fetchJson<T>(url: string, apiKey?: string): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(url, { headers: authHeaders(apiKey), signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
  if (!res.ok) throw new OpenRouterApiError(res.status, res.statusText, "OpenRouter API");
  return (await res.json()) as T;
}

// ---------------------------------------------------------------------------
// Model catalog caches
//
// Three independently invalidated caches (public models, account models,
// per-model endpoints) with one shared rule: the cache key includes a hash
// of the API key, so switching accounts via /login never serves the previous
// account's data.
// ---------------------------------------------------------------------------

interface ModelCache {
  models: OpenRouterModel[];
  timestamp: number;
  keyHash: string;
}

let publicModelsCache: ModelCache | null = null;
let userModelsCache: ModelCache | null = null;
const endpointCache = new Map<string, EndpointCacheEntry>();

function cacheIsFresh(cache: ModelCache | null, keyHash: string, force: boolean): cache is ModelCache & { models: OpenRouterModel[] } {
  return (
    !force &&
    cache !== null &&
    cache.keyHash === keyHash &&
    cache.models.length >= 0 &&
    Date.now() - cache.timestamp < CACHE_TTL_MS
  );
}

export function invalidateModelCache(): void {
  publicModelsCache = null;
}

export function invalidateUserModelCache(): void {
  userModelsCache = null;
}

export function invalidateEndpointCache(modelId?: string): void {
  if (modelId) endpointCache.delete(modelId);
  else endpointCache.clear();
}

export function invalidateAllCaches(): void {
  invalidateModelCache();
  invalidateUserModelCache();
  invalidateEndpointCache();
}

/** Public catalog: every model OpenRouter lists, no auth required. */
async function fetchPublicModels(apiKey: string | undefined, force: boolean): Promise<OpenRouterModel[]> {
  const keyHash = hashKey(apiKey);
  if (cacheIsFresh(publicModelsCache, keyHash, force)) return publicModelsCache.models;
  const json = await fetchJson<{ data?: OpenRouterModel[] }>(OPENROUTER_MODELS_URL, apiKey);
  publicModelsCache = { models: json.data ?? [], timestamp: Date.now(), keyHash };
  return publicModelsCache.models;
}

/**
 * Account catalog (GET /models/user): the models this key can actually
 * reach, including BYOK-provisioned models and `~` fallback aliases. Requires
 * an API key.
 */
async function fetchAccountModels(apiKey: string, force: boolean): Promise<OpenRouterModel[]> {
  const keyHash = hashKey(apiKey);
  if (cacheIsFresh(userModelsCache, keyHash, force)) return userModelsCache.models;
  const json = await fetchJson<{ data?: OpenRouterModel[] }>(`${OPENROUTER_BASE_URL}/models/user`, apiKey);
  userModelsCache = { models: json.data ?? [], timestamp: Date.now(), keyHash };
  return userModelsCache.models;
}

/**
 * Catalog source selection: the account catalog when a key is available,
 * the public catalog otherwise. Only an auth rejection (401/403) justifies
 * that fallback — a key without /models/user access still yields a valid
 * sync. Timeouts and 5xx must surface: silently swapping to the public
 * catalog would replace the account's BYOK/private models with ones the
 * key may not be able to use.
 */
export async function fetchCatalogModels(apiKey?: string, force = false): Promise<OpenRouterModel[]> {
  if (!apiKey) return fetchPublicModels(apiKey, force);
  try {
    return await fetchAccountModels(apiKey, force);
  } catch (err) {
    if (err instanceof OpenRouterApiError && (err.status === 401 || err.status === 403)) {
      return fetchPublicModels(apiKey, force);
    }
    throw err;
  }
}

/** Endpoints of one model, from GET /models/{author}/{slug}/endpoints. */
export async function fetchModelEndpoints(
  modelId: string,
  apiKey?: string,
  force = false,
): Promise<OpenRouterEndpoint[]> {
  const cached = endpointCache.get(modelId);
  if (!force && cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
    return cached.endpoints;
  }

  const path = modelId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
  const json = await fetchJson<OpenRouterEndpointsResponse>(
    `${OPENROUTER_BASE_URL}/models/${path}/endpoints`,
    apiKey,
  );
  const endpoints = json.data?.endpoints ?? [];
  endpointCache.set(modelId, { timestamp: Date.now(), endpoints });
  return endpoints;
}

/** GET /key: the requesting key's limits and usage. */
export async function fetchKeyInfo(apiKey: string): Promise<OpenRouterKeyInfo> {
  const json = await fetchJson<{ data?: OpenRouterKeyInfo }>(`${OPENROUTER_BASE_URL}/key`, apiKey);
  return json.data ?? {};
}

/**
 * GET /credits: account balance. Requires a management key — a regular key
 * gets rejected, which is reported as "no data" rather than an error.
 */
export async function fetchCredits(apiKey: string): Promise<OpenRouterCreditsInfo | null> {
  try {
    const json = await fetchJson<{ data?: OpenRouterCreditsInfo }>(`${OPENROUTER_BASE_URL}/credits`, apiKey);
    return json.data ?? null;
  } catch {
    return null;
  }
}

/** Snapshot of the public catalog cache, when one is held. */
export function getCachedModels(): OpenRouterModel[] | null {
  return publicModelsCache?.models ?? null;
}

// ---------------------------------------------------------------------------
// Just-in-time selection data
//
// OpenRouter website APIs under /api/frontend/v1 — undocumented, unauthenticated.
// Shapes verified against live responses; re-verify there when they misbehave.
// ---------------------------------------------------------------------------

const FRONTEND_BASE_URL = "https://openrouter.ai/api/frontend/v1";
const PERMASLUG_TTL_MS = 24 * 60 * 60 * 1000;

let permaslugCache: { map: Map<string, string>; timestamp: number } | null = null;

/**
 * slug -> permaslug map, from the frontend catalog (over a thousand entries;
 * cached for a day). The stats endpoints key on permaslug, not slug.
 */
export async function fetchPermaslugMap(force = false): Promise<Map<string, string>> {
  const now = Date.now();
  if (!force && permaslugCache && now - permaslugCache.timestamp < PERMASLUG_TTL_MS) {
    return permaslugCache.map;
  }
  const res = await fetch(`${FRONTEND_BASE_URL}/catalog/models`);
  if (!res.ok) throw new OpenRouterApiError(res.status, res.statusText, "OpenRouter frontend catalog");
  const json = (await res.json()) as {
    data?: Array<{ slug?: string; permaslug?: string }>;
  };
  const map = new Map<string, string>();
  for (const item of json.data ?? []) {
    if (item.slug && item.permaslug) map.set(item.slug, item.permaslug);
  }
  permaslugCache = { map, timestamp: now };
  return map;
}

export function invalidatePermaslugCache(): void {
  permaslugCache = null;
}

/** Age of the cached permaslug map in ms; Infinity when uncached. */
export function permaslugCacheAge(): number {
  return permaslugCache ? Date.now() - permaslugCache.timestamp : Infinity;
}

/**
 * Per-endpoint selection data for one model: provider slug, context length,
 * quantization, pricing, supported params, status, and 30-minute
 * latency/throughput percentiles with request counts.
 */
export async function fetchEndpointStats(permaslug: string): Promise<StatEndpoint[]> {
  const query = new URLSearchParams({
    permaslug,
    variant: "standard",
    latencyMetric: "latency",
    perfWorkload: "text_generation",
  });
  const res = await fetch(`${FRONTEND_BASE_URL}/stats/endpoint?${query}`);
  if (!res.ok) throw new OpenRouterApiError(res.status, res.statusText, "OpenRouter endpoint stats");
  const json = (await res.json()) as { data?: StatEndpoint[] };
  return json.data ?? [];
}

/**
 * Latest tool-call error rate (%) per endpoint UUID, from the daily series.
 */
export async function fetchToolCallErrorRates(permaslug: string): Promise<Map<string, number>> {
  const query = new URLSearchParams({ permaslug, variant: "standard" });
  const res = await fetch(`${FRONTEND_BASE_URL}/stats/tool-call-error-rate?${query}`);
  if (!res.ok) throw new OpenRouterApiError(res.status, res.statusText, "OpenRouter tool-call error stats");
  const json = (await res.json()) as { data?: Array<{ x: string; y: Record<string, number> }> };
  const series = json.data ?? [];
  const rates = new Map<string, number>();
  if (series.length === 0) return rates;
  for (const [uuid, rate] of Object.entries(series[series.length - 1].y ?? {})) {
    if (typeof rate === "number") rates.set(uuid, rate);
  }
  return rates;
}
