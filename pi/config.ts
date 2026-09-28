import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type {
  LoadedRequestConfig,
  ModelRequestConfig,
  RequestConfig,
  SelectionTuning,
  SelectionWeights,
} from "./types.js";

/**
 * Fourth fork fix: per-request routing configuration.
 *
 * OpenRouter accepts a `provider` routing object and a handful of extra
 * top-level fields on every chat-completions request. This module loads a
 * shipped default config plus an optional user config, merges them
 * (user wins), validates them against what OpenRouter actually accepts,
 * and exposes the per-model entries the stream factory merges into each
 * outgoing payload.
 *
 * Why validation matters: OpenRouter validates the `provider` object with a
 * strict schema — an unknown sub-key returns `400 provider: Unrecognized
 * key` and would fail every request to that model. A typo in the JSON file
 * must therefore be caught locally, not on the wire.
 */

// Every field OpenRouter's provider routing object accepts (docs/provider-routing).
// Anything else in the `provider` object is dropped with a warning.
const PROVIDER_FIELD_ALLOWLIST = new Set([
  "order",
  "only",
  "ignore",
  "allow_fallbacks",
  "require_parameters",
  "data_collection",
  "zdr",
  "enforce_distillable_text",
  "quantizations",
  "sort",
  "preferred_min_throughput",
  "preferred_max_latency",
  "max_price",
]);

// Fields pi-ai builds from the conversation; the config must never override them.
export const RESERVED_PAYLOAD_KEYS = new Set([
  "model",
  "messages",
  "stream",
  "tools",
  "tool_choice",
  "response_format",
  "n",
  "max_tokens",
]);

// Router slugs, ~-prefixed aliases, and routing-variant suffixes are not real
// models: never store endpoint data or apply config entries for them.
// (:free ids are genuine selectable offerings — same model at zero cost — and
// stay in the picker; :batch is a discounted async tier; :nitro/:floor are
// routing shortcuts that never appear as catalog ids but are excluded for
// safety.)
const VARIANT_SUFFIXES = [":batch", ":nitro", ":floor"];

export function isRealModelId(id: string): boolean {
  if (!id || typeof id !== "string") return false;
  if (id.startsWith("~")) return false;
  if (id.startsWith("openrouter/")) return false;
  return !isVariantModelId(id);
}

/** Catalog ids with routing-variant suffixes (:batch, :nitro, :floor) — excluded from the picker. */
export function isVariantModelId(id: string): boolean {
  if (!id) return false;
  const lower = id.toLowerCase();
  return VARIANT_SUFFIXES.some((suffix) => lower.endsWith(suffix));
}

/**
 * Merge one model's config entry into the outgoing request payload.
 * `provider` merges over any existing provider object; other entry fields
 * pass through as top-level request fields except the pi-ai-owned keys.
 * `context_window` / `max_completion_tokens` are plugin-side only and never
 * reach the wire. Shape-agnostic: works for both chat-completions and
 * responses payloads (the provider object is identical on both endpoints).
 */
export function mergeRequestConfig(payload: unknown, entry: ModelRequestConfig): unknown {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return payload;
  }
  const record: Record<string, unknown> = { ...(payload as Record<string, unknown>) };

  if (entry.provider && typeof entry.provider === "object") {
    const existing =
      record.provider && typeof record.provider === "object" && !Array.isArray(record.provider)
        ? { ...(record.provider as Record<string, unknown>) }
        : {};
    record.provider = { ...existing, ...entry.provider };
  }

  for (const [key, value] of Object.entries(entry)) {
    if (key === "provider" || key === "context_window" || key === "max_completion_tokens") continue;
    if (RESERVED_PAYLOAD_KEYS.has(key)) continue;
    record[key] = value;
  }
  return record;
}

// ---------------------------------------------------------------------------
// Provider-selection tuning
// ---------------------------------------------------------------------------

/**
 * Built-in selection tuning — the values the algorithm shipped with, kept
 * here so a user file overrides any single field without naming the rest.
 * The shipped `openrouter-advanced-provider.json` carries the same numbers;
 * this constant is the base of the merge and the fallback for invalid input.
 */
export const DEFAULT_SELECTION: SelectionTuning = {
  enabled: true,
  priceAnchor: 3.0,
  budgetWeights: { throughput: 0.55, latency: 0.2, toolCall: 0.15, price: 0.1 },
  flagshipWeights: { throughput: 0.3, latency: 0.25, toolCall: 0.2, price: 0.25 },
};

const WEIGHT_KEYS = ["throughput", "latency", "toolCall", "price"] as const;

function parseWeights(
  tier: string,
  raw: unknown,
  base: SelectionWeights,
  warnings: string[],
): SelectionWeights {
  if (raw === undefined) return { ...base };
  if (!isPlainObject(raw)) {
    warnings.push(`selection.weights.${tier} must be an object — default weights kept`);
    return { ...base };
  }
  const out: SelectionWeights = { ...base };
  let sum = 0;
  for (const key of WEIGHT_KEYS) {
    const value = raw[key];
    if (value === undefined) continue;
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      warnings.push(`selection.weights.${tier}.${key} must be a non-negative number — default kept`);
      continue;
    }
    out[key] = value;
  }
  for (const key of Object.keys(raw)) {
    if (!WEIGHT_KEYS.includes(key as (typeof WEIGHT_KEYS)[number])) {
      warnings.push(`selection.weights.${tier}.${key} is not a selection axis — dropped`);
    }
  }
  for (const key of WEIGHT_KEYS) sum += out[key];
  if (sum <= 0) {
    warnings.push(`selection.weights.${tier} disables every axis — default weights kept`);
    return { ...base };
  }
  return out;
}

/**
 * Parse the config file's `selection` section over the built-in defaults.
 * Per-field semantics: a provided field replaces that field only; unknown
 * or invalid fields are dropped with a warning and never reach the wire
 * (they cannot — selection tuning is client-side only).
 */
export function parseSelectionSection(raw: unknown, warnings: string[]): SelectionTuning {
  if (raw === undefined) return structuredClone(DEFAULT_SELECTION);
  if (!isPlainObject(raw)) {
    warnings.push(`"selection" section must be an object — default tuning kept`);
    return structuredClone(DEFAULT_SELECTION);
  }
  const tuning: SelectionTuning = structuredClone(DEFAULT_SELECTION);

  if (raw.enabled !== undefined) {
    if (typeof raw.enabled === "boolean") tuning.enabled = raw.enabled;
    else warnings.push(`selection.enabled must be a boolean — default kept`);
  }
  if (raw.priceAnchor !== undefined) {
    if (typeof raw.priceAnchor === "number" && Number.isFinite(raw.priceAnchor) && raw.priceAnchor > 0) {
      tuning.priceAnchor = raw.priceAnchor;
    } else {
      warnings.push(`selection.priceAnchor must be a positive number — default kept`);
    }
  }
  if (raw.weights !== undefined) {
    if (isPlainObject(raw.weights)) {
      for (const key of Object.keys(raw.weights)) {
        if (key !== "budget" && key !== "flagship") {
          warnings.push(`selection.weights.${key} is not a weight tier ("budget"|"flagship") — dropped`);
        }
      }
      tuning.budgetWeights = parseWeights("budget", raw.weights.budget, tuning.budgetWeights, warnings);
      tuning.flagshipWeights = parseWeights("flagship", raw.weights.flagship, tuning.flagshipWeights, warnings);
    } else {
      warnings.push(`selection.weights must be an object with "budget" and "flagship" — default weights kept`);
    }
  }
  for (const key of Object.keys(raw)) {
    if (!["enabled", "priceAnchor", "weights"].includes(key)) {
      warnings.push(`selection.${key} is not a selection option — dropped`);
    }
  }
  return tuning;
}

function defaultConfigPath(): string {
  // Shipped alongside the extension; the same-named user file in the global
  // pi directory overrides it.
  return path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "openrouter-advanced-provider.json");
}

function userConfigPath(): string {
  return path.join(os.homedir(), ".pi", "agent", "openrouter-advanced-provider.json");
}

function readJsonFile(
    filePath: string,
    warnings: string[],
    label: string,
): Record<string, unknown> | undefined {
    if (!fs.existsSync(filePath)) return undefined;
    try {
        const raw = fs.readFileSync(filePath, "utf8");
        const parsed = JSON.parse(raw);
        if (!isPlainObject(parsed)) {
            warnings.push(`${label} (${filePath}) must be a JSON object — ignored`);
            return undefined;
        }
        return parsed;
    } catch (err) {
        // Unreadable or malformed: say so instead of silently disabling every
        // routing override the file carried.
        warnings.push(`${label} (${filePath}) could not be read/parsed — ignored: ${err instanceof Error ? err.message : String(err)}`);
        return undefined;
    }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** defaults merged under override; plain objects recurse, arrays and scalars replace. */
function deepMerge(defaults: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...defaults };
  for (const [key, value] of Object.entries(override)) {
    const base = out[key];
    if (isPlainObject(base) && isPlainObject(value)) {
      out[key] = deepMerge(base, value);
    } else {
      out[key] = value;
    }
  }
  return out;
}

function validateProviderObject(
  modelId: string,
  provider: unknown,
  warnings: string[],
): Record<string, unknown> | undefined {
  if (provider === undefined) return undefined;
  if (!isPlainObject(provider)) {
    warnings.push(`${modelId}: "provider" must be an object — ignored`);
    return undefined;
  }
  const clean: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(provider)) {
    if (!PROVIDER_FIELD_ALLOWLIST.has(key)) {
      warnings.push(`${modelId}: provider.${key} is not an OpenRouter routing field — dropped (OpenRouter 400s on unknown provider keys)`);
      continue;
    }
    if (["order", "only", "ignore", "quantizations"].includes(key)) {
      if (!Array.isArray(value) || !value.every((v) => typeof v === "string")) {
        warnings.push(`${modelId}: provider.${key} must be an array of strings — dropped`);
        continue;
      }
    } else if (["allow_fallbacks", "require_parameters", "zdr", "enforce_distillable_text"].includes(key)) {
      if (typeof value !== "boolean") {
        warnings.push(`${modelId}: provider.${key} must be a boolean — dropped`);
        continue;
      }
    } else if (key === "data_collection" && value !== "allow" && value !== "deny") {
      warnings.push(`${modelId}: provider.data_collection must be "allow" or "deny" — dropped`);
      continue;
    } else if (key === "sort" && !isPlainObject(value) && typeof value !== "string") {
      warnings.push(`${modelId}: provider.sort must be a string or object — dropped`);
      continue;
    } else if (key === "preferred_min_throughput" || key === "preferred_max_latency") {
      if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
        warnings.push(`${modelId}: provider.${key} must be a positive number — dropped`);
        continue;
      }
    } else if (key === "max_price") {
      // {prompt, completion, ...}: every value a finite non-negative number.
      if (!isPlainObject(value) || !Object.values(value).every((v) => typeof v === "number" && Number.isFinite(v) && v >= 0)) {
        warnings.push(`${modelId}: provider.max_price must be an object of non-negative numbers — dropped`);
        continue;
      }
    }
    clean[key] = value;
  }
  return Object.keys(clean).length > 0 ? clean : undefined;
}

function validateEntry(modelId: string, entry: unknown, warnings: string[]): ModelRequestConfig | undefined {
  if (!isPlainObject(entry)) {
    warnings.push(`${modelId}: config entry must be an object — ignored`);
    return undefined;
  }
  const clean: ModelRequestConfig = {};
  const provider = validateProviderObject(modelId, entry.provider, warnings);
  if (provider) clean.provider = provider;

  for (const numericKey of ["context_window", "max_completion_tokens"] as const) {
    const value = entry[numericKey];
    if (value === undefined) continue;
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      warnings.push(`${modelId}: ${numericKey} must be a positive number — ignored`);
      continue;
    }
    clean[numericKey] = value;
  }

  for (const [key, value] of Object.entries(entry)) {
    if (key === "provider" || key === "context_window" || key === "max_completion_tokens") continue;
    if (RESERVED_PAYLOAD_KEYS.has(key)) {
      warnings.push(`${modelId}: "${key}" is owned by pi-ai and cannot be overridden — dropped`);
      continue;
    }
    clean[key] = value;
  }

  return Object.keys(clean).length > 0 ? clean : undefined;
}

/** Extract the "requests" section from a loaded config file. */
function readRequestsSection(
  raw: Record<string, unknown> | undefined,
  warnings: string[],
): { defaults: Record<string, unknown>; models: Record<string, unknown> } {
  const empty = { defaults: {}, models: {} };
  if (!raw) return empty;
  if (raw.requests === undefined) return empty;
  if (!isPlainObject(raw.requests)) {
    warnings.push(`"requests" section must be an object — ignored`);
    return empty;
  }
  const requests = raw.requests;
  return {
    defaults: isPlainObject(requests.defaults) ? requests.defaults : {},
    models: isPlainObject(requests.models) ? requests.models : {},
  };
}

export function loadRequestConfig(
  shippedPath: string = defaultConfigPath(),
  userPath: string = userConfigPath(),
): LoadedRequestConfig {
  const warnings: string[] = [];

  const defaultFile = readJsonFile(shippedPath, warnings, "shipped config");
  const userFile = readJsonFile(userPath, warnings, "user config");
  const defaultRaw = readRequestsSection(defaultFile, warnings);
  const userRaw = readRequestsSection(userFile, warnings);

  for (const [label, file] of [
    ["shipped config", defaultFile],
    ["user config", userFile],
  ] as const) {
    if (file?.selection !== undefined && !isPlainObject(file.selection)) {
      warnings.push(`${label}: "selection" section must be an object — default tuning kept`);
    }
  }
  const mergedSelectionRaw =
    isPlainObject(userFile?.selection) || isPlainObject(defaultFile?.selection)
      ? deepMerge(
          isPlainObject(defaultFile?.selection) ? defaultFile.selection : {},
          isPlainObject(userFile?.selection) ? userFile.selection : {},
        )
      : undefined;
  const selection = parseSelectionSection(mergedSelectionRaw, warnings);

  const defaultModels = defaultRaw.models;
  const userModels = userRaw.models;

  const mergedDefaults = deepMerge(defaultRaw.defaults, userRaw.defaults);
  // Defaults ride every request, so they pass the same validation as a
  // per-model entry (provider allowlist, reserved keys) — an unvalidated
  // default would reach the wire on models with no explicit entry.
  const validatedDefaults =
    Object.keys(mergedDefaults).length > 0
      ? validateEntry("requests.defaults", mergedDefaults, warnings)
      : undefined;

  const mergedModels: Record<string, ModelRequestConfig> = {};
  for (const id of new Set([...Object.keys(defaultModels), ...Object.keys(userModels)])) {
    if (!isRealModelId(id)) {
      warnings.push(`${id}: not a real model (router, alias, or variant) — entry skipped`);
      continue;
    }
    const defaultEntry = defaultModels[id];
    const userEntry = userModels[id];
    const rawEntry =
      isPlainObject(defaultEntry) && isPlainObject(userEntry)
        ? deepMerge(defaultEntry, userEntry)
        : (userEntry ?? defaultEntry);
    const entry = validateEntry(id, rawEntry, warnings);
    if (entry) mergedModels[id] = entry;
  }

  const config: RequestConfig = {
    defaults: validatedDefaults,
    models: Object.keys(mergedModels).length > 0 ? new Map(Object.entries(mergedModels)) : undefined,
  };
  return { config, selection, warnings };
}