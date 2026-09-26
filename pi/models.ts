import { getModels } from "@earendil-works/pi-ai";
import {
  type OpenRouterModel,
  type OpenRouterEndpoint,
  type OpenRouterArchitecture,
  type ProviderModelConfig,
  type InputType,
  type ModelRequestConfig,
} from "./types.js";

// ---------------------------------------------------------------------------
// Reasoning detection
//
// OpenRouter signals reasoning support inconsistently across providers: some
// declare a supported parameter, some ship reasoning defaults, some only
// hint at it in the id or display name. A model counts as a reasoning model
// when any signal fires.
// ---------------------------------------------------------------------------

const REASONING_ID_PATTERNS = [
  ":thinking",
  "-r1",
  "/r1",
  "o1-",
  "o3-",
  "o4-",
  "reasoner",
  "-thinking",
  "qwq-",
  "/qwq",
];

const REASONING_NAME_PATTERNS = ["thinking", "reasoner", "chain-of-thought"];

const REASONING_PARAMETER_NAMES = new Set([
  "include_reasoning",
  "reasoning",
  "reasoning_effort",
]);

function declaresReasoningParameter(supportedParameters?: string[]): boolean {
  return supportedParameters?.some((p) => REASONING_PARAMETER_NAMES.has(p)) ?? false;
}

function shipsReasoningDefaults(defaultParameters?: Record<string, unknown> | null): boolean {
  return (
    defaultParameters?.include_reasoning !== undefined ||
    defaultParameters?.reasoning !== undefined ||
    defaultParameters?.reasoning_effort !== undefined
  );
}

function idOrNameSuggestsReasoning(id: string, name?: string): boolean {
  const lowerId = id.toLowerCase();
  const lowerName = (name || "").toLowerCase();
  return (
    REASONING_ID_PATTERNS.some((pattern) => lowerId.includes(pattern)) ||
    REASONING_NAME_PATTERNS.some((pattern) => lowerName.includes(pattern))
  );
}

export function isReasoningModel(m: OpenRouterModel): boolean {
  return (
    declaresReasoningParameter(m.supported_parameters) ||
    shipsReasoningDefaults(m.default_parameters) ||
    idOrNameSuggestsReasoning(m.id, m.name)
  );
}

// ---------------------------------------------------------------------------
// Input modality
// ---------------------------------------------------------------------------

export function supportsImages(architecture?: OpenRouterArchitecture): boolean {
  if (architecture?.input_modalities) {
    return architecture.input_modalities.includes("image");
  }
  return architecture?.modality?.includes("multimodal") ?? false;
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

/**
 * Parse an OpenRouter pricing string into a number. Missing is not zero:
 * an absent price stays undefined so callers can distinguish it from free.
 * The "-1" sentinel (router slugs like openrouter/auto, where the price
 * varies with the routed model) is also not a price — treating it as one
 * registers roughly -$1M/M-token rates.
 */
export function parseCost(value?: string): number | undefined {
  if (value == null || value === "") return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return undefined;
  if (parsed === -1) return undefined;
  return parsed * 1_000_000;
}

function costOrFallback(cost: number | undefined, fallback: number): number {
  return cost !== undefined ? cost : fallback;
}

function minPositive(values: Array<number | undefined>, fallback: number): number {
  const present = values.filter((v): v is number => typeof v === "number" && v > 0);
  return present.length > 0 ? Math.min(...present) : fallback;
}

// ---------------------------------------------------------------------------
// Config-approved endpoint narrowing
// ---------------------------------------------------------------------------

// Service-tier endpoints (flex/fast/priority/highspeed, per OpenRouter's
// service-tiers docs) are addressed by tier-qualified slugs, so a base
// provider slug must not match them.
const TIER_SEGMENTS = new Set(["flex", "fast", "priority", "highspeed"]);

/** Does an endpoint tag ("wafer", "parasail/fp8", "openai/flex") match a config slug? */
export function endpointMatchesSlug(tag: string | undefined, slug: string): boolean {
  if (!tag) return false;
  if (tag === slug) return true;
  if (slug.includes("/")) return false;
  const segments = tag.split("/");
  if (segments[0] !== slug) return false;
  return !TIER_SEGMENTS.has(segments[segments.length - 1].toLowerCase());
}

/**
 * The endpoints a config entry actually approves: the `only`/`order` slugs
 * (minus `ignore`) when present, otherwise all endpoints minus `ignore`,
 * always narrowed to the configured quantizations. Drives the effective
 * context window below.
 */
export function approvedEndpoints(
  entry: ModelRequestConfig,
  endpoints: OpenRouterEndpoint[],
): OpenRouterEndpoint[] {
  const provider = entry.provider;
  if (!provider) return endpoints;

  // Requests only route to endpoints matching the configured quantizations,
  // so variants outside the filter must not shape the derived window.
  const quantizations = Array.isArray(provider.quantizations)
    ? (provider.quantizations as string[]).map((q) => q.toLowerCase())
    : undefined;
  const quantizationAllowed = (e: OpenRouterEndpoint) =>
    !quantizations || quantizations.includes((e.quantization || "unknown").toLowerCase());

  const ignore = new Set(Array.isArray(provider.ignore) ? (provider.ignore as string[]) : []);
  const isIgnored = (e: OpenRouterEndpoint) =>
    ignore.has((e.tag || "").split("/")[0]) || ignore.has(e.tag || "");

  const only = Array.isArray(provider.only) ? (provider.only as string[]) : [];
  const order = Array.isArray(provider.order) ? (provider.order as string[]) : [];
  const approvedSlugs = only.length > 0 ? only : order;

  if (approvedSlugs.length === 0) {
    return endpoints.filter((e) => !isIgnored(e) && quantizationAllowed(e));
  }
  return endpoints.filter(
    (e) => !isIgnored(e) && quantizationAllowed(e) && approvedSlugs.some((slug) => endpointMatchesSlug(e.tag, slug)),
  );
}

/**
 * Replace the catalog-advertised contextWindow/maxTokens with the smallest
 * value across the approved provider set, so pi's context budgeting and
 * autocompact reflect what the routed provider can actually serve.
 * Explicit entry overrides win when present.
 */
export function applyModelConfigOverrides(
  model: ProviderModelConfig,
  entry: ModelRequestConfig,
  endpoints: OpenRouterEndpoint[],
): ProviderModelConfig {
  const approved = approvedEndpoints(entry, endpoints);
  let contextWindow = model.contextWindow;
  let maxTokens = model.maxTokens;
  if (approved.length > 0) {
    contextWindow = minPositive(approved.map((e) => e.context_length), model.contextWindow);
    maxTokens = minPositive(approved.map((e) => e.max_completion_tokens), model.maxTokens);
  }
  if (typeof entry.context_window === "number" && entry.context_window > 0) {
    contextWindow = entry.context_window;
  }
  if (typeof entry.max_completion_tokens === "number" && entry.max_completion_tokens > 0) {
    maxTokens = entry.max_completion_tokens;
  }
  if (contextWindow === model.contextWindow && maxTokens === model.maxTokens) return model;
  return { ...model, contextWindow, maxTokens };
}

// ---------------------------------------------------------------------------
// Model conversion
// ---------------------------------------------------------------------------

/**
 * pi ships its own OpenRouter model list with transport metadata (api,
 * baseUrl, thinking-level maps, compat quirks, provider headers). Our
 * catalog entries merge over that metadata so converted models keep
 * everything pi knows about serving them.
 */
const BUILTIN_OPENROUTER_MODELS = new Map(getModels("openrouter").map((model) => [model.id, model]));

function withBuiltinServingMetadata(model: ProviderModelConfig): ProviderModelConfig {
  const builtin = BUILTIN_OPENROUTER_MODELS.get(model.id);
  if (!builtin) return model;

  const merged: ProviderModelConfig = {
    ...model,
    api: model.api ?? builtin.api,
    baseUrl: model.baseUrl ?? builtin.baseUrl,
    reasoning: model.reasoning || builtin.reasoning,
  };

  if (builtin.thinkingLevelMap || model.thinkingLevelMap) {
    merged.thinkingLevelMap = {
      ...(builtin.thinkingLevelMap ?? {}),
      ...(model.thinkingLevelMap ?? {}),
    };
  }

  if (builtin.headers || model.headers) {
    merged.headers = {
      ...(builtin.headers ?? {}),
      ...(model.headers ?? {}),
    };
  }

  if (builtin.compat || model.compat) {
    merged.compat = {
      ...((builtin.compat ?? {}) as Record<string, unknown>),
      ...((model.compat ?? {}) as Record<string, unknown>),
    } as ProviderModelConfig["compat"];
  }

  return merged;
}

export function toProviderModel(m: OpenRouterModel): ProviderModelConfig {
  const input: InputType[] = supportsImages(m.architecture) ? ["text", "image"] : ["text"];
  return withBuiltinServingMetadata({
    id: m.id,
    name: m.name || m.id,
    reasoning: isReasoningModel(m),
    input,
    cost: {
      input: costOrFallback(parseCost(m.pricing?.prompt), 0),
      output: costOrFallback(parseCost(m.pricing?.completion), 0),
      cacheRead: costOrFallback(parseCost(m.pricing?.input_cache_read), 0),
      cacheWrite: costOrFallback(parseCost(m.pricing?.input_cache_write), 0),
    },
    contextWindow: m.context_length || 128_000,
    maxTokens: m.top_provider?.max_completion_tokens || 16_384,
  });
}
