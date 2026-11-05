/**
 * Shared fixture contract for the containerized mock OpenRouter harness.
 *
 * Imported by:
 *   - test/mock-openrouter.mjs   (the mock service)
 *   - scripts/seed-omp-db.mjs    (omp agent.db credential seeding)
 *   - the per-host scenario script (assertions read these constants)
 *
 * Everything here is deterministic and free of real credentials.
 */

/** Bearer key the mock accepts on keyed /api/v1 endpoints; also seeded into auth stores. */
export const FIXTURE_KEY = "sk-or-test-container";

/**
 * Second fixture bearer key, used by the discovery-overlap scenario to give
 * two concurrent refreshes two DIFFERENT credentials. It is additive: FIXTURE_KEY
 * keeps its exact meaning, so every existing case is unaffected.
 */
export const FIXTURE_KEY_B = "sk-or-test-container-b";

/** The glm-like fixture model a scenario turn targets. */
export const MODEL_ID = "z-ai/glm-container";

/** Router slug exercising the pi router-cost restore path. */
export const ROUTER_ID = "openrouter/auto";

/** Sole content delta the SSE stream emits. */
export const RESPONSE_TEXT = "CONTAINER_MOCK_OK";

/** Usage chunk cost, asserted to land in both hosts' accounting. */
export const COST = 0.012345;

/** X-Title header value the extension registers (pi/stream.ts APP_TITLE). */
export const APP_TITLE = "openrouter-advanced-provider";

/**
 * Account-scoped catalog served by GET /api/v1/models/user — ten agentic
 * text models, all tools-capable and advertising the reasoning_effort
 * parameter (deterministic reasoning signal — no ":thinking", "-r1",
 * "o1-", "qwq-", "reasoner" id/name markers), so conversion is
 * deterministic.
 */
const TOOLS_PARAMETERS = ["tools", "temperature", "top_p", "stop"];

function catalogRow(overrides) {
  return {
    created: 1700000000,
    description: "Container fixture model.",
    architecture: { input_modalities: ["text"], modality: "text->text" },
    supported_parameters: TOOLS_PARAMETERS,
    ...overrides,
  };
}

export const ACCOUNT_MODELS = [
  catalogRow({
    id: MODEL_ID,
    canonical_slug: `${MODEL_ID}-20261001`,
    name: "GLM Container",
    context_length: 1048576,
    top_provider: { max_completion_tokens: 32768 },
    pricing: { prompt: "0.000003", completion: "0.000008", input_cache_read: "0.0000003", input_cache_write: "0.0000015" },
  }),
  catalogRow({
    id: ROUTER_ID,
    canonical_slug: "openrouter/auto",
    name: "Auto (Container)",
    context_length: 131072,
    top_provider: { max_completion_tokens: 32768 },
    // "-1" sentinels: routed price varies, must stay unpriced (models.ts parseCost).
    pricing: { prompt: "-1", completion: "-1" },
  }),
  catalogRow({
    id: "anthropic/claude-container",
    name: "Claude Container",
    context_length: 200000,
    top_provider: { max_completion_tokens: 8192 },
    pricing: { prompt: "0.000004", completion: "0.000012" },
  }),
  catalogRow({
    id: "openai/gpt-container",
    name: "GPT Container",
    context_length: 128000,
    top_provider: { max_completion_tokens: 16384 },
    pricing: { prompt: "0.0000025", completion: "0.00001" },
  }),
  catalogRow({
    id: "google/gemini-container",
    name: "Gemini Container",
    context_length: 1000000,
    top_provider: { max_completion_tokens: 8192 },
    pricing: { prompt: "0.000001", completion: "0.000004" },
  }),
  catalogRow({
    id: "meta-llama/llama-container",
    name: "Llama Container",
    context_length: 131072,
    top_provider: { max_completion_tokens: 4096 },
    pricing: { prompt: "0.0000006", completion: "0.0000022" },
  }),
  catalogRow({
    id: "mistralai/mistral-container",
    name: "Mistral Container",
    context_length: 131072,
    top_provider: { max_completion_tokens: 8192 },
    pricing: { prompt: "0.000002", completion: "0.000006" },
  }),
  catalogRow({
    id: "deepseek/deepseek-container",
    name: "DeepSeek Container",
    context_length: 65536,
    top_provider: { max_completion_tokens: 8192 },
    pricing: { prompt: "0.0000015", completion: "0.000005" },
  }),
  catalogRow({
    id: "qwen/qwen-container",
    name: "Qwen Container",
    context_length: 131072,
    top_provider: { max_completion_tokens: 8192 },
    pricing: { prompt: "0.0000018", completion: "0.0000062" },
  }),
  catalogRow({
    id: "moonshotai/kimi-container",
    name: "Kimi Container",
    context_length: 131072,
    top_provider: { max_completion_tokens: 8192 },
    pricing: { prompt: "0.0000022", completion: "0.0000075" },
  }),
  // Every account row advertises reasoning_effort: a deterministic reasoning
  // signal (no ":thinking"/"-r1"/"o1-"/"qwq-"/"reasoner" id/name markers), so
  // all ten classify as agentic-reasoning models.
].map((row) => ({ ...row, supported_parameters: [...row.supported_parameters, "reasoning_effort"] }));

/**
 * Account catalog served to FIXTURE_KEY_B. Distinct ids from ACCOUNT_MODELS so
 * a scenario can prove which credential's rows ended up in the host registry.
 * Same shape (ten agentic rows) as ACCOUNT_MODELS, so counts stay comparable.
 */
export const ACCOUNT_MODELS_B = ACCOUNT_MODELS.map((row) => ({
  ...row,
  id: `${row.id}-b`,
  canonical_slug: `${row.canonical_slug ?? `${row.id}-20261001`}-b`,
  name: `${row.name} B`,
}));

/** Public catalog served by GET /api/v1/models — deliberately disjoint from the account list. */
export const PUBLIC_MODELS = [
  catalogRow({
    id: "public/alpha-container",
    name: "Public Alpha",
    context_length: 8192,
    top_provider: { max_completion_tokens: 2048 },
    pricing: { prompt: "0", completion: "0" },
  }),
  catalogRow({
    id: "public/beta-container",
    name: "Public Beta",
    context_length: 16384,
    top_provider: { max_completion_tokens: 4096 },
    pricing: { prompt: "0", completion: "0" },
  }),
  catalogRow({
    id: "public/gamma-container",
    name: "Public Gamma",
    context_length: 32768,
    top_provider: { max_completion_tokens: 4096 },
    pricing: { prompt: "0", completion: "0" },
  }),
];

/** slug (model id) -> permaslug for GET /api/frontend/v1/catalog/models. */
export const PERMASLUGS = new Map(
  ACCOUNT_MODELS.map((m) => [m.id, m.canonical_slug ?? `${m.id}-20261001`]),
);

/**
 * Deterministic frontend stats rows per permaslug. Shapes mirror
 * StatEndpoint (pi/types.ts, verified 2026-09) and pass scoreProviders'
 * gates: status 0, non-tier provider_slug, explicit non-heavy
 * quantization, "tools" parameter, stats.request_count >= SAMPLE_GATE
 * (100) and p50_throughput > 0 so the sampled branch scores.
 *
 * Row 1 wins deterministically (highest effective throughput, lowest
 * latency); row 2 is the slower healthy alternative; row 3 is heavy
 * quantization (filtered by hasBetterQuant); row 4 is a tier endpoint
 * (excluded by isTier).
 */
function statsRows(modelId) {
  const k = PERMASLUGS.get(modelId) ?? modelId;
  const params = [...TOOLS_PARAMETERS];
  return [
    {
      id: `${k}-lambda`,
      provider_slug: "lambda",
      provider_name: "Lambda",
      provider_display_name: "Lambda",
      context_length: modelId === MODEL_ID ? 524288 : 131072,
      max_completion_tokens: modelId === MODEL_ID ? 512 : 32768,
      quantization: "fp8",
      pricing: { prompt: "0.000003", completion: "0.000008", input_cache_read: "0.0000003", discount: 0 },
      supported_parameters: params,
      status: 0,
      stats: {
        p50_latency: 0.35, p75_latency: 0.42, p90_latency: 0.5, p95_latency: 0.52, p99_latency: 0.55,
        p50_throughput: 180, p99_throughput: 240,
        request_count: 5400, latency_request_count: 5400, throughput_request_count: 5400,
        window_minutes: 30,
      },
    },
    {
      id: `${k}-gmicloud`,
      provider_slug: "gmicloud",
      provider_name: "GMICloud",
      provider_display_name: "GMICloud",
      context_length: modelId === MODEL_ID ? 524288 : 131072,
      max_completion_tokens: modelId === MODEL_ID ? 512 : 32768,
      quantization: "fp8",
      pricing: { prompt: "0.0000024", completion: "0.000007", input_cache_read: "0.0000002", discount: 0 },
      supported_parameters: params,
      status: 0,
      stats: {
        p50_latency: 0.9, p75_latency: 1.1, p90_latency: 1.3, p95_latency: 1.4, p99_latency: 1.6,
        p50_throughput: 90, p99_throughput: 130,
        request_count: 700, latency_request_count: 700, throughput_request_count: 700,
        window_minutes: 30,
      },
    },
    {
      id: `${k}-parasail-nvfp4`,
      provider_slug: "parasail",
      provider_name: "Parasail",
      provider_display_name: "Parasail",
      context_length: modelId === MODEL_ID ? 524288 : 131072,
      max_completion_tokens: modelId === MODEL_ID ? 512 : 32768,
      quantization: "nvfp4",
      pricing: { prompt: "0.000001", completion: "0.000003", input_cache_read: "0.0000001", discount: 0 },
      supported_parameters: params,
      status: 0,
      stats: {
        p50_latency: 0.2, p99_latency: 0.3,
        p50_throughput: 400, p99_throughput: 500,
        request_count: 900, latency_request_count: 900, throughput_request_count: 900,
        window_minutes: 30,
      },
    },
    {
      id: `${k}-lambda-fast`,
      provider_slug: "lambda/fast",
      provider_name: "Lambda",
      provider_display_name: "Lambda",
      context_length: modelId === MODEL_ID ? 524288 : 131072,
      max_completion_tokens: modelId === MODEL_ID ? 512 : 32768,
      quantization: "fp8",
      pricing: { prompt: "0.000006", completion: "0.000016", input_cache_read: "0.0000006", discount: 0 },
      supported_parameters: params,
      status: 0,
      stats: {
        p50_latency: 0.3, p99_latency: 0.4,
        p50_throughput: 220, p99_throughput: 260,
        request_count: 1200, latency_request_count: 1200, throughput_request_count: 1200,
        window_minutes: 30,
      },
    },
  ];
}

export function endpointStatsFor(permaslug) {
  const modelId = [...PERMASLUGS.entries()].find(([, p]) => p === permaslug)?.[0];
  return modelId ? statsRows(modelId) : null;
}

/** Tool-call error-rate series per permaslug: { data: [{ x, y: { <endpoint id>: rate } }] }. */
export function toolCallErrorRateFor(permaslug) {
  const rows = endpointStatsFor(permaslug);
  if (!rows) return null;
  const y = Object.fromEntries(rows.filter((r) => !r.provider_slug.includes("/")).map((r) => [r.id, 0.4]));
  return { data: [{ x: "2026-11-04", y }] };
}

/** Per-model /api/v1 endpoints list (fetchModelEndpoints shape). */
export function modelEndpointsFor(modelId) {
  const rows = endpointStatsFor(PERMASLUGS.get(modelId) ?? modelId);
  if (!rows) return null;
  return rows
    .filter((r) => !r.provider_slug.includes("/"))
    .map((r) => ({
      name: r.provider_name,
      tag: r.provider_slug,
      context_length: r.context_length,
      max_completion_tokens: r.max_completion_tokens,
      quantization: r.quantization,
      pricing: {
        prompt: r.pricing.prompt,
        completion: r.pricing.completion,
        input_cache_read: r.pricing.input_cache_read,
      },
      supported_parameters: r.supported_parameters,
      status: r.status,
    }));
}
