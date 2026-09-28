import type {
  ScoredProvider,
  SelectionResult,
  SelectionWeights,
  StatEndpoint,
  SelectionTuning,
} from "./types.js";

type Row = StatEndpoint & { _teff?: number; _tail?: number };

/**
 * Just-in-time provider selection: adaptive weights with a user-tunable
 * price anchor.
 *
 * Below the anchor (budget tier — Luna, GLM-5.3, Flash, DeepSeek, Qwen):
 *   - Price differences are pennies per million tokens and should not override speed.
 *   - Default weights: Throughput 0.55, Latency 0.20, Tool-call 0.15, Price 0.10.
 *
 * At or above the anchor (flagship tier — Terra, Sonnet, Opus):
 *   - Price spreads are real dollars ($1-4+/M) that justify speed tradeoffs.
 *   - Default weights: Throughput 0.30, Latency 0.25, Tool-call 0.20, Price 0.25.
 *
 * All of it is tunable per config file (`selection` section): the anchor,
 * each weight set, and an enable/disable switch. Weight values are ratios —
 * only their relative sizes matter, and a zero weight removes the axis from
 * scoring AND tiebreaks (single-axis tuning is `{throughput: 1, rest: 0}`).
 *
 * Gates (applied in order — a gated endpoint is out entirely, never tunable):
 *   - disabled / private endpoints
 *   - service tiers (flex/fast/priority/highspeed) — never pin
 *   - no `tools` in supported_parameters
 *   - heavy quantization (fp4/int4/mxfp4/nvfp4) when any candidate serves
 *     the model in a lighter quantization
 *   - stunted context: context_length below 50% of the median across
 *     candidates — such providers are also returned in `ignore` so
 *     OpenRouter's fallbacks cannot land on them
 */

const LATENCY_P50_WEIGHT = 0.5; // within the latency axis: p50 vs tail
const TIE = 0.05;
const STUNTED_RATIO = 0.5;
const SAMPLE_GATE = 100;
const ADVISORY_TPS_RATIO = 1.75;

const AXIS_ORDER = ["throughput", "latency", "toolCall", "price"] as const;
type Axis = (typeof AXIS_ORDER)[number];

/**
 * Tiebreak priority: descending weight among positively-weighted axes,
 * ties in declaration order. The default weights reproduce the shipped
 * tiebreaks (budget favored throughput first, flagship favored price
 * ahead of tool-call via its heavier weight).
 */
export function tiebreakAxes(weights: SelectionWeights): Axis[] {
  return [...AXIS_ORDER]
    .sort((a, b) => weights[b] - weights[a])
    .filter((axis) => weights[axis] > 0);
}

const TIER_SEGMENTS = new Set(["flex", "fast", "priority", "highspeed"]);
const HEAVY_QUANT = new Set(["fp4", "nvfp4", "int4", "mxfp4"]);
function quantOf(e: StatEndpoint): string {
  return (e.quantization || "unknown").toLowerCase();
}

function isTier(e: StatEndpoint): boolean {
  const last = (e.provider_slug || "").split("/").pop() ?? "";
  return TIER_SEGMENTS.has(last.toLowerCase());
}

function blendedPrice(e: StatEndpoint): number | undefined {
  const p = e.pricing?.prompt;
  const c = e.pricing?.completion;
  if (p === undefined || c === undefined) return undefined;
  const np = Number(p);
  const nc = Number(c);
  // Malformed strings would poison the median and every normalized score
  // downstream with NaN; treat them as unpriced.
  if (!Number.isFinite(np) || !Number.isFinite(nc)) return undefined;
  return (3 * np + nc) / 4;
}

function minMax(values: number[], v: number): number {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return max > min ? (v - min) / (max - min) : 1;
}

function invMinMax(values: number[], v: number): number {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return max > min ? (max - v) / (max - min) : 1;
}

/** True median: the average of the two middle values for even-length input. */
function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Soft-floor scoring for tool-call error rates.
 * Error rates under 3.0% are solid passes (>97% reliability) and receive 0.90 to 1.0
 * to avoid amplifying small statistical fluctuations between reliable hosts.
 */
function toolQualityScore(rate: number | undefined): number {
  if (rate === undefined) return 0.85; // neutral default when unmeasured
  if (rate <= 3.0) return 1.0 - rate / 30.0;
  return Math.max(0.0, 0.9 - (rate - 3.0) / 10.0);
}

export function scoreProviders(
  modelId: string,
  endpoints: StatEndpoint[],
  toolCallRates: Map<string, number>,
  tuning: SelectionTuning,
): SelectionResult | undefined {
  // ---- gates ----
  let candidates = endpoints.filter(
    (e) =>
      !e.is_disabled &&
      !e.is_private &&
      e.provider_slug &&
      e.status === 0 &&
      !isTier(e) &&
      (e.supported_parameters ?? []).includes("tools"),
  );
  if (candidates.length === 0) {
    // Nothing passed the full gate set. Drop only the `tools` requirement
    // (endpoint stats can under-report it) — never re-admit disabled,
    // private, down, or tier endpoints just to produce a pick. If this set
    // is empty too, scored ends up empty and the caller treats it as no
    // selection (OpenRouter's default routing serves the request).
    candidates = endpoints.filter(
      (e) => !e.is_disabled && !e.is_private && e.provider_slug && e.status === 0 && !isTier(e),
    );
  }

  // "unknown" is not evidence of a lighter quantization — only an explicit
  // non-heavy level (bf16/fp8/...) proves one exists.
  const hasBetterQuant = candidates.some((e) => {
    const q = quantOf(e);
    return q !== "unknown" && !HEAVY_QUANT.has(q);
  });
  if (hasBetterQuant) {
    candidates = candidates.filter((e) => !HEAVY_QUANT.has(quantOf(e)));
  }

  // ---- stunted context: below half the median context of the candidate set ----
  const contexts = candidates
    .map((e) => e.context_length)
    .filter((v): v is number => typeof v === "number" && v > 0)
    .sort((a, b) => a - b);
  const medianContext = median(contexts);
  // Endpoint-level exclusion: only the stunted variants drop out of
  // candidacy. A provider joins the request `ignore` list solely when ALL
  // of its candidate rows were stunted — one low-context quantization must
  // not disqualify the same provider's healthy variant, and fallbacks only
  // need steering away from providers with no healthy row at all.
  const stuntedRows = new Set<StatEndpoint>();
  const ignoreProviders = new Set<string>();
  if (medianContext) {
    for (const e of candidates) {
      if (e.context_length && e.context_length < STUNTED_RATIO * medianContext) {
        stuntedRows.add(e);
      }
    }
    if (stuntedRows.size > 0) {
      const byProvider = new Map<string, StatEndpoint[]>();
      for (const e of candidates) {
        const p = e.provider_slug.split("/")[0];
        const rows = byProvider.get(p) ?? [];
        rows.push(e);
        byProvider.set(p, rows);
      }
      for (const [provider, rows] of byProvider) {
        if (rows.every((r) => stuntedRows.has(r))) ignoreProviders.add(provider);
      }
      candidates = candidates.filter((e) => !stuntedRows.has(e));
    }
  }

  // ---- sampled vs fallback-only ----
  const sampled = candidates.filter(
    (e) => (e.stats?.request_count ?? 0) >= SAMPLE_GATE && (e.stats?.p50_throughput ?? 0) > 0,
  );

  // ---- scoring ----
  const scored: Array<ScoredProvider & { rawTps: number; mct?: number }> = [];
  if (sampled.length > 0) {
    const tEffs: number[] = [];
    const latP50s: number[] = [];
    const tails: number[] = [];
    for (const e of sampled as Row[]) {
      const s = e.stats!;
      const p50t = s.p50_throughput ?? 0;
      const p99t = s.p99_throughput || p50t;
      const p50l = s.p50_latency || 1;
      const p99l = s.p99_latency || p50l;
      e._teff = p50t * Math.sqrt(p50t / p99t);
      e._tail = p99l / Math.max(p50l, 1);
      tEffs.push(e._teff);
      latP50s.push(p50l);
      tails.push(e._tail);
    }
    // Anchor the price axes on the SAMPLED rows only: unsampled endpoints are
    // fallback-only, and letting an unmeasured cheap row into the median can
    // flip an expensive model under the budget anchor and distort every
    // sampled row's price score.
    const prices = sampled
      .map((e) => blendedPrice(e))
      .filter((v): v is number => v !== undefined);
    const medianPrice = median(prices);
    const priceCap = medianPrice ? 1.5 * medianPrice : undefined;
    const cheapest = prices.length > 0 ? Math.min(...prices) : undefined;

    const isBudget =
      medianPrice !== undefined && medianPrice * 1e6 < tuning.priceAnchor;
    const weights = isBudget ? tuning.budgetWeights : tuning.flagshipWeights;
    const tieOrder = tiebreakAxes(weights);

    for (const e of sampled as Row[]) {
      const tNorm = minMax(tEffs, e._teff ?? 0);
      const latP50Norm = invMinMax(latP50s, e.stats!.p50_latency || 1);
      const tailNorm = invMinMax(tails, e._tail ?? 1);
      const latencyScore = LATENCY_P50_WEIGHT * latP50Norm + (1 - LATENCY_P50_WEIGHT) * tailNorm;

      const rate = toolCallRates.get(e.id);
      const toolScore = toolQualityScore(rate);

      const price = blendedPrice(e);
      let priceScore = 0.5;
      if (price !== undefined && priceCap !== undefined && cheapest !== undefined) {
        priceScore = priceCap > cheapest
          ? Math.max(0, (priceCap - price) / (priceCap - cheapest))
          : price <= priceCap
            ? 1
            : 0;
      }

      const score =
        weights.throughput * tNorm +
        weights.latency * latencyScore +
        weights.toolCall * toolScore +
        weights.price * priceScore;

      scored.push({
        tag: e.provider_slug,
        contextLength: e.context_length,
        quantization: quantOf(e),
        blendedPrice: price,
        tpsP50: e.stats!.p50_throughput,
        latP50: e.stats!.p50_latency,
        tailRatio: e._tail,
        toolCallErrorRate: rate,
        score,
        rawTps: e.stats!.p50_throughput ?? 0,
        mct: e.max_completion_tokens,
      });
    }
    const topScore = Math.max(...scored.map((r) => r.score ?? 0));
    scored.sort((a, b) => {
      // Ties are scores within 5% of the top (relative, per the README) — an
      // absolute 0.05 gap would tie 0.20 vs 0.15, a 25% difference.
      const aTied = topScore > 0 ? (topScore - (a.score ?? 0)) / topScore <= TIE : true;
      const bTied = topScore > 0 ? (topScore - (b.score ?? 0)) / topScore <= TIE : true;
      if (aTied !== bTied) return aTied ? -1 : 1;
      if (aTied && bTied) {
        // Weight-desc axis order: the user's dominant axis decides ties.
        // Undefined values carry no ordering information — the axis is
        // skipped rather than letting undefined sort as free or fastest.
        for (const axis of tieOrder) {
          if (axis === "throughput") {
            if ((a.tpsP50 ?? 0) !== (b.tpsP50 ?? 0)) return (b.tpsP50 ?? 0) - (a.tpsP50 ?? 0);
          } else if (axis === "latency") {
            if ((a.latP50 ?? 0) !== (b.latP50 ?? 0)) return (a.latP50 ?? 0) - (b.latP50 ?? 0);
          } else if (axis === "price") {
            if (a.blendedPrice !== undefined && b.blendedPrice !== undefined
                && a.blendedPrice !== b.blendedPrice) {
              return a.blendedPrice - b.blendedPrice;
            }
          } else if (axis === "toolCall") {
            const ar = a.toolCallErrorRate;
            const br = b.toolCallErrorRate;
            if (ar !== undefined && br !== undefined && ar !== br) return ar - br;
          }
        }
      }
      return (b.score ?? 0) - (a.score ?? 0);
    });
  } else {
    // no sampled endpoints: price-first among compliant, unpriced last
    for (const e of candidates) {
      scored.push({
        tag: e.provider_slug,
        contextLength: e.context_length,
        quantization: quantOf(e),
        blendedPrice: blendedPrice(e),
        rawTps: 0,
        mct: e.max_completion_tokens,
      });
    }
    // An endpoint with unknown pricing is not free: priced rows sort first,
    // unpriced rows after them (insertion order among unpriced).
    scored.sort((a, b) => {
      if (a.blendedPrice === undefined && b.blendedPrice === undefined) return 0;
      if (a.blendedPrice === undefined) return 1;
      if (b.blendedPrice === undefined) return -1;
      return a.blendedPrice - b.blendedPrice;
    });
  }

  // An empty score table (no endpoints at all, or every candidate gated out
  // and the relaxed fallback also empty) is no selection: returning a result
  // with an empty pick would register a zero context window and send
  // provider.order: [""] on the wire.
  if (scored.length === 0) return undefined;

  const pickRow = scored[0];
  const pick = pickRow?.tag ?? "";

  // ---- advisory: a compliant alternative much faster right now ----
  let advisory: string | undefined;
  if (pickRow && sampled.length > 1) {
    const bestAlt = scored
      .slice(1)
      .filter((r) => r.rawTps > 0)
      .sort((a, b) => b.rawTps - a.rawTps)[0];
    if (bestAlt && bestAlt.rawTps >= ADVISORY_TPS_RATIO * (pickRow.rawTps || 0) && pickRow.rawTps > 0) {
      advisory = `${modelId}: picked ${pick} (${Math.round(pickRow.rawTps)} tok/s) but ${bestAlt.tag} is serving ${Math.round(bestAlt.rawTps)} tok/s right now`;
    }
  }

  return {
    pick,
    // "unknown" constrains nothing meaningful — only pin a real quant level.
    quantization: pickRow && pickRow.quantization !== "unknown" ? pickRow.quantization : undefined,
    ignore: [...ignoreProviders],
    contextLength: pickRow?.contextLength ?? medianContext ?? 0,
    // The SELECTED row's own limit — re-searching the raw endpoint list
    // could read a different variant's (disabled/stunted/other-quant row
    // of the same provider).
    maxCompletionTokens: pickRow?.mct,
    advisory,
    table: scored.map(({ rawTps: _raw, mct: _mct, ...rest }) => rest),
    sampledCount: sampled.length,
  };
}
