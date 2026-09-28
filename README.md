# pi-openrouter-advanced-provider

An [OpenRouter](https://openrouter.ai) provider extension for the
[pi coding agent](https://github.com/earendil-works/pi): account-scoped live
model catalog, per-model provider routing config, just-in-time provider
selection, and real context windows.

## What it does

- **Live account catalog.** Syncs OpenRouter's `/models/user` endpoint (the
  models your key can actually use, including BYOK models and `~` fallback
  aliases; the public `/models` list otherwise), plus endpoint health data
  and credit balance. The interactive model picker is filtered to the
  agentic-usable capability set (`tools` + `reasoning`); catalog ids with the
  `:batch` call-method suffix are not registered, while `:free` — a price
  indicator, not a tier — remains selectable.
- **Real context windows.** A model's registered context window comes from
  the serving provider's endpoint data, not the catalog maximum, so pi
  budgets and compacts against what the route actually offers.
- **Just-in-time provider selection.** See below.
- **Router cost accounting.** OpenRouter reports `"-1"` pricing for
  `openrouter/auto` and `openrouter/auto-beta` because their price varies
  with the routed model. The extension captures the billed cost from each
  router response stream and puts it on the completed assistant message,
  so pi's session total uses the actual charge instead of zero. If the
  stream does not report a cost, it leaves the message unchanged rather
  than guessing from another model's list price.
- **Clean info channel.** Extension info messages (`/openrouter-status`,
  `/openrouter-balance`) are kept out of the LLM context, so account and
  status text never pollutes prompts.

## API key

One of:

- `OPENROUTER_API_KEY` set in the environment launching pi, or
- an OpenRouter key stored via `/login` in `~/.pi/agent/auth.json`.

The extension resolves the key at load time (env first, then the auth
store). Without a key no registration happens — falling back to the public
catalog would silently swap account-scoped data for the generic list.

## Just-in-time provider selection

OpenRouter's built-in provider auto-selection options (balanced, cheapest,
highest throughput, lowest latency, tool-call quality) are single-axis
sorts. This extension computes a weighted composite instead — not from a
static config, but from the live 30-minute rolling window at the moment a
model is selected:

- At `before_agent_start` (the earliest point pi exposes after a model
  switch — pi has no model-selection event, and pre-warming happens at
  prompt submit), the extension fetches the model's per-endpoint stats (one
  frontend call, plus one for tool-call error rates) and scores every
  compliant endpoint.
- The winner is pinned for that model: the request carries
  `provider: {order: [pick], ignore: [...], allow_fallbacks: true}` — the
  single pick first, and if it becomes unavailable, OpenRouter's own
  fallback balancing takes over (minus ignored providers).
- The picked provider's `context_length` becomes the model's registered
  context window.
- If a compliant alternative is serving ≥1.75× the winner's p50 throughput,
  you get an advisory notification at switch time. Advisory only — the pick
  stands.

Scoring (weights are constants at the top of `pi/scoring.ts`):

- Gates: disabled/private endpoints, service tiers, no-`tools` endpoints,
  heavy quantization (4-bit) when any candidate serves a lighter
  quantization, and stunted context — endpoints below 50% of the median
  `context_length` of the candidate set are excluded from the pick AND added
  to the request's `ignore` list, so fallbacks cannot land on a 256K
  provider for a 1M model.
- Adaptive axes by price anchor (blended $/M, default $3.00 — tunable via
  the [selection config](#selection-tuning)):
  - **Under the anchor** (budget/workhorse models): price differences
    are pennies per million tokens and should not override speed.
    Default weights: Throughput 0.55 (consistency-adjusted: `p50 × √(p50/p99)`),
    Latency 0.20 (half inverse p50 latency, half inverse latency tail
    `p99/p50`), Tool-call quality 0.15 (soft-floor: ≤3% error rate receives
    near-full credit), Price 0.10.
  - **At or above the anchor** (flagships): price spreads are real
    dollars that justify speed tradeoffs. Default weights: Throughput 0.30,
    Latency 0.25, Tool-call quality 0.20, Price 0.25.
- Endpoints without samples are fallback-only; a model with no sampled
  endpoints picks price-first.

Ties within 5% of the top score break in descending weight order — the
tier's dominant axis decides first (throughput under both default weight
sets). See [Selection tuning](#selection-tuning) for customizing all of
it.

## Configuration

One file name, two locations — the location differentiates, nothing else:

- `pi/openrouter-advanced-provider.json` — shipped defaults for the routing
  config and the selection tuning below.
- `~/.pi/agent/openrouter-advanced-provider.json` — user overrides, merged
  per-field on top. Reload via `/openrouter-sync` after editing.

### Selection tuning

The `selection` section tunes the just-in-time provider selection
algorithm. Every field is optional and overrides only itself; these are
the shipped values:

```json
{
  "selection": {
    "enabled": true,
    "priceAnchor": 3.0,
    "weights": {
      "budget":   { "throughput": 0.55, "latency": 0.2, "toolCall": 0.15, "price": 0.1 },
      "flagship": { "throughput": 0.3, "latency": 0.25, "toolCall": 0.2, "price": 0.25 }
    }
  }
}
```

- `enabled: false` turns the client-side selection off entirely — no
  provider pinning and no stats fetches; OpenRouter's default routing
  serves every request. The `requests` entries below keep working.
- `priceAnchor` is the blended $/M price splitting budget-tier weights
  from flagship-tier weights. A model whose sampled endpoints' median
  blended price is below the anchor scores as budget, at or above it as
  flagship.
- `weights` replaces the per-tier axis weights. Values are ratios — only
  relative size matters, so `2` and `0.2` behave identically. A zero
  weight removes the axis from scoring and from tiebreaks, which is how
  you disable a data point: `{ "throughput": 1, "latency": 0,
  "toolCall": 0, "price": 0 }` pins strictly the highest-throughput
  provider. Ties within 5% of the top score break in descending weight
  order — your dominant axis decides.

Invalid values are dropped with a warning (surfaced on `/openrouter-sync`
and `/openrouter-status`); an all-zero weight set keeps the default
weights for that tier. `/openrouter-status` shows the active tuning.

The goal is to rely on the algorithm, not the file — but when you want to
force routing for a model, an entry under `requests.models` keyed by exact
OpenRouter model id bypasses the algorithm entirely:

```json
{
  "requests": {
    "defaults": { "provider": { "data_collection": "deny" } },
    "models": {
      "z-ai/glm-5.3": {
        "provider": {
          "order": ["deepinfra", "parasail"],
          "ignore": ["io-net"],
          "allow_fallbacks": true
        },
        "context_window": 1048576
      }
    }
  }
}
```

- `requests.defaults` applies to every OpenRouter request first (same
  validation as a model entry); per-model entries and the just-in-time pick
  merge on top, so a defaults-level `provider.data_collection` survives
  alongside a per-model `order`.
- The entry's `provider` object merges into every request for that model;
  other entry fields pass through as top-level request fields. Keys pi-ai
  owns (`model`, `messages`, `stream`, `tools`, `tool_choice`,
  `response_format`, `n`, `max_tokens`) are rejected at load time. The merge
  is payload-shape agnostic — `/chat/completions` and `/responses` accept
  the identical `provider` object, both validated strictly.
- `context_window` / `max_completion_tokens` are plugin-side only (never on
  the wire): at sync time a configured model's registration uses the
  explicit entry value, else the smallest value across the entry's approved
  endpoints.
- OpenRouter rejects unknown `provider` sub-keys with a 400, so the loader
  drops unknown sub-keys and type-violating values with a warning (surfaced
  on `/openrouter-sync`) instead of failing every request.
- Only real model ids get entries: router slugs (`openrouter/auto`),
  `~`-prefixed aliases, and `:batch` ids are skipped with a warning. (`:free`
  is a price indicator, not a tier — those stay registered.)

## Cost tier

The auto routers (`openrouter/auto`, `openrouter/auto-beta`) accept a named
`cost_tier` that selects a cost-percentile band of average generation cost
for the classified task: `low` [0, 20), `medium` [20, 40), `high` [40, 60),
`xhigh` [60, 80), `max` [80, 100]. `/openrouter-tier` sets that tier for the
current session; run it with no arguments in an interactive session to pick
from a visual list, or pass an argument directly:

```
/openrouter-tier max    # top band, quality-first
/openrouter-tier low    # cheapest band
/openrouter-tier off    # clear back to the workspace default
```

The selection is session-scoped like pi's `/thinking`: it resets when the
session restarts and never changes stored settings. The default state sends
nothing, so OpenRouter's workspace routing settings (including its saved
cost tier and "prevent overrides") apply untouched. On concrete models the
tier is inert — it is only attached when the request targets a router slug,
with the plugin id (`auto-router` / `auto-beta-router`) matched to that slug.
An explicit selection replaces any auto-router plugin entry from the
[configuration](#configuration) layers; unrelated plugin entries are
preserved.

## Install

```
pi install git:github.com/mitchellnemitz/pi-openrouter-advanced-provider
```

Then reload pi.

## Commands

- `/openrouter-sync` — refresh the catalog and re-run provider selection.
- `/openrouter-balance` — show your OpenRouter credit balance and usage.
- `/openrouter-status` — show sync state and current selections.
- `/openrouter-tier` — set the session's auto-router cost tier (interactive picker, or `low|medium|high|xhigh|max|off`).

## Development

```
npm install
npm run typecheck
npm test
```

The typecheck maps `@earendil-works/pi-ai` the same way pi's extension
loader resolves it at runtime, so the types you compile against are the
types that serve.
