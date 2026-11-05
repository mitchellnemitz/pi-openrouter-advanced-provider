# pi-openrouter-advanced-provider

An [OpenRouter](https://openrouter.ai) provider extension for the
[pi coding agent](https://github.com/earendil-works/pi) and its
[omp](https://omp.sh) fork: account-scoped live model catalog, per-model
provider routing config, just-in-time provider selection, and real context
windows. One codebase, both hosts.

## What it does

- **Live account catalog.** Syncs OpenRouter's `/models/user` endpoint (the
  models your key can actually use, including BYOK models and `~` fallback
  aliases; the public `/models` list otherwise), plus endpoint health data
  and credit balance. The interactive model picker is filtered to the
  agentic-usable capability set (`tools` + `reasoning`); catalog ids with the
  `:batch` call-method suffix are not registered, while `:free` — a price
  indicator, not a tier — remains selectable.
- **Real context windows.** The model's registered context window comes from
  the serving endpoint's selected context limit, not the catalog maximum.
  Under both pi and omp, JIT prewarming lowers the model's `contextWindow`
  and `maxTokens` before dispatch; omp refreshes its dynamic rows using the
  host-resolved credential without pinning a key. Requests also cap the
  output budget to the lower of the selected endpoint and registered model
  ceilings in the field selected by the transport. Before dispatch, a pi-ai
  token estimate checks the transcript and separate system prompt; the final
  payload hook checks outgoing messages and tool schemas with protocol
  headroom after any selection refresh. The final payload is measured in its
  OpenAI wire shape (role strings, string-or-part content, `tool_calls`),
  and the output budget is read from both `max_tokens` and
  `max_completion_tokens`, so either field is enforced against the selected
  ceiling. Requests are rejected if their output budget exceeds
  the selected ceiling or input plus output exceeds the endpoint context.
  Estimates can conservatively reject prompts that would otherwise fit.
- **Just-in-time provider selection.** See below.
- **Router cost accounting.** OpenRouter reports `"-1"` pricing for
  `openrouter/auto` and `openrouter/auto-beta` because their price varies
  with the routed model. The extension captures the billed cost from each
  router response stream and puts it on the completed assistant message,
  so pi's session total uses the actual charge instead of zero. If the
  stream does not report a cost, it leaves the message unchanged rather
  than guessing from another model's list price. (omp already applies the
  billed cost natively, so this capture path runs only under pi.)
- **Clean info channel.** Extension info messages (`/openrouter-status`,
  `/openrouter-balance`) are kept out of the LLM context, so account and
  status text never pollutes prompts.
- **Thought signature failover recovery.** Pi replays prior assistant reasoning
  as reasoning_details with encrypted signatures. When OpenRouter fails over
  between a model's backing providers (such as Gemini across Google AI Studio and
  Vertex AI), the new provider cannot validate the prior provider's signature
  and rejects the request with an invalid thought signature 400 error. The
  extension automatically catches this error, strips reasoning replay signatures
  from the request context, and retries once without signatures.

## API key

One of:

- `OPENROUTER_API_KEY` set in the environment launching the agent, or
- an OpenRouter key stored via `/login` in the host's auth store.

The extension resolves the key at load time (env first, then the auth
store). Under pi that is `~/.pi/agent/auth.json`; under omp it also peeks
read-only at omp's credential store (`~/.omp/agent/agent.db`). Without a key
no registration happens — falling back to the public catalog would silently
swap account-scoped data for the generic list.

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
  The active model's `contextWindow` and `maxTokens` are lowered to the
  selected endpoint's advertised limits.
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

- `pi/openrouter-advanced-provider.json` — shipped defaults for the selection
  tuning below.
- `~/.pi/agent/openrouter-advanced-provider.json` (pi) or
  `~/.omp/agent/openrouter-advanced-provider.json` (omp) — user overrides,
  merged per-field on top. Reload via `/openrouter-sync` after editing.

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
pi install npm:pi-openrouter-advanced-provider     # pi
omp install npm:pi-openrouter-advanced-provider    # omp
```

Or from git:

```
pi install git:github.com/mitchellnemitz/pi-openrouter-advanced-provider
```

Then reload the agent.

## Commands

- `/openrouter-sync` — refresh the catalog and re-run provider selection.
- `/openrouter-balance` — show your OpenRouter credit balance and usage.
- `/openrouter-status` — show sync state and current selections.
- `/openrouter-tier` — set the session's auto-router cost tier (interactive picker, or `low|medium|high|xhigh|max|off`).

## Development

```
npm ci
npm run typecheck
npm test
```

A synchronized `package-lock.json` is tracked, so `npm ci` installs the
exact pinned dependency tree (and is what CI runs, with npm caching).

`npm test` runs the unit tests on plain Node — no Bun needed. Node ≥ 22.18
strips TypeScript natively (`scripts/run-tests.mjs` enforces this);
`scripts/ts-resolve.mjs` remaps the sources' TypeScript-style `.js`
specifiers to `.ts` the same way the extension loader resolves them at
runtime. The full dual-host suite is in the container harness below.

The typecheck maps `@earendil-works/pi-ai` the same way pi's extension
loader resolves it at runtime, so the types you compile against are the
types that serve.

## Container test harness

`npm run test:container` exercises the extension end-to-end inside a pinned
dual-host container — no OpenRouter account or key required. The image
(`oven/bun:1.4.3-debian`) installs pi 1.5.0 and omp 18.16.0 from their
official releases, checksum-verified at build time. The mock OpenRouter API
runs **inside** the container and the container runs with `--network=none`,
so no live API call can happen and your real credentials are never mounted
or copied in — the harness seeds fixture credential stores (fixture key
`sk-or-test-container`) into an isolated tmpfs HOME.

```
npm run test:container                    # both hosts (pi + omp)
npm run test:container -- --host=pi       # single host (or --host=omp)
npm run test:container -- --no-cache      # rebuild the image without layer cache
```

`--live=false` is the only permitted `--live` setting (the harness is
mock-only); any other value exits with an error. The runner asserts against
the mock's request log (fixture key used, public catalog never fetched while
keyed), registry/session artifacts, and process exits; it exits nonzero on
any failed assertion.

In CI, the same command runs in the `container-matrix` workflow job (one leg
per host) with the Docker build cache warmed from a saved image — see
`.github/workflows/ci.yml`. Requires Docker locally.
