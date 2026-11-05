# Changelog

Notable changes to pi-openrouter-advanced-provider.

## [Unreleased]

### omp compatibility

The extension now runs on both hosts from one codebase: the pi coding agent
(1.5.0, `@earendil-works/*`) and omp (18.16.0, `oh-my-pi`),
a fork with a legacy-compat loader but divergent credential storage,
registration validation, and transport dispatch.

What was done:

- **Platform adapter** (`pi/platform.ts`): host detection from the
  ExtensionAPI itself (omp-only `registerComposerShape`/`arktype` markers —
  directory probing misclassifies dual-installed machines), agent-dir
  resolution (`PI_CODING_AGENT_DIR` / host config dir), and load-time key
  resolution: env → host auth store (`readStoredCredential`) → `auth.json`.
  omp keys live in its SQLite credential DB; the host API is used, never raw
  DB access.
- **Dynamic provider registration under omp**: omp requires `baseUrl` +
  `apiKey` whenever static models are registered, and a config `apiKey`
  shadows stored `/login` credentials in omp's key cascade (rotation and
  logout would be ignored). The extension therefore registers with
  `fetchDynamicModels` and no static list — the catalog is fetched with the
  host-resolved key on every refresh, and `/openrouter-sync` maps to
  `refreshProvider`. Rotation and logout stay authoritative.
- **Recovery-wrapper engagement under omp**: omp dispatches by the
  per-model `api` and reserves built-in api names for custom registration,
  so the provider registers under the unique api id
  `openrouter-recovery-completions` and every dynamic row is remapped to
  it; the wrapper hands off to omp's dispatcher with the model's `api`
  rewritten to `openai-completions`, getting omp's native option
  normalization (and avoiding recursion through its custom-API registry).
- **Cost accounting is native under omp**: omp's OpenRouter transport
  already stamps the billed `usage.cost` onto messages, so the fetch-tee +
  `message_end` cost-restore path runs only under pi.
- **No-key sessions no longer masquerade the public catalog**: without an
  OpenRouter key, the account-scoped sync is skipped entirely (pi keeps its
  builtin OpenRouter list; omp's fetcher returns an empty row set) instead
  of registering the generic `/models` list under this provider's name —
  which silently swapped account-scoped data for models the key cannot use.
  A manual `/openrouter-sync` without a key now points at
  `/login openrouter`.
- **Registration/build**: `omp.extensions` manifest field added; peer deps
  retargeted to the `~1.5.0` minor line (the generation actually verified by
  the container harness), with devDeps following so typecheck compiles
  against pi-ai 1.5.0 types — a regenerated, tracked `package-lock.json`
  now pins the dependency tree to that generation instead of the stale one
  that had typecheck resolving against pi-ai 0.74; `npm test` runs on plain
  Node (>= 22.18) via `scripts/run-tests.mjs`, with `scripts/ts-resolve.mjs`
  remapping the sources' `.js`→`.ts` specifiers — no Bun required. Because
  the lockfile is tracked, CI installs with `npm ci` and npm caching.

- **Selected endpoint limits on both hosts**: JIT prewarming lowers the
  registered `contextWindow` and `maxTokens` to the selected endpoint's
  advertised ceilings. Under omp, the extension refreshes its dynamic model
  rows through the host registry using the host-resolved credential; it does
  not pin credentials. The outgoing output budget is capped to the lower of
  selected endpoint and registered model output ceilings in the field
  selected by the transport. A pre-dispatch pi-ai token estimate checks the
  transcript and separate system prompt; the final payload hook checks the
  actual outgoing messages and tool schemas, plus protocol headroom, after
  any selection refresh. The final payload is measured in its serialized
  OpenAI wire shape (plain role strings, string-or-part content, `tool_calls`)
  rather than cast to pi message blocks. The guard reads both `max_tokens` and
  `max_completion_tokens` and enforces the selected output ceiling. Requests
  exceeding either endpoint context or completion ceilings are rejected.
  Estimates can conservatively reject prompts that would otherwise fit.

### Container test harness and CI

- **`npm run test:container`**: a one-command, reproducible harness that
  builds a pinned dual-host image (`oven/bun:1.4.3-debian` with pi 1.5.0 and
  omp 18.16.0, checksum-verified from the official release checksums), boots
  both hosts headlessly against a mock OpenRouter server, and asserts the
  compatibility behaviors found during the port: host detection, key
  resolution, account-scoped catalog sync, just-in-time selection, recovery
  wrapper engagement (via the mock `X-Title` header), cost accounting, and
  the no-key masquerade guard. Mock only — the container runs with
  `--network=none`, no credentials are mounted, and `--live=false` is the
  only permitted mode. Flags: `--host=pi|omp|both`, `--no-cache`.
- **GitHub Actions** (`.github/workflows/ci.yml`): a `gates` job
  (typecheck + unit tests) and a `container-matrix` job running the same
  harness command once per host. No secrets; the Docker build cache is
  warmed by saving the built image (with inline cache metadata) to the
  GitHub Actions cache and restoring it on later runs.
