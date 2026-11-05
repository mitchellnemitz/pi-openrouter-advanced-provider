#!/usr/bin/env bun
/**
 * Per-host container scenario for the dual-host test harness.
 *
 * Runs INSIDE the container (no network; the mock comes up here). The
 * container runner invokes it as:
 *
 *   bun "$SCENARIO_REPO_DIR/test/container-scenario.mjs"
 *
 * Contract (set by scripts/container-test.mjs / Dockerfile entrypoint):
 *   SCENARIO_HOSTS         comma-separated subset of "pi,omp"
 *   SCENARIO_REPO_DIR      repo root in the container (pi/, test/, scripts/)
 *   SCENARIO_FIXTURE_KEY   the fixture bearer key (sk-or-test-container)
 *   SCENARIO_HOME          writable temporary HOME/native cache (tmpfs)
 *   SCENARIO_ARTIFACTS_DIR writable dir inside SCENARIO_HOME for artifacts
 *
 * pi/omp binaries are on PATH. For each requested host this script:
 *
 *   1. keyed case — isolated agent dir in the shared temporary HOME, seeded credential store
 *      (pi: agent/auth.json; omp: scripts/seed-omp-db.mjs), two one-shot
 *      turns through the host binary:
 *        a. the fixture glm model  -> catalog key resolution, wrapper
 *           engagement (X-Title), registered row set + registry api identity,
 *           JIT selection observable (frontend stats fetched with 200, and
 *           the picked provider order on the wire), streamed response text;
 *        b. the router slug        -> usage.cost lands in session accounting
 *           (pi: message_end restore; omp: native wire stamp).
 *   2. stats-down case — the same fixture-model turn against a second mock
 *      whose frontend stats endpoints answer 404: the completion still
 *      succeeds and carries NO provider order (selection skipped). Proves
 *      the keyed case's provider-order assertion is not vacuous.
 *   3. no-key case — fresh unseeded home; /openrouter-sync must exit 0 with
 *      zero requests to the mock and no extension-registered rows.
 *
 * Assertion surfaces: the mock's NDJSON request log (per-run slices), the
 * probe extension artifact (test/fixtures/probe-extension.mjs), host exit
 * codes and print output. Nothing here touches real credentials; the only
 * key ever used is the fixture key. Diagnostics name the failing assertion
 * and include output tails; exit code is nonzero on any failure.
 */

import { spawn } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  APP_TITLE,
  COST,
  MODEL_ID,
  RESPONSE_TEXT,
  ROUTER_ID,
  startMockOpenRouter,
} from "./mock-openrouter.mjs";
import { ACCOUNT_MODELS, PUBLIC_MODELS } from "./fixtures/openrouter.mjs";
import { seedOmpDb } from "../scripts/seed-omp-db.mjs";
import { runDiscoveryCases } from "./discovery-cases.mjs";

/**
 * The unique api id the provider extension registers under omp
 * (pi/stream.ts OPENROUTER_RECOVERY_API). Kept as a literal so the scenario
 * stays import-light; a drift fails the registry-api assertion below with
 * the mismatch spelled out.
 */
const OPENROUTER_RECOVERY_API = "openrouter-recovery-completions";

/**
 * omp's cross-process permit segment for our provider id:
 * base64url(sha256(provider)) — packages/ai/src/stream.ts:160-163
 * (providerInFlightSegment) at v18.16.0. The lease root is HOME-scoped, so
 * omp processes sharing one case home share their permits; the limit-one case
 * samples that directory to prove limiting stays enforced.
 */
const OPENROUTER_PROVIDER_SEGMENT = crypto.createHash("sha256").update("openrouter").digest("base64url");

// ---------------------------------------------------------------------------
// Environment contract
// ---------------------------------------------------------------------------

const repoDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const probePath = path.join(repoDir, "test", "fixtures", "probe-extension.mjs");
const HOSTS = ["pi", "omp"];
const SPAWN_TIMEOUT_MS = 240_000;
const PROMPT = "Reply with exactly one short sentence.";
const JIT_PICK = "lambda"; // provider_slug of the winning fixture stats row (test/fixtures/openrouter.mjs statsRows: row 1 wins deterministically).

function fail(message) {
  console.error(`container-scenario: ${message}`);
  process.exit(2);
}

const scenarioHome = process.env.SCENARIO_HOME;
if (!scenarioHome) fail("SCENARIO_HOME is not set — the runner must provide the writable isolated HOME");
if (!fs.existsSync(scenarioHome)) fail(`SCENARIO_HOME does not exist: ${scenarioHome}`);

const artifactsDir = process.env.SCENARIO_ARTIFACTS_DIR ?? path.join(scenarioHome, "artifacts");
fs.mkdirSync(artifactsDir, { recursive: true });

const fixtureKey = process.env.SCENARIO_FIXTURE_KEY;
if (!fixtureKey) fail("SCENARIO_FIXTURE_KEY is not set — the runner must provide the fixture bearer key");

const hosts = (process.env.SCENARIO_HOSTS ?? "")
  .split(",")
  .map((h) => h.trim().toLowerCase())
  .filter(Boolean);
if (hosts.length === 0) fail("SCENARIO_HOSTS is empty — expected a comma-separated subset of \"pi,omp\"");
for (const host of hosts) {
  if (!HOSTS.includes(host)) fail(`SCENARIO_HOSTS contains unknown host "${host}" — expected one of: ${HOSTS.join(", ")}`);
}

/**
 * The corrected fixture (test/fixtures/openrouter.mjs) carries ten account
 * rows — every non-router row advertises both "tools" and "reasoning_effort",
 * which is what pi/state.ts requires for catalog registration, so all ten
 * must survive, not just the router. Pinned here so a fixture shrink fails
 * the scenario instead of quietly narrowing these assertions.
 */
const EXPECTED_ACCOUNT_COUNT = 10;
const accountIds = ACCOUNT_MODELS.map((m) => m.id);
if (accountIds.length !== EXPECTED_ACCOUNT_COUNT) {
  fail(`account fixture drifted: expected ${EXPECTED_ACCOUNT_COUNT} account rows, got ${accountIds.length} (${accountIds.join(", ")})`);
}
if (!accountIds.includes(MODEL_ID)) {
  fail(`account fixture is missing the fixture GLM model "${MODEL_ID}"`);
}
if (!accountIds.includes(ROUTER_ID)) {
  fail(`account fixture is missing the router slug "${ROUTER_ID}"`);
}
if (PUBLIC_MODELS.some((m) => accountIds.includes(m.id))) {
  fail("account fixture leaks a public-catalog model id");
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** Collect named assertions for one case; failures print immediately. */
class Checks {
  constructor(prefix) {
    this.prefix = prefix;
    this.passed = 0;
    this.failures = [];
  }

  check = (name, ok, detail = "") => {
    if (ok) {
      this.passed += 1;
      return true;
    }
    this.failures.push({ name, detail });
    console.error(`FAIL [${this.prefix}] ${name}${detail ? `\n  ${detail}` : ""}`);
    return false;
  };
}

function clip(text, max = 1600) {
  const s = String(text ?? "");
  return s.length > max ? `${s.slice(0, max)}…[clipped]` : s;
}

/** Isolated temp home under SCENARIO_HOME; agent dir honors PI_CODING_AGENT_DIR on both hosts. */
function makeCaseHome(host, mode) {
  const caseDir = fs.mkdtempSync(path.join(scenarioHome, `${host}-${mode}-`));
  const agentDir = path.join(caseDir, "agent");
  fs.mkdirSync(agentDir, { recursive: true });
  return { caseDir, agentDir };
}

/**
 * Seed the fixture key into the host's own credential store — never a real
 * credential, never the user's home. pi reads agent/auth.json
 * ({ "<provider>": { type: "api_key", key } }); omp reads agent.db
 * (scripts/seed-omp-db.mjs, pinned 18.16 auth schema).
 */
async function seedCredentials(host, agentDir, key) {
  if (host === "pi") {
    fs.writeFileSync(
      path.join(agentDir, "auth.json"),
      `${JSON.stringify({ openrouter: { type: "api_key", key } }, null, 2)}\n`,
    );
    return;
  }
  await runHost("omp", ["--no-extensions", "--print", "initialize"], hostEnv({
    caseDir: path.dirname(agentDir), agentDir, probeArtifact: "", label: "init", baseUrl: "http://127.0.0.1:9/api/v1",
  }), path.dirname(agentDir));
  seedOmpDb(agentDir, key);
}

function hostEnv({ caseDir, agentDir, probeArtifact, label, baseUrl }) {
  const env = { ...process.env };
  // The hosts must resolve the key from the seeded store, never from env,
  // and the runtime-override test seam must stay off.
  delete env.OPENROUTER_API_KEY;
  delete env.OPENROUTER_ADVANCED_RUNTIME;
  env.HOME = scenarioHome;
  env.PI_CODING_AGENT_DIR = agentDir; // honored by both hosts
  env.OMP_CODING_AGENT_DIR = agentDir;
  env.OPENROUTER_BASE_URL = baseUrl; // aims catalog, frontend stats, and completions at the mock
  env.OPENROUTER_PROBE_ARTIFACT = probeArtifact;
  env.OPENROUTER_PROBE_LABEL = label;
  env.NO_COLOR = "1";
  env.FORCE_COLOR = "0";
  env.TERM = "dumb";
  return env;
}

/** Spawn a host one-shot; resolves { code, stdout, stderr, timedOut, missing }. */
function runHost(bin, args, env, cwd, timeoutMs = SPAWN_TIMEOUT_MS) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn(bin, args, { env, cwd, stdio: ["ignore", "pipe", "pipe"] });
    } catch (error) {
      resolve({ code: null, stdout: "", stderr: `spawn failed: ${error?.message}`, timedOut: false, missing: true });
      return;
    }
    let stdout = "";
    let stderr = "";
    child.stdout?.setEncoding("utf8");
    child.stderr?.setEncoding("utf8");
    child.stdout?.on("data", (d) => { stdout += d; });
    child.stderr?.on("data", (d) => { stderr += d; });
    let timedOut = false;
    let settled = false;
    const killer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000).unref?.();
    }, timeoutMs);
    killer.unref?.();
    const done = (code, missing = false) => {
      if (settled) return;
      settled = true;
      clearTimeout(killer);
      resolve({ code, stdout, stderr, timedOut, missing });
    };
    child.on("error", (error) => {
      stderr += `\nspawn error: ${error?.message ?? error}`;
      done(null, error?.code === "ENOENT");
    });
    child.on("close", (code) => done(code));
  });
}

/**
 * Sample omp's cross-process permit leases for our provider while turns run.
 * Each in-flight request holds one lease directory under the provider segment,
 * so under `maxInFlightRequests.openrouter: 1` the sampled count must never
 * exceed 1 — two leases at once would mean the recovery wrapper's nested
 * dispatch took a second permit, i.e. a silent limit bypass.
 *
 * The lease root is keyed by omp's CONFIG root (~/.omp, i.e. $HOME/.omp), not
 * by the per-case agent dir: the harness gives every case its own agent dir
 * (PI_CODING_AGENT_DIR) but shares one HOME, so the leases land under the
 * SHARED home. Sampling the per-case dir would observe nothing and let
 * `maxConcurrent <= 1` pass vacuously.
 *
 * Real-interval polling is required here (no deterministic substitute): the
 * observed state is live omp processes creating/removing lease directories on
 * the real filesystem.
 */
function startLeaseSampler(homeRoot, intervalMs = 5) {
  // omp's state root prefers XDG_STATE_HOME and falls back to the config root
  // (~/.omp); sample every root that exists so the sampler stays correct
  // whichever layout the container resolves. Roots are de-duplicated by real
  // path so one lease directory is never counted twice.
  const stateRoot = process.env.XDG_STATE_HOME;
  const configDir = process.env.PI_CONFIG_DIR;
  const candidateRoots = [
    path.join(homeRoot, ".omp"),
    ...(stateRoot ? [path.join(stateRoot, "omp")] : []),
    // omp composes its config root as join(homedir(), PI_CONFIG_DIR ?? ".omp"),
    // so a custom PI_CONFIG_DIR is resolved relative to HOME the same way.
    ...(configDir ? [path.join(homeRoot, configDir)] : []),
  ];
  const segmentDirs = [];
  const seenRoots = new Set();
  for (const root of candidateRoots) {
    let key;
    try {
      key = fs.realpathSync(root);
    } catch {
      key = path.resolve(root);
    }
    if (seenRoots.has(key)) continue;
    seenRoots.add(key);
    segmentDirs.push(path.join(root, "run", "provider-inflight", OPENROUTER_PROVIDER_SEGMENT));
  }
  let maxConcurrent = 0;
  let samples = 0;
  let observedLease = false;
  let observedSegment = false;
  const timer = setInterval(() => {
    let count = 0;
    for (const segmentDir of segmentDirs) {
      try {
        const entries = fs.readdirSync(segmentDir, { withFileTypes: true });
        observedSegment = true;
        // One lease DIRECTORY per in-flight request; the files inside it
        // (info.json, .lock) live under the lease, not beside it.
        count += entries.filter((e) => e.isDirectory()).length;
      } catch {
        // No segment dir yet (nothing in flight): contributes zero.
      }
    }
    samples += 1;
    if (count > 0) observedLease = true;
    if (count > maxConcurrent) maxConcurrent = count;
  }, intervalMs);
  return {
    stop: () => clearInterval(timer),
    result: () => ({ maxConcurrent, samples, observedLease, observedSegment }),
  };
}

/**
 * Spawn a host turn and SIGKILL it once `killWhen` sees the target state
 * (bounded by capMs), then resolve with the outcome. Polls the mock log
 * slice AND the probe artifact: the kill target is a live host process
 * mid-request (permit held), and the probe's before_provider_request record
 * is written at request START — mock log rows only appear once a response
 * has finished, which is always too late to catch the request in flight.
 */
function runHostUntilKill(bin, args, env, cwd, { logPath, probePath = "", killWhen, capMs = 60_000, pollMs = 100 }) {
  return new Promise((resolve) => {
    const since = logSize(logPath);
    let child;
    try {
      child = spawn(bin, args, { env, cwd, stdio: ["ignore", "pipe", "pipe"] });
    } catch (error) {
      resolve({ code: null, stdout: "", stderr: `spawn failed: ${error?.message}`, killed: false, killReason: "spawn-failed" });
      return;
    }
    let stdout = "";
    let stderr = "";
    child.stdout?.setEncoding("utf8");
    child.stderr?.setEncoding("utf8");
    child.stdout?.on("data", (d) => { stdout += d; });
    child.stderr?.on("data", (d) => { stderr += d; });
    let settled = false;
    let killed = false;
    let killReason = "exited-first";
    const startedAt = Date.now();
    const poll = setInterval(() => {
      const records = [...readLogSlice(logPath, since), ...readNdjson(probePath)];
      if (killWhen(records)) {
        killed = true;
        killReason = "request-observed";
        clearInterval(poll);
        child.kill("SIGKILL");
      } else if (Date.now() - startedAt > capMs) {
        killed = true;
        killReason = "cap";
        clearInterval(poll);
        child.kill("SIGKILL");
      }
    }, pollMs);
    const done = (code) => {
      if (settled) return;
      settled = true;
      clearInterval(poll);
      resolve({ code, stdout, stderr, killed, killReason });
    };
    child.on("error", (error) => { stderr += `\nspawn error: ${error?.message ?? error}`; done(null); });
    child.on("close", (code) => done(code));
  });
}

// Mock NDJSON log slicing: one independent slice per spawn so assertions and
// diagnostics never mix two runs.
function logSize(file) {
  try {
    return fs.statSync(file).size;
  } catch {
    return 0;
  }
}

function readLogSlice(file, since) {
  const size = logSize(file);
  if (size <= since) return [];
  const buffer = Buffer.alloc(size - since);
  const fd = fs.openSync(file, "r");
  try {
    fs.readSync(fd, buffer, 0, buffer.length, since);
  } finally {
    fs.closeSync(fd);
  }
  return buffer
    .toString("utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function readNdjson(file) {
  let text;
  try {
    text = fs.readFileSync(file, "utf8");
  } catch {
    return [];
  }
  return text
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

const isPublicCatalog = (row) => row.method === "GET" && row.path === "/api/v1/models";
const isAccountCatalog = (row) => row.method === "GET" && row.path === "/api/v1/models/user";
const isCompletion = (row) => row.method === "POST" && row.path.endsWith("/chat/completions");
const isFrontend = (row) => row.path.startsWith("/api/frontend/");

/** The frontend website APIs the JIT selection reads (pi/api.ts, unauthenticated). */
const FRONTEND_CATALOG_PATH = "/api/frontend/v1/catalog/models";
const FRONTEND_STATS_PATHS = ["/api/frontend/v1/stats/endpoint", "/api/frontend/v1/stats/tool-call-error-rate"];
const isFrontendCatalog = (row) => row.method === "GET" && row.path === FRONTEND_CATALOG_PATH;
const isFrontendStats = (row) => row.method === "GET" && FRONTEND_STATS_PATHS.includes(row.path);

function completionSummary(rows) {
  return JSON.stringify(rows.map((r) => ({ model: r.body?.model, title: r.title, auth: r.authorization })));
}

/**
 * JIT-selection assertions over one turn's mock log slice.
 *
 * The extension reads the frontend website APIs (catalog + both stats
 * endpoints) without a bearer token, exactly like the real service — so
 * those rows must be unauthenticated. `expectOrder` is the provider order
 * the completion body must carry, or null when the turn must carry none
 * (selection skipped). `expectStatsStatus` is the status the two stats
 * endpoints must answer with.
 */
function assertJitSelection(check, slice, { expectOrder, expectStatsStatus }) {
  const catalogRows = slice.filter(isFrontendCatalog);
  const statsRows = slice.filter(isFrontendStats);

  check.check(
    "JIT fetched the frontend catalog",
    catalogRows.length >= 1,
    "GET /api/frontend/v1/catalog/models missing — selection never ran",
  );
  check.check(
    "frontend catalog answered 200",
    catalogRows.length >= 1 && catalogRows.every((r) => r.status === 200),
    `catalog statuses seen: ${JSON.stringify(catalogRows.map((r) => r.status))}`,
  );
  check.check(
    "JIT fetched the frontend endpoint stats",
    statsRows.some((r) => r.path === "/api/frontend/v1/stats/endpoint"),
    `stats paths seen: ${JSON.stringify(statsRows.map((r) => r.path))}`,
  );
  if (expectStatsStatus === 200) {
    // Both stats endpoints are awaited on the success path, so both must be
    // logged. On the failure path only the endpoint-stats read is guaranteed
    // to have been logged (its sibling may still be in flight).
    check.check(
      "JIT fetched both frontend stats endpoints",
      FRONTEND_STATS_PATHS.every((p) => statsRows.some((r) => r.path === p)),
      `stats paths seen: ${JSON.stringify(statsRows.map((r) => r.path))}`,
    );
  }
  check.check(
    `frontend stats answered ${expectStatsStatus}`,
    statsRows.length >= 1 && statsRows.every((r) => r.status === expectStatsStatus),
    `stats statuses seen: ${JSON.stringify(statsRows.map((r) => ({ path: r.path, status: r.status })))}`,
  );

  const frontendRows = slice.filter(isFrontend);
  const unauthenticated = frontendRows.every((r) => r.authorization === null);
  check.check(
    "frontend calls carried no bearer token",
    unauthenticated,
    `authorizations seen: ${JSON.stringify(frontendRows.map((r) => r.authorization))}`,
  );

  const glmPosts = slice.filter((r) => isCompletion(r) && r.body?.model === MODEL_ID);
  if (glmPosts.length === 0) return null;
  const post = glmPosts[glmPosts.length - 1];

  if (expectOrder === null) {
    check.check(
      "completion carried no JIT provider order",
      post.body?.provider?.order === undefined,
      `wire provider=${JSON.stringify(post.body?.provider)} — selection must be skipped when the stats fail`,
    );
  } else {
    // The frontend reads must precede the completion they steer.
    const postIndex = slice.indexOf(post);
    const frontendIndexes = frontendRows.map((r) => slice.indexOf(r)).filter((i) => i >= 0);
    check.check(
      "frontend reads preceded the fixture-model completion",
      frontendIndexes.length > 0 && frontendIndexes.every((i) => i < postIndex),
      `frontend row indexes ${JSON.stringify(frontendIndexes)} vs completion index ${postIndex}`,
    );
    check.check(
      "completion carried the selected provider order",
      JSON.stringify(post.body?.provider?.order) === JSON.stringify(expectOrder),
      `wire provider.order=${JSON.stringify(post.body?.provider?.order)} expected ${JSON.stringify(expectOrder)}`,
    );
    check.check(
      "completion allowed provider fallbacks",
      post.body?.provider?.allow_fallbacks === true,
      `wire provider=${JSON.stringify(post.body?.provider)}`,
    );
  }

  return post;
}

// ---------------------------------------------------------------------------
// Registry/session assertions (probe artifact)
// ---------------------------------------------------------------------------

function registryRecords(records) {
  return records.filter((r) => r.type === "registry" && Array.isArray(r.rows));
}

function assistantMessages(records) {
  return records.filter((r) => r.type === "message_end").map((r) => r.message ?? {});
}

/**
 * Registry assertions per source:
 *   - snapshot (both hosts): the extension's committed catalog — exactly the
 *     account fixture set (public rows would prove the masquerade guard broke).
 *   - provider-config (pi): registered ProviderConfig — exactly the account
 *     set, no per-row api override (a per-row api would bypass the wrapper),
 *     provider-level wrapper api + streamSimple attached.
 *   - host-registry (omp): live registry rows for the provider — omp's merge
 *     keeps its native non-completions rows, so assert on the account-id
 *     subset: exactly the account set, each carrying the recovery api id.
 */
function assertRegistry(check, host, records) {
  const regs = registryRecords(records);
  check.check("probe captured registry state", regs.length > 0, "no registry records in the probe artifact");
  if (host === "pi") {
    for (const reg of regs.filter((record) => record.source === "snapshot")) {
      const ids = [...new Set(reg.rows.map((row) => row.id).filter(Boolean))];
      check.check("snapshot rows == account fixture catalog", reg.rows.length === accountIds.length && ids.length === accountIds.length && ids.every((id) => accountIds.includes(id)), `snapshot has ${reg.rows.length} rows`);
    }
    const configs = regs.filter((record) => record.source === "provider-config");
    check.check("pi registered the account catalog through the wrapper config", configs.length > 0 && configs.every((record) => record.rows.length === accountIds.length && record.providerApi === "openai-completions" && record.hasStreamSimple === true), "provider-config records missing or incomplete");
    return;
  }
  const registries = regs.filter((record) => record.source === "host-registry");
  check.check("omp registered account rows carry the recovery api", registries.length > 0 && registries.every((record) => {
    const mine = record.rows.filter((row) => accountIds.includes(row.id));
    return mine.length === accountIds.length && new Set(mine.map((row) => row.api)).size === 1 && mine[0]?.api === OPENROUTER_RECOVERY_API;
  }), "host-registry records missing or incomplete");
  const selectedRows = registries.flatMap((record) => record.rows).filter((row) => row.id === MODEL_ID);
  check.check(
    "omp selected endpoint limits are present in the live model registry",
    selectedRows.length > 0 &&
      selectedRows.at(-1)?.contextWindow === 524_288 &&
      selectedRows.at(-1)?.maxTokens === 512,
    `selected rows: ${JSON.stringify(selectedRows)}`,
  );
}

// ---------------------------------------------------------------------------
// Per-host cases
// ---------------------------------------------------------------------------

/** One one-shot turn; returns { result, logSlice, probeRecords, probeArtifact }. */
async function runTurn({ host, bin, caseDir, agentDir, mode, modelArg, prompt, baseUrl, logPath }) {
  const turnLabel = `${host}/${mode}${modelArg ? `:${modelArg}` : ""}`;
  const probeArtifact = path.join(artifactsDir, `${host}-${mode}${modelArg ? `-${san(modelArg)}` : ""}-probe.ndjson`);
  fs.rmSync(probeArtifact, { force: true });
  const env = hostEnv({ caseDir, agentDir, probeArtifact, label: turnLabel, baseUrl });
  const promptFile = prompt.length > 100_000 ? path.join(caseDir, `${mode}-prompt.txt`) : undefined;
  if (promptFile) fs.writeFileSync(promptFile, prompt);
  const args = [
    "--no-extensions",
    "-e", path.join(repoDir, "pi/index.ts"),
    "-e", probePath,
    ...(modelArg ? ["--model", modelArg] : []),
    "-p", promptFile ? `@${promptFile}` : prompt,
  ];
  const stdoutArtifact = path.join(artifactsDir, `${host}-${mode}${modelArg ? `-${san(modelArg)}` : ""}.out`);
  const stderrArtifact = stdoutArtifact.replace(/\.out$/, ".err");

  const since = logSize(logPath);
  const result = await runHost(bin, args, env, caseDir);
  const logSlice = readLogSlice(logPath, since);
  const probeRecords = readNdjson(probeArtifact);

  // Artifacts of record for offline debugging.
  fs.writeFileSync(stdoutArtifact, result.stdout);
  fs.writeFileSync(stderrArtifact, result.stderr);

  return { result, logSlice, probeRecords, probeArtifact, args, stdoutArtifact, stderrArtifact };
}

function san(value) {
  return String(value).replace(/[^a-zA-Z0-9._-]+/g, "_");
}

function exitCheck(check, turn, what) {
  const { result } = turn;
  const detail = [
    `exit=${result.code} timedOut=${result.timedOut}`,
    `args: ${JSON.stringify(turn.args)}`,
    `stdout tail: ${clip(result.stdout)}`,
    `stderr tail: ${clip(result.stderr)}`,
    `artifacts: ${turn.stdoutArtifact} / ${turn.stderrArtifact} / ${turn.probeArtifact}`,
  ].join("\n  ");
  return check.check(
    `${what} exited 0`,
    result.code === 0 && !result.timedOut && !result.missing,
    result.missing ? `host binary "${hostBin}" not found on PATH\n  ${detail}` : detail,
  );
}

let hostBin; // module-level for exitCheck diagnostics

async function runKeyedCase(host, bin, baseUrl, logPath) {
  const check = new Checks(`${host}/keyed`);
  const { caseDir, agentDir } = makeCaseHome(host, "keyed");
  await seedCredentials(host, agentDir, fixtureKey);
  const modelArg = `openrouter/${MODEL_ID}`;
  const prewarm = await runTurn({ host, bin, caseDir, agentDir, mode: "keyed", modelArg, prompt: "Warm the selected model.", baseUrl, logPath });
  exitCheck(check, prewarm, `${host} JIT prewarm`);
  const turn1 = await runTurn({ host, bin, caseDir, agentDir, mode: "keyed", modelArg, prompt: PROMPT, baseUrl, logPath });
  const ok1 = exitCheck(check, turn1, `${host} fixture-model turn`);
  if (ok1) {
    check.check(
      `${host} fixture-model turn printed the fixture response`,
      turn1.result.stdout.includes(RESPONSE_TEXT),
      `stdout tail: ${clip(turn1.result.stdout)}`,
    );

    const slice = turn1.logSlice;
    const account = slice.filter(isAccountCatalog);
    check.check(
      "account catalog fetched with the fixture key",
      account.length >= 1 && account.every((r) => r.authorization === fixtureKey),
      account.length === 0
        ? "GET /api/v1/models/user never hit the mock — key resolution failed"
        : `authorizations seen: ${JSON.stringify(account.map((r) => r.authorization))}`,
    );
    check.check(
      "public catalog never requested while keyed",
      !slice.some(isPublicCatalog),
      `public-catalog requests: ${JSON.stringify(slice.filter(isPublicCatalog).map((r) => r.path))}`,
    );

    const glmPosts = slice.filter((r) => isCompletion(r) && r.body?.model === MODEL_ID);
    check.check(
      "fixture-model completion reached the mock",
      glmPosts.length >= 1,
      `completion POSTs: ${completionSummary(slice.filter(isCompletion))}`,
    );
    if (glmPosts.length >= 1) {
      const post = glmPosts[glmPosts.length - 1];
      check.check(
        "X-Title wrapper header on completion",
        post.title === APP_TITLE,
        `x-title=${JSON.stringify(post.title)} expected ${JSON.stringify(APP_TITLE)}`,
      );
      check.check("fixture key used on completion", post.authorization === fixtureKey, `authorization=${JSON.stringify(post.authorization)}`);
      const outputBudget = post.body?.max_completion_tokens ?? post.body?.max_tokens;
      check.check(
        "fixture-model completion stayed within selected endpoint output limit",
        typeof outputBudget === "number" && outputBudget <= 512,
        `wire output budget=${JSON.stringify(outputBudget)} expected <= 512`,
      );
    }

    // Real just-in-time verification: the frontend reads must have succeeded
    // unauthenticated and the completion must carry the provider order the
    // fixture stats produce (JIT_PICK). A logged-but-401 frontend read is
    // NOT a selection — it silently yields nothing.
    assertJitSelection(check, slice, { expectOrder: [JIT_PICK], expectStatsStatus: 200 });

    const glmMessages = assistantMessages(turn1.probeRecords).filter((m) => String(m.model ?? "").endsWith(MODEL_ID));
    check.check("probe saw the fixture-model assistant message", glmMessages.length >= 1,
      `probe artifact: ${turn1.probeArtifact}; message_end records: ${JSON.stringify(assistantMessages(turn1.probeRecords).map((m) => ({ model: m.model, stopReason: m.stopReason, error: m.errorMessage })))}`);
    if (glmMessages.length >= 1) {
      const message = glmMessages[glmMessages.length - 1];
      check.check("fixture-model response text streamed", String(message.text ?? "").includes(RESPONSE_TEXT),
        `text=${JSON.stringify(clip(message.text, 200))} stopReason=${JSON.stringify(message.stopReason)} errorMessage=${JSON.stringify(message.errorMessage)}`);
      check.check("fixture-model assistant carried the openrouter provider", message.provider === "openrouter",
        `provider=${JSON.stringify(message.provider)}`);
    }
  }

  // --- Turn 2: router slug — cost accounting (both hosts) ---
  // Model-string matching differs per host: omp matches bare registry ids
  // (live-verified), pi splits provider/id — so the id-qualified form
  // "openrouter/openrouter/auto" is the canonical pin and the bare slug is
  // the omp fallback. First candidate that posts the router completion wins.
  const routerCandidates = host === "omp" ? [ROUTER_ID, `openrouter/${ROUTER_ID}`] : [`openrouter/${ROUTER_ID}`, ROUTER_ID];
  let turn2 = null;
  for (const candidate of routerCandidates) {
    const attempt = await runTurn({ host, bin, caseDir, agentDir, mode: "keyed-router", modelArg: candidate, prompt: PROMPT, baseUrl, logPath });
    const posted = attempt.logSlice.some((r) => isCompletion(r) && r.body?.model === ROUTER_ID);
    if (attempt.result.code === 0 && posted) {
      turn2 = attempt;
      break;
    }
    turn2 = attempt; // keep the latest attempt for diagnostics when all fail
  }
  const ok2 = turn2 !== null && exitCheck(check, turn2, `${host} router-slug turn`);
  if (ok2) {
    const slice = turn2.logSlice;
    check.check("public catalog never requested while keyed (router turn)", !slice.some(isPublicCatalog),
      `public-catalog requests: ${JSON.stringify(slice.filter(isPublicCatalog).map((r) => r.path))}`);
    const routerPosts = slice.filter((r) => isCompletion(r) && r.body?.model === ROUTER_ID);
    check.check("router-slug completion reached the mock", routerPosts.length >= 1,
      `completion POSTs: ${completionSummary(slice.filter(isCompletion))}`);
    if (routerPosts.length >= 1) {
      const post = routerPosts[routerPosts.length - 1];
      check.check("X-Title wrapper header on router completion", post.title === APP_TITLE,
        `x-title=${JSON.stringify(post.title)}`);
      check.check("fixture key used on router completion", post.authorization === fixtureKey,
        `authorization=${JSON.stringify(post.authorization)}`);
      check.check(
        "router slug carried no JIT provider order",
        post.body?.provider?.order === undefined,
        `wire provider=${JSON.stringify(post.body?.provider)} — JIT must skip openrouter/ slugs`,
      );
    }
    check.check(
      "router slug made no JIT stats calls",
      !slice.some(isFrontend),
      `frontend requests on the router turn: ${JSON.stringify(slice.filter(isFrontend).map((r) => r.path))}`,
    );

    const routerMessages = assistantMessages(turn2.probeRecords).filter((m) => String(m.model ?? "").endsWith(ROUTER_ID));
    check.check("probe saw the router-slug assistant message", routerMessages.length >= 1,
      `probe artifact: ${turn2.probeArtifact}; message_end records: ${JSON.stringify(assistantMessages(turn2.probeRecords).map((m) => ({ model: m.model, stopReason: m.stopReason, error: m.errorMessage })))}`);
    if (routerMessages.length >= 1) {
      const message = routerMessages[routerMessages.length - 1];
      const billed = message.usage?.cost?.total;
      check.check(
        "router usage.cost landed in session accounting",
        typeof billed === "number" && Math.abs(billed - COST) < 1e-9,
        `usage.cost.total=${JSON.stringify(billed)} expected ${COST} (stopReason=${JSON.stringify(message.stopReason)} errorMessage=${JSON.stringify(message.errorMessage)})`,
      );
    }
  }

  // --- Registry identity (turn-1 probe artifact holds the catalog) ---
  if (ok1) {
    assertRegistry(check, host, turn1.probeRecords);
  }

  return check;
}

async function runLimitBoundaryCase(host, bin, baseUrl, logPath) {
  const check = new Checks(`${host}/limit-boundary`);
  const { caseDir, agentDir } = makeCaseHome(host, "limit-boundary");
  await seedCredentials(host, agentDir, fixtureKey);
  const turn = await runTurn({
    host,
    bin,
    caseDir,
    agentDir,
    mode: "limit-boundary",
    modelArg: `openrouter/${MODEL_ID}`,
    prompt: "boundary-token ".repeat(200_000),
    baseUrl,
    logPath,
  });
  const rejectedBySelectedEndpoint = assistantMessages(turn.probeRecords).some(
    (message) =>
      message.stopReason === "error" &&
      typeof message.errorMessage === "string" &&
      message.errorMessage.includes("Request exceeds the selected OpenRouter endpoint context limit (524288 tokens)"),
  );
  check.check(
    "boundary input was rejected by the selected 524288-token endpoint context guard",
    rejectedBySelectedEndpoint,
    `assistant errors: ${JSON.stringify(assistantMessages(turn.probeRecords).map((message) => message.errorMessage))}`,
  );
  const completions = turn.logSlice.filter((row) => isCompletion(row) && row.body?.model === MODEL_ID);
  check.check(
    "catalog-sized input above the selected endpoint context was not sent",
    completions.length === 0,
    `exit=${turn.result.code} timedOut=${turn.result.timedOut} completion POSTs=${completionSummary(completions)}`,
  );
  return check;
}

/**
 * Negative twin of the fixture-model turn: the frontend stats endpoints
 * answer 404, so the extension must skip provider selection. The completion
 * must still succeed and must carry NO provider order — if "provider order
 * present" ever became unconditional, this case fails.
 *
 * It runs against the stats-down mock (own port, own log file) in its own
 * isolated home, so neither the primary mock's log slice nor the host's
 * cached dynamic models leak into the observation.
 */
async function runStatsDownCase(host, bin, statsDown) {
  const check = new Checks(`${host}/stats-down`);
  const { caseDir, agentDir } = makeCaseHome(host, "statsdown");
  await seedCredentials(host, agentDir, fixtureKey);
  const modelArg = `openrouter/${MODEL_ID}`;

  const turn = await runTurn({
    host,
    bin,
    caseDir,
    agentDir,
    mode: "statsdown",
    modelArg,
    prompt: PROMPT,
    baseUrl: statsDown.baseUrl,
    logPath: statsDown.logPath,
  });
  const ok = exitCheck(check, turn, `${host} stats-unavailable turn`);
  if (ok) {
    check.check(
      `${host} stats-unavailable turn printed the fixture response`,
      turn.result.stdout.includes(RESPONSE_TEXT),
      `stdout tail: ${clip(turn.result.stdout)}`,
    );
    assertJitSelection(check, turn.logSlice, { expectOrder: null, expectStatsStatus: 404 });
  }
  return check;
}

async function runNoKeyCase(host, bin, baseUrl, logPath) {
  const check = new Checks(`${host}/no-key`);
  const { caseDir, agentDir } = makeCaseHome(host, "nokey");
  // Deliberately unseeded: a fresh home must yield no key, no catalog fetch,
  // no registration, and still a clean exit (masquerade guard).

  const turn = await runTurn({ host, bin, caseDir, agentDir, mode: "nokey", modelArg: "openrouter/auto", prompt: "/openrouter-sync", baseUrl, logPath });
  const ok = exitCheck(check, turn, `${host} no-key run`);
  if (ok) {
    check.check(
      "no-key run made zero requests to the mock",
      turn.logSlice.length === 0,
      `requests seen: ${JSON.stringify(turn.logSlice.map((r) => `${r.method} ${r.path}`))}`,
    );
    const regs = registryRecords(turn.probeRecords);
    check.check("no-key probe loaded", turn.probeRecords.some((record) => record.type === "start"),
      `probe artifact: ${turn.probeArtifact} (${turn.probeRecords.length} records) — probe did not load`);
    const offenders = regs.flatMap((r) => r.rows).filter((row) => accountIds.includes(row.id));
    check.check(
      "no-key run registered no extension rows",
      offenders.length === 0,
      `extension rows present without a key: ${JSON.stringify(offenders.map((r) => r.id))}`,
    );

  }
  return check;
}

/**
 * omp keyed case under `providers.maxInFlightRequests.openrouter = 1` — the
 * review's P1 deadlock regression. omp reads providers.* settings from the
 * agent dir's config.yml (the same store the credential seeding targets), so
 * the limit is written into this case home before any turn. Prewarm +
 * fixture-model turn + router turn must COMPLETE within the bounded timeout:
 * the pre-fix deadlock blocked every request before any HTTP call, so a
 * timeout here is the regression signal.
 *
 * Enforcement stays observable: two omp processes sharing this case home share
 * the HOME-scoped lease root, and the sampled lease count must never exceed 1
 * (two leases at once would mean the recovery wrapper's nested dispatch took a
 * second permit for the provider the enclosing dispatch already holds). A
 * mid-completion SIGKILL leaves a dead-pid lease that omp's stale-lease
 * reaping must clear: a follow-up turn in the same case home must complete
 * promptly, proving cancellation does not wedge the permit store.
 */
async function runLimitOneCase(host, bin, baseUrl, logPath) {
  const check = new Checks(`${host}/limit-one`);
  const { caseDir, agentDir } = makeCaseHome(host, "limit-one");
  await seedCredentials(host, agentDir, fixtureKey);
  fs.writeFileSync(
    path.join(agentDir, "config.yml"),
    ["providers:", "  maxInFlightRequests:", "    openrouter: 1", ""].join("\n"),
  );

  // Fail loudly if the limit never reached omp: every later assertion in this
  // case would silently exercise no permit path at all.
  const configProbe = await runHost(
    "omp",
    ["config", "get", "providers.maxInFlightRequests"],
    hostEnv({ caseDir, agentDir, probeArtifact: "", label: "limit-one-config", baseUrl }),
    caseDir,
    60_000,
  );
  check.check(
    `${host} limit-one setting visible to omp`,
    configProbe.code === 0 && configProbe.stdout.includes('"openrouter":1'),
    `exit=${configProbe.code} stdout=${JSON.stringify(clip(configProbe.stdout, 300))} stderr=${JSON.stringify(clip(configProbe.stderr, 300))}`,
  );

  const modelArg = `openrouter/${MODEL_ID}`;
  const prewarm = await runTurn({ host, bin, caseDir, agentDir, mode: "limit-one-prewarm", modelArg, prompt: "Warm the selected model.", baseUrl, logPath });
  exitCheck(check, prewarm, `${host} limit-one prewarm (deadlock regression = timeout)`);
  const turn1 = await runTurn({ host, bin, caseDir, agentDir, mode: "limit-one", modelArg, prompt: PROMPT, baseUrl, logPath });
  const ok1 = exitCheck(check, turn1, `${host} limit-one fixture-model turn`);
  if (ok1) {
    check.check(
      `${host} limit-one completion reached the mock`,
      turn1.logSlice.some((r) => isCompletion(r) && r.body?.model === MODEL_ID),
      `completion POSTs: ${completionSummary(turn1.logSlice.filter(isCompletion))}`,
    );
    const streamed = assistantMessages(turn1.probeRecords).filter((m) => String(m.model ?? "").endsWith(MODEL_ID));
    check.check(
      `${host} limit-one assistant message completed`,
      streamed.length >= 1 && streamed[streamed.length - 1].stopReason === "stop",
      `probe messages: ${JSON.stringify(assistantMessages(turn1.probeRecords).map((m) => ({ model: m.model, stopReason: m.stopReason, errorMessage: m.errorMessage })))}`,
    );
  }

  const turn2 = await runTurn({ host, bin, caseDir, agentDir, mode: "limit-one-router", modelArg: `openrouter/${ROUTER_ID}`, prompt: PROMPT, baseUrl, logPath });
  const ok2 = exitCheck(check, turn2, `${host} limit-one router turn`);
  if (ok2) {
    check.check(
      `${host} limit-one router completion reached the mock`,
      turn2.logSlice.some((r) => isCompletion(r) && r.body?.model === ROUTER_ID),
      `completion POSTs: ${completionSummary(turn2.logSlice.filter(isCompletion))}`,
    );
  }

  // --- Enforcement: two concurrent omp processes, one shared lease root ---
  // The concurrency check runs against its own mock with a held-open
  // completion response: the permit is held for the whole in-flight request,
  // so the lease is observable on disk for the delay window. Without the
  // delay the request can finish between two samples and the sampler would
  // never see a lease — the limit check would pass vacuously.
  const concurrencyLogPath = path.join(artifactsDir, "mock-openrouter-limit-one-concurrent.ndjson");
  fs.rmSync(concurrencyLogPath, { force: true });
  const concurrencyMock = await startMockOpenRouter({
    logPath: concurrencyLogPath,
    expectedKey: fixtureKey,
    completionDelayMs: 3_000,
  });
  // FRESH agent dirs for both concurrent turns: omp caches the provider
  // config (including baseUrl) with its dynamic-models catalog, so reusing an
  // agent dir that already pointed at the primary mock would send these
  // requests there instead of the concurrency mock (same leak the kill case
  // isolates). Each agent dir carries the same limit=1 config.
  const agentConcurrencyA = path.join(caseDir, "agent-concurrency-a");
  const agentConcurrencyB = path.join(caseDir, "agent-concurrency-b");
  for (const dir of [agentConcurrencyA, agentConcurrencyB]) {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(
      path.join(dir, "config.yml"),
      ["providers:", "  maxInFlightRequests:", "    openrouter: 1", ""].join("\n"),
    );
    await seedCredentials(host, dir, fixtureKey);
  }
  const sampler = startLeaseSampler(scenarioHome);
  let concurrentA;
  let concurrentB;
  try {
    [concurrentA, concurrentB] = await Promise.all([
      runTurn({ host, bin, caseDir, agentDir: agentConcurrencyA, mode: "limit-one-concurrent-a", modelArg, prompt: PROMPT, baseUrl: concurrencyMock.baseUrl, logPath: concurrencyLogPath }),
      runTurn({ host, bin, caseDir, agentDir: agentConcurrencyB, mode: "limit-one-concurrent-b", modelArg, prompt: PROMPT, baseUrl: concurrencyMock.baseUrl, logPath: concurrencyLogPath }),
    ]);
  } finally {
    sampler.stop();
    await concurrencyMock.close();
  }
  const sampling = sampler.result();
  const okA = exitCheck(check, concurrentA, `${host} limit-one concurrent turn A`);
  const okB = exitCheck(check, concurrentB, `${host} limit-one concurrent turn B`);
  check.check(
    `${host} limit-one sampler observed the permit store`,
    sampling.observedLease && sampling.samples > 0,
    `observed a lease=${sampling.observedLease} segment dir=${sampling.observedSegment} over ${sampling.samples} samples (no lease ever observed = the permit limit was never exercised, so maxConcurrent would pass vacuously)`,
  );
  check.check(
    `${host} limit-one sampled permits never exceed the limit of one`,
    sampling.maxConcurrent <= 1,
    `max sampled leases=${sampling.maxConcurrent} over ${sampling.samples} samples (limit 1; two leases at once = nested bypass)`,
  );
  if (okA && okB) {
    check.check(
      `${host} limit-one both concurrent turns posted a completion`,
      concurrentA.logSlice.some(isCompletion) && concurrentB.logSlice.some(isCompletion),
      `A POSTs=${concurrentA.logSlice.filter(isCompletion).length} B POSTs=${concurrentB.logSlice.filter(isCompletion).length}`,
    );
  }

  // --- Permit release after cancellation: bounded kill path ---
  const killLogPath = path.join(artifactsDir, "mock-openrouter-limit-one-kill.ndjson");
  fs.rmSync(killLogPath, { force: true });
  // A FRESH agent dir for the killed turn: omp caches the provider config
  // (including baseUrl) with its dynamic-models catalog, so reusing the
  // original agent dir would send the completion to the primary mock and the
  // kill log would never see it (same leak the stats-down case isolates).
  // The kill fires on the probe's before_provider_request record — written at
  // request START, while the dispatch still holds our provider's permit — so
  // the SIGKILL lands mid-request instead of racing an already-finished run
  // (mock log rows only appear after a response completes). The kill turn
  // shares the case home, so its permit lease lands in the same cross-process
  // lease root.
  const agentKill = path.join(caseDir, "agent-kill");
  fs.mkdirSync(agentKill, { recursive: true });
  await seedCredentials(host, agentKill, fixtureKey);
  const killMock = await startMockOpenRouter({
    logPath: killLogPath,
    expectedKey: fixtureKey,
    // Hold the completion RESPONSE open: the kill fires on the probe's
    // request-start record, and the delay guarantees the SIGKILL lands while
    // the host is still awaiting the response (permit held), not after it.
    completionDelayMs: 4_000,
  });
  const killProbeArtifact = path.join(artifactsDir, `${host}-limit-one-kill-probe.ndjson`);
  let killed;
  try {
    killed = await runHostUntilKill(bin, [
      "--no-extensions",
      "-e", path.join(repoDir, "pi/index.ts"),
      "-e", probePath,
      "--model", modelArg,
      "-p", PROMPT,
    ], hostEnv({ caseDir, agentDir: agentKill, probeArtifact: killProbeArtifact, label: `${host}/limit-one-kill`, baseUrl: killMock.baseUrl }), caseDir, {
      logPath: killLogPath,
      probePath: killProbeArtifact,
      killWhen: (records) =>
        records.some((r) => r.type === "before_provider_request" && String(r.model ?? "").endsWith(MODEL_ID)),
    });
  } finally {
    await killMock.close();
  }
  check.check(
    `${host} limit-one kill fired on a live host mid-request (permit held)`,
    killed.killReason === "request-observed",
    `exit=${killed.code} killReason=${killed.killReason} stderr=${JSON.stringify(clip(killed.stderr, 300))}`,
  );

  const followUp = await runTurn({ host, bin, caseDir, agentDir, mode: "limit-one-followup", modelArg, prompt: PROMPT, baseUrl, logPath });
  const okFollow = exitCheck(check, followUp, `${host} limit-one follow-up turn after kill`);
  if (okFollow) {
    check.check(
      `${host} limit-one follow-up completion reached the mock`,
      followUp.logSlice.some(isCompletion),
      `completion POSTs: ${completionSummary(followUp.logSlice.filter(isCompletion))}`,
    );
  }

  return check;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const logPath = path.join(artifactsDir, "mock-openrouter.ndjson");
  fs.rmSync(logPath, { force: true });
  // The stats-down twin: its own port and its own log file, so its requests
  // never mix with the primary mock's slice.
  const statsDownLogPath = path.join(artifactsDir, "mock-openrouter-stats-down.ndjson");
  fs.rmSync(statsDownLogPath, { force: true });

  const mock = await startMockOpenRouter({ logPath, expectedKey: fixtureKey });
  const statsDownMock = await startMockOpenRouter({
    logPath: statsDownLogPath,
    expectedKey: fixtureKey,
    frontendStatsUnavailable: true,
  });
  const statsDown = { baseUrl: statsDownMock.baseUrl, logPath: statsDownLogPath };
  const allChecks = [];
  try {
    for (const host of hosts) {
      const bin = host;
      hostBin = bin;
      allChecks.push(await runKeyedCase(host, bin, mock.baseUrl, logPath));
      allChecks.push(await runLimitBoundaryCase(host, bin, mock.baseUrl, logPath));
      allChecks.push(await runStatsDownCase(host, bin, statsDown));
      allChecks.push(await runNoKeyCase(host, bin, mock.baseUrl, logPath));
      // The omp-only permit case: pi's pi-ai has no provider permit, so the
      // limit-one config and the cross-process lease sampling are omp-only.
      if (host === "omp") allChecks.push(await runLimitOneCase(host, bin, mock.baseUrl, logPath));
      // Discovery consistency (failed refresh reporting, prior-catalog
      // retention, overlap, logout): self-contained module with its own
      // mocks and case homes under SCENARIO_HOME.
      const discovery = await runDiscoveryCases({
        host,
        bin,
        rootDir: scenarioHome,
        artifactsDir,
        timeoutMs: SPAWN_TIMEOUT_MS,
      });
      allChecks.push({
        prefix: `${host}/discovery`,
        passed: discovery.passed,
        failures: discovery.failures.map((f) => ({ name: `[${f.case}] ${f.name}`, detail: f.detail })),
      });
    }
  } finally {
    await mock.close();
    await statsDownMock.close();
  }

  const passed = allChecks.reduce((n, c) => n + c.passed, 0);
  const failed = allChecks.reduce((n, c) => n + c.failures.length, 0);
  const summary = {
    hosts,
    checksPassed: passed,
    checksFailed: failed,
    failures: allChecks.flatMap((c) => c.failures.map((f) => ({ case: c.prefix, ...f }))),
  };
  fs.writeFileSync(path.join(artifactsDir, "scenario-summary.json"), `${JSON.stringify(summary, null, 2)}\n`);

  if (failed > 0) {
    console.error(`container-scenario: ${failed} assertion(s) failed, ${passed} passed — details above; artifacts in ${artifactsDir}`);
    process.exitCode = 1;
  } else {
    console.log(`container-scenario: all ${passed} assertion(s) passed for hosts ${hosts.join(", ")}`);
  }
}

try {
  await main();
} catch (error) {
  console.error(`container-scenario: crashed: ${error?.stack ?? error}`);
  process.exit(2);
}
