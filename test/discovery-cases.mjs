#!/usr/bin/env bun
/**
 * Discovery-consistency cases for the dual-host harness.
 *
 * These cases cover the phase-4 review findings:
 *
 *   1. A failed refresh must never masquerade success: a first-ever failure
 *      leaves NO account rows anywhere, and a later successful refresh
 *      registers the new catalog.
 *   2. A failed refresh after a successful one must keep the previous
 *      catalog usable (never half-updated, never cleared where the host
 *      persists it): the extension serves the credential's last-known-good
 *      rows on a failed omp discovery, because omp 18.16 otherwise EVICTS
 *      its persisted cache — the failure is still reported and nothing new
 *      is registered.
 *   3. Overlapping same-key discoveries in ONE process must converge to one
 *      consistent catalog (no dupes, no mixes).
 *   4. A logout (no credential) must not restore account rows.
 *
 * OBSERVABILITY MODEL (corrected 2026-11-05 after the orchestrator's
 * container run; see .context/2026-11-04.6-dual-host-harness/
 * research-discovery-consistency.md):
 *
 *   - Wire (mock NDJSON log slices): statuses, bearer keys, and the number
 *     of discovery fetches per run. Always available, always asserted.
 *   - Follow-up observation runs: an extra probe-carrying host run in the
 *     SAME home. The probe dumps the host registry only from event handlers
 *     that carry a ctx, and its `session_start` dump reflects the COMPLETED
 *     discovery of that run (omp merges the refresh result before the dump;
 *     pi's registration happens at extension load). So a follow-up run's
 *     registry record shows what a fresh process computes/persists under the
 *     CURRENT mock+credential state — it cannot show mid-run states.
 *   - Extension-attributed rows are discriminated per host:
 *       omp: rows with api === "openrouter-recovery-completions" (the
 *         fetcher's remap; omp's builtin openrouter/auto row shares
 *         ROUTER_ID but carries a different api — id matching alone
 *         false-positives).
 *       pi: the registered ProviderConfig IS the extension's when it
 *         registered; a record whose rows are exactly the account fixture
 *         set is extension attribution (builtin catalogs never equal the
 *         fixture set).
 *   - The probe's "snapshot" source is NEVER asserted: in `-e` mode the
 *     probe's `import("../../pi/state.js")` resolves to a separate module
 *     instance from the extension's, so it always reads an empty snapshot.
 *
 * Notifications are NOT asserted: both hosts drop `ctx.ui.notify` in
 * one-shot print mode, so the "N models synced" claim is unassertable at
 * host level. The registered catalog state IS the claim.
 *
 * The module is self-contained: it starts its own mock instances, keeps its
 * own log slices, and returns a plain result object. The orchestrator wires
 * it into the scenario main.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ACCOUNT_MODELS, FIXTURE_KEY } from "./fixtures/openrouter.mjs";
import { startMockOpenRouter } from "./mock-openrouter.mjs";
import { seedOmpDb } from "../scripts/seed-omp-db.mjs";

const repoDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const probePath = path.join(repoDir, "test", "fixtures", "probe-extension.mjs");

const accountIds = ACCOUNT_MODELS.map((m) => m.id);
const EXPECTED_ACCOUNT_COUNT = ACCOUNT_MODELS.length;

/**
 * The api id the omp dynamic fetcher pins on every row (pi/stream.ts
 * OPENROUTER_RECOVERY_API). Kept as a literal so this module stays decoupled
 * from the compiled extension the same way container-scenario.mjs is.
 */
const OPENROUTER_RECOVERY_API = "openrouter-recovery-completions";

/** The probe registry source that carries per-host registration state. */
const OBSERVABLE_SOURCE = { omp: "host-registry", pi: "provider-config" };

/** Collected assertions for one case. */
class Checks {
  constructor(prefix) {
    this.prefix = prefix;
    this.passed = 0;
    this.failures = [];
  }

  check(name, ok, detail = "") {
    if (ok) {
      this.passed += 1;
      return true;
    }
    this.failures.push({ name, detail });
    console.error(`FAIL [${this.prefix}] ${name}${detail ? `\n  ${detail}` : ""}`);
    return false;
  }
}

function clip(text, max = 1200) {
  const s = String(text ?? "");
  return s.length > max ? `${s.slice(0, max)}…[clipped]` : s;
}

/**
 * Run one one-shot host command; returns { code, stdout, stderr, timedOut }.
 * The child runs with an isolated HOME and agent dir under `caseDir`. The
 * hard-kill fallback matters: an omp run has been observed to hang, so
 * SIGTERM is followed by SIGKILL instead of waiting forever.
 */
function runHost(bin, args, { caseDir, agentDir, baseUrl, probeArtifact, label }, timeoutMs) {
  return new Promise((resolve) => {
    const env = { ...process.env };
    // The hosts must resolve the key from the seeded store, never from env,
    // and the runtime-override test seam must stay off.
    // Keep omp's extracted native addon in the shared temporary HOME cache;
    // each case still has a distinct agent dir for credentials and sessions.
    env.HOME = process.env.SCENARIO_HOME || caseDir;
    env.PI_CODING_AGENT_DIR = agentDir;
    env.OMP_CODING_AGENT_DIR = agentDir;
    env.OPENROUTER_BASE_URL = baseUrl;
    env.OPENROUTER_PROBE_ARTIFACT = probeArtifact;
    env.OPENROUTER_PROBE_LABEL = label;
    env.NO_COLOR = "1";
    env.FORCE_COLOR = "0";
    env.TERM = "dumb";

    const child = spawn(bin, args, { env, cwd: caseDir, stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    let settled = false;
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (d) => { stdout += d; });
    child.stderr.on("data", (d) => { stderr += d; });
    const killer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 5_000).unref?.();
    }, timeoutMs);
    killer.unref?.();
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(killer);
      resolve({ code, stdout, stderr, timedOut });
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(killer);
      resolve({ code: null, stdout, stderr: `${stderr}\nspawn error: ${error?.message ?? error}`, timedOut });
    });
  });
}

/** Read the NDJSON log rows appended since byte offset `since`. */
function readLogSlice(file, since) {
  let size = 0;
  try {
    size = fs.statSync(file).size;
  } catch {
    return [];
  }
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
  let text = "";
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

const isAccountCatalog = (row) => row.method === "GET" && row.path === "/api/v1/models/user";

/** Isolated temp home under rootDir; agent dir honors PI_CODING_AGENT_DIR on both hosts. */
function makeCaseHome(rootDir, prefix) {
  const caseDir = fs.mkdtempSync(path.join(rootDir, `${prefix}-`));
  const agentDir = path.join(caseDir, "agent");
  fs.mkdirSync(agentDir, { recursive: true });
  const logPath = path.join(caseDir, "mock.ndjson");
  fs.writeFileSync(logPath, "");
  return { caseDir, agentDir, logPath };
}

/**
 * Seed the fixture key into the host's own credential store.
 * pi reads agent/auth.json; omp reads agent.db — schema-initialized by a
 * short one-shot extension run (see below), then seeded with the pinned
 * 18.16 auth schema upsert.
 */
async function seedCredentials(host, bin, agentDir, key, caseDir, timeoutMs) {
  if (host === "pi") {
    fs.writeFileSync(
      path.join(agentDir, "auth.json"),
      `${JSON.stringify({ openrouter: { type: "api_key", key } }, null, 2)}\n`,
    );
    return;
  }
  // Attempt host initialization first. If omp cannot populate the native
  // database in this offline run, the pinned seeder creates its auth schema.
  await runHost(
    bin,
    ["--no-extensions", "--print", "initialize"],
    { caseDir, agentDir, baseUrl: "http://127.0.0.1:9/api/v1", probeArtifact: "", label: "init" },
    timeoutMs,
  );
  seedOmpDb(agentDir, key);
}

/**
 * Run one host turn and collect every observable: exit status, the mock
 * requests it caused (an exact byte-offset slice of the shared log), and the
 * probe's registry records. Every case turn runs `-p /openrouter-sync`.
 */
async function runTurn({ host, bin, caseDir, agentDir, baseUrl, logPath, timeoutMs, artifactsDir, mode }) {
  const probeArtifact = path.join(artifactsDir, `${host}-${mode}-probe.ndjson`);
  fs.rmSync(probeArtifact, { force: true });
  const since = (() => {
    try {
      return fs.statSync(logPath).size;
    } catch {
      return 0;
    }
  })();

  const result = await runHost(
    bin,
    [
      "--no-extensions",
      "-e", path.join(repoDir, "pi", "index.ts"),
      "-e", probePath,
      "-p", "/openrouter-sync",
    ],
    { caseDir, agentDir, baseUrl, probeArtifact, label: `${host}/${mode}` },
    timeoutMs,
  );

  const logSlice = readLogSlice(logPath, since);
  const records = readNdjson(probeArtifact);
  const observable = records.filter((r) => r.type === "registry" && r.source === OBSERVABLE_SOURCE[host] && Array.isArray(r.rows));
  return { result, logSlice, records, observable, probeArtifact };
}

/**
 * Extension-attributed rows in one observable registry record, per host
 * shape. See the module doc block for why attribution keys on the recovery
 * api (omp) / the exact fixture set (pi) instead of raw ids.
 */
function extensionRows(host, record) {
  if (!record || !Array.isArray(record.rows)) return [];
  if (host === "omp") {
    return record.rows.filter((row) => row.api === OPENROUTER_RECOVERY_API);
  }
  const rows = record.rows;
  const attributed = rows.length === EXPECTED_ACCOUNT_COUNT && rows.every((row) => accountIds.includes(row.id));
  return attributed ? rows : [];
}

/**
 * Assert one observable record carries EXACTLY the expected account set:
 * every extension row's id is in `expectedIds`, the deduped set equals it
 * (no dupes), and the counts agree.
 */
function checkExactCatalog(check, host, label, record, expectedIds) {
  const rows = extensionRows(host, record);
  const ids = rows.map((row) => row.id);
  const unique = [...new Set(ids.filter(Boolean))];
  check.check(
    `${label}: exactly the expected account rows`,
    rows.length === expectedIds.length && unique.length === expectedIds.length && unique.every((id) => expectedIds.includes(id)),
    `extension rows: ${JSON.stringify(ids)}`,
  );
}

/** Assert every observable record of the turn is free of extension rows. */
function checkNoExtensionRowsAnywhere(check, host, label, turn) {
  const offenders = turn.observable.flatMap((record) => extensionRows(host, record).map((row) => row.id));
  check.check(
    `${label}: no extension-attributed rows in any registry record`,
    offenders.length === 0,
    `extension rows seen: ${JSON.stringify(offenders)}`,
  );
}

/** The account-catalog wire rows of one turn. */
const accountFetches = (turn) => turn.logSlice.filter(isAccountCatalog);

/**
 * The probe appends a `start` record at load unconditionally; assert it so a
 * missing registry record means "the host registered nothing", never
 * "the probe died".
 */
function checkProbeAlive(check, label, turn) {
  check.check(
    `${label}: probe ran`,
    turn.records.some((r) => r.type === "start"),
    `probe artifact: ${JSON.stringify(turn.records.map((r) => r.type))}`,
  );
}

/** Assert the turn reached the account catalog and every fetch used `status`. */
function checkAccountFetchStatus(check, label, turn, status) {
  const fetches = accountFetches(turn);
  const ok = fetches.length >= 1 && fetches.every((r) => (status === "5xx" ? r.status >= 500 : r.status === status));
  check.check(
    `${label}: account catalog answered ${status === "5xx" ? "5xx" : status} (${fetches.length} fetch${fetches.length === 1 ? "" : "es"})`,
    ok,
    `statuses: ${JSON.stringify(fetches.map((r) => r.status))}`,
  );
}

/** Assert exit 0 within the deadline. */
function checkExitZero(check, label, turn) {
  check.check(
    `${label}: exited 0`,
    turn.result.code === 0 && !turn.result.timedOut,
    `exit=${turn.result.code} timedOut=${turn.result.timedOut} stderr=${clip(turn.result.stderr)}`,
  );
}

/**
 * Case 1+2: a failed refresh must not masquerade success, and a later
 * successful refresh must register the new catalog.
 *
 *   - A failing sync on a FRESH home: the wire shows the 5xx account-catalog
 *     fetches and the run still exits 0. A follow-up observation run in the
 *     same home (against the still-failing mock) registers NO extension
 *     account rows — a first-ever failure must not masquerade.
 *   - A healthy sync on the same home: the wire shows 200s. A follow-up
 *     observation run registers exactly the 10 extension account rows.
 *
 * The 'new count' notification itself is unassertable in print mode (both
 * hosts drop ctx.ui.notify there); the registered-catalog state is the
 * claim.
 */
async function runFailureThenRecoveryCase({ host, bin, rootDir, artifactsDir, timeoutMs }) {
  const check = new Checks(`${host}/discovery-failure`);
  const { caseDir, agentDir, logPath } = makeCaseHome(rootDir, `${host}-discovery-`);

  // The mock fails the account catalog first, then recovers.
  const failing = await startMockOpenRouter({ logPath, expectedKey: FIXTURE_KEY, accountCatalogFail: true });
  const healthy = await startMockOpenRouter({ logPath, expectedKey: FIXTURE_KEY });
  try {
    await seedCredentials(host, bin, agentDir, FIXTURE_KEY, caseDir, timeoutMs);

    // --- 1. initial failure on a fresh home ---
    const failed = await runTurn({ host, bin, caseDir, agentDir, baseUrl: failing.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-fail" });
    checkExitZero(check, "failed refresh", failed);
    checkAccountFetchStatus(check, "failed refresh", failed, "5xx");

    // --- 2. follow-up observation run: the failure left nothing behind ---
    const failedFollowUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: failing.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-fail-followup" });
    checkExitZero(check, "post-failure follow-up", failedFollowUp);
    checkProbeAlive(check, "post-failure follow-up", failedFollowUp);
    checkNoExtensionRowsAnywhere(check, host, "post-failure follow-up", failedFollowUp);

    // --- 3. recovery on the SAME home ---
    const recovered = await runTurn({ host, bin, caseDir, agentDir, baseUrl: healthy.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-recover" });
    checkExitZero(check, "recovered refresh", recovered);
    checkAccountFetchStatus(check, "recovered refresh", recovered, 200);

    // --- 4. follow-up observation run: the healthy catalog is registered ---
    const recoveredFollowUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: healthy.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-recover-followup" });
    checkExitZero(check, "post-recovery follow-up", recoveredFollowUp);
    check.check(
      "post-recovery follow-up captured registry state",
      recoveredFollowUp.observable.length >= 1,
      `no ${OBSERVABLE_SOURCE[host]} records in the probe artifact`,
    );
    checkExactCatalog(check, host, "post-recovery follow-up (session start)", recoveredFollowUp.observable[0], accountIds);
    checkExactCatalog(check, host, "post-recovery follow-up (final)", recoveredFollowUp.observable.at(-1), accountIds);
  } finally {
    await failing.close();
    await healthy.close();
  }
  return check;
}

/**
 * Case 2: a failed refresh AFTER a successful one must not masquerade and
 * must leave a usable prior catalog.
 *
 * Pinned policy (corrective round 2, 2026-11-05): the extension serves the
 * credential's LAST-KNOWN-GOOD rows on a failed omp discovery. omp 18.16
 * evicts both the live catalog and its persisted registry cache
 * (`agent/models.db`) when discovery fails, so without serving, a fresh
 * process could no longer resolve prior account models ("Model not found").
 * With serving, omp treats the served rows as the discovery result and
 * RETAINS the catalog. The pinned assertions:
 *
 *   - the healthy catalog is registered and persisted (follow-up + cache);
 *   - the failed sync exits 0 with 5xx wire evidence (failure surfaced, no
 *     crash, no fake success);
 *   - the failing run's registry still shows EXACTLY the prior account set:
 *     the served rows keep the catalog registered, and nothing NEW may
 *     appear;
 *   - omp's persisted cache STILL holds exactly the account catalog after
 *     the failed refresh, and the follow-up run (still-failing mock, fresh
 *     process serving from the agent-dir mirror) still registers and
 *     persists the same set;
 *   - pi: registration is per-process; a fresh load with a failing mock
 *     registers nothing, and pi persists no catalog, so there is nothing to
 *     serve across processes (honest pi behavior — the extension snapshot
 *     is pi's in-process preservation, unit-tested in
 *     test/discovery-consistency.test.ts).
 */
async function runFailureAfterSuccessCase({ host, bin, rootDir, artifactsDir, timeoutMs }) {
  const check = new Checks(`${host}/discovery-failure-after-success`);
  const { caseDir, agentDir, logPath } = makeCaseHome(rootDir, `${host}-discovery2-`);

  const healthy = await startMockOpenRouter({ logPath, expectedKey: FIXTURE_KEY });
  const failing = await startMockOpenRouter({ logPath, expectedKey: FIXTURE_KEY, accountCatalogFail: true });
  try {
    await seedCredentials(host, bin, agentDir, FIXTURE_KEY, caseDir, timeoutMs);

    const first = await runTurn({ host, bin, caseDir, agentDir, baseUrl: healthy.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-ok" });
    checkExitZero(check, "initial sync", first);
    checkAccountFetchStatus(check, "initial sync", first, 200);

    const okFollowUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: healthy.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-ok-followup" });
    checkExitZero(check, "post-ok follow-up", okFollowUp);
    check.check(
      "post-ok follow-up captured registry state",
      okFollowUp.observable.length >= 1,
      `no ${OBSERVABLE_SOURCE[host]} records in the probe artifact`,
    );
    checkExactCatalog(check, host, "post-ok follow-up (session start)", okFollowUp.observable[0], accountIds);
    if (host === "omp") {
      // The healthy catalog must be PERSISTED, not just merged into the live
      // registry: this is the state a later process would reload.
      const cached = ompModelCacheIds(agentDir);
      const unique = [...new Set(cached)];
      check.check(
        "omp persisted cache holds the healthy account catalog",
        cached.length === EXPECTED_ACCOUNT_COUNT && unique.length === EXPECTED_ACCOUNT_COUNT && unique.every((id) => accountIds.includes(id)),
        `persisted cache ids: ${JSON.stringify(cached.slice(0, 20))}`,
      );
    }

    const second = await runTurn({ host, bin, caseDir, agentDir, baseUrl: failing.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-fail2" });
    checkExitZero(check, "failed refresh", second);
    checkAccountFetchStatus(check, "failed refresh", second, "5xx");
    if (host === "omp") {
      // The extension serves the last-known-good rows on the failed
      // discovery, so the catalog stays REGISTERED in the failing run
      // itself — exactly the prior set, nothing new. (Before the serving
      // fix, this dump showed 0 recovery rows: omp evicted on error.)
      check.check(
        "failing run captured registry state",
        second.observable.length >= 1,
        `no ${OBSERVABLE_SOURCE[host]} records in the probe artifact`,
      );
      checkExactCatalog(check, host, "failing run (session start)", second.observable[0], accountIds);
      checkExactCatalog(check, host, "failing run (final)", second.observable.at(-1), accountIds);
      // omp RETAINS the persisted catalog: the served rows are consumed as
      // the discovery result and re-persisted. (omp 18.16 would wipe the
      // cache if the fetch error propagated — pinned so a regression in the
      // serving path is noticed.)
      const cached = ompModelCacheIds(agentDir);
      const unique = [...new Set(cached)];
      check.check(
        "omp persisted cache still holds the account catalog after the failed refresh",
        cached.length === EXPECTED_ACCOUNT_COUNT && unique.length === EXPECTED_ACCOUNT_COUNT && unique.every((id) => accountIds.includes(id)),
        `persisted cache ids: ${JSON.stringify(cached.slice(0, 20))}`,
      );
    }

    const failFollowUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: failing.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-fail2-followup" });
    checkExitZero(check, "post-failure follow-up", failFollowUp);
    checkProbeAlive(check, "post-failure follow-up", failFollowUp);
    if (host === "omp") {
      // The follow-up is a FRESH process: its first discovery fails against
      // the still-failing mock, and the extension serves the mirrored
      // last-good rows — the prior catalog stays registered (resolvable)
      // and persisted, with NOTHING NEW (the set is exactly the fixture).
      check.check(
        "post-failure follow-up captured registry state",
        failFollowUp.observable.length >= 1,
        `no ${OBSERVABLE_SOURCE[host]} records in the probe artifact`,
      );
      checkExactCatalog(check, host, "post-failure follow-up (session start)", failFollowUp.observable[0], accountIds);
      checkExactCatalog(check, host, "post-failure follow-up (final)", failFollowUp.observable.at(-1), accountIds);
      const cached = ompModelCacheIds(agentDir);
      const unique = [...new Set(cached)];
      check.check(
        "omp persisted cache still holds the account catalog after the follow-up",
        cached.length === EXPECTED_ACCOUNT_COUNT && unique.length === EXPECTED_ACCOUNT_COUNT && unique.every((id) => accountIds.includes(id)),
        `persisted cache ids: ${JSON.stringify(cached.slice(0, 20))}`,
      );
    } else {
      checkNoExtensionRowsAnywhere(check, host, "post-failure follow-up", failFollowUp);
    }
  } finally {
    await healthy.close();
    await failing.close();
  }
  return check;
}

/**
 * Case 4: repeated same-key discoveries inside ONE process must converge to
 * one consistent catalog.
 *
 * Both hosts fire more than one discovery per `-p /openrouter-sync` run (pi:
 * load bootstrap + session_start sync + command sync; omp: startup refresh +
 * session_start/command refreshes — the orchestrator's container run saw 4
 * concurrent same-key fetches in one omp process). The mock delays key A's
 * account catalog so the discoveries genuinely interleave; whichever order
 * they settle in, every fetch must succeed with key A and the follow-up run
 * must show exactly ONE consistent set of the 10 account rows (no dupes, no
 * mixes).
 *
 * The superseded-attempt publication rules (including the cross-credential
 * veto) are invariant-tested in test/discovery-consistency.test.ts;
 * host-level cross-CREDENTIAL overlap is unreachable (see the research
 * note).
 */
async function runSameKeyOverlapCase({ host, bin, rootDir, artifactsDir, timeoutMs }) {
  const check = new Checks(`${host}/discovery-same-key-overlap`);
  const { caseDir, agentDir, logPath } = makeCaseHome(rootDir, `${host}-discovery3-`);

  // Key A is delayed: several same-key discoveries are in flight per run.
  const mock = await startMockOpenRouter({ logPath, expectedKey: FIXTURE_KEY, keyDelays: { [FIXTURE_KEY]: 1500 } });
  try {
    await seedCredentials(host, bin, agentDir, FIXTURE_KEY, caseDir, timeoutMs);

    const overlap = await runTurn({ host, bin, caseDir, agentDir, baseUrl: mock.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-overlap" });
    checkExitZero(check, "overlap run", overlap);
    const fetches = accountFetches(overlap);
    check.check(
      `overlap run fired repeated same-key discoveries (${fetches.length} account fetches)`,
      fetches.length >= 2,
      `account fetches: ${JSON.stringify(fetches.map((r) => ({ status: r.status, auth: r.authorization })))}`,
    );
    check.check(
      "every overlapping discovery fetched 200 with the current key",
      fetches.length >= 1 && fetches.every((r) => r.status === 200 && r.authorization === FIXTURE_KEY),
      `statuses/keys: ${JSON.stringify(fetches.map((r) => ({ status: r.status, auth: r.authorization })))}`,
    );

    const followUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: mock.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-overlap-followup" });
    checkExitZero(check, "overlap follow-up", followUp);
    check.check(
      "overlap follow-up captured registry state",
      followUp.observable.length >= 1,
      `no ${OBSERVABLE_SOURCE[host]} records in the probe artifact`,
    );
    checkExactCatalog(check, host, "overlap follow-up (session start)", followUp.observable[0], accountIds);
    checkExactCatalog(check, host, "overlap follow-up (final)", followUp.observable.at(-1), accountIds);
    if (host === "omp") {
      const cached = ompModelCacheIds(agentDir);
      const unique = [...new Set(cached)];
      check.check(
        "omp persisted cache holds one consistent set of account rows",
        cached.length === EXPECTED_ACCOUNT_COUNT && unique.length === EXPECTED_ACCOUNT_COUNT && unique.every((id) => accountIds.includes(id)),
        `persisted cache ids: ${JSON.stringify(cached.slice(0, 20))}`,
      );
    }
  } finally {
    await mock.close();
  }
  return check;
}

/**
 * Case 5: a logout (no credential) must not restore account rows.
 *
 * The credential is dropped from the host's own store (omp: the pinned
 * seedOmpDb upsert with an empty key — it DELETES the openrouter credential
 * rows and inserts one with key ""; pi: an empty auth.json). The logout run
 * must exit 0 without touching the account catalog (both hosts short-circuit
 * before any fetch), and a follow-up observation run must show NO
 * extension-attributed account rows: omp 18.16 EVICTS the persisted dynamic
 * rows when a discovery returns none (probed: live registry and
 * agent/models.db both end up without them), and pi registers nothing
 * without a key. Builtin rows (different api / never the fixture set) are
 * always allowed.
 */
async function runLogoutCase({ host, bin, rootDir, artifactsDir, timeoutMs }) {
  const check = new Checks(`${host}/discovery-logout`);
  const { caseDir, agentDir, logPath } = makeCaseHome(rootDir, `${host}-discovery4-`);

  const mock = await startMockOpenRouter({ logPath, expectedKey: FIXTURE_KEY });
  try {
    await seedCredentials(host, bin, agentDir, FIXTURE_KEY, caseDir, timeoutMs);

    const before = await runTurn({ host, bin, caseDir, agentDir, baseUrl: mock.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-before-logout" });
    checkExitZero(check, "pre-logout sync", before);
    checkAccountFetchStatus(check, "pre-logout sync", before, 200);

    const beforeFollowUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: mock.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-before-logout-followup" });
    checkExitZero(check, "pre-logout follow-up", beforeFollowUp);
    check.check(
      "pre-logout follow-up captured registry state",
      beforeFollowUp.observable.length >= 1,
      `no ${OBSERVABLE_SOURCE[host]} records in the probe artifact`,
    );
    checkExactCatalog(check, host, "pre-logout follow-up (session start)", beforeFollowUp.observable[0], accountIds);

    // Drop the credential from the host's store: the next discovery has none.
    if (host === "pi") {
      fs.writeFileSync(
        path.join(agentDir, "auth.json"),
        `${JSON.stringify({}, null, 2)}\n`,
      );
    } else {
      seedOmpDb(agentDir, "");
    }

    const after = await runTurn({ host, bin, caseDir, agentDir, baseUrl: mock.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-after-logout" });
    checkExitZero(check, "logout run", after);
    const fetches = accountFetches(after);
    check.check(
      "logout fetched no account catalog",
      fetches.length === 0,
      `account fetches: ${JSON.stringify(fetches.map((r) => ({ status: r.status, auth: r.authorization })))}`,
    );

    const afterFollowUp = await runTurn({ host, bin, caseDir, agentDir, baseUrl: mock.baseUrl, logPath, timeoutMs, artifactsDir, mode: "discovery-after-logout-followup" });
    checkExitZero(check, "post-logout follow-up", afterFollowUp);
    checkProbeAlive(check, "post-logout follow-up", afterFollowUp);
    const postFetches = accountFetches(afterFollowUp);
    check.check(
      "post-logout follow-up fetched no account catalog",
      postFetches.length === 0,
      `account fetches: ${JSON.stringify(postFetches.map((r) => ({ status: r.status, auth: r.authorization })))}`,
    );
    checkNoExtensionRowsAnywhere(check, host, "post-logout follow-up", afterFollowUp);
    if (host === "omp") {
      const cached = ompModelCacheIds(agentDir);
      check.check(
        "omp persisted cache holds no account rows after logout",
        cached.filter((id) => accountIds.includes(id)).length === 0,
        `persisted cache ids: ${JSON.stringify(cached.slice(0, 20))}`,
      );
    }
  } finally {
    await mock.close();
  }
  return check;
}

/**
 * Read the openrouter row of omp's persisted dynamic-model registry cache
 * (`agent/models.db`, table model_cache) — the same observable the hosts
 * themselves reload at startup. Read-only, best-effort: an absent or
 * unreadable cache reads as no persisted rows (never throws into a case).
 */
function ompModelCacheIds(agentDir) {
  let Database;
  try {
    // bun:sqlite ships with the bun runtime the scenario runs under.
    Database = require("bun:sqlite").Database;
  } catch {
    return [];
  }
  let db;
  try {
    db = new Database(path.join(agentDir, "models.db"), { readonly: true });
    const row = db.query("SELECT models FROM model_cache WHERE provider_id = ?").get("openrouter");
    if (!row || typeof row.models !== "string") return [];
    const parsed = JSON.parse(row.models);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((m) => (m && typeof m === "object" ? m.id : String(m))).filter((id) => typeof id === "string");
  } catch {
    return [];
  } finally {
    try {
      db?.close();
    } catch {
      // already closed
    }
  }
}

/**
 * Run every discovery case for one host.
 *
 * @param {object} options
 * @param {string} options.host            "omp" | "pi"
 * @param {string} options.bin             the host binary on PATH
 * @param {string} options.rootDir         writable directory for the case homes
 * @param {string} options.artifactsDir    writable directory for the probe artifacts
 * @param {number} [options.timeoutMs]     per-host-command deadline
 * @returns {Promise<{passed: number, failures: Array<{case: string, name: string, detail: string}>}>}
 */
export async function runDiscoveryCases({ host, bin, rootDir, artifactsDir, timeoutMs = 180_000 }) {
  if (!fs.existsSync(artifactsDir)) fs.mkdirSync(artifactsDir, { recursive: true });
  const checks = [];
  checks.push(await runFailureThenRecoveryCase({ host, bin, rootDir, artifactsDir, timeoutMs }));
  checks.push(await runFailureAfterSuccessCase({ host, bin, rootDir, artifactsDir, timeoutMs }));
  checks.push(await runSameKeyOverlapCase({ host, bin, rootDir, artifactsDir, timeoutMs }));
  checks.push(await runLogoutCase({ host, bin, rootDir, artifactsDir, timeoutMs }));

  const passed = checks.reduce((n, c) => n + c.passed, 0);
  const failures = checks.flatMap((c) => c.failures.map((f) => ({ case: c.prefix, ...f })));
  return { passed, failures };
}

export default runDiscoveryCases;
