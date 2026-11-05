/**
 * Unit tests for the discovery-attempt ledger (pi/discovery.ts).
 *
 * These cover the three behaviours the review findings turn on:
 *   - a failed fetch reports FAILURE (and keeps the previous catalog),
 *   - a superseded attempt never publishes an older account's rows,
 *   - the no-key attempt publishes no rows.
 *
 * Run: npm test   (from this package directory) — or: node --test test/
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { describe, it } from "node:test";
import {
  beginDiscoveryAttempt,
  completeDiscoveryAttempt,
  configureLastGoodPersistence,
  getDiscoveryOutcomesSince,
  keyFingerprint,
  lastGoodRowsForFailedAttempt,
  publishDiscoveryRows,
  resetDiscoveryLedger,
  summarizeDiscovery,
} from "../pi/discovery.ts";
import type { ProviderModelConfig } from "../pi/types.ts";

const row = (id: string): ProviderModelConfig =>
  ({ id, name: id, api: "openai-completions" } as unknown as ProviderModelConfig);

const rowsA = [row("a/one"), row("a/two")];
const rowsB = [row("b/one")];

/** Attempt ids share the snapshot generation ordering (pi/state.ts). */
let attemptCounter = 0;
const nextId = (): number => ++attemptCounter;

describe("discovery attempt ledger", () => {
  it("starts each attempt with a fresh id and reports the newest one", () => {
    resetDiscoveryLedger();
    const first = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const second = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    assert.ok(second.id > first.id, "attempt ids increase");
  });

  it("records a successful attempt with the count it published", () => {
    resetDiscoveryLedger();
    const attempt = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(attempt, { status: "ok", modelCount: 2, rows: rowsA });
    const outcomes = getDiscoveryOutcomesSince(attempt.id - 1);
    assert.equal(outcomes.length, 1);
    assert.equal(outcomes[0].status, "ok");
    assert.equal(outcomes[0].modelCount, 2);
  });

  it("reports a failed attempt as failed, with its error, and no count", () => {
    resetDiscoveryLedger();
    const attempt = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(attempt, { status: "failed", error: "503 Service Unavailable" });
    const report = summarizeDiscovery(getDiscoveryOutcomesSince(attempt.id - 1));
    assert.equal(report.ok, false);
    assert.equal(report.noKey, false);
    assert.match(report.error ?? "", /503/);
    assert.equal(report.modelCount, 0);
  });

  it("keeps the previous catalog usable when a refresh fails", () => {
    resetDiscoveryLedger();
    const ok = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(ok, { status: "ok", modelCount: 2, rows: rowsA });
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });

    // The failed attempt contributes no rows, so the summary of the SECOND
    // refresh (the failed one alone) is a failure — never a count from the
    // previous, still-valid catalog.
    const report = summarizeDiscovery(getDiscoveryOutcomesSince(failed.id - 1));
    assert.equal(report.ok, false);
    assert.match(report.error ?? "", /boom/);
  });

  it("publishes no rows for the no-key attempt", () => {
    resetDiscoveryLedger();
    const attempt = beginDiscoveryAttempt(keyFingerprint(undefined), nextId());
    completeDiscoveryAttempt(attempt, { status: "no-key", modelCount: 0, rows: [] });
    const report = summarizeDiscovery(getDiscoveryOutcomesSince(attempt.id - 1));
    assert.equal(report.noKey, true);
    assert.equal(report.ok, false);
  });

  it("summarizes an empty window as a failure rather than a zero-count success", () => {
    resetDiscoveryLedger();
    const report = summarizeDiscovery([]);
    assert.equal(report.ok, false);
    assert.equal(report.modelCount, 0);
    assert.match(report.error ?? "", /no catalog discovery/);
  });
});

describe("supersession guard", () => {
  it("publishes its own rows when no newer attempt started", async () => {
    resetDiscoveryLedger();
    const attempt = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const published = await publishDiscoveryRows(attempt, rowsA);
    assert.equal(published.status, "ok");
    assert.deepEqual(published.rows, rowsA);
  });

  it("publishes the NEWER attempt's rows, never the older account's", async () => {
    resetDiscoveryLedger();
    const older = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const newer = beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    // The newer attempt wins the race and publishes B's rows.
    completeDiscoveryAttempt(newer, { status: "ok", modelCount: 1, rows: rowsB });

    const published = await publishDiscoveryRows(older, rowsA);
    assert.equal(published.status, "superseded");
    assert.deepEqual(published.rows, rowsB, "the retired account's rows must not be published");
    assert.ok(
      !published.rows.some((r) => r.id.startsWith("a/")),
      "no row from the superseded account may reach the host",
    );
  });

  it("publishes nothing when the newer attempt used another credential and failed", async () => {
    resetDiscoveryLedger();
    const older = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const newer = beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    completeDiscoveryAttempt(newer, { status: "failed", error: "503 Service Unavailable" });

    const published = await publishDiscoveryRows(older, rowsA);
    assert.equal(published.status, "superseded");
    assert.deepEqual(published.rows, [], "a retired account's rows must not be restored");
  });

  it("keeps the older attempt's rows when the newer attempt failed for the SAME account", async () => {
    resetDiscoveryLedger();
    const older = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const newer = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(newer, { status: "failed", error: "503 Service Unavailable" });

    const published = await publishDiscoveryRows(older, rowsA);
    assert.equal(published.status, "ok");
    assert.deepEqual(published.rows, rowsA, "a same-account failure must not drop the usable catalog");
  });

  it("keeps the same-key overlap harmless: both attempts publish the same catalog", async () => {
    resetDiscoveryLedger();
    const first = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const second = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(second, { status: "ok", modelCount: 2, rows: rowsA });

    const published = await publishDiscoveryRows(first, rowsA);
    assert.equal(published.status, "superseded");
    assert.deepEqual(published.rows, rowsA);
    // The superseded attempt contributes no count of its own; the newer
    // attempt reports the count.
    const report = summarizeDiscovery(getDiscoveryOutcomesSince(0));
    assert.equal(report.ok, true);
    assert.equal(report.modelCount, 2);
  });

  it("does not let a superseded attempt's own outcome decide the report", async () => {
    resetDiscoveryLedger();
    const older = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const newer = beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    completeDiscoveryAttempt(newer, { status: "ok", modelCount: 1, rows: rowsB });
    completeDiscoveryAttempt(older, { status: "superseded", modelCount: 2, rows: rowsB });

    const report = summarizeDiscovery(getDiscoveryOutcomesSince(0));
    assert.equal(report.ok, true);
    assert.equal(report.modelCount, 1, "the count comes from the attempt that published");
  });
});

describe("key fingerprints", () => {
  it("distinguishes different credentials and matches identical ones", () => {
    assert.notEqual(keyFingerprint("key-a"), keyFingerprint("key-b"));
    assert.equal(keyFingerprint("key-a"), keyFingerprint("key-a"));
  });

  it("treats an absent key as its own fingerprint", () => {
    assert.equal(keyFingerprint(undefined), "");
    assert.notEqual(keyFingerprint(undefined), keyFingerprint("key-a"));
  });
});

describe("last-known-good serving", () => {
  it("serves the committed catalog when the same credential fails, while the outcome stays failed", () => {
    resetDiscoveryLedger();
    const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "503 Service Unavailable" });

    // The fetcher serves the prior catalog on the failed fetch: the host
    // keeps a usable catalog.
    assert.deepEqual(lastGoodRowsForFailedAttempt(failed), rowsA);
    // But the recorded outcome is the FAILURE: reporting reads the ledger,
    // never the served rows, so the failure is still surfaced.
    const report = summarizeDiscovery(getDiscoveryOutcomesSince(good.id));
    assert.equal(report.ok, false);
    assert.match(report.error ?? "", /503/);
    assert.equal(report.modelCount, 0);
  });

  it("propagates on a first-ever failure (nothing to preserve)", () => {
    resetDiscoveryLedger();
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
    assert.equal(lastGoodRowsForFailedAttempt(failed), undefined);
  });

  it("never serves another credential's committed rows", () => {
    resetDiscoveryLedger();
    const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
    const failedB = beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    completeDiscoveryAttempt(failedB, { status: "failed", error: "boom" });
    assert.equal(lastGoodRowsForFailedAttempt(failedB), undefined);
  });

  it("serves nothing for a no-key attempt even with rows committed", () => {
    resetDiscoveryLedger();
    const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
    const noKey = beginDiscoveryAttempt(keyFingerprint(undefined), nextId());
    completeDiscoveryAttempt(noKey, { status: "no-key", modelCount: 0, rows: [] });
    // Logout never restores account rows: the empty-fingerprint attempt has
    // no entry to serve.
    assert.equal(lastGoodRowsForFailedAttempt(noKey), undefined);
  });

  it("serves nothing when a newer different-credential attempt owns the catalog (veto)", () => {
    resetDiscoveryLedger();
    const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
    // A rotation is in flight: key-a is retired, its last-known-good rows
    // must not be served even though they exist.
    beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    assert.deepEqual(lastGoodRowsForFailedAttempt(failed), [], "a retired account's rows must not be served");
  });

  it("still serves its own rows when a newer SAME-credential attempt is in flight", () => {
    resetDiscoveryLedger();
    const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
    beginDiscoveryAttempt(keyFingerprint("key-a"), nextId()); // overlapping same-account refresh
    assert.deepEqual(lastGoodRowsForFailedAttempt(failed), rowsA);
  });

  it("a superseded completion must not poison the store with the newer account's rows", () => {
    resetDiscoveryLedger();
    const older = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    const newer = beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    completeDiscoveryAttempt(newer, { status: "ok", modelCount: 1, rows: rowsB });
    completeDiscoveryAttempt(older, { status: "superseded", modelCount: 1, rows: rowsB });
    // key-a never committed its OWN catalog; a later key-a failure must
    // propagate — serving key-b's rows under key-a would be cross-account.
    const failedA = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failedA, { status: "failed", error: "boom" });
    assert.equal(lastGoodRowsForFailedAttempt(failedA), undefined);
    const failedB = beginDiscoveryAttempt(keyFingerprint("key-b"), nextId());
    completeDiscoveryAttempt(failedB, { status: "failed", error: "boom" });
    assert.deepEqual(lastGoodRowsForFailedAttempt(failedB), rowsB);
  });

  it("an empty successful catalog does not evict the last-good entry", () => {
    resetDiscoveryLedger();
    const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
    const empty = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(empty, { status: "ok", modelCount: 0, rows: [] });
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
    // "Last-known-good" means a catalog with content; an empty success must
    // not destroy the rows that carry the account through a later failure.
    assert.deepEqual(lastGoodRowsForFailedAttempt(failed), rowsA);
  });

  it("a newer successful refresh replaces the stored rows", () => {
    resetDiscoveryLedger();
    const first = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(first, { status: "ok", modelCount: 2, rows: rowsA });
    const updated = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(updated, { status: "ok", modelCount: 1, rows: rowsB });
    const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
    completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
    assert.deepEqual(lastGoodRowsForFailedAttempt(failed), rowsB);
  });

  it("stays bounded across refreshes and credentials", () => {
    resetDiscoveryLedger();
    const keys = ["k1", "k2", "k3", "k4", "k5", "k6"];
    for (const k of keys) {
      const attempt = beginDiscoveryAttempt(keyFingerprint(k), nextId());
      completeDiscoveryAttempt(attempt, { status: "ok", modelCount: 1, rows: [row(`${k}/one`)] });
    }
    // The store is capped at 4 credentials: k1 and k2 were evicted, the
    // rest still serve their own rows.
    for (const k of keys) {
      const failed = beginDiscoveryAttempt(keyFingerprint(k), nextId());
      completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
      const served = lastGoodRowsForFailedAttempt(failed);
      if (k === "k1" || k === "k2") {
        assert.equal(served, undefined, `${k} was evicted by the cap`);
      } else {
        assert.deepEqual(served, [row(`${k}/one`)], `${k} still serves`);
      }
    }
    // Re-committing k6 refreshes its recency: the next credential evicts
    // k3 (the oldest survivor), not k6.
    const again = beginDiscoveryAttempt(keyFingerprint("k6"), nextId());
    completeDiscoveryAttempt(again, { status: "ok", modelCount: 1, rows: [row("k6/two")] });
    const k7 = beginDiscoveryAttempt(keyFingerprint("k7"), nextId());
    completeDiscoveryAttempt(k7, { status: "ok", modelCount: 1, rows: [row("k7/one")] });
    const failedK3 = beginDiscoveryAttempt(keyFingerprint("k3"), nextId());
    completeDiscoveryAttempt(failedK3, { status: "failed", error: "boom" });
    assert.equal(lastGoodRowsForFailedAttempt(failedK3), undefined, "k3 was evicted");
    const failedK6 = beginDiscoveryAttempt(keyFingerprint("k6"), nextId());
    completeDiscoveryAttempt(failedK6, { status: "failed", error: "boom" });
    assert.deepEqual(lastGoodRowsForFailedAttempt(failedK6), [row("k6/two")]);
  });

  it("mirrors last-good rows for fresh processes and tolerates a corrupt mirror", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dc-lastgood-"));
    const mirror = path.join(dir, "last-good.json");
    try {
      resetDiscoveryLedger();
      configureLastGoodPersistence(mirror);
      const good = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
      completeDiscoveryAttempt(good, { status: "ok", modelCount: 2, rows: rowsA });
      assert.ok(fs.existsSync(mirror), "commit must write the mirror");
      assert.ok(!fs.readFileSync(mirror, "utf8").includes("key-a"), "the mirror must never store raw keys");

      // Fresh process: ledger wiped, seam re-attached to the same mirror.
      resetDiscoveryLedger();
      configureLastGoodPersistence(mirror);
      const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
      completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
      assert.deepEqual(lastGoodRowsForFailedAttempt(failed), rowsA);

      // A corrupt or unreadable mirror must not serve garbage or crash.
      fs.writeFileSync(mirror, "{not json");
      resetDiscoveryLedger();
      configureLastGoodPersistence(mirror);
      const failed2 = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
      completeDiscoveryAttempt(failed2, { status: "failed", error: "boom" });
      assert.equal(lastGoodRowsForFailedAttempt(failed2), undefined);
    } finally {
      resetDiscoveryLedger();
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("an absent mirror behaves like first-ever failure", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "dc-lastgood-"));
    try {
      resetDiscoveryLedger();
      configureLastGoodPersistence(path.join(dir, "never-written.json"));
      const failed = beginDiscoveryAttempt(keyFingerprint("key-a"), nextId());
      completeDiscoveryAttempt(failed, { status: "failed", error: "boom" });
      assert.equal(lastGoodRowsForFailedAttempt(failed), undefined);
    } finally {
      resetDiscoveryLedger();
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
});
