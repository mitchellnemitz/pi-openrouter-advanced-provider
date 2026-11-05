/**
 * Discovery-attempt ledger.
 *
 * A "discovery" is one attempt to fetch a credential's account catalog. The
 * host owns the refresh schedule, so the extension cannot know in advance how
 * many attempts one /openrouter-sync produces, nor which credential each one
 * used.
 *
 * Every attempt therefore records its outcome here, and the caller reports
 * the outcome of the attempt ITS refresh produced instead of inferring success
 * from the committed snapshot. The snapshot is only committed on success, so
 * after a failed refresh it still holds the PREVIOUS catalog: reporting its
 * size would announce the old catalog as a fresh success (and announce zero
 * on a first-ever failure).
 *
 * The ledger also decides which attempt may publish rows. A host may run
 * several discoveries for one provider concurrently (verified against omp
 * 18.16.0: its startup discovery and the extension's own refresh overlap), so
 * a slower attempt can finish after a newer one. Its rows describe an older
 * catalog — and possibly an older ACCOUNT — so the host must not consume
 * them. A superseded attempt defers to the newer attempt: it publishes the
 * newer attempt's rows, so every attempt of one refresh publishes the same,
 * newest catalog.
 *
 * Cross-credential supersession is the dangerous case: the newer attempt may
 * use a DIFFERENT credential (a login/logout or key rotation between two
 * refreshes). The older attempt must not publish its own rows at all — they
 * belong to an account the user is no longer using. It publishes the newer
 * attempt's rows instead; if the newer attempt used a different credential
 * and failed, the older attempt publishes NOTHING, so the host's registry
 * keeps no rows of a retired account.
 *
 * Successful attempts also commit their own rows as the credential's
 * LAST-KNOWN-GOOD catalog. omp evicts both its live catalog and its
 * persisted cache when a discovery fails, so the fetcher serves those
 * last-known-good rows on a failed attempt for the same credential — the
 * host keeps a usable prior catalog while the attempt's outcome stays
 * "failed". The store (and its optional on-disk mirror for fresh omp
 * processes) is keyed by fingerprint and capped, never holds raw keys, and
 * never serves across credentials.
 *
 * This module deliberately imports nothing but node:crypto, node:fs and
 * types: the unit tests drive it directly, without a live host.
 */

import fs from "node:fs";
import { createHash } from "node:crypto";
import type { ProviderModelConfig } from "./types.js";

/** What one discovery attempt achieved. */
export type DiscoveryStatus =
  /** The attempt fetched a catalog and it may be published. */
  | "ok"
  /**
   * The attempt failed. The failure is what gets reported; the credential's
   * last-known-good catalog, if one exists, is served again so the host
   * keeps a usable catalog.
   */
  | "failed"
  /**
   * The attempt had no credential to fetch with, so it published no rows.
   */
  | "no-key"
  /**
   * The attempt was superseded by a newer one. It published the newer
   * attempt's rows (or none at all), so it contributes no count of its own.
   */
  | "superseded";

export interface DiscoveryOutcome {
  /** Attempt id, increasing with every discovery attempt. */
  attempt: number;
  /** Status of the attempt. */
  status: DiscoveryStatus;
  /** Rows the attempt published (0 unless it published rows). */
  modelCount: number;
  /** Failure message (failed attempts only). */
  error?: string;
  /**
   * Fingerprint of the credential the attempt used. Two attempts with
   * different fingerprints are different accounts, so an older account's rows
   * must never reach the host registry.
   */
  keyFingerprint: string;
}

/** Summary of the attempts belonging to one refresh. */
export interface DiscoveryReport {
  /** True when a catalog was fetched and published. */
  ok: boolean;
  /** Rows the successful attempt published (0 when not ok). */
  modelCount: number;
  /** Failure message (not ok). */
  error?: string;
  /** True when the attempt had no credential. */
  noKey: boolean;
}

/** A discovery attempt in flight. */
export interface DiscoveryAttempt {
  /** Attempt id, increasing with every discovery attempt. */
  id: number;
  /** Fingerprint of the credential this attempt used. */
  keyFingerprint: string;
  /**
   * Resolves with the rows this attempt publishes. A superseded attempt
   * resolves with the rows of the newer attempt it deferred to, so every
   * attempt of one refresh publishes the same, newest catalog.
   */
  published: Promise<ProviderModelConfig[]>;
}

interface AttemptRecord extends DiscoveryAttempt {
  resolvePublished: (rows: ProviderModelConfig[]) => void;
  rejectPublished: (error: unknown) => void;
  settled: boolean;
}

/**
 * A discovery attempt failed. Carries the fingerprint of the credential that
 * failed, so a superseded attempt can tell whether the failure belonged to
 * its own account (its rows stay publishable) or to a different one (they
 * must not be published).
 */
export class DiscoveryFailedError extends Error {
  readonly keyFingerprint: string;

  constructor(
    message: string,
    keyFingerprint: string,
  ) {
    super(message);
    this.keyFingerprint = keyFingerprint;
    this.name = "DiscoveryFailedError";
  }
}

const attempts = new Map<number, AttemptRecord>();
const outcomes: DiscoveryOutcome[] = [];

// ---------------------------------------------------------------------------
// Last-known-good catalogs
//
// omp evicts BOTH its live catalog and its persisted `agent/models.db` cache
// when a discovery fails, so a failed fetch for a credential that previously
// committed a good catalog must serve those rows again or the account loses
// its usable catalog. One entry per credential fingerprint; a successful
// commit replaces its entry, and the store is capped so neither repeated
// refreshes nor distinct credentials can grow it unboundedly.
// ---------------------------------------------------------------------------

const LAST_GOOD_MAX_KEYS = 4;
const lastGoodRowsByKey = new Map<string, ProviderModelConfig[]>();

/**
 * Optional on-disk mirror of the last-good store. omp runs are one-shot
 * processes; without a mirror a FRESH process has no last-good rows to serve
 * and its first failed discovery would evict the persisted catalog the
 * previous process left behind. The path is supplied by pi/index.ts (omp
 * only); null disables persistence. Raw keys are never stored — only
 * fingerprints.
 */
let lastGoodFilePath: string | null = null;
let lastGoodHydrated = true;

/** Attach (or detach, with null) the on-disk mirror of the last-good store. */
export function configureLastGoodPersistence(filePath: string | null): void {
  lastGoodFilePath = filePath;
  lastGoodHydrated = filePath === null;
}

/** Load the on-disk mirror once, for credentials this process has not committed itself. */
function hydrateLastGoodRows(): void {
  if (lastGoodHydrated) return;
  lastGoodHydrated = true;
  if (!lastGoodFilePath) return;
  let entries: unknown[] = [];
  try {
    const parsed: unknown = JSON.parse(fs.readFileSync(lastGoodFilePath, "utf8"));
    if (typeof parsed === "object" && parsed !== null && "entries" in parsed && Array.isArray(parsed.entries)) {
      entries = parsed.entries;
    }
  } catch {
    return; // Absent or unreadable mirror: nothing to restore.
  }
  for (const entry of entries) {
    if (typeof entry !== "object" || entry === null) continue;
    if (!("keyFingerprint" in entry) || !("rows" in entry)) continue;
    const fingerprint = typeof entry.keyFingerprint === "string" ? entry.keyFingerprint : "";
    const rawRows: unknown[] = Array.isArray(entry.rows) ? entry.rows : [];
    const rows = rawRows.filter((row): row is ProviderModelConfig => {
      // Structural guard: only accept entries that look like model configs.
      return typeof row === "object" && row !== null && "id" in row && typeof row.id === "string";
    });
    // Rows this process already committed are at least as fresh as the mirror.
    if (!fingerprint || rows.length === 0 || lastGoodRowsByKey.has(fingerprint)) continue;
    lastGoodRowsByKey.set(fingerprint, rows);
  }
}

/** Write the whole store to the mirror. Best-effort: a failed write must never break discovery. */
function persistLastGoodRows(): void {
  if (!lastGoodFilePath) return;
  try {
    const payload = JSON.stringify({
      version: 1,
      entries: [...lastGoodRowsByKey.entries()].map(([keyFingerprint, rows]) => ({ keyFingerprint, rows })),
    });
    const tmp = `${lastGoodFilePath}.tmp`;
    fs.writeFileSync(tmp, payload);
    fs.renameSync(tmp, lastGoodFilePath);
  } catch {
    // The mirror is an optimization for fresh processes; ignore write errors.
  }
}

/**
 * Commit a successful catalog as the credential's last-known-good rows.
 *
 * Only non-empty catalogs commit: "last-known-good" means a catalog with
 * content, and an empty successful discovery must not destroy the rows that
 * keep the account usable through a later failure.
 */
function noteLastGoodRows(fingerprint: string, rows: ProviderModelConfig[]): void {
  if (!fingerprint || rows.length === 0) return;
  // Re-insert to refresh recency, then cap the store.
  lastGoodRowsByKey.delete(fingerprint);
  lastGoodRowsByKey.set(fingerprint, rows);
  while (lastGoodRowsByKey.size > LAST_GOOD_MAX_KEYS) {
    const oldest = lastGoodRowsByKey.keys().next().value;
    if (oldest === undefined) break;
    lastGoodRowsByKey.delete(oldest);
  }
  persistLastGoodRows();
}

/**
 * Rows to serve for a FAILED discovery attempt.
 *
 * omp wipes the served catalog AND its persisted `agent/models.db` cache when
 * discovery fails, so a failed fetch for a credential that previously
 * committed a good catalog must serve those last-known-good rows again: the
 * host keeps a usable prior catalog while the attempt's recorded outcome
 * stays "failed" (reporting never announces the served rows as a success).
 *
 * Returns undefined when there is nothing to preserve — a first-ever failure
 * propagates its error. Returns [] when a newer attempt with a DIFFERENT
 * credential owns the catalog: this account is retired and its rows must
 * never be published, last-known-good or not.
 */
export function lastGoodRowsForFailedAttempt(attempt: DiscoveryAttempt): ProviderModelConfig[] | undefined {
  hydrateLastGoodRows();
  const rows = lastGoodRowsByKey.get(attempt.keyFingerprint);
  if (!rows || rows.length === 0) return undefined;
  const newer = getNewerDiscoveryAttempt(attempt.id);
  if (newer && newer.keyFingerprint !== attempt.keyFingerprint) return [];
  return rows;
}

/**
 * Fingerprint of a credential: a short digest that identifies "same
 * credential" without storing the key itself. Empty for an absent key.
 */
export function keyFingerprint(apiKey: string | undefined): string {
  if (!apiKey) return "";
  return createHash("sha256").update(apiKey).digest("hex").slice(0, 16);
}

/**
 * Start a discovery attempt; returns the attempt handle.
 *
 * `nextId` is supplied by the caller so the attempt id shares one ordering
 * with the committed catalog's generation: "a newer attempt started" and
 * "this snapshot is stale" can then never disagree about which is newer.
 */
export function beginDiscoveryAttempt(keyFingerprintValue: string, nextId: number): DiscoveryAttempt {
  const id = nextId;
  // Promise.withResolvers is ES2024; this package compiles against lib
  // es2022, so the executor form is the one that type-checks here.
  let resolvePublished!: (rows: ProviderModelConfig[]) => void;
  let rejectPublished!: (error: unknown) => void;
  const published = new Promise<ProviderModelConfig[]>((resolve, reject) => {
    resolvePublished = resolve;
    rejectPublished = reject;
  });
  // An attempt nobody awaits (the common case) must not raise an unhandled
  // rejection when it fails.
  published.catch(() => {});
  const record: AttemptRecord = {
    id,
    keyFingerprint: keyFingerprintValue,
    published,
    resolvePublished,
    rejectPublished,
    settled: false,
  };
  attempts.set(id, record);
  return record;
}

/** The newest attempt started after the given one, if any. */
export function getNewerDiscoveryAttempt(attemptId: number): DiscoveryAttempt | undefined {
  let newest: AttemptRecord | undefined;
  for (const record of attempts.values()) {
    if (record.id > attemptId && (!newest || record.id > newest.id)) newest = record;
  }
  return newest;
}

/**
 * Finish a discovery attempt.
 *
 * `superseded` means the attempt published another attempt's rows instead of
 * its own, so it reports no count of its own: the newer attempt is what the
 * refresh actually delivered.
 */
export function completeDiscoveryAttempt(
  attempt: DiscoveryAttempt,
  result: {
    status: DiscoveryStatus;
    modelCount?: number;
    error?: string;
    /** Rows this attempt publishes to the host. */
    rows?: ProviderModelConfig[];
  },
): void {
  const record = attempts.get(attempt.id);
  const modelCount = result.modelCount ?? 0;
  recordDiscoveryOutcome({
    attempt: attempt.id,
    status: result.status,
    modelCount,
    error: result.error,
    keyFingerprint: attempt.keyFingerprint,
  });
  if (!record || record.settled) return;
  record.settled = true;
  if (result.status === "failed") {
    record.rejectPublished(new DiscoveryFailedError(result.error ?? "catalog fetch failed", attempt.keyFingerprint));
    return;
  }
  // A successful attempt commits its OWN rows as its credential's
  // last-known-good catalog. A superseded attempt publishes ANOTHER
  // attempt's rows — committing them here would file the newer account's
  // catalog under the older credential's fingerprint, so only "ok" commits.
  if (result.status === "ok") noteLastGoodRows(attempt.keyFingerprint, result.rows ?? []);
  record.resolvePublished(result.rows ?? []);
}

/**
 * Await the rows a newer attempt publishes.
 *
 * Used when a cross-credential attempt finds itself superseded: it must not
 * publish its own (retired account's) rows, so it publishes the newer
 * attempt's rows instead. If the newer attempt failed, the rejection carries
 * the failure so the caller can surface it instead of publishing stale rows.
 */
export async function awaitNewerDiscoveryRows(newer: DiscoveryAttempt): Promise<ProviderModelConfig[]> {
  return newer.published;
}

/**
 * Decide what a finished attempt may publish.
 *
 * A host may run several discoveries for one provider concurrently, so an
 * attempt can find that a NEWER attempt already started. The newer attempt
 * owns the catalog then, and this attempt must not publish its own rows:
 *
 *   - Same credential: the newer attempt's rows are this account's catalog
 *     too, so publishing them keeps every attempt of the refresh consistent.
 *   - Different credential (a login/logout or key rotation between the two
 *     refreshes): the newer attempt's rows are the CURRENT account's. The
 *     retired account's rows must never reach the host registry, so this
 *     attempt publishes the newer attempt's rows — or, when that attempt
 *     failed, nothing at all.
 */
export async function publishDiscoveryRows(
  attempt: DiscoveryAttempt,
  ownRows: ProviderModelConfig[],
): Promise<{ status: DiscoveryStatus; rows: ProviderModelConfig[] }> {
  const newer = getNewerDiscoveryAttempt(attempt.id);
  if (!newer) return { status: "ok", rows: ownRows };
  try {
    const rows = await awaitNewerDiscoveryRows(newer);
    return { status: "superseded", rows };
  } catch (err) {
    if (newer.keyFingerprint === attempt.keyFingerprint) {
      // The newer attempt failed for the SAME account: this attempt's rows are
      // still the current account's catalog, so they stay publishable.
      return { status: "ok", rows: ownRows };
    }
    // The newer attempt used a DIFFERENT credential and failed. Publishing
    // this attempt's rows would restore the retired account's catalog.
    return { status: "superseded", rows: [] };
  }
}

/** Record the outcome of a discovery attempt. */
export function recordDiscoveryOutcome(outcome: DiscoveryOutcome): void {
  outcomes.push(outcome);
  // The ledger only needs the attempts of the refresh in flight; keep it bounded.
  if (outcomes.length > 64) outcomes.splice(0, outcomes.length - 64);
}

/** Outcomes recorded after the given attempt id. */
export function getDiscoveryOutcomesSince(attemptId: number): DiscoveryOutcome[] {
  return outcomes.filter((outcome) => outcome.attempt > attemptId);
}

/**
 * Summarize the attempts of one refresh.
 *
 * The newest attempt that actually published or failed decides: it is the
 * most recent thing the host asked for, so its outcome is the refresh's
 * outcome. An attempt that published no rows because it had no credential is
 * reported as such rather than as a failure, and a superseded attempt
 * contributes no count of its own — the newer attempt it deferred to reports
 * the count.
 */
export function summarizeDiscovery(outcomesForRefresh: DiscoveryOutcome[]): DiscoveryReport {
  const decisive = outcomesForRefresh.filter((outcome) => outcome.status !== "superseded");
  const newest = decisive[decisive.length - 1];
  if (!newest) {
    return {
      ok: false,
      modelCount: 0,
      error: "no catalog discovery completed",
      noKey: false,
    };
  }
  if (newest.status === "ok") {
    return { ok: true, modelCount: newest.modelCount, noKey: false };
  }
  if (newest.status === "no-key") {
    return { ok: false, modelCount: 0, error: "no API key configured", noKey: true };
  }
  return { ok: false, modelCount: 0, error: newest.error ?? "catalog fetch failed", noKey: false };
}

/** Drop all recorded attempts and outcomes (module reload / test isolation). */
export function resetDiscoveryLedger(): void {
  attempts.clear();
  outcomes.length = 0;
  lastGoodRowsByKey.clear();
  lastGoodFilePath = null;
  lastGoodHydrated = true;
}
