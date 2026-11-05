#!/usr/bin/env bun
/**
 * Seed fixture OpenRouter credentials into an omp agent directory (pinned
 * omp 18.16 auth schema, introspected read-only from a live 18.16 agent.db;
 * no real credentials are ever written).
 *
 *   bun scripts/seed-omp-db.mjs <agentDir> [apiKey]
 *
 * `agentDir` is the directory that receives agent.db (create it inside a
 * temporary fixture HOME -- never the user's real ~/.omp). `apiKey` defaults
 * to FIXTURE_KEY (sk-or-test-container).
 *
 * Exports seedOmpDb(agentDir, apiKey):
 *   - if agent.db exists and omp already initialized it, only the openrouter
 *     credential row is upserted;
 *   - if the db is absent, a realistic auth-slice schema is created first.
 *     Recommended path is still to let omp initialize the directory and seed
 *     afterwards, so every table omp owns exists.
 */

import { mkdirSync } from "node:fs";
import path from "node:path";
import { Database } from "bun:sqlite";
import { FIXTURE_KEY } from "../test/fixtures/openrouter.mjs";

// Pinned omp 18.16 auth schema (verbatim introspection, minus the
// openai-codex oauth mirror triggers, which only guard oauth rows).
const AUTH_SCHEMA_DDL = `
CREATE TABLE IF NOT EXISTS auth_schema_version (
				id INTEGER PRIMARY KEY CHECK (id = 1),
				version INTEGER NOT NULL
			);
CREATE TABLE IF NOT EXISTS auth_change_revision (
				id INTEGER PRIMARY KEY CHECK (id = 1),
				revision INTEGER NOT NULL
			);
CREATE TABLE IF NOT EXISTS auth_credentials (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				provider TEXT NOT NULL,
				credential_type TEXT NOT NULL,
				data TEXT NOT NULL,
				disabled_cause TEXT DEFAULT NULL,
				identity_key TEXT DEFAULT NULL,
				created_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER)),
				updated_at INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
			);
CREATE TABLE IF NOT EXISTS auth_credential_blocks (
				credential_id INTEGER NOT NULL,
				provider_key TEXT NOT NULL,
				block_scope TEXT NOT NULL DEFAULT '',
				blocked_until_ms INTEGER NOT NULL,
				updated_at INTEGER NOT NULL,
				PRIMARY KEY (credential_id, provider_key, block_scope)
			);
CREATE TABLE IF NOT EXISTS auth_credential_refresh_leases (
				credential_id INTEGER PRIMARY KEY,
				owner TEXT NOT NULL,
				expires_at_ms INTEGER NOT NULL,
				updated_at INTEGER NOT NULL
			);
CREATE TABLE IF NOT EXISTS auth_credential_block_mirror_guard (
				credential_id INTEGER PRIMARY KEY
			) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_auth_provider ON auth_credentials(provider);
CREATE INDEX IF NOT EXISTS idx_auth_provider_identity ON auth_credentials(provider, identity_key) WHERE identity_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_auth_credential_blocks_expires ON auth_credential_blocks(blocked_until_ms);
CREATE INDEX IF NOT EXISTS idx_auth_credential_refresh_leases_expires ON auth_credential_refresh_leases(expires_at_ms);
CREATE TRIGGER IF NOT EXISTS auth_change_revision_auth_credentials_insert
					AFTER INSERT ON auth_credentials
					BEGIN
						UPDATE auth_change_revision SET revision = revision + 1 WHERE id = 1;
					END;
CREATE TRIGGER IF NOT EXISTS auth_change_revision_auth_credentials_update
					AFTER UPDATE ON auth_credentials
					BEGIN
						UPDATE auth_change_revision SET revision = revision + 1 WHERE id = 1;
					END;
CREATE TRIGGER IF NOT EXISTS auth_change_revision_auth_credentials_delete
					AFTER DELETE ON auth_credentials
					BEGIN
						UPDATE auth_change_revision SET revision = revision + 1 WHERE id = 1;
					END;
`;

/** Pinned omp 18.16 schema version. */
const AUTH_SCHEMA_VERSION = 8;

export function seedOmpDb(agentDir, apiKey = FIXTURE_KEY) {
  const dbPath = path.join(agentDir, "agent.db");
  mkdirSync(agentDir, { recursive: true });

  const db = new Database(dbPath, { create: true });
  try {
    db.exec("PRAGMA journal_mode = WAL;");

    // A prior omp startup can leave a partial db. The auth tables, not the
    // unrelated settings table, determine whether credential upserts are safe.
    const hasAuthSchema = Boolean(
      db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='auth_credentials'").get(),
    );
    if (!hasAuthSchema) {
      db.exec(AUTH_SCHEMA_DDL);
      db.query("INSERT OR REPLACE INTO auth_schema_version (id, version) VALUES (1, ?)").run(AUTH_SCHEMA_VERSION);
      db.query("INSERT OR REPLACE INTO auth_change_revision (id, revision) VALUES (1, 0)").run();
    }

    // Live 18.16 row shape: credential_type 'api_key', data JSON with key and
    // source:'login'. Replace any existing openrouter rows for idempotency.
    db.query("DELETE FROM auth_credentials WHERE provider = 'openrouter'").run();
    db.query(
      "INSERT INTO auth_credentials (provider, credential_type, data) VALUES ('openrouter', 'api_key', ?)",
    ).run(JSON.stringify({ key: apiKey, source: "login" }));

    // The auth_change_revision triggers only fire through omp's own writes;
    // bump the revision directly so cached readers re-check credentials.
    if (hasAuthSchema) {
      db.query("UPDATE auth_change_revision SET revision = revision + 1 WHERE id = 1").run();
    }

    const row = db
      .query("SELECT provider, credential_type, data FROM auth_credentials WHERE provider = 'openrouter'")
      .get();
    if (!row || JSON.parse(row.data).key !== apiKey) {
      throw new Error(`seedOmpDb: verification failed for ${dbPath}`);
    }
    return dbPath;
  } finally {
    db.close();
  }
}

function main() {
  const agentDir = process.argv[2];
  const key = process.argv[3] ?? FIXTURE_KEY;
  if (!agentDir) {
    console.error("Usage: bun scripts/seed-omp-db.mjs <agentDir> [apiKey]");
    process.exit(2);
  }
  const dbPath = seedOmpDb(agentDir, key);
  console.log(`Seeded ${dbPath} with fixture credential for provider "openrouter".`);
}

if (import.meta.main) main();
