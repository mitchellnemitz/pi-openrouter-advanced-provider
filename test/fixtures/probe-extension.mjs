/**
 * Test-only probe extension for the containerized dual-host harness.
 *
 * Loaded by BOTH hosts next to the provider extension via explicit
 * `--extension` flags (never shipped, never discovered: the scenario runs
 * hosts with `--no-extensions` plus explicit paths). It is a pure observer:
 * it registers event handlers and appends NDJSON records to the artifact
 * file named by OPENROUTER_PROBE_ARTIFACT so the container scenario can
 * assert on registry/session observables the host itself produced.
 *
 *   export OPENROUTER_PROBE_ARTIFACT=/path/probe.ndjson   # required; inert without it
 *   export OPENROUTER_PROBE_LABEL="omp/keyed"             # optional tag for diagnostics
 *
 * Records (one JSON object per line):
 *   { type: "start", label, host, pid }
 *   { type: "session_start" }
 *   { type: "registry", source, rows: [{ id, api, ... }], ...source extras }
 *   { type: "before_provider_request", model, provider }
 *   { type: "message_end", message: { provider, model, api, stopReason, errorMessage, usage, text } }
 *   { type: "agent_end" } / { type: "session_shutdown" }
 *   { type: "probe_error", where, message }
 *
 * Registry observables come from up to three independent sources, each
 * feature-detected and best-effort:
 *   - "host-registry": ctx.modelRegistry.getProviderModels("openrouter")
 *     (omp only; rows carry the recovery api id after the fetcher remap)
 *   - "provider-config": ctx.modelRegistry.getRegisteredProviderConfig
 *     ("openrouter") (pi; the registered ProviderConfig incl. per-row api)
 *   - "snapshot": the provider extension's committed catalog snapshot via
 *     the shared pi/state module (works only when the host's extension
 *     loader dedupes module instances by resolved path; probed last).
 *
 * Everything is guarded: a probe failure must degrade to fewer records,
 * never break the host run.
 */

import fs from "node:fs";

const PROVIDER_NAME = "openrouter";

/** Project a registry/model row down to the observable fields we assert on. */
function normalizeRow(row) {
  if (!row || typeof row !== "object") return { id: String(row) };
  return {
    id: typeof row.id === "string" ? row.id : undefined,
    api: typeof row.api === "string" ? row.api : undefined,
    provider: typeof row.provider === "string" ? row.provider : undefined,
    name: typeof row.name === "string" ? row.name : undefined,
    contextWindow: typeof row.contextWindow === "number" ? row.contextWindow : undefined,
    maxTokens: typeof row.maxTokens === "number" ? row.maxTokens : undefined,
  };
}

/** Join assistant content blocks into plain text for response assertions. */
function contentText(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => (block && typeof block === "object" && block.type === "text" && typeof block.text === "string" ? block.text : ""))
    .join("");
}

export default async function probeExtension(pi) {
  const artifactPath = process.env.OPENROUTER_PROBE_ARTIFACT ?? "";
  if (!artifactPath) return; // Inert outside the container scenario.

  const label = process.env.OPENROUTER_PROBE_LABEL ?? "";
  let broken = false;

  const write = (record) => {
    if (broken) return;
    try {
      fs.appendFileSync(artifactPath, `${JSON.stringify(record)}\n`);
    } catch {
      // One failed append disables the probe rather than spamming the host.
      broken = true;
    }
  };

  const guard = (where, fn) => {
    try {
      return fn();
    } catch (error) {
      write({ type: "probe_error", where, message: error instanceof Error ? error.message : String(error) });
      return undefined;
    }
  };

  // Host discrimination mirrors pi/platform.ts initHost: omp-only
  // ExtensionAPI markers, never directory probing.
  const api = pi ?? {};
  const host = "registerComposerShape" in api || "arktype" in api ? "omp" : "pi";

  // Shared catalog snapshot: same module instance as the provider extension
  // whenever the host loader dedupes by resolved path. A plain relative
  // literal keeps omp's legacy-compat rewriting out of the way; under omp the
  // host-registry source covers registry observables regardless.
  let stateModule;
  try {
    stateModule = await import("../../pi/state.js");
  } catch {
    stateModule = undefined;
  }

  const lastBySource = new Map();

  /**
   * Dump every available registry source; emit only changed sources so the
   * artifact stays small and the scenario can assert "no rows ever appeared"
   * without a flood of duplicates.
   */
  const dumpRegistries = (ctx) =>
    guard("registry", () => {
      const dumps = [];

      const hostRows = guard("host-registry", () =>
        ctx?.modelRegistry?.getProviderModels?.(PROVIDER_NAME));
      if (Array.isArray(hostRows)) {
        dumps.push({ source: "host-registry", rows: hostRows.map(normalizeRow) });
      }

      const providerConfig = guard("provider-config", () =>
        ctx?.modelRegistry?.getRegisteredProviderConfig?.(PROVIDER_NAME));
      if (providerConfig && typeof providerConfig === "object") {
        dumps.push({
          source: "provider-config",
          providerApi: typeof providerConfig.api === "string" ? providerConfig.api : undefined,
          hasStreamSimple: typeof providerConfig.streamSimple === "function",
          headers: providerConfig.headers && typeof providerConfig.headers === "object" ? providerConfig.headers : undefined,
          rows: Array.isArray(providerConfig.models) ? providerConfig.models.map(normalizeRow) : [],
        });
      }

      const snapshot = guard("snapshot", () => stateModule?.getSnapshot?.());
      if (snapshot && Array.isArray(snapshot.models)) {
        dumps.push({
          source: "snapshot",
          generation: guard("snapshot-generation", () => stateModule?.getGeneration?.()),
          rows: snapshot.models.map(normalizeRow),
        });
      }

      for (const dump of dumps) {
        const line = JSON.stringify(dump);
        if (lastBySource.get(dump.source) === line) continue;
        lastBySource.set(dump.source, line);
        write({ type: "registry", ...dump });
      }
    });

  write({ type: "start", label, host, pid: process.pid });
  dumpRegistries();

  pi.on("session_start", (_event, ctx) => {
    write({ type: "session_start" });
    dumpRegistries(ctx);
  });

  pi.on("before_agent_start", (_event, ctx) => {
    dumpRegistries(ctx);
  });

  pi.on("before_provider_request", (event, ctx) => {
    const payload = event?.payload;
    write({
      type: "before_provider_request",
      model: payload && typeof payload === "object" ? payload.model : undefined,
      provider: payload && typeof payload === "object" ? payload.provider : undefined,
    });
    dumpRegistries(ctx);
  });

  pi.on("message_end", (event, ctx) => {
    const message = event?.message;
    if (!message || message.role !== "assistant") return;
    write({
      type: "message_end",
      message: {
        provider: message.provider,
        model: message.model,
        api: message.api,
        stopReason: message.stopReason,
        errorMessage: message.errorMessage,
        responseId: message.responseId,
        usage: message.usage,
        text: contentText(message.content),
      },
    });
    dumpRegistries(ctx);
  });

  pi.on("agent_end", () => {
    write({ type: "agent_end" });
    dumpRegistries();
  });

  pi.on("session_shutdown", () => {
    write({ type: "session_shutdown" });
    dumpRegistries();
  });

  // Safety net for hosts that never fire a late event in print mode, and for
  // registry states that change outside events (omp startup refresh). The
  // unref keeps the interval from delaying host exit.
  const timer = setInterval(() => dumpRegistries(), 500);
  timer.unref?.();
}
