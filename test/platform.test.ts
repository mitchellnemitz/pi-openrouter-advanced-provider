/**
 * Unit tests for the platform adapter (pi/platform.ts): host detection from
 * a real ExtensionAPI shape, agent-root resolution, and auth.json key
 * parsing.
 *
 * Run: npm test   (bun test test/)
 */

import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";
import os from "node:os";
import path from "node:path";
import fs from "node:fs";

import { agentRoot, bootstrapApiKey, detectRuntime, initHost } from "../pi/platform.ts";

const RUNTIME_ENV = "OPENROUTER_ADVANCED_RUNTIME";

const ompApi = {
  registerComposerShape: () => {},
  arktype: {},
};
const piApi = {
  typebox: {},
  // Deliberately NOT a function: a non-omp host API must classify as pi
  // even when ~/.omp/agent exists on this machine.
};

afterEach(() => {
  delete process.env[RUNTIME_ENV];
});

describe("initHost", () => {
  it("classifies an omp-shaped ExtensionAPI as omp", () => {
    assert.equal(initHost(ompApi), "omp");
    assert.equal(detectRuntime(), "omp");
  });

  it("classifies a pi-shaped ExtensionAPI as pi without consulting the filesystem", () => {
    // A supplied host API must decide the classification; the storage-root
    // probe is only for no-API contexts. On a dual-installed machine
    // (~/.omp/agent present) the probe would wrongly say omp — which is
    // exactly why it must not run here.
    assert.equal(initHost(piApi), "pi");
    assert.equal(detectRuntime(), "pi");
  });

  it("host detection wins over later probe fallbacks", () => {
    assert.equal(initHost(ompApi), "omp");
    assert.equal(initHost(piApi), "pi");
    // No-arg calls keep the last real decision; they never re-probe over it.
    assert.equal(initHost(), "pi");
  });
});

describe("detectRuntime without initHost", () => {
  it("honors the env override", () => {
    process.env[RUNTIME_ENV] = "omp";
    // initHost with no argument reads the override before probing.
    assert.equal(initHost(), "omp");
    process.env[RUNTIME_ENV] = "pi";
    assert.equal(initHost(), "pi");
  });
});

describe("agentRoot", () => {
  it("uses PI_CODING_AGENT_DIR when set", () => {
    process.env.PI_CODING_AGENT_DIR = "/tmp/fake-agent-dir";
    try {
      assert.equal(agentRoot(), "/tmp/fake-agent-dir");
    } finally {
      delete process.env.PI_CODING_AGENT_DIR;
    }
  });

  it("falls back to homedir/CONFIG_DIR_NAME/agent", () => {
    delete process.env.PI_CODING_AGENT_DIR;
    const root = agentRoot();
    assert.ok(root.startsWith(os.homedir()));
    assert.ok(root.endsWith(path.join("agent")));
  });
});

describe("bootstrapApiKey auth.json parsing", () => {
  // Hermetic: a developer/CI environment may export OPENROUTER_API_KEY or
  // have host stores with real keys; save/clear what the resolution reads so
  // assertions never depend on — or print — real credentials. Env handling
  // is inline (no hooks) to keep it obvious and shim-proof.
  it("reads an openrouter key from the host auth store", async () => {
    const savedKey = process.env.OPENROUTER_API_KEY;
    const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
    const savedRuntime = process.env[RUNTIME_ENV];
    delete process.env.OPENROUTER_API_KEY;
    delete process.env[RUNTIME_ENV];
    // Isolated agent dir: point agentRoot at a temp dir via the env var.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "oradv-auth-"));
    process.env.PI_CODING_AGENT_DIR = dir;
    try {
      fs.writeFileSync(
        path.join(dir, "auth.json"),
        JSON.stringify({ openrouter: { key: "sk-or-test-123" } }),
      );
      assert.equal(await bootstrapApiKey(), "sk-or-test-123");
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
      if (savedKey !== undefined) process.env.OPENROUTER_API_KEY = savedKey;
      else delete process.env.OPENROUTER_API_KEY;
      if (savedAgentDir !== undefined) process.env.PI_CODING_AGENT_DIR = savedAgentDir;
      else delete process.env.PI_CODING_AGENT_DIR;
      if (savedRuntime !== undefined) process.env[RUNTIME_ENV] = savedRuntime;
      else delete process.env[RUNTIME_ENV];
    }
  });

  it("returns undefined when no store has a key", async () => {
    const savedKey = process.env.OPENROUTER_API_KEY;
    const savedAgentDir = process.env.PI_CODING_AGENT_DIR;
    const savedRuntime = process.env[RUNTIME_ENV];
    delete process.env.OPENROUTER_API_KEY;
    delete process.env[RUNTIME_ENV];
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "oradv-auth-"));
    process.env.PI_CODING_AGENT_DIR = dir;
    try {
      assert.equal(await bootstrapApiKey(), undefined);
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
      if (savedKey !== undefined) process.env.OPENROUTER_API_KEY = savedKey;
      else delete process.env.OPENROUTER_API_KEY;
      if (savedAgentDir !== undefined) process.env.PI_CODING_AGENT_DIR = savedAgentDir;
      else delete process.env.PI_CODING_AGENT_DIR;
      if (savedRuntime !== undefined) process.env[RUNTIME_ENV] = savedRuntime;
      else delete process.env[RUNTIME_ENV];
    }
  });
});
