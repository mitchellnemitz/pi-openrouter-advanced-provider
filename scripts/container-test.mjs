#!/usr/bin/env node
/**
 * Containerized dual-host test harness runner (local dev + CI).
 *
 * Usage:
 *   npm run test:container                  # both hosts (pi + omp)
 *   npm run test:container -- --host=omp    # single host (or --host=pi)
 *   npm run test:container -- --no-cache    # rebuild the image without layer cache
 *   npm run test:container -- --live=false  # permitted no-op; any other --live value exits 2
 *
 * What it does:
 *   1. Builds the pinned image from ./Dockerfile (oven/bun:1.4.3-debian with
 *      pi 1.5.0 and omp 18.16.0, checksum-verified from release checksums;
 *      the repo is COPYed to /app for CI determinism).
 *   2. Runs one container with:
 *        --network=none  — the mock OpenRouter server runs INSIDE the
 *                          container; no route to the real API exists
 *        --tmpfs HOME    — isolated fixture HOME on a fresh tmpfs; user
 *                          credentials are never mounted, copied or forwarded
 *        env-whitelisted contract variables only (no host env passthrough)
 *   3. Delegates everything else to test/container-scenario.mjs, executed
 *      with bun inside the container. The scenario starts the mock, seeds
 *      fixture credential stores (pi auth.json / omp agent.db via
 *      scripts/seed-omp-db.mjs), drives the selected hosts headlessly,
 *      asserts against the mock NDJSON log, registry/session artifacts and
 *      process exits, and exits nonzero on any failure.
 *
 * Local vs CI: in a local checkout the repo is bind-mounted read-only at
 * /repo and the scenario runs from the mount (live code under test). If the
 * checkout has node_modules it is mounted read-only at /repo/node_modules;
 * otherwise dependencies fall back to the image-installed copy (symlinked at
 * /node_modules by the Dockerfile). In CI (GITHUB_ACTIONS=true) nothing is
 * mounted and the COPYed repo at /app is used.
 *
 * Environment contract passed to the scenario (documented here because this
 * runner is its only caller):
 *   SCENARIO_HOSTS         comma-separated subset of "pi,omp" for this run
 *   SCENARIO_REPO_DIR      repo root inside the container, containing pi/,
 *                          test/, scripts/ and package.json
 *   SCENARIO_FIXTURE_KEY   bearer key the mock accepts on keyed endpoints
 *                          (taken from test/fixtures/openrouter.mjs, so the
 *                          value never drifts from the fixture)
 *   SCENARIO_HOME          writable isolated HOME (tmpfs); fixture credential
 *                          stores and session files live only here
 *   SCENARIO_ARTIFACTS_DIR directory for the mock NDJSON request log and any
 *                          other assertion artifacts (inside SCENARIO_HOME)
 * The scenario must not receive or forward any other host environment.
 */

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { FIXTURE_KEY } from "../test/fixtures/openrouter.mjs";

const REPO_ROOT = path.resolve(
	path.dirname(fileURLToPath(import.meta.url)),
	"..",
);
const pkg = JSON.parse(readFileSync(path.join(REPO_ROOT, "package.json"), "utf8"));
const IMAGE = `${pkg.name}:container-test-${pkg.version}`;

/** Print a failure and exit with the given code. */
function fail(message, code = 2) {
	console.error(`[container-test] ${message}`);
	process.exit(code);
}

/** Parse and validate CLI arguments; anything unlisted is rejected. */
function parseArgs(argv) {
	const opts = { hosts: ["pi", "omp"], noCache: false };
	for (const arg of argv) {
		if (arg === "--no-cache") {
			opts.noCache = true;
		} else if (arg === "--live=false") {
			// Mock-only harness: this is the only permitted --live setting.
		} else if (arg.startsWith("--host=")) {
			const value = arg.slice("--host=".length);
			if (value === "both") {
				opts.hosts = ["pi", "omp"];
			} else if (value === "pi" || value === "omp") {
				opts.hosts = [value];
			} else {
				fail(`unsupported --host value: ${value} (expected pi, omp or both)`);
			}
		} else {
			fail(
				`unsupported argument: ${arg}\n` +
					"supported: --host=pi|omp|both, --no-cache, --live=false (live mode is not implemented)",
			);
		}
	}
	return opts;
}

/** Verify a docker CLI is available before doing any work. */
function requireDocker() {
	const probe = spawnSync("docker", ["info", "--format", "{{.ServerVersion}}"], {
		encoding: "utf8",
	});
	if (probe.error || probe.status !== 0) {
		fail(
			"docker is required but not usable (is the daemon running and the CLI on PATH?)",
			1,
		);
	}
	return (probe.stdout || "").trim();
}

/** Build the pinned image; streams build output. */
function buildImage(noCache) {
	console.log(`[container-test] building ${IMAGE}${noCache ? " (--no-cache)" : ""} ...`);
	const build = spawnSync(
		"docker",
		["build", ...(noCache ? ["--no-cache"] : []), "--tag", IMAGE, REPO_ROOT],
		{ stdio: "inherit" },
	);
	if (build.error) {
		fail(`failed to start docker build: ${build.error.message}`, 1);
	}
	if (build.status !== 0) {
		console.error(
			"[container-test] image build failed; rerun with --no-cache if layers are suspect",
		);
		process.exit(build.status ?? 1);
	}
}

/** Run the scenario container; streams its output as diagnostics. */
function runScenario(hosts) {
	const ciMode = process.env.GITHUB_ACTIONS === "true";
	const repoDir = ciMode ? "/app" : "/repo";
	const home = "/fixture-home";
	const artifactsDir = `${home}/artifacts`;

	const dockerArgs = [
		"run",
		"--rm",
		"--network=none",
		"--tmpfs",
		`${home}:rw,exec,nosuid,size=1g`,
		"--workdir",
		"/app",
		// Contract variables only — no host environment passthrough, so real
		// credentials (OPENROUTER_API_KEY and friends) cannot leak in.
		"--env",
		`HOME=${home}`,
		"--env",
		`SCENARIO_HOME=${home}`,
		"--env",
		`SCENARIO_HOSTS=${hosts.join(",")}`,
		"--env",
		`SCENARIO_REPO_DIR=${repoDir}`,
		"--env",
		`SCENARIO_FIXTURE_KEY=${FIXTURE_KEY}`,
		"--env",
		`SCENARIO_ARTIFACTS_DIR=${artifactsDir}`,
	];

	if (!ciMode) {
		// Local run: test live checkout code; the image copy is only the
		// dependency fallback (and the CI layout).
		dockerArgs.push("--volume", `${REPO_ROOT}:/repo:ro`);
		const nodeModules = path.join(REPO_ROOT, "node_modules");
		if (existsSync(nodeModules)) {
			dockerArgs.push("--volume", `${nodeModules}:/repo/node_modules:ro`);
		}
	}

	// bun is required: the scenario imports bun:sqlite (via seed-omp-db.mjs)
	// and both pinned host binaries are self-contained, so no node is needed.
	dockerArgs.push(
		IMAGE,
		"/bin/sh",
		"-c",
		'exec bun "$SCENARIO_REPO_DIR/test/container-scenario.mjs"',
	);

	console.log(
		`[container-test] running scenario for hosts: ${hosts.join(",")}` +
			` (${ciMode ? "CI layout /app, no mounts" : "local mount /repo read-only"}, network=none, isolated HOME)`,
	);
	const run = spawnSync("docker", dockerArgs, { stdio: "inherit" });
	if (run.error) {
		fail(`failed to start container: ${run.error.message}`, 1);
	}
	return run.status ?? 1;
}

function main() {
	const opts = parseArgs(process.argv.slice(2));
	const serverVersion = requireDocker();
	console.log(`[container-test] docker server ${serverVersion}, image ${IMAGE}`);

	buildImage(opts.noCache);
	const status = runScenario(opts.hosts);
	if (status !== 0) {
		console.error(
			`\n[container-test] FAILED (exit ${status}) for hosts: ${opts.hosts.join(",")}`,
		);
		console.error(
			"[container-test] diagnostics: the full container output is above; " +
				"the scenario's mock NDJSON log and artifacts live under SCENARIO_ARTIFACTS_DIR inside the (removed) container. " +
				"To keep them for inspection, rerun the docker command printed with -v added, or drop --rm:",
		);
		console.error(
			`  docker run --network=none -v "$PWD":/repo:ro ${IMAGE} /bin/sh -c 'exec bun "$SCENARIO_REPO_DIR/test/container-scenario.mjs"'`,
		);
	} else {
		console.log(
			`[container-test] PASSED for hosts: ${opts.hosts.join(",")}`,
		);
	}
	process.exit(status);
}

main();
