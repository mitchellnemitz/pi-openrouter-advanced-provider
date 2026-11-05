/**
 * Unit-test runner: plain Node, no Bun required.
 *
 * Runs `node --test` over the TypeScript test files. Node ≥ 22.18 strips
 * types natively; scripts/ts-resolve.mjs covers the ".js" → ".ts" specifier
 * remap, propagated to spawned child processes through NODE_OPTIONS.
 *
 * Usage: node scripts/run-tests.mjs [extra node --test args]
 */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

const [major, minor] = process.versions.node.split(".").map(Number);
const tooOld = major < 22 || (major === 22 && minor < 18);
if (tooOld) {
	console.error(
		`npm test needs Node >= 22.18 (native TypeScript stripping); running ${process.versions.node}. ` +
			`Alternatively run the full suite in the pinned image: npm run test:container.`,
	);
	process.exit(1);
}

const hook = path.join(root, "scripts", "ts-resolve.mjs");
const existing = process.env.NODE_OPTIONS ?? "";
process.env.NODE_OPTIONS = existing
	? `${existing} --import ${JSON.stringify(hook)}`
	: `--import ${JSON.stringify(hook)}`;

const args = ["--test", path.posix.join("test", "*.test.ts"), ...process.argv.slice(2)];
const result = spawnSync(process.execPath, args, { cwd: root, stdio: "inherit" });
process.exit(result.status ?? 1);
