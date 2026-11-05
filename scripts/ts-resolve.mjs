/**
 * Node module-resolution hook for the unit-test runner.
 *
 * The extension sources use TypeScript-style ".js" specifiers
 * (`import ... from "./types.js"` resolving to "./types.ts") because pi's
 * extension loader resolves them that way at runtime and `tsc` requires it.
 * Bun remaps ".js" → ".ts" automatically; Node's built-in type stripping
 * (Node ≥ 22.18) does not, so this hook retries the ".ts" specifier when
 * the ".js" one cannot be found.
 *
 * The bare "@earendil-works/pi-ai" specifier resolves to the compat
 * entrypoint, mirroring pi/tsconfig.json's path mapping (the extension
 * loader honours the same mapping at runtime): the root entrypoint does not
 * re-export the compat-only helpers the sources import.
 *
 * Registered via NODE_OPTIONS by scripts/run-tests.mjs so child processes
 * spawned by tests (test/base-url.test.ts uses `node --eval`) inherit it.
 */
import { register } from "node:module";

register("./ts-resolve.mjs", import.meta.url);

export async function resolve(specifier, context, nextResolve) {
	if (specifier === "@earendil-works/pi-ai") {
		return nextResolve("@earendil-works/pi-ai/compat", context);
	}
	try {
		return await nextResolve(specifier, context);
	} catch (error) {
		if (
			error.code === "ERR_MODULE_NOT_FOUND" &&
			specifier.startsWith(".") &&
			specifier.endsWith(".js")
		) {
			return nextResolve(specifier.slice(0, -3) + ".ts", context);
		}
		throw error;
	}
}
