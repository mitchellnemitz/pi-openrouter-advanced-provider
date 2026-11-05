import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { once } from "node:events";
import { createServer } from "node:http";
import { it } from "node:test";
import { promisify } from "node:util";

const exec = promisify(execFile);

it("routes account catalog, endpoint metadata and JIT stats to the configured OpenRouter origin", async () => {
  const requests: Array<{ path: string; authorization?: string }> = [];
  const server = createServer((request, response) => {
    const path = new URL(request.url!, "http://localhost").pathname;
    requests.push({ path, authorization: request.headers.authorization });
    const data = path.endsWith("/models/user")
      ? [{ id: "private/test-model" }]
      : path.endsWith("/endpoints")
        ? { endpoints: [{ provider_name: "Fixture" }] }
        : path.endsWith("/catalog/models")
          ? [{ slug: "private/test-model", permaslug: "fixture-permaslug" }]
          : path.endsWith("/stats/endpoint")
            ? [{ id: "fixture-endpoint" }]
            : path.endsWith("/tool-call-error-rate")
              ? [{ x: "2026-11-04", y: { "fixture-endpoint": 2.5 } }]
              : { limit: 42 };
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({ data }));
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  const address = server.address() as { port: number };
  try {
    const { stdout } = await exec(process.execPath, ["--eval", `
      const originalFetch = globalThis.fetch;
      globalThis.fetch = (input, init) => {
        if (new URL(input).origin !== new URL(process.env.OPENROUTER_BASE_URL).origin) {
          throw new Error('Unexpected live OpenRouter request: ' + input);
        }
        return originalFetch(input, init);
      };
      const api = await import('./pi/api.ts');
      const catalog = await api.fetchCatalogModels('fixture-key', true);
      const endpoints = await api.fetchModelEndpoints('private/test-model', 'fixture-key', true);
      const key = await api.fetchKeyInfo('fixture-key');
      const credits = await api.fetchCredits('fixture-key');
      const permaslugs = await api.fetchPermaslugMap(true);
      const stats = await api.fetchEndpointStats('fixture-permaslug');
      const rates = await api.fetchToolCallErrorRates('fixture-permaslug');
      console.log(JSON.stringify({catalog, endpoints, key, credits, permaslugs: [...permaslugs], stats, rates: [...rates]}));
    `], {
      cwd: new URL("..", import.meta.url).pathname,
      env: { ...process.env, OPENROUTER_BASE_URL: `http://127.0.0.1:${address.port}/api/v1/` },
      timeout: 20_000,
    });
    assert.deepEqual(JSON.parse(stdout), {
      catalog: [{ id: "private/test-model" }],
      endpoints: [{ provider_name: "Fixture" }],
      key: { limit: 42 }, credits: { limit: 42 },
      permaslugs: [["private/test-model", "fixture-permaslug"]],
      stats: [{ id: "fixture-endpoint" }], rates: [["fixture-endpoint", 2.5]],
    });
    assert.deepEqual(requests, [
      { path: "/api/v1/models/user", authorization: "Bearer fixture-key" },
      { path: "/api/v1/models/private/test-model/endpoints", authorization: "Bearer fixture-key" },
      { path: "/api/v1/key", authorization: "Bearer fixture-key" },
      { path: "/api/v1/credits", authorization: "Bearer fixture-key" },
      { path: "/api/frontend/v1/catalog/models", authorization: undefined },
      { path: "/api/frontend/v1/stats/endpoint", authorization: undefined },
      { path: "/api/frontend/v1/stats/tool-call-error-rate", authorization: undefined },
    ]);
  } finally {
    await promisify(server.close.bind(server))();
  }
});

it("restores router cost only from the configured completion endpoint", async () => {
  const { stdout } = await exec(process.execPath, ["--eval", `
    const {wrapRouterCostFetch, restoreRouterCost} = await import('./pi/auto-cost.ts');
    const results = [];
    for (const url of ['http://127.0.0.1:49152/api/v1/chat/completions', 'http://127.0.0.1:49153/api/v1/chat/completions']) {
      const captures = new Map();
      const fetch = wrapRouterCostFetch(async () => new Response(
        'data: {"id":"gen-override","choices":[],"usage":{"cost":0.012345}}\\n\\ndata: [DONE]\\n\\n',
        {headers: {'content-type': 'text/event-stream'}}
      ), captures);
      await (await fetch(url, {method: 'POST', body: JSON.stringify({model: 'openrouter/auto'})})).text();
      const message = {provider: 'openrouter', model: 'openrouter/auto', responseId: 'gen-override', stopReason: 'stop', usage: {cost: {total: 0}}};
      const restored = await restoreRouterCost(message, captures);
      results.push(restored?.usage.cost.total ?? null);
    }
    console.log(JSON.stringify(results));
  `], {
    cwd: new URL("..", import.meta.url).pathname,
    env: {...process.env, OPENROUTER_BASE_URL: "http://127.0.0.1:49152/api/v1"},
    timeout: 5_000,
  });
  assert.deepEqual(JSON.parse(stdout), [0.012345, null]);
});
