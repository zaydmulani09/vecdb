# vecdb-client (TypeScript)

TypeScript client SDK for the [vecdb](https://github.com/vecdb/vecdb) vector database.

Zero runtime dependencies. Uses native `fetch` (Node 18+, all modern browsers).

## Install

```bash
npm install vecdb-client
```

## Quick Start

```typescript
import { VecDbClient } from "vecdb-client";

const client = new VecDbClient({ baseUrl: "http://localhost:8080", apiKey: "secret" });

await client.createCollection({ name: "docs", dimension: 1536 });

await client.upsert("docs", [
  { id: "1", vector: new Array(1536).fill(0.1), text: "hello world" },
]);

const results = await client.searchDense("docs", new Array(1536).fill(0.1), 5);
for (const r of results.results) {
  console.log(r.id, r.score);
}
```

## Dev

```bash
npm install
npm test
npm run typecheck
npm run build
```
