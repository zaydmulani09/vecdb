import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";
import { NotFoundError, UnauthorizedError, VecDbClient } from "../src/index.js";
import { server } from "./setup.js";

const BASE_URL = "http://test.vecdb";
const client = new VecDbClient({ baseUrl: BASE_URL });

const sampleCollection = {
  name: "test",
  dimension: 3,
  metric: "cosine",
  index_type: "hnsw",
  vector_count: 0,
  created_at: "2026-01-01T00:00:00Z",
};

const sampleSearchResponse = {
  results: [
    { id: "v1", score: 0.99, dense_score: 0.99, payload: { title: "a" }, text: "hello" },
    { id: "v2", score: 0.85, dense_score: 0.85, payload: null, text: null },
    { id: "v3", score: 0.70, dense_score: 0.70, payload: null, text: null },
  ],
  count: 3,
  time_ms: 5,
};

describe("VecDbClient", () => {
  it("health returns HealthResponse", async () => {
    server.use(
      http.get(`${BASE_URL}/health`, () =>
        HttpResponse.json({ status: "ok", version: "0.1.0", vector_count: 0, collections: 0 }),
      ),
    );
    const result = await client.health();
    expect(result.status).toBe("ok");
  });

  it("createCollection returns CollectionInfo", async () => {
    server.use(
      http.post(`${BASE_URL}/collections`, () =>
        HttpResponse.json(sampleCollection, { status: 201 }),
      ),
    );
    const result = await client.createCollection({ name: "test", dimension: 3 });
    expect(result.name).toBe("test");
    expect(result.dimension).toBe(3);
    expect(result.metric).toBe("cosine");
  });

  it("listCollections returns array", async () => {
    const second = { ...sampleCollection, name: "test2" };
    server.use(
      http.get(`${BASE_URL}/collections`, () =>
        HttpResponse.json({ collections: [sampleCollection, second], count: 2 }),
      ),
    );
    const results = await client.listCollections();
    expect(results).toHaveLength(2);
    expect(results[0].name).toBe("test");
    expect(results[1].name).toBe("test2");
  });

  it("upsert sends correct body and returns UpsertResponse", async () => {
    let capturedBody: unknown;
    server.use(
      http.post(`${BASE_URL}/collections/test/vectors`, async ({ request }) => {
        capturedBody = await request.json();
        return HttpResponse.json({ inserted: 1, updated: 0, errors: [], time_ms: 3 });
      }),
    );
    const result = await client.upsert("test", [
      { id: "v1", vector: [0.1, 0.2, 0.3], payload: { k: "v" } },
    ]);
    expect(result.inserted).toBe(1);
    expect((capturedBody as { records: unknown[] }).records).toHaveLength(1);
  });

  it("searchDense returns SearchResponse", async () => {
    server.use(
      http.post(`${BASE_URL}/collections/test/search/dense`, () =>
        HttpResponse.json(sampleSearchResponse),
      ),
    );
    const result = await client.searchDense("test", [0.1, 0.2, 0.3], 3);
    expect(result.count).toBe(3);
    expect(typeof result.results[0].score).toBe("number");
  });

  it("searchHybrid sends vector and query", async () => {
    let capturedBody: unknown;
    server.use(
      http.post(`${BASE_URL}/collections/test/search/hybrid`, async ({ request }) => {
        capturedBody = await request.json();
        return HttpResponse.json(sampleSearchResponse);
      }),
    );
    const result = await client.searchHybrid("test", {
      vector: [0.1, 0.2, 0.3],
      query: "hello world",
      k: 3,
    });
    const body = capturedBody as { vector: unknown; query: unknown };
    expect(body.vector).toBeDefined();
    expect(body.query).toBe("hello world");
    expect(result.count).toBe(3);
  });

  it("getCollection throws NotFoundError on 404", async () => {
    server.use(
      http.get(`${BASE_URL}/collections/ghost`, () =>
        HttpResponse.json(
          { error: { code: "not_found", message: "collection 'ghost' not found" } },
          { status: 404 },
        ),
      ),
    );
    await expect(client.getCollection("ghost")).rejects.toThrow(NotFoundError);
  });

  it("unauthorized throws UnauthorizedError", async () => {
    server.use(
      http.get(`${BASE_URL}/health`, () =>
        HttpResponse.json(
          { error: { code: "unauthorized", message: "invalid api key" } },
          { status: 401 },
        ),
      ),
    );
    await expect(client.health()).rejects.toThrow(UnauthorizedError);
  });
});
