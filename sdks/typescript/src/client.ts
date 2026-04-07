import { ConnectionError, raiseForStatus } from "./errors.js";
import type {
  CollectionConfig,
  CollectionInfo,
  HealthResponse,
  SearchResponse,
  UpsertResponse,
  VectorRecord,
} from "./models.js";

export class VecDbClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;

  constructor(options?: {
    baseUrl?: string;
    apiKey?: string;
    timeout?: number;
  }) {
    this.baseUrl = (options?.baseUrl ?? "http://localhost:8080").replace(/\/$/, "");
    this.timeout = options?.timeout ?? 30000;
    this.headers = { "Content-Type": "application/json" };
    if (options?.apiKey) {
      this.headers["X-Api-Key"] = options.apiKey;
    }
  }

  private _url(path: string): string {
    return `${this.baseUrl}/${path.replace(/^\//, "")}`;
  }

  private async _request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    let response: Response;
    try {
      response = await fetch(this._url(path), {
        method,
        headers: this.headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } catch (err) {
      clearTimeout(timeoutId);
      throw new ConnectionError(String(err));
    }

    clearTimeout(timeoutId);
    await raiseForStatus(response);
    return (await response.json()) as T;
  }

  async health(): Promise<HealthResponse> {
    return this._request<HealthResponse>("GET", "/health");
  }

  async createCollection(config: CollectionConfig): Promise<CollectionInfo> {
    const body: Record<string, unknown> = {
      name: config.name,
      dimension: config.dimension,
    };
    if (config.metric !== undefined) body["metric"] = config.metric;
    if (config.indexType !== undefined) body["index_type"] = config.indexType;
    return this._request<CollectionInfo>("POST", "/collections", body);
  }

  async listCollections(): Promise<CollectionInfo[]> {
    const data = await this._request<{ collections: CollectionInfo[] }>("GET", "/collections");
    return data.collections;
  }

  async getCollection(name: string): Promise<CollectionInfo> {
    return this._request<CollectionInfo>("GET", `/collections/${name}`);
  }

  async deleteCollection(name: string): Promise<boolean> {
    await this._request<unknown>("DELETE", `/collections/${name}`);
    return true;
  }

  async upsert(collection: string, records: VectorRecord[]): Promise<UpsertResponse> {
    return this._request<UpsertResponse>(
      "POST",
      `/collections/${collection}/vectors`,
      { records },
    );
  }

  async getVector(collection: string, id: string): Promise<VectorRecord> {
    return this._request<VectorRecord>("GET", `/collections/${collection}/vectors/${id}`);
  }

  async deleteVectors(collection: string, ids: string[]): Promise<Record<string, unknown>> {
    return this._request<Record<string, unknown>>(
      "DELETE",
      `/collections/${collection}/vectors`,
      { ids },
    );
  }

  async searchDense(collection: string, vector: number[], k?: number): Promise<SearchResponse> {
    return this._request<SearchResponse>(
      "POST",
      `/collections/${collection}/search/dense`,
      { vector, k: k ?? 10 },
    );
  }

  async searchSparse(collection: string, query: string, k?: number): Promise<SearchResponse> {
    return this._request<SearchResponse>(
      "POST",
      `/collections/${collection}/search/sparse`,
      { query, k: k ?? 10 },
    );
  }

  async searchHybrid(
    collection: string,
    options: {
      vector?: number[];
      query?: string;
      k?: number;
      alpha?: number;
    },
  ): Promise<SearchResponse> {
    return this._request<SearchResponse>(
      "POST",
      `/collections/${collection}/search/hybrid`,
      { ...options, k: options.k ?? 10, alpha: options.alpha ?? 0.7 },
    );
  }

  async querySQL(sql: string): Promise<SearchResponse> {
    return this._request<SearchResponse>("POST", "/query", { sql });
  }
}
