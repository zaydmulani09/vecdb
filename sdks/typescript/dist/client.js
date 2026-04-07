import { ConnectionError, raiseForStatus } from "./errors.js";
export class VecDbClient {
    constructor(options) {
        this.baseUrl = (options?.baseUrl ?? "http://localhost:8080").replace(/\/$/, "");
        this.timeout = options?.timeout ?? 30000;
        this.headers = { "Content-Type": "application/json" };
        if (options?.apiKey) {
            this.headers["X-Api-Key"] = options.apiKey;
        }
    }
    _url(path) {
        return `${this.baseUrl}/${path.replace(/^\//, "")}`;
    }
    async _request(method, path, body) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), this.timeout);
        let response;
        try {
            response = await fetch(this._url(path), {
                method,
                headers: this.headers,
                body: body !== undefined ? JSON.stringify(body) : undefined,
                signal: controller.signal,
            });
        }
        catch (err) {
            clearTimeout(timeoutId);
            throw new ConnectionError(String(err));
        }
        clearTimeout(timeoutId);
        await raiseForStatus(response);
        return (await response.json());
    }
    async health() {
        return this._request("GET", "/health");
    }
    async createCollection(config) {
        const body = {
            name: config.name,
            dimension: config.dimension,
        };
        if (config.metric !== undefined)
            body["metric"] = config.metric;
        if (config.indexType !== undefined)
            body["index_type"] = config.indexType;
        return this._request("POST", "/collections", body);
    }
    async listCollections() {
        const data = await this._request("GET", "/collections");
        return data.collections;
    }
    async getCollection(name) {
        return this._request("GET", `/collections/${name}`);
    }
    async deleteCollection(name) {
        await this._request("DELETE", `/collections/${name}`);
        return true;
    }
    async upsert(collection, records) {
        return this._request("POST", `/collections/${collection}/vectors`, { records });
    }
    async getVector(collection, id) {
        return this._request("GET", `/collections/${collection}/vectors/${id}`);
    }
    async deleteVectors(collection, ids) {
        return this._request("DELETE", `/collections/${collection}/vectors`, { ids });
    }
    async searchDense(collection, vector, k) {
        return this._request("POST", `/collections/${collection}/search/dense`, { vector, k: k ?? 10 });
    }
    async searchSparse(collection, query, k) {
        return this._request("POST", `/collections/${collection}/search/sparse`, { query, k: k ?? 10 });
    }
    async searchHybrid(collection, options) {
        return this._request("POST", `/collections/${collection}/search/hybrid`, { ...options, k: options.k ?? 10, alpha: options.alpha ?? 0.7 });
    }
    async querySQL(sql) {
        return this._request("POST", "/query", { sql });
    }
}
