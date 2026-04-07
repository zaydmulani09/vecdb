import type { CollectionConfig, CollectionInfo, HealthResponse, SearchResponse, UpsertResponse, VectorRecord } from "./models.js";
export declare class VecDbClient {
    private readonly baseUrl;
    private readonly headers;
    private readonly timeout;
    constructor(options?: {
        baseUrl?: string;
        apiKey?: string;
        timeout?: number;
    });
    private _url;
    private _request;
    health(): Promise<HealthResponse>;
    createCollection(config: CollectionConfig): Promise<CollectionInfo>;
    listCollections(): Promise<CollectionInfo[]>;
    getCollection(name: string): Promise<CollectionInfo>;
    deleteCollection(name: string): Promise<boolean>;
    upsert(collection: string, records: VectorRecord[]): Promise<UpsertResponse>;
    getVector(collection: string, id: string): Promise<VectorRecord>;
    deleteVectors(collection: string, ids: string[]): Promise<Record<string, unknown>>;
    searchDense(collection: string, vector: number[], k?: number): Promise<SearchResponse>;
    searchSparse(collection: string, query: string, k?: number): Promise<SearchResponse>;
    searchHybrid(collection: string, options: {
        vector?: number[];
        query?: string;
        k?: number;
        alpha?: number;
    }): Promise<SearchResponse>;
    querySQL(sql: string): Promise<SearchResponse>;
}
