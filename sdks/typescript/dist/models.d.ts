export interface CollectionConfig {
    name: string;
    dimension: number;
    metric?: string;
    indexType?: string;
}
export interface CollectionInfo {
    name: string;
    dimension: number;
    metric: string;
    index_type: string;
    vector_count: number;
    created_at: string;
}
export interface VectorRecord {
    id: string;
    vector: number[];
    payload?: Record<string, unknown>;
    text?: string;
}
export interface SearchResult {
    id: string;
    score: number;
    dense_score?: number;
    sparse_score?: number;
    payload?: Record<string, unknown>;
    text?: string;
}
export interface UpsertResponse {
    inserted: number;
    updated: number;
    errors: unknown[];
    time_ms: number;
}
export interface SearchResponse {
    results: SearchResult[];
    count: number;
    time_ms: number;
}
export interface HealthResponse {
    status: string;
    version: string;
    vector_count: number;
    collections: number;
}
