export { VecDbClient } from "./client.js";
export type {
  CollectionConfig,
  CollectionInfo,
  HealthResponse,
  SearchResponse,
  SearchResult,
  UpsertResponse,
  VectorRecord,
} from "./models.js";
export {
  ConnectionError,
  InvalidRequestError,
  NotFoundError,
  ServerError,
  UnauthorizedError,
  VecDbError,
} from "./errors.js";
