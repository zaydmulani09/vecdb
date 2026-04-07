export declare class VecDbError extends Error {
    readonly statusCode?: number | undefined;
    readonly errorCode?: string | undefined;
    constructor(message: string, statusCode?: number | undefined, errorCode?: string | undefined);
}
export declare class NotFoundError extends VecDbError {
    constructor(message: string, statusCode?: number, errorCode?: string);
}
export declare class UnauthorizedError extends VecDbError {
    constructor(message: string, statusCode?: number, errorCode?: string);
}
export declare class InvalidRequestError extends VecDbError {
    constructor(message: string, statusCode?: number, errorCode?: string);
}
export declare class ServerError extends VecDbError {
    constructor(message: string, statusCode?: number, errorCode?: string);
}
export declare class ConnectionError extends VecDbError {
    constructor(message: string, statusCode?: number, errorCode?: string);
}
export declare function raiseForStatus(response: Response): Promise<void>;
