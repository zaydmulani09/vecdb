export class VecDbError extends Error {
    constructor(message, statusCode, errorCode) {
        super(message);
        this.statusCode = statusCode;
        this.errorCode = errorCode;
        this.name = "VecDbError";
    }
}
export class NotFoundError extends VecDbError {
    constructor(message, statusCode, errorCode) {
        super(message, statusCode, errorCode);
        this.name = "NotFoundError";
    }
}
export class UnauthorizedError extends VecDbError {
    constructor(message, statusCode, errorCode) {
        super(message, statusCode, errorCode);
        this.name = "UnauthorizedError";
    }
}
export class InvalidRequestError extends VecDbError {
    constructor(message, statusCode, errorCode) {
        super(message, statusCode, errorCode);
        this.name = "InvalidRequestError";
    }
}
export class ServerError extends VecDbError {
    constructor(message, statusCode, errorCode) {
        super(message, statusCode, errorCode);
        this.name = "ServerError";
    }
}
export class ConnectionError extends VecDbError {
    constructor(message, statusCode, errorCode) {
        super(message, statusCode, errorCode);
        this.name = "ConnectionError";
    }
}
export async function raiseForStatus(response) {
    if (response.ok)
        return;
    let message = response.statusText || `HTTP ${response.status}`;
    let errorCode;
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
        try {
            const body = (await response.json());
            if (body.error) {
                if (body.error.message)
                    message = body.error.message;
                if (body.error.code)
                    errorCode = body.error.code;
            }
        }
        catch {
            // use statusText if JSON parse fails
        }
    }
    const status = response.status;
    if (status === 401)
        throw new UnauthorizedError(message, status, errorCode);
    if (status === 404)
        throw new NotFoundError(message, status, errorCode);
    if (status === 400 || status === 422)
        throw new InvalidRequestError(message, status, errorCode);
    if (status >= 500)
        throw new ServerError(message, status, errorCode);
    throw new VecDbError(message, status, errorCode);
}
