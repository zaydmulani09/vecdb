from __future__ import annotations

import httpx


class VecDbError(Exception):
    def __init__(self, message: str, status_code: int | None = None, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


class NotFoundError(VecDbError):
    pass


class UnauthorizedError(VecDbError):
    pass


class InvalidRequestError(VecDbError):
    pass


class ServerError(VecDbError):
    pass


class ConnectionError(VecDbError):
    pass


def raise_for_status(response: httpx.Response) -> None:
    if response.status_code < 400:
        return

    message = f"HTTP {response.status_code}"
    error_code: str | None = None

    try:
        body = response.json()
        err = body.get("error", {})
        if isinstance(err, dict):
            message = err.get("message", message)
            error_code = err.get("code")
    except Exception:
        pass

    status = response.status_code
    if status == 401:
        raise UnauthorizedError(message, status_code=status, error_code=error_code)
    if status == 404:
        raise NotFoundError(message, status_code=status, error_code=error_code)
    if status in (400, 422):
        raise InvalidRequestError(message, status_code=status, error_code=error_code)
    if status >= 500:
        raise ServerError(message, status_code=status, error_code=error_code)
    raise VecDbError(message, status_code=status, error_code=error_code)
