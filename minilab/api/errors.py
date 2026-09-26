"""Errors in OpenAI's shape, so the official SDKs raise the right exception class.

    {"error": {"message": "...", "type": "invalid_request_error", "param": null, "code": "model_not_found"}}

The SDK picks the exception from the HTTP status (401 -> AuthenticationError,
404 -> NotFoundError, 429 -> RateLimitError, ...) and exposes `type`/`code`.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(Exception):
    def __init__(self, status: int, message: str, type: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type, code, param
        self.headers = headers or {}

    def to_json(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


def invalid_request(message: str, param: str | None = None, code: str | None = None) -> APIError:
    return APIError(400, message, param=param, code=code)


def model_not_found(model: str) -> APIError:
    return APIError(404, f"The model '{model}' does not exist or you do not have access to it.",
                    code="model_not_found", param="model")


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        return JSONResponse(exc.to_json(), status_code=exc.status, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        # OpenAI answers malformed requests with 400, not FastAPI's default 422.
        err = exc.errors()[0] if exc.errors() else {"loc": (), "msg": "Invalid request."}
        param = ".".join(str(p) for p in err["loc"] if p != "body") or None
        return JSONResponse(invalid_request(err["msg"], param=param).to_json(), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        # Unknown routes and wrong methods, in the same shape as everything else.
        if exc.status_code in (404, 405):
            message = f"Invalid URL ({request.method} {request.url.path})"
        else:
            message = str(exc.detail)
        body = APIError(exc.status_code, message).to_json()
        return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        # A bug: Starlette still logs the traceback; the client gets JSON its SDK understands.
        return JSONResponse(APIError(500, "The server had an error while processing your request.",
                                     type="server_error").to_json(), status_code=500)
