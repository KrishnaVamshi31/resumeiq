"""Application error taxonomy and the single JSON error envelope."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logging_conf import get_logger, request_id_var

logger = get_logger(__name__)


class ResumeIQError(Exception):
    """Base class for every error this service raises deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class UnsupportedFileType(ResumeIQError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    code = "unsupported_file_type"


class FileTooLarge(ResumeIQError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "file_too_large"


class ExtractionFailed(ResumeIQError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "extraction_failed"


class EmptyDocument(ResumeIQError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "empty_document"


class ResourceNotFound(ResumeIQError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class RateLimited(ResumeIQError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"


class LLMUnavailable(ResumeIQError):
    """Raised only when the caller explicitly demanded LLM output."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "llm_unavailable"


def _envelope(code: str, message: str, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": request_id_var.get(),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so clients always see the same error shape."""

    @app.exception_handler(ResumeIQError)
    async def _handle_app_error(_: Request, exc: ResumeIQError) -> JSONResponse:
        logger.warning(
            "handled_error",
            extra={"code": exc.code, "detail": exc.message, "status": exc.status_code},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(
                "validation_error", "Request failed validation.", {"errors": exc.errors()}
            ),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to the client; the request_id ties the response
        # back to the full traceback in the logs.
        logger.exception("unhandled_error", extra={"exc_type": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "An unexpected error occurred.", {}),
        )
