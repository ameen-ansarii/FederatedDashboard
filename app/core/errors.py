from typing import Any, Dict, Optional
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse


class AppException(HTTPException):
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = "INTERNAL_ERROR",
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.extra = extra or {}


class NotFoundException(AppException):
    def __init__(self, detail: str = "Resource not found", error_code: str = "NOT_FOUND"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail, error_code=error_code)


class BadRequestException(AppException):
    def __init__(self, detail: str = "Bad request", error_code: str = "BAD_REQUEST"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail, error_code=error_code)


class UnauthorizedException(AppException):
    def __init__(self, detail: str = "Could not validate credentials", error_code: str = "UNAUTHORIZED"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            extra={"headers": {"WWW-Authenticate": "Bearer"}},
        )


class ForbiddenException(AppException):
    def __init__(self, detail: str = "Operation not permitted", error_code: str = "FORBIDDEN"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail, error_code=error_code)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "extra": exc.extra,
            }
        },
    )
