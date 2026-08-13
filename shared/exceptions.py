"""Application-level exception types. `StorageError` (via its
`ApplicationError` base) is what `text_extractors.py` raises for any
extraction failure — invalid/corrupted files, empty output, a missing
LibreOffice binary — each carrying a stable `error_code` and HTTP-shaped
`status_code` for callers that want to distinguish failure kinds."""

from http import HTTPStatus

from shared.constants import ErrorMessages


class ApplicationError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_code: str = ErrorMessages.APPLICATION_ERROR,
        status_code: int = HTTPStatus.SERVICE_UNAVAILABLE,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


class StorageError(ApplicationError):
    def __init__(self, message: str):
        super().__init__(
            message,
            error_code=ErrorMessages.STORAGE_ERROR,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )
