"""
pipeline/models.py
Data models and schemas for Dispatch Table OCR extraction and validation.
"""

from enum import Enum
from typing import Optional, Union, List
from pydantic import BaseModel, Field


class RecordStatus(str, Enum):
    VALID = "VALID"
    RECONCILIATION_FAILED = "RECONCILIATION_FAILED"
    UNREADABLE_CELL = "UNREADABLE_CELL"
    INVALID_FORMAT = "INVALID_FORMAT"
    MISSING_VALUE = "MISSING_VALUE"


class CellData(BaseModel):
    raw_text: Optional[str] = Field(default=None, description="Original OCR text recognized in the cell")
    normalized_value: Optional[Union[int, str]] = Field(
        default=None, description="Normalized integer or string, or null if unreadable/invalid"
    )
    confidence: float = Field(default=0.0, description="OCR recognition confidence score [0.0 - 1.0]")
    is_readable: bool = Field(default=True, description="Whether the cell text was reliably detected and parsed")


class Record(BaseModel):
    row_index: int = Field(description="1-based visual row index in data table")
    product_code: CellData = Field(description="Product identifier code (e.g., PRD-101)")
    dispatched: CellData = Field(description="Quantity dispatched in units")
    returned: CellData = Field(description="Quantity returned in units")
    net_dispatch: CellData = Field(description="Reported net dispatch in units (preserved regardless of reconciliation)")
    reconciled: Optional[bool] = Field(
        default=None,
        description="True if Dispatched - Returned == Net Dispatch, False if mismatch, null if unreadable/missing",
    )
    status: RecordStatus = Field(description="Record processing and reconciliation status")
    findings: List[str] = Field(default_factory=list, description="Validation messages or error descriptions")


class ImageMetadata(BaseModel):
    width: int
    height: int
    estimated_skew_degrees: float
    rotation_corrected: bool
    contrast_enhancement_applied: bool


class PipelineSummary(BaseModel):
    total_rows_detected: int
    valid_rows: int
    reconciliation_failed_rows: int
    unreadable_or_invalid_rows: int
    batch_status: str


class PipelineResult(BaseModel):
    source_image: str
    processing_timestamp: str
    processing_time_ms: float
    image_metadata: ImageMetadata
    summary: PipelineSummary
    records: List[Record]
