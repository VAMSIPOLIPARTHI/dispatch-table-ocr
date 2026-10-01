"""
Unit test suite for dispatch table validation logic.
Covers:
1. A row that reconciles
2. A row that does not reconcile (reported Net Dispatch preserved)
3. A missing quantity
4. An invalid quantity
5. A comma-separated quantity
6. Preservation of unreadable cell (no synthetic auto-fill)
7. Multi-row continuation when one row fails
"""

import pytest
from pipeline.validator import RecordValidator
from pipeline.models import RecordStatus


def test_row_that_reconciles():
    """Verify that Dispatched - Returned == Net Dispatch results in VALID status."""
    record = RecordValidator.validate_row(
        row_index=1,
        raw_product_code="PRD-101",
        raw_dispatched="1,250",
        raw_returned="50",
        raw_net_dispatch="1,200",
    )
    assert record.status == RecordStatus.VALID
    assert record.reconciled is True
    assert record.dispatched.normalized_value == 1250
    assert record.returned.normalized_value == 50
    assert record.net_dispatch.normalized_value == 1200
    assert len(record.findings) == 0


def test_row_that_does_not_reconcile():
    """
    Verify that Dispatched - Returned != Net Dispatch flags RECONCILIATION_FAILED,
    and crucially preserves the reported Net Dispatch without altering it.
    """
    # 800 - 25 should be 775, but reported is 750
    record = RecordValidator.validate_row(
        row_index=2,
        raw_product_code="PRD-102",
        raw_dispatched="800",
        raw_returned="25",
        raw_net_dispatch="750",
    )
    assert record.status == RecordStatus.RECONCILIATION_FAILED
    assert record.reconciled is False
    assert record.dispatched.normalized_value == 800
    assert record.returned.normalized_value == 25
    assert record.net_dispatch.normalized_value == 750  # Must PRESERVE reported 750!
    assert any("Reconciliation discrepancy" in f for f in record.findings)


def test_missing_quantity():
    """
    Verify that when a quantity is missing (None or empty),
    status is flagged as UNREADABLE_CELL, reconciled is None,
    and NO synthetic back-calculation is performed.
    """
    # Returned is None
    record = RecordValidator.validate_row(
        row_index=3,
        raw_product_code="PRD-103",
        raw_dispatched="600",
        raw_returned=None,
        raw_net_dispatch="570",
    )
    assert record.status == RecordStatus.UNREADABLE_CELL
    assert record.reconciled is None
    assert record.returned.normalized_value is None
    assert record.returned.is_readable is False
    # CRITICAL: Net Dispatch must be preserved, and Returned must NOT be auto-filled to 30!
    assert record.net_dispatch.normalized_value == 570
    assert record.returned.normalized_value is None
    assert any("reconstruction prohibited" in f for f in record.findings)


def test_invalid_quantity():
    """Verify that a malformed or non-numeric quantity flags INVALID_FORMAT."""
    record = RecordValidator.validate_row(
        row_index=4,
        raw_product_code="PRD-104",
        raw_dispatched="450",
        raw_returned="INVALID_TEXT",
        raw_net_dispatch="450",
    )
    assert record.status == RecordStatus.INVALID_FORMAT
    assert record.reconciled is None
    assert record.returned.normalized_value is None
    assert record.returned.is_readable is False
    assert any("Cannot parse" in f for f in record.findings)


def test_comma_separated_quantity():
    """Verify comma-separated quantities are normalized accurately into integers."""
    record = RecordValidator.validate_row(
        row_index=1,
        raw_product_code="PRD-999",
        raw_dispatched=" 12,345,678 ",
        raw_returned=" 1,000 ",
        raw_net_dispatch=" 12,344,678 ",
    )
    assert record.status == RecordStatus.VALID
    assert record.reconciled is True
    assert record.dispatched.normalized_value == 12345678
    assert record.returned.normalized_value == 1000
    assert record.net_dispatch.normalized_value == 12344678


def test_no_synthetic_replacement_when_obscured():
    """
    Explicitly test the assessment constraint:
    'Do not calculate a replacement for an unreadable cell from the other values.'
    """
    record = RecordValidator.validate_row(
        row_index=2,
        raw_product_code="PRD-102",
        raw_dispatched="800",
        raw_returned="[OBSCURED]",
        raw_net_dispatch="775",
    )
    assert record.status == RecordStatus.UNREADABLE_CELL
    assert record.reconciled is None
    assert record.returned.normalized_value is None
    assert record.dispatched.normalized_value == 800
    assert record.net_dispatch.normalized_value == 775


def test_multi_row_continuation():
    """
    Verify that in a table with multiple rows, an error on row 2 does not stop
    processing of subsequent rows 3 and 4.
    """
    raw_rows = [
        ("PRD-101", "1,250", "50", "1,200"),
        ("PRD-102", "800", None, "775"),        # Obscured/missing
        ("PRD-103", "600", "40", "570"),
        ("PRD-104", "450", "0", "450"),
    ]

    records = [
        RecordValidator.validate_row(i + 1, r[0], r[1], r[2], r[3])
        for i, r in enumerate(raw_rows)
    ]

    assert len(records) == 4
    assert records[0].status == RecordStatus.VALID
    assert records[1].status == RecordStatus.UNREADABLE_CELL
    assert records[2].status == RecordStatus.RECONCILIATION_FAILED
    assert records[3].status == RecordStatus.VALID
