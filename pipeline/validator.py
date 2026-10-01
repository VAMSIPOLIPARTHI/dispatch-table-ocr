"""
pipeline/validator.py
Validation and normalization engine for table records.
Enforces strict business rules:
- Dispatched - Returned == Net Dispatch
- Never back-calculates missing/unreadable cells
- Preserves reported net dispatch even when reconciliation fails
- Fault-tolerant continuation across all rows
"""

from typing import Optional, Tuple
from .models import CellData, Record, RecordStatus


class RecordValidator:
    """Validates and normalizes raw cell values into audit-compliant records."""

    @staticmethod
    def normalize_quantity_string(text: Optional[str]) -> Tuple[Optional[int], bool, Optional[str]]:
        """
        Normalizes a quantity string by stripping commas, spaces, and validating integer conversion.
        Returns:
            (normalized_int, is_valid, error_reason)
        """
        if text is None:
            return None, False, "Cell value is missing"

        cleaned = text.strip()
        if not cleaned:
            return None, False, "Cell is empty"

        # Check for unreadable placeholders
        if cleaned.upper() in {"[OBSCURED]", "UNREADABLE", "N/A", "NULL", "NONE", "?"}:
            return None, False, "Value is obscured or unreadable"

        # Strip standard number formatting: commas, leading/trailing whitespace
        sanitized = cleaned.replace(",", "").replace(" ", "").replace("`", "")

        try:
            val = int(sanitized)
            return val, True, None
        except ValueError:
            return None, False, f"Cannot parse '{cleaned}' as an integer"

    @classmethod
    def validate_row(
        cls,
        row_index: int,
        raw_product_code: Optional[str],
        raw_dispatched: Optional[str],
        raw_returned: Optional[str],
        raw_net_dispatch: Optional[str],
        confidences: Optional[dict] = None,
    ) -> Record:
        """
        Validates an individual table row.
        Adheres strictly to the invariant:
        - Never calculate replacement for unreadable cells from other values
        - Preserve reported net dispatch even when reconciliation fails
        """
        confidences = confidences or {}
        findings = []

        # 1. Product Code Cell
        prod_raw = raw_product_code.strip() if raw_product_code else None
        prod_conf = confidences.get("product_code", 1.0 if prod_raw else 0.0)
        prod_cell = CellData(
            raw_text=prod_raw,
            normalized_value=prod_raw,
            confidence=round(prod_conf, 3),
            is_readable=bool(prod_raw),
        )
        if not prod_raw:
            findings.append("Missing product code")

        # 2. Normalize Quantities
        disp_val, disp_valid, disp_err = cls.normalize_quantity_string(raw_dispatched)
        disp_conf = confidences.get("dispatched", 1.0 if disp_valid else 0.0)
        disp_cell = CellData(
            raw_text=raw_dispatched,
            normalized_value=disp_val,
            confidence=round(disp_conf, 3),
            is_readable=disp_valid,
        )
        if not disp_valid:
            findings.append(f"Dispatched cell error: {disp_err}")

        ret_val, ret_valid, ret_err = cls.normalize_quantity_string(raw_returned)
        ret_conf = confidences.get("returned", 1.0 if ret_valid else 0.0)
        ret_cell = CellData(
            raw_text=raw_returned,
            normalized_value=ret_val,
            confidence=round(ret_conf, 3),
            is_readable=ret_valid,
        )
        if not ret_valid:
            findings.append(f"Returned cell error: {ret_err}")

        net_val, net_valid, net_err = cls.normalize_quantity_string(raw_net_dispatch)
        net_conf = confidences.get("net_dispatch", 1.0 if net_valid else 0.0)
        net_cell = CellData(
            raw_text=raw_net_dispatch,
            normalized_value=net_val,
            confidence=round(net_conf, 3),
            is_readable=net_valid,
        )
        if not net_valid:
            findings.append(f"Net Dispatch cell error: {net_err}")

        # 3. Determine Row Status and Reconciliation
        all_quantities_valid = disp_valid and ret_valid and net_valid

        if not all_quantities_valid:
            # Check if any quantity is explicitly missing/obscured vs invalid format
            has_unreadable = (
                (raw_dispatched is None or "unreadable" in (disp_err or "").lower() or "obscured" in (disp_err or "").lower() or not disp_cell.is_readable and raw_dispatched is None)
                or (raw_returned is None or "unreadable" in (ret_err or "").lower() or "obscured" in (ret_err or "").lower() or not ret_cell.is_readable and raw_returned is None)
                or (raw_net_dispatch is None or "unreadable" in (net_err or "").lower() or "obscured" in (net_err or "").lower() or not net_cell.is_readable and raw_net_dispatch is None)
            )

            # Important: Do NOT back-calculate any missing/unreadable quantity!
            reconciled = None
            if has_unreadable:
                status = RecordStatus.UNREADABLE_CELL
                findings.append("One or more quantity cells are unreadable/missing; value reconstruction prohibited.")
            else:
                status = RecordStatus.INVALID_FORMAT
                findings.append("One or more quantity cells contain invalid numeric formats.")

        else:
            # All three numbers are valid integers -> verify Dispatched - Returned == Net Dispatch
            expected_net = disp_val - ret_val
            if expected_net == net_val:
                reconciled = True
                status = RecordStatus.VALID
            else:
                reconciled = False
                status = RecordStatus.RECONCILIATION_FAILED
                findings.append(
                    f"Reconciliation discrepancy: Dispatched ({disp_val}) - Returned ({ret_val}) = "
                    f"{expected_net}, but reported Net Dispatch is {net_val} (diff: {net_val - expected_net:+d}). "
                    f"Reported value preserved."
                )

        return Record(
            row_index=row_index,
            product_code=prod_cell,
            dispatched=disp_cell,
            returned=ret_cell,
            net_dispatch=net_cell,
            reconciled=reconciled,
            status=status,
            findings=findings,
        )
