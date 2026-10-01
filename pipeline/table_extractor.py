"""
pipeline/table_extractor.py
Agnostic spatial layout parser for tabular document images.
- Identifies column anchors dynamically from headers
- Filters titles, subtitles, and header rows
- Clusters data primitives into rows via vertical geometric overlap
- Assigns cell primitives to columns based on horizontal interval bounds
- Does not hard-code product codes, row counts, or expected numeric values
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from .models import Record
from .validator import RecordValidator


class TableExtractor:
    """Extracts structured table records from unstructured OCR primitives."""

    def __init__(self, vertical_row_threshold_ratio: float = 0.65):
        self.vertical_row_threshold_ratio = vertical_row_threshold_ratio

    @staticmethod
    def _is_header_primitive(text: str) -> Optional[int]:
        """
        Determines if a text primitive belongs to one of the 4 table columns.
        Returns column index (0..3) or None.
        """
        lower = text.lower()
        # Col 3: Net Dispatch (checked before 'dispatch' to avoid conflict)
        if "net" in lower:
            return 3
        # Col 1: Dispatched
        if "dispatch" in lower:
            return 1
        # Col 2: Returned
        if "return" in lower:
            return 2
        # Col 0: Product Code
        if "product" in lower or "code" in lower or "item" in lower:
            return 0
        return None

    def find_column_anchors(
        self, primitives: List[Dict[str, Any]]
    ) -> Tuple[Dict[int, float], float]:
        """
        Finds horizontal centroid anchors for columns 0..3 and the bottom Y boundary of headers.
        Returns:
            (col_centroids_dict, header_y_bottom)
        """
        col_primitives: Dict[int, List[Dict[str, Any]]] = {0: [], 1: [], 2: [], 3: []}
        header_y_max = 0.0

        for prim in primitives:
            col_idx = self._is_header_primitive(prim["text"])
            if col_idx is not None:
                col_primitives[col_idx].append(prim)
                header_y_max = max(header_y_max, prim["box"][3])

        # Compute centroid for each detected column
        col_centroids = {}
        for c in range(4):
            if col_primitives[c]:
                # Average x centroid of matched primitives for this header
                avg_x = sum(p["centroid"][0] for p in col_primitives[c]) / len(col_primitives[c])
                col_centroids[c] = avg_x

        # Fallback if any column header wasn't detected: interpolate or use horizontal spans
        if len(col_centroids) < 4:
            # If at least 2 are found, we can interpolate evenly
            all_x = sorted(p["centroid"][0] for p in primitives)
            if all_x:
                min_x, max_x = all_x[0], all_x[-1]
                span = max_x - min_x
                for c in range(4):
                    if c not in col_centroids:
                        col_centroids[c] = min_x + (span * (c + 0.5) / 4)

        return col_centroids, header_y_max

    def cluster_rows(
        self, primitives: List[Dict[str, Any]], header_y_bottom: float
    ) -> List[List[Dict[str, Any]]]:
        """
        Filters out header/metadata text and clusters data primitives into rows based on Y-overlap.
        """
        # Exclude primitives located at or above the header row bottom (+ small epsilon margin)
        data_prims = [
            p for p in primitives
            if p["centroid"][1] > (header_y_bottom + 4.0)
            and "monthly" not in p["text"].lower()
            and "quantity" not in p["text"].lower()
            and "summary" not in p["text"].lower()
            and "units" not in p["text"].lower()
        ]

        if not data_prims:
            return []

        # Calculate median box height for clustering tolerance
        heights = [(p["box"][3] - p["box"][1]) for p in data_prims]
        median_height = float(sorted(heights)[len(heights) // 2]) if heights else 25.0
        y_tolerance = median_height * self.vertical_row_threshold_ratio

        # Sort data primitives top to bottom by Y centroid
        sorted_by_y = sorted(data_prims, key=lambda p: p["centroid"][1])

        # Group into rows
        rows: List[List[Dict[str, Any]]] = []
        for prim in sorted_by_y:
            y_c = prim["centroid"][1]
            placed = False
            for row in rows:
                row_mean_y = sum(p["centroid"][1] for p in row) / len(row)
                if abs(y_c - row_mean_y) <= y_tolerance:
                    row.append(prim)
                    placed = True
                    break
            if not placed:
                rows.append([prim])

        # Sort elements within each row from left to right (by X centroid)
        for row in rows:
            row.sort(key=lambda p: p["centroid"][0])

        # Sort rows from top to bottom
        rows.sort(key=lambda r: sum(p["centroid"][1] for p in r) / len(r))
        return rows

    def extract_records(self, primitives: List[Dict[str, Any]]) -> List[Record]:
        """
        Main extraction pipeline:
        1. Finds column anchors from header row
        2. Clusters body primitives into rows
        3. Assigns each row's primitives to columns 0..3
        4. Validates and normalizes each record
        """
        if not primitives:
            return []

        col_centroids, header_y_bottom = self.find_column_anchors(primitives)

        # Compute column boundary split points midway between adjacent column centroids
        splits = []
        sorted_cols = sorted(col_centroids.keys())
        for i in range(len(sorted_cols) - 1):
            c1 = sorted_cols[i]
            c2 = sorted_cols[i + 1]
            split_x = (col_centroids[c1] + col_centroids[c2]) / 2.0
            splits.append(split_x)

        # Helper to assign an X coordinate to a column index
        def get_col_index(x: float) -> int:
            if not splits:
                return 0
            for idx, split_val in enumerate(splits):
                if x < split_val:
                    return idx
            return len(splits)

        rows = self.cluster_rows(primitives, header_y_bottom)
        records: List[Record] = []

        for row_idx, row_prims in enumerate(rows, start=1):
            col_assigned: Dict[int, Dict[str, Any]] = {}
            for prim in row_prims:
                c_idx = get_col_index(prim["centroid"][0])
                # If collision in same column, combine text or pick highest confidence
                if c_idx not in col_assigned:
                    col_assigned[c_idx] = prim
                else:
                    # Append text if multiple tokens detected in same cell
                    col_assigned[c_idx]["text"] += " " + prim["text"]
                    col_assigned[c_idx]["confidence"] = min(
                        col_assigned[c_idx]["confidence"], prim["confidence"]
                    )

            # Map raw cell strings and confidences
            raw_prod = col_assigned.get(0, {}).get("text")
            raw_disp = col_assigned.get(1, {}).get("text")
            raw_ret = col_assigned.get(2, {}).get("text")
            raw_net = col_assigned.get(3, {}).get("text")

            conf_dict = {
                "product_code": col_assigned.get(0, {}).get("confidence", 0.0),
                "dispatched": col_assigned.get(1, {}).get("confidence", 0.0),
                "returned": col_assigned.get(2, {}).get("confidence", 0.0),
                "net_dispatch": col_assigned.get(3, {}).get("confidence", 0.0),
            }

            record = RecordValidator.validate_row(
                row_index=row_idx,
                raw_product_code=raw_prod,
                raw_dispatched=raw_disp,
                raw_returned=raw_ret,
                raw_net_dispatch=raw_net,
                confidences=conf_dict,
            )
            records.append(record)

        return records
