"""
main.py
CLI entrypoint for Dispatch Table OCR extraction and validation pipeline.
Accepts any image file path, processes it, and outputs standardized audit JSON.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from pipeline.models import PipelineResult, PipelineSummary, ImageMetadata
from pipeline.preprocessor import ImagePreprocessor
from pipeline.ocr_engine import OCREngine
from pipeline.table_extractor import TableExtractor


def process_image(image_path: str, preprocessor: ImagePreprocessor = None, ocr_engine: OCREngine = None, table_extractor: TableExtractor = None) -> PipelineResult:
    """
    Processes a single image file and returns a structured PipelineResult.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    preprocessor = preprocessor or ImagePreprocessor()
    ocr_engine = ocr_engine or OCREngine()
    table_extractor = table_extractor or TableExtractor()

    start_time = time.perf_counter()

    # 1. Image Loading and Adaptive Preprocessing
    raw_img = preprocessor.load_image(image_path)
    preprocessed_img, prep_meta = preprocessor.preprocess(raw_img)

    # 2. OCR Primitive Extraction
    primitives = ocr_engine.detect_and_recognize(preprocessed_img)

    # 3. Spatial Parsing and Validation
    records = table_extractor.extract_records(primitives)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # 4. Summarize Batch Status
    total = len(records)
    valid_count = sum(1 for r in records if r.status == "VALID")
    failed_recon = sum(1 for r in records if r.status == "RECONCILIATION_FAILED")
    unreadable_invalid = sum(1 for r in records if r.status in {"UNREADABLE_CELL", "INVALID_FORMAT", "MISSING_VALUE"})

    if valid_count == total and total > 0:
        batch_status = "ALL_VALID"
    elif unreadable_invalid > 0:
        batch_status = "REQUIRES_REVIEW_UNREADABLE"
    elif failed_recon > 0:
        batch_status = "REQUIRES_REVIEW_RECONCILIATION"
    else:
        batch_status = "NO_RECORDS_DETECTED"

    summary = PipelineSummary(
        total_rows_detected=total,
        valid_rows=valid_count,
        reconciliation_failed_rows=failed_recon,
        unreadable_or_invalid_rows=unreadable_invalid,
        batch_status=batch_status,
    )

    image_meta = ImageMetadata(
        width=prep_meta["original_width"],
        height=prep_meta["original_height"],
        estimated_skew_degrees=prep_meta["estimated_skew_degrees"],
        rotation_corrected=prep_meta["rotation_corrected"],
        contrast_enhancement_applied=prep_meta["contrast_enhancement_applied"],
    )

    result = PipelineResult(
        source_image=os.path.abspath(image_path),
        processing_timestamp=datetime.utcnow().isoformat() + "Z",
        processing_time_ms=elapsed_ms,
        image_metadata=image_meta,
        summary=summary,
        records=records,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Dispatch Table OCR & Validation Pipeline")
    parser.add_argument("image_path", help="Path to input image file (e.g. data/sample_a_baseline.png)")
    parser.add_argument("-o", "--output", help="Optional path to write JSON output file")
    parser.add_argument("--json", action="store_true", help="Print raw JSON to stdout")
    args = parser.parse_args()

    try:
        result = process_image(args.image_path)
    except Exception as e:
        print(f"Error processing image: {e}", file=sys.stderr)
        sys.exit(1)

    result_json = result.model_dump_json(indent=2)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result_json)
        print(f"JSON output successfully saved to: {args.output}")

    if args.json or not args.output:
        print(result_json)


if __name__ == "__main__":
    main()
