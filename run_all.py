"""
run_all.py
Executes the Dispatch Table OCR pipeline across all three sample variants:
1. Version A: Baseline
2. Version B: Rotated & Low Contrast
3. Version C: Obscured Cell
Generates audit JSON files in results/ and prints a comparative summary.
"""

import os
import json
from main import process_image
from generate_samples import main as generate_all_samples

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_pipeline():
    sample_a = os.path.join(DATA_DIR, "sample_a_baseline.png")
    sample_b = os.path.join(DATA_DIR, "sample_b_rotated_low_contrast.png")
    sample_c = os.path.join(DATA_DIR, "sample_c_obscured.png")

    if not all(os.path.exists(p) for p in [sample_a, sample_b, sample_c]):
        print("Sample images not found. Generating them now...")
        generate_all_samples()

    samples = [
        ("Version A (Baseline)", sample_a, "result_a.json"),
        ("Version B (Rotated & Low Contrast)", sample_b, "result_b.json"),
        ("Version C (Obscured Data)", sample_c, "result_c.json"),
    ]

    print("=" * 80)
    print("DISPATCH TABLE OCR & VALIDATION PIPELINE EXECUTION")
    print("=" * 80)

    for label, img_path, out_filename in samples:
        print(f"\nProcessing {label}...")
        print(f"  Input:  {img_path}")
        out_path = os.path.join(RESULTS_DIR, out_filename)

        result = process_image(img_path)

        # Write result to JSON
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))

        print(f"  Output: {out_path}")
        print(f"  Processing Time: {result.processing_time_ms} ms")
        print(f"  Preprocessing: Skew={result.image_metadata.estimated_skew_degrees}°, "
              f"Deskewed={result.image_metadata.rotation_corrected}, "
              f"CLAHE_Boost={result.image_metadata.contrast_enhancement_applied}")
        print(f"  Batch Status: {result.summary.batch_status}")
        print(f"  Rows: Total={result.summary.total_rows_detected}, "
              f"Valid={result.summary.valid_rows}, "
              f"Recon_Failed={result.summary.reconciliation_failed_rows}, "
              f"Unreadable/Invalid={result.summary.unreadable_or_invalid_rows}")

        print("  Extracted Records:")
        for r in result.records:
            prod = r.product_code.normalized_value or "None"
            disp = r.dispatched.normalized_value if r.dispatched.normalized_value is not None else "None"
            ret = r.returned.normalized_value if r.returned.normalized_value is not None else "None"
            net = r.net_dispatch.normalized_value if r.net_dispatch.normalized_value is not None else "None"
            print(f"    - [{r.status:<21}] {prod:<8} | Disp: {disp:>5} | Ret: {ret:>5} | Net: {net:>5} | Findings: {len(r.findings)}")
            for f in r.findings:
                print(f"        -> {f}")

    print("\n" + "=" * 80)
    print("All 3 sample images processed successfully. Results saved in 'results/' directory.")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
