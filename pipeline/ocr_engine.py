"""
pipeline/ocr_engine.py
OCR Engine interface wrapping RapidOCR (DBNet detection + SVTR text recognition).
Produces standardized bounding boxes, detected text strings, and confidence scores.
"""

from typing import List, Dict, Any, Union
import numpy as np
from rapidocr_onnxruntime import RapidOCR


class OCREngine:
    """Encapsulates RapidOCR engine execution with structured primitive outputs."""

    def __init__(self):
        # RapidOCR initializes ONNX runtime models for text detection, direction, and recognition
        self.engine = RapidOCR()

    def detect_and_recognize(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Runs OCR detection and recognition on input image.
        Returns:
            List of detected text primitives:
            [
                {
                    "text": str,
                    "confidence": float,
                    "box": [x_min, y_min, x_max, y_max],
                    "centroid": [x_center, y_center],
                    "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                },
                ...
            ]
        """
        raw_result, _ = self.engine(img)

        primitives = []
        if not raw_result:
            return primitives

        for item in raw_result:
            # item format: (poly_points, text, confidence)
            poly, text, conf = item
            poly_np = np.array(poly, dtype=np.float32)

            x_min = float(np.min(poly_np[:, 0]))
            y_min = float(np.min(poly_np[:, 1]))
            x_max = float(np.max(poly_np[:, 0]))
            y_max = float(np.max(poly_np[:, 1]))

            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            primitives.append({
                "text": text.strip(),
                "confidence": float(conf),
                "box": [round(x_min, 1), round(y_min, 1), round(x_max, 1), round(y_max, 1)],
                "centroid": [round(x_center, 1), round(y_center, 1)],
                "polygon": poly,
            })

        return primitives
