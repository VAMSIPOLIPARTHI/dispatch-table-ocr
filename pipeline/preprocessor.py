"""
pipeline/preprocessor.py
Adaptive image preprocessing for table OCR:
- Dynamic contrast enhancement (CLAHE) applied selectively when low dynamic range is detected
- Hough line-based skew estimation and affine rotation correction
- Preserves raw gradient anti-aliasing for neural OCR models
"""

import math
from typing import Tuple
import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocesses photographed or scanned table images adaptively."""

    def __init__(
        self,
        min_skew_angle_deg: float = 0.5,
        low_contrast_std_thresh: float = 15.0,
    ):
        self.min_skew_angle_deg = min_skew_angle_deg
        self.low_contrast_std_thresh = low_contrast_std_thresh

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from file path as BGR / RGB numpy array."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image from: {image_path}")
        return img

    def estimate_skew(self, gray: np.ndarray) -> float:
        """
        Estimates skew angle in degrees using Probabilistic Hough Transform on horizontal lines.
        Returns estimated angle in degrees (-45.0 to +45.0).
        Positive angle means rotated counter-clockwise.
        """
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=60,
            maxLineGap=10,
        )

        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    continue
                angle_deg = math.degrees(math.atan2(dy, dx))
                # Only keep near-horizontal lines within +/- 30 degrees
                if abs(angle_deg) <= 30.0:
                    angles.append(angle_deg)

        if len(angles) >= 3:
            return float(np.median(angles))

        # Secondary fallback: Minimum Area Bounding Box on thresholded text components
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 100:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            elif angle > 45:
                angle = 90 - angle
            if abs(angle) <= 30:
                return float(angle)

        return 0.0

    def rotate_image(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotates image around its center by angle_deg with white background border replication.
        """
        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        # Calculate new bounding dimensions to avoid clipping
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            img,
            rot_mat,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated

    def preprocess(self, img_input: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Runs adaptive preprocessing:
        1. Checks contrast standard deviation; applies CLAHE if contrast is low.
        2. Detects skew angle; corrects rotation if > min_skew_angle_deg.
        Returns:
            (preprocessed_bgr_img, metadata_dict)
        """
        h_orig, w_orig = img_input.shape[:2]
        img = img_input.copy()

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # 1. Dynamic Contrast Assessment
        intensity_std = float(np.std(gray))
        contrast_boost_applied = False

        if intensity_std < self.low_contrast_std_thresh:
            # Low contrast detected (e.g. Version B flat scan) -> Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            if len(img.shape) == 3:
                # Convert to LAB color space to preserve color balance while boosting luminance
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_enhanced = clahe.apply(l_channel)
                lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
                img = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            else:
                img = clahe.apply(img)
            contrast_boost_applied = True
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

        # 2. Skew Angle Detection
        skew_angle = self.estimate_skew(gray)
        rotation_corrected = False

        # OpenCV rotation is counter-clockwise for positive angles.
        # If Hough detected a positive tilt angle dy/dx, we rotate by -skew_angle to level it.
        if abs(skew_angle) >= self.min_skew_angle_deg:
            img = self.rotate_image(img, skew_angle)
            rotation_corrected = True

        metadata = {
            "original_width": w_orig,
            "original_height": h_orig,
            "contrast_std": round(intensity_std, 2),
            "contrast_enhancement_applied": contrast_boost_applied,
            "estimated_skew_degrees": round(skew_angle, 2),
            "rotation_corrected": rotation_corrected,
        }
        return img, metadata
