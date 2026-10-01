"""
generate_samples.py
Script to programmatically generate the three required sample images:
1. Version A: Clear baseline summary table
2. Version B: Rotated slightly (~3.8 degrees) with reduced contrast
3. Version C: Obscured 'Returned' cell for product PRD-102
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def get_font(size: int, bold: bool = False):
    """Attempt to load a standard system TrueType font, falling back to default."""
    font_candidates = [
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibrib.ttf" if bold else "C:\\Windows\\Fonts\\calibri.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf" if bold else "C:\\Windows\\Fonts\\segoeui.ttf",
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_baseline_table() -> tuple[Image.Image, dict]:
    """
    Renders the clean baseline table image.
    Returns:
        (image, metadata_dict) where metadata_dict contains cell coordinates.
    """
    width, height = 1000, 580
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = get_font(26, bold=True)
    subtitle_font = get_font(16, bold=False)
    header_font = get_font(18, bold=True)
    body_font = get_font(18, bold=False)

    # Document Header
    draw.text((60, 40), "Monthly Dispatch Summary", fill=(20, 25, 35), font=title_font)
    draw.text((60, 80), "Quantity in units", fill=(90, 95, 105), font=subtitle_font)

    # Table Layout
    table_left = 60
    table_top = 130
    row_height = 55
    col_widths = [220, 220, 220, 220]
    col_x = [table_left + sum(col_widths[:i]) for i in range(len(col_widths))]
    headers = ["Product Code", "Dispatched", "Returned", "Net Dispatch"]
    rows = [
        ["PRD-101", "1,250", "50", "1,200"],
        ["PRD-102", "800", "25", "775"],
        ["PRD-103", "600", "40", "570"],
        ["PRD-104", "450", "0", "450"],
    ]

    total_width = sum(col_widths)
    total_height = row_height * (len(rows) + 1)

    # Header Background
    draw.rectangle(
        [table_left, table_top, table_left + total_width, table_top + row_height],
        fill=(244, 246, 250),
    )

    # Outer table border
    draw.rectangle(
        [table_left, table_top, table_left + total_width, table_top + total_height],
        outline=(200, 205, 215),
        width=2,
    )

    # Header divider
    draw.line(
        [(table_left, table_top + row_height), (table_left + total_width, table_top + row_height)],
        fill=(180, 185, 195),
        width=2,
    )

    # Render Header Text
    for i, header in enumerate(headers):
        x = col_x[i] + 25
        y = table_top + (row_height - 20) // 2
        draw.text((x, y), header, fill=(30, 40, 55), font=header_font)

    # Render Rows and record cell bounding boxes
    cell_boxes = {}
    for r_idx, row in enumerate(rows):
        y_top = table_top + (r_idx + 1) * row_height
        y_bottom = y_top + row_height

        # Alternating subtle row background
        if r_idx % 2 == 1:
            draw.rectangle(
                [table_left + 1, y_top + 1, table_left + total_width - 1, y_bottom - 1],
                fill=(250, 252, 255),
            )

        # Horizontal row divider line
        draw.line(
            [(table_left, y_bottom), (table_left + total_width, y_bottom)],
            fill=(225, 230, 238),
            width=1,
        )

        for c_idx, val in enumerate(row):
            x = col_x[c_idx] + 25
            y = y_top + (row_height - 20) // 2
            draw.text((x, y), val, fill=(35, 40, 50), font=body_font)
            cell_boxes[(r_idx, c_idx)] = (col_x[c_idx], y_top, col_x[c_idx] + col_widths[c_idx], y_bottom)

    # Vertical column dividers
    for i in range(1, len(col_widths)):
        x_div = col_x[i]
        draw.line(
            [(x_div, table_top), (x_div, table_top + total_height)],
            fill=(230, 234, 242),
            width=1,
        )

    metadata = {
        "headers": headers,
        "rows": rows,
        "cell_boxes": cell_boxes,
        "table_bounds": (table_left, table_top, table_left + total_width, table_top + total_height),
    }
    return image, metadata


def generate_version_a(base_img: Image.Image) -> str:
    """Saves baseline image."""
    path = os.path.join(DATA_DIR, "sample_a_baseline.png")
    base_img.save(path, format="PNG")
    print(f"Generated Version A (Baseline): {path}")
    return path


def generate_version_b(base_img: Image.Image) -> str:
    """
    Version B: Rotates slightly (~3.8 degrees) and reduces contrast.
    """
    # 1. Rotate by ~3.8 degrees with white expansion background
    angle = 3.8
    rotated = base_img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))

    # 2. Reduce contrast significantly
    enhancer = ImageEnhance.Contrast(rotated)
    low_contrast = enhancer.enhance(0.40)

    # 3. Adjust brightness / gamma slightly towards gray to simulate flatbed scan loss
    enhancer_b = ImageEnhance.Brightness(low_contrast)
    low_contrast = enhancer_b.enhance(0.92)

    path = os.path.join(DATA_DIR, "sample_b_rotated_low_contrast.png")
    low_contrast.save(path, format="PNG")
    print(f"Generated Version B (Rotated {angle} deg & Low Contrast): {path}")
    return path


def generate_version_c(base_img: Image.Image, metadata: dict) -> str:
    """
    Version C: Obscures the 'Returned' value for PRD-102 (Row index 1, Col index 2).
    """
    img_c = base_img.copy()
    draw = ImageDraw.Draw(img_c)

    # PRD-102 is row 1 (0-indexed: PRD-101=0, PRD-102=1). Returned is col 2.
    box = metadata["cell_boxes"][(1, 2)]
    x1, y1, x2, y2 = box

    # Target specific text area within the cell
    text_x1 = x1 + 20
    text_y1 = y1 + 12
    text_x2 = text_x1 + 65
    text_y2 = text_y1 + 30

    # Draw heavy black marker scribble / blotch
    draw.rectangle([text_x1 - 4, text_y1 - 2, text_x2 + 4, text_y2 + 2], fill=(15, 15, 20))
    # Add uneven smudge edges
    for i in range(-5, 35, 6):
        draw.line([(text_x1 - 6, text_y1 + i), (text_x2 + 8, text_y1 + i + 2)], fill=(10, 10, 15), width=4)

    # Apply localized blur to simulate ink bleed/smudge
    crop_region = img_c.crop((text_x1 - 10, text_y1 - 8, text_x2 + 12, text_y2 + 8))
    blurred = crop_region.filter(ImageFilter.GaussianBlur(radius=1.8))
    img_c.paste(blurred, (text_x1 - 10, text_y1 - 8))

    path = os.path.join(DATA_DIR, "sample_c_obscured.png")
    img_c.save(path, format="PNG")
    print(f"Generated Version C (Obscured PRD-102 Returned): {path}")
    return path


def main():
    print("Generating synthetic sample images...")
    base_img, metadata = render_baseline_table()
    generate_version_a(base_img)
    generate_version_b(base_img)
    generate_version_c(base_img, metadata)
    print("All sample images generated successfully in 'data/' directory.")


if __name__ == "__main__":
    main()
