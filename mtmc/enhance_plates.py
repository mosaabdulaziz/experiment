"""Aggressive enhancement pipeline for low-res CCTV license plate crops.

Applies multiple morphological + thresholding strategies to maximize readability.

Usage:
    .venv/bin/python -m mtmc.enhance_plates --input mtmc/output/plate_crops/plates
    .venv/bin/python -m mtmc.enhance_plates --input mtmc/output/plate_crops/plates/frame01649_best.jpg --scale 6
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def upscale(img, scale):
    """High-quality upscale with INTER_LANCZOS4."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def morphological_enhance(img):
    """Contrast stretch + CLAHE + denoise + sharpen."""
    # Contrast stretch per channel
    stretched = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        lo, hi = np.percentile(ch, (1, 99))
        stretched[:, :, c] = np.clip((ch - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)

    # CLAHE on L channel
    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Bilateral denoise
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=50, sigmaSpace=50)

    # Sharpen via unsharp mask
    blurred = cv2.GaussianBlur(denoised, (0, 0), 2)
    sharpened = cv2.addWeighted(denoised, 1.8, blurred, -0.8, 0)

    return sharpened


def method_otsu(img_gray):
    """Standard Otsu threshold."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def method_adaptive_gaussian(img_gray):
    """Adaptive Gaussian threshold."""
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 8)


def method_adaptive_mean(img_gray):
    """Adaptive mean threshold."""
    return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 25, 12)


def method_morph_close(img_gray):
    """Otsu + morphological close to fill gaps."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def method_morph_open_close(img_gray):
    """Open then close to clean noise then fill gaps."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def method_dilate(img_gray):
    """Otsu + dilate to thicken characters."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated


def method_erode_dilate(img_gray):
    """Erode then dilate (opening) to remove noise, then close."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Opening: erode then dilate (removes small noise)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k1, iterations=1)
    # Closing: dilate then erode (fills gaps)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k2, iterations=2)
    return closed


def method_blackhat(img_gray):
    """Black-hat to extract dark features (text) on lighter background."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    _, binary = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def method_tophat(img_gray):
    """Top-hat to extract light features."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def method_gradient(img_gray):
    """Morphological gradient to extract edges."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def method_canny_morph(img_gray):
    """Canny edges + dilate to fill in character strokes."""
    edges = cv2.Canny(img_gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    # Invert so text is black on white
    return cv2.bitwise_not(dilated)


def method_invert_otsu(img_gray):
    """Otsu with inversion (handles light text on dark bg)."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def method_adaptive_morph(img_gray):
    """Adaptive threshold + morphological cleanup."""
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def method_contrast_binary(img_gray):
    """Aggressive contrast stretch then Otsu."""
    # Percentile-based stretch
    lo = np.percentile(img_gray, 5)
    hi = np.percentile(img_gray, 95)
    stretched = np.clip((img_gray.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def method_adaptive_dilate(img_gray):
    """Adaptive threshold + dilate to connect broken strokes."""
    binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return dilated


def method_heavy_morph(img_gray):
    """Heavy morphological processing: close → open → close."""
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Close to fill gaps between character parts
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=3)
    # Open to remove noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k, iterations=1)
    # Close again to fill any new gaps
    final = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k, iterations=2)
    return final


ALL_METHODS = {
    "otsu": method_otsu,
    "adaptive_gaussian": method_adaptive_gaussian,
    "adaptive_mean": method_adaptive_mean,
    "morph_close": method_morph_close,
    "morph_open_close": method_morph_open_close,
    "dilate": method_dilate,
    "erode_dilate": method_erode_dilate,
    "blackhat": method_blackhat,
    "tophat": method_tophat,
    "gradient": method_gradient,
    "canny_morph": method_canny_morph,
    "invert_otsu": method_invert_otsu,
    "adaptive_morph": method_adaptive_morph,
    "contrast_binary": method_contrast_binary,
    "adaptive_dilate": method_adaptive_dilate,
    "heavy_morph": method_heavy_morph,
}


def main():
    parser = argparse.ArgumentParser(description="Enhance CCTV plate crops — all methods")
    parser.add_argument("--input", required=True, help="Input directory or single image")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--scale", type=int, default=4, help="Upscale factor")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        images = [input_path]
        out_dir = Path(args.output) if args.output else input_path.parent / "enhanced_all"
    else:
        images = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))
        out_dir = Path(args.output) if args.output else input_path / "enhanced_all"

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(images)} images × {len(ALL_METHODS)} methods")
    print(f"Scale: {args.scale}x → {out_dir}")

    for img_path in images:
        stem = img_path.stem
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_path.name}")
            continue

        # Save original upscaled
        upscaled = upscale(img, args.scale)
        cv2.imwrite(str(out_dir / f"{stem}_original.jpg"), upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Save morphological enhanced (color)
        morph = morphological_enhance(img)
        morph_up = upscale(morph, args.scale)
        cv2.imwrite(str(out_dir / f"{stem}_morph_enhanced.jpg"), morph_up, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Convert to grayscale for thresholding methods
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_up = upscale(cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY), args.scale)

        # Apply each binary method
        for name, method in ALL_METHODS.items():
            try:
                result = method(gray_up)
                cv2.imwrite(str(out_dir / f"{stem}_{name}.jpg"), result)
            except Exception as e:
                print(f"  {stem}/{name}: FAILED — {e}")

        print(f"  {img_path.name}: {img.shape[1]}x{img.shape[0]} → {len(ALL_METHODS)+2} outputs")

    print(f"\nDone! All results in: {out_dir}")
    print(f"Methods tested: {', '.join(ALL_METHODS.keys())}")


if __name__ == "__main__":
    main()
