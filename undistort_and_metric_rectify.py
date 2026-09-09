from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from config import (
    CAMERA_IMAGES_DIR,
    CAMERA_UNDISTORTED_DIR,
    METRIC_PLANE_NPZ,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def configure_console_encoding() -> None:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch camera undistortion + metric-plane homography rectification."
    )
    parser.add_argument("--input-dir", type=Path, default=CAMERA_IMAGES_DIR)
    parser.add_argument("--plane-calib", type=Path, default=METRIC_PLANE_NPZ)
    parser.add_argument("--output-dir", type=Path, default=CAMERA_UNDISTORTED_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_metric_calibration(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Plane calibration file not found: {path}")

    with np.load(str(path)) as data:
        required = {
            "camera_matrix",
            "distortion_coefficients",
            "new_camera_matrix",
            "image_size",
            "h_undistorted_image_to_metric_px",
            "rectified_output_size_px",
            "px_per_mm",
            "mm_per_px",
            "rectified_origin_mm",
            "rectified_field_size_mm",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError("Plane calibration is missing fields: " + ", ".join(missing))

        return {
            "camera_matrix": data["camera_matrix"].astype(np.float64),
            "distortion": data["distortion_coefficients"].astype(np.float64),
            "new_camera_matrix": data["new_camera_matrix"].astype(np.float64),
            "image_size": tuple(int(v) for v in data["image_size"]),
            "homography": data["h_undistorted_image_to_metric_px"].astype(np.float64),
            "output_size": tuple(int(v) for v in data["rectified_output_size_px"]),
            "px_per_mm": float(data["px_per_mm"]),
            "mm_per_px": float(data["mm_per_px"]),
            "origin_mm": [float(v) for v in data["rectified_origin_mm"]],
            "field_size_mm": [float(v) for v in data["rectified_field_size_mm"]],
        }


def list_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input image folder not found: {input_dir}")
    return [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def save_scale_metadata(output_dir: Path, calibration_path: Path, calib: dict[str, object]) -> None:
    metadata = {
        "processing": "camera undistortion + metric plane rectification",
        "plane_calibration_file": str(calibration_path.resolve()),
        "px_per_mm": calib["px_per_mm"],
        "mm_per_px": calib["mm_per_px"],
        "rectified_origin_mm": calib["origin_mm"],
        "rectified_field_size_mm": calib["field_size_mm"],
        "rectified_output_size_px": list(calib["output_size"]),
        "note": "Pixel scale is read directly from metric_plane_calibration.npz; no manual rescaling is applied.",
    }
    (output_dir / "metric_scale.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def process_folder(
    input_dir: Path, calibration_path: Path, output_dir: Path, overwrite: bool
) -> None:
    input_dir = input_dir.resolve()
    calibration_path = calibration_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calib = load_metric_calibration(calibration_path)
    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    expected_size = calib["image_size"]
    output_size = calib["output_size"]
    map1, map2 = cv2.initUndistortRectifyMap(
        calib["camera_matrix"],
        calib["distortion"],
        None,
        calib["new_camera_matrix"],
        expected_size,
        cv2.CV_16SC2,
    )

    print("\nUndistortion + metric-plane rectification started")
    print("-" * 68)
    print(f"Input folder:       {input_dir}")
    print(f"Output folder:      {output_dir}")
    print(f"Plane calibration:  {calibration_path}")
    print(f"Required input size:{expected_size[0]} x {expected_size[1]} px")
    print(f"Output size:        {output_size[0]} x {output_size[1]} px")
    print(f"True output scale:  1 px = {calib['mm_per_px']:.9g} mm")
    print(f"Field of view:      {calib['field_size_mm'][0]:.6g} x {calib['field_size_mm'][1]:.6g} mm")
    print(f"Image count:        {len(image_paths)}")
    print("-" * 68)

    saved = skipped = failed = 0
    for index, image_path in enumerate(image_paths, start=1):
        output_path = output_dir / image_path.name
        if output_path.exists() and not overwrite:
            print(f"[{index}/{len(image_paths)}] Exists, skipped: {output_path.name}")
            skipped += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[{index}/{len(image_paths)}] Read failed: {image_path.name}")
            failed += 1
            continue
        current_size = (image.shape[1], image.shape[0])
        if current_size != expected_size:
            print(
                f"[{index}/{len(image_paths)}] Size mismatch, skipped: {image_path.name} "
                f"({current_size[0]}x{current_size[1]}, expected "
                f"{expected_size[0]}x{expected_size[1]})"
            )
            skipped += 1
            continue

        undistorted = cv2.remap(
            image, map1, map2, interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        metric_rectified = cv2.warpPerspective(
            undistorted,
            calib["homography"],
            output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if not cv2.imwrite(str(output_path), metric_rectified):
            print(f"[{index}/{len(image_paths)}] Save failed: {output_path}")
            failed += 1
            continue

        print(f"[{index}/{len(image_paths)}] Saved: {output_path.name}")
        saved += 1

    save_scale_metadata(output_dir, calibration_path, calib)
    print("\nDone")
    print("-" * 68)
    print(f"Saved:   {saved}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print(f"Output:  {output_dir}")
    print(f"Scale:   1 px = {calib['mm_per_px']:.9g} mm")


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    try:
        process_folder(args.input_dir, args.plane_calib, args.output_dir, args.overwrite)
        return 0
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
