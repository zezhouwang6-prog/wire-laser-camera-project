from __future__ import annotations

"""
RUN20 camera stitch - standalone edition
=========================================

Purpose
-------
Reproduce the camera-stitching pipeline used by the verified run20 result,
without importing pinjie.py.

The important RUN20 order is preserved:
    1. load source camera frames in capture/image-index order;
    2. use real camera motor feedback as the global circumferential coordinate;
    3. use the verified RUN20 camera ROI at a permanently fixed pixel position;
    4. apply the same cylindrical correction idea as pinjie.py;
    5. build a visually clean NATURAL stitch with pinjie-style overlap matching:
           global overlap search -> repair/median smoothing
           -> local overlap + dx refinement
           -> full overlap linear blend (aligned-blend)
    6. only AFTER all visual joins are complete, piecewise-warp the natural
       stitched image onto the real-encoder metric Y grid;
    7. save the same diagnostic families used by run20.

This file intentionally has NO `import pinjie` dependency.
It still uses config.py for the experiment input/output folders, exactly like
other programs in this project.
"""

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from statistics import median

import cv2
import numpy as np

try:
    from config import CAMERA_STITCHED_DIR, CAMERA_UNDISTORTED_DIR, LASER_RESTORED_DIR
except Exception:
    # Fallback only when this file is copied outside the original project.
    # In the normal project, config.py still remains the preferred single place
    # for changing experiment paths.
    _HERE = Path(__file__).resolve().parent
    CAMERA_UNDISTORTED_DIR = _HERE / "undistorted_images"
    CAMERA_STITCHED_DIR = _HERE / "stitched"
    LASER_RESTORED_DIR = _HERE / "laser_restored"


# =============================================================================
# RUN20 SETTINGS - KEEP THESE UNCHANGED WHEN REPRODUCING RUN20
# =============================================================================
INPUT_IMG_DIR = CAMERA_UNDISTORTED_DIR
OUTPUT_DIR = CAMERA_STITCHED_DIR

MATCH_LASER_HEIGHT_AFTER_STITCH = False
LASER_HEIGHT_REFERENCE_NAMES = (
    "laser_height_mm_true_scale_0p0504mm_detail_height.png",
    "laser_height_mm_true_scale_detail_height.png",
    "laser_height_mm_detail_height.png",
)

ROI_JSON_PATH = ""

# -------------------------------------------------------------------------
# PERMANENTLY LOCKED CAMERA ROI
# -------------------------------------------------------------------------
# These four values come from the experiment where horizontal fusion was good.
# Every experiment must use the SAME source-image pixel coordinate system.
#
# IMPORTANT:
#   - Normal execution does NOT open an ROI selection window.
#   - Existing roi.json files are ignored while ROI_SELECTION_MODE == "fixed".
#   - The final stitched image width is therefore always 339 px.
#   - Do not change these values when changing wire type.
#
# Verified good ROI:
#     X = 1038
#     Y = 1035
#     W = 339
#     H = 85
ROI_SELECTION_MODE = "fixed"
FIXED_ROI_X_PX = 1056
FIXED_ROI_Y_PX = 1040
FIXED_ROI_WIDTH_PX = 358
FIXED_ROI_HEIGHT_PX = 79

SHOW_PREVIEW_CENTER_GUIDES = True
ROI_SELECT_IMAGE_INDEX = 0

# Before stitching starts, always show the LOCKED ROI for visual confirmation.
# This preview NEVER changes the fixed ROI coordinates.
PREVIEW_FIXED_ROI_BEFORE_STITCH = True
SAVE_CONFIRMED_ROI_PREVIEW = True
ROI_PREVIEW_MAX_FULL_WIDTH = 1400
ROI_PREVIEW_MAX_FULL_HEIGHT = 820
ROI_PREVIEW_ZOOM_SCALE = 3.0

IMAGE_EXT = ".png"
WHEEL_DIAMETER_MM = 150.0
STITCH_SURFACE_MM_PER_PIXEL = 0.05473705
FALLBACK_ANGLE_STEP_DEG = 1.0
TOTAL_ROTATION_DEG = 360.0
DROP_REAL_360_DUPLICATE = True

# ---- real encoder ----
USE_REAL_ENCODER_FEEDBACK = True
ENCODER_FEEDBACK_FILENAME = "camera_motor_feedback.csv"
ENCODER_REAL_FEEDBACK_ARBITRATION_ID = 0
ENCODER_FEEDBACK_INDEX_SHIFT = -1
ENCODER_EXPECTED_STEP_DEG = 1.0
ENCODER_MAX_SINGLE_STEP_DEVIATION_DEG = 0.45
ENCODER_MAX_ABS_TARGET_ERROR_DEG = 2.0
SAVE_ENCODER_ANGLE_TABLE = True
LOCK_REAL_ENCODER_GLOBAL_CENTERS = True

# ---- image preparation / cylindrical correction ----
USE_CLAHE = True
USE_CYLINDRICAL_CORRECTION = True
CYLINDER_RADIUS_PX = 0.0
CYLINDER_CENTER_OFFSET_Y_PX = 0.0
CURVATURE_CALIBRATION_HEIGHT = 120
CURVATURE_CALIBRATION_PAIRS = 32
CURVATURE_MAX_SHIFT_RATIO = 0.45
CURVATURE_MIN_SCORE = 0.12

# ---- run20 seam / overlap settings ----
WEIGHTED_OVERLAP_BLEND = False
BLEND_EDGE_WEIGHT = 0.08
SEAM_FEATHER_ROWS = 1
USE_CONTENT_AWARE_SEAM = False
SEAM_SEARCH_HALF_ROWS = 10
SEAM_PATH_MAX_STEP = 1
SEAM_PATH_SMOOTHNESS = 3.0
SEAM_CENTER_PENALTY = 0.45
SEAM_EDGE_WEIGHT = 0.75
SEAM_BRIGHTNESS_WEIGHT = 0.30
SEAM_BLOCK_WIDTH = 24
SEAM_SMOOTH_WINDOW = 61
SEAM_MAX_PATH_OFFSET_ROWS = 6
SEAM_OUT_OF_LIMIT_PENALTY = 20.0
SEAM_FOREGROUND_PENALTY = 1.10
SEAM_FOREGROUND_BRIGHT_PERCENTILE = 68.0
SEAM_FOREGROUND_EDGE_PERCENTILE = 65.0
SEAM_FOREGROUND_DILATE_PX = 1
SEAM_FEATHER_DARK_PERCENTILE = 55.0
SEAM_FEATHER_EDGE_PERCENTILE = 45.0
SEAM_MAX_LOCAL_SOURCE_CORRECTION_PX = 2.0
SEAM_USE_LOCAL_X_CORRECTION = False
OVERLAP_KEEP_MODE = "keep_previous_full_overlap"
OVERLAP_CUT_MARGIN_SOURCE_ROWS = 0.0
MATCH_COLOR_TO_PREVIOUS = False
SAVE_DEBUG_STRIPS = True
DEBUG_STRIP_COUNT = 12

# ---- paper / thesis preprocessing figures ----
# These outputs are diagnostics only. They DO NOT change the stitching result.
# The program input is CAMERA_UNDISTORTED_DIR, so 01_rectified_input.png is
# already the image after lens-undistortion + metric-plane rectification.
SAVE_PAPER_PREPROCESS_IMAGES = True
PAPER_EXAMPLE_IMAGE_INDEX = 180  # representative frame; clamped automatically
PAPER_PREPROCESS_DIRNAME = "paper_preprocess_examples"

# ---- pinjie.py visual matcher constants used by run20 ----
GLOBAL_MIN_OVERLAP_RATIO = 0.30
GLOBAL_MAX_OVERLAP_RATIO = 0.70
MATCH_CENTER_RATIO = 0.85
SCORE_TRUST_THRESHOLD = 0.12
BOUNDARY_PENALTY = True
LOCAL_REPAIR_RADIUS = 3
MEDIAN_FILTER_WINDOW = 15
LOCAL_ALIGN_MAX_SHIFT_X = 2
LOCAL_ALIGN_MAX_SHIFT_OVERLAP_ROWS = 3

USE_METRIC_PHYSICAL_STRIP_STITCH = False
PHYSICAL_STRIP_TAKE = "center"
USE_PINJIE_STYLE_OVERLAP_STITCH = True
PINJIE_STYLE_COMPOSE_MODE = "aligned-blend"  # verified run20
SAVE_PINJIE_STYLE_OVERLAP_TABLE = True

# Visual overlap first, then real-encoder piecewise Y warp.
USE_ENCODER_CONSTRAINED_OVERLAP_WARP = True
ENCODER_OVERLAP_WARP_CHUNK_ROWS = 512
ENCODER_OVERLAP_REQUIRE_REAL_FEEDBACK = True

# Registration is diagnostic / source-scale estimation in run20.
AUTO_SEAM_REGISTRATION = True
REGISTRATION_SEARCH_ROWS = 22
REGISTRATION_SEARCH_COLS = 6
REGISTRATION_MIN_SCORE = 0.55
REGISTRATION_BOUNDARY_MARGIN_PX = 2
REGISTRATION_MAX_INITIAL_STEP_ERROR_PX = 20.0
REGISTRATION_MAX_CENTER_CORRECTION_PX = 12.0
REGISTRATION_MIN_OVERLAP_ROWS = 12
REGISTRATION_X_MARGIN_PX = 20
REGISTRATION_MIN_VALID_PAIRS = 6


# =============================================================================
# GENERAL / IO HELPERS INLINED FROM pinjie.py FAMILY
# =============================================================================
def natural_sort_key(path_obj: Path):
    s = path_obj.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def parse_angle_from_filename(path_obj: Path) -> float | None:
    match = re.search(r"angle_([+-]?\d+(?:\.\d+)?)", path_obj.name, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def find_capture_info_path(img_dir: Path) -> Path | None:
    for path in (img_dir / "capture_info.csv", img_dir.parent / "capture_info.csv"):
        if path.exists():
            return path
    return None


def load_images(img_dir: Path, ext: str) -> list[Path]:
    info_path = find_capture_info_path(img_dir)
    if info_path is not None:
        rows: list[tuple[Path, float | None]] = []
        with info_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = str(row.get("filename", "")).strip()
                if not filename:
                    continue
                p = img_dir / filename
                if not p.exists() or p.suffix.lower() != ext.lower():
                    continue
                if not p.name.lower().startswith("img_"):
                    continue
                angle_text = row.get("angle_deg", "") or row.get("target_angle_deg", "")
                try:
                    angle = float(angle_text)
                except (TypeError, ValueError):
                    angle = None
                rows.append((p, angle))
        if rows:
            return [p for p, _ in sorted(rows, key=lambda item: natural_sort_key(item[0]))]

    return sorted(
        [p for p in img_dir.glob(f"img_*{ext}") if p.is_file()],
        key=natural_sort_key,
    )


def load_capture_angles(img_dir: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    info_path = find_capture_info_path(img_dir)
    if info_path is not None:
        try:
            with info_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = str(row.get("filename", "")).strip()
                    if not filename:
                        continue
                    value = (
                        row.get("angle_deg", "")
                        or row.get("camera_sweep_angle_deg", "")
                        or row.get("target_angle_deg", "")
                        or row.get("shared_zero_target_angle_deg", "")
                    )
                    try:
                        out[filename] = float(value)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass

    for p in img_dir.glob(f"img_*{IMAGE_EXT}"):
        if p.name not in out:
            angle = parse_angle_from_filename(p)
            if angle is not None:
                out[p.name] = angle
    return out


def preprocess(img: np.ndarray, use_clahe: bool = True, resize_scale: float = 1.0) -> np.ndarray:
    out = img
    if use_clahe:
        # Preserve color while enhancing local luminance contrast.
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if abs(float(resize_scale) - 1.0) > 1e-9:
        out = cv2.resize(
            out,
            None,
            fx=float(resize_scale),
            fy=float(resize_scale),
            interpolation=cv2.INTER_CUBIC,
        )
    return out


def crop_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    x = max(0, min(int(x), iw - 1))
    y = max(0, min(int(y), ih - 1))
    w = max(1, min(int(w), iw - x))
    h = max(1, min(int(h), ih - y))
    return img[y:y + h, x:x + w].copy()


def crop_centered_band(img: np.ndarray, x: int, center_y: float, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    w = max(1, min(int(w), iw))
    h = max(2, min(int(h), ih))
    x0 = int(np.clip(int(round(x)), 0, iw - w))
    y0 = int(round(float(center_y) - (h - 1) / 2.0))
    y0 = int(np.clip(y0, 0, ih - h))
    return img[y0:y0 + h, x0:x0 + w].copy()


def force_band_size(band: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if band.shape[:2] == (int(target_h), int(target_w)):
        return band.copy()
    return cv2.resize(band, (int(target_w), int(target_h)), interpolation=cv2.INTER_CUBIC)


def parse_roi_text(text: str) -> tuple[int, int, int, int]:
    parts = [int(v.strip()) for v in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return x, y, w, h


def save_roi(x: int, y: int, w: int, h: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"x": int(x), "y": int(y), "w": int(w), "h": int(h)}, indent=2),
        encoding="utf-8",
    )


def load_roi(path: Path) -> tuple[int, int, int, int]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        return int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])
    if isinstance(data, (list, tuple)) and len(data) >= 4:
        return tuple(int(v) for v in data[:4])
    raise ValueError(f"Invalid ROI JSON: {path}")


def draw_preview_center_guides(image: np.ndarray) -> None:
    h, w = image.shape[:2]
    cx = w // 2
    cy = h // 2
    green = (0, 255, 0)
    cv2.line(image, (cx, 0), (cx, h - 1), green, 1, cv2.LINE_AA)
    cv2.line(image, (0, cy), (w - 1, cy), green, 1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), 8, green, 1, cv2.LINE_AA)
    cv2.putText(
        image, "CENTER", (cx + 10, max(20, cy - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, green, 1, cv2.LINE_AA,
    )


# =============================================================================
# ROI SELECTION
# =============================================================================
def clamp_fixed_roi_position(
    x: int | float,
    y: int | float,
    image_width: int,
    image_height: int,
    roi_width: int,
    roi_height: int,
) -> tuple[int, int, int, int]:
    roi_width = int(roi_width)
    roi_height = int(roi_height)
    if image_width < roi_width or image_height < roi_height:
        raise RuntimeError(
            f"Source image {image_width}x{image_height} is smaller than ROI "
            f"{roi_width}x{roi_height}."
        )
    x = int(np.clip(int(round(x)), 0, image_width - roi_width))
    y = int(np.clip(int(round(y)), 0, image_height - roi_height))
    return x, y, roi_width, roi_height


def select_free_manual_roi(image: np.ndarray) -> tuple[int, int, int, int] | None:
    if SHOW_PREVIEW_CENTER_GUIDES:
        preview = image.copy()
        draw_preview_center_guides(preview)
    else:
        preview = image
    window = "RUN20 - select ROI, Enter/Space confirm, Esc cancel"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    h, w = image.shape[:2]
    scale = min(1.0, 1500 / max(w, 1), 900 / max(h, 1))
    cv2.resizeWindow(window, max(1, int(w * scale)), max(1, int(h * scale)))
    try:
        roi = cv2.selectROI(window, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
    x, y, rw, rh = [int(v) for v in roi]
    if rw <= 0 or rh <= 0:
        return None
    return x, y, rw, rh


def select_fixed_movable_roi(
    image: np.ndarray,
    initial_xy: tuple[int, int] | None = None,
    roi_width: int = FIXED_ROI_WIDTH_PX,
    roi_height: int = FIXED_ROI_HEIGHT_PX,
) -> tuple[int, int, int, int] | None:
    if image is None or image.size == 0:
        raise RuntimeError("Cannot select ROI from an empty image.")
    ih, iw = image.shape[:2]
    if initial_xy is None:
        start_x, start_y = (iw - roi_width) // 2, (ih - roi_height) // 2
    else:
        start_x, start_y = initial_xy
    x, y, _, _ = clamp_fixed_roi_position(start_x, start_y, iw, ih, roi_width, roi_height)
    state = {
        "x": x, "y": y, "dragging": False,
        "drag_offset_x": roi_width // 2, "drag_offset_y": roi_height // 2,
    }
    window_name = f"Move fixed ROI {roi_width}x{roi_height} - Enter/Space confirm, Esc cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    scale = min(1.0, 1500 / max(iw, 1), 900 / max(ih, 1))
    cv2.resizeWindow(window_name, max(1, int(iw * scale)), max(1, int(ih * scale)))

    def mouse_callback(event, mx, my, _flags, _param):
        mx = int(np.clip(mx, 0, iw - 1)); my = int(np.clip(my, 0, ih - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            inside = state["x"] <= mx < state["x"] + roi_width and state["y"] <= my < state["y"] + roi_height
            if inside:
                state["drag_offset_x"] = mx - state["x"]
                state["drag_offset_y"] = my - state["y"]
            else:
                state["drag_offset_x"] = roi_width // 2
                state["drag_offset_y"] = roi_height // 2
                nx, ny, _, _ = clamp_fixed_roi_position(
                    mx - state["drag_offset_x"], my - state["drag_offset_y"], iw, ih, roi_width, roi_height
                )
                state["x"], state["y"] = nx, ny
            state["dragging"] = True
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            nx, ny, _, _ = clamp_fixed_roi_position(
                mx - state["drag_offset_x"], my - state["drag_offset_y"], iw, ih, roi_width, roi_height
            )
            state["x"], state["y"] = nx, ny
        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging"] = False

    cv2.setMouseCallback(window_name, mouse_callback)
    confirmed = False
    try:
        while True:
            preview = image.copy()
            if SHOW_PREVIEW_CENTER_GUIDES:
                draw_preview_center_guides(preview)
            x0, y0 = int(state["x"]), int(state["y"])
            x1, y1 = x0 + roi_width - 1, y0 + roi_height - 1
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 0), 4)
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 255), 2)
            label = f"ROI {roi_width}x{roi_height} x={x0}, y={y0}  Drag | Enter/Space=OK | Esc=Cancel | R=Center"
            cv2.putText(preview, label, (10, max(24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, label, (10, max(24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window_name, preview)
            key = cv2.waitKeyEx(20)
            if key in (13, 10, 32):
                confirmed = True; break
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                state["x"], state["y"] = (iw - roi_width) // 2, (ih - roi_height) // 2
                continue
            if key in (2424832, 65361): state["x"] -= 1
            elif key in (2555904, 65363): state["x"] += 1
            elif key in (2490368, 65362): state["y"] -= 1
            elif key in (2621440, 65364): state["y"] += 1
            else: continue
            state["x"], state["y"], _, _ = clamp_fixed_roi_position(
                state["x"], state["y"], iw, ih, roi_width, roi_height
            )
    finally:
        cv2.destroyWindow(window_name); cv2.waitKey(1)
    if not confirmed:
        return None
    return int(state["x"]), int(state["y"]), int(roi_width), int(roi_height)


def auto_roi_json(img_dir: Path) -> Path:
    return img_dir.parent / "roi.json"


def choose_roi(args: argparse.Namespace, image_paths: list[Path]) -> tuple[int, int, int, int]:
    """Resolve the camera ROI.

    Default behavior in this version is a TRUE fixed ROI:
        (X, Y, W, H) = (1038, 1035, 339, 85)

    In fixed mode:
      - no selection window is opened;
      - previously saved roi.json files are ignored;
      - the ROI is not re-centered or clamped to a different location;
      - if the source image is incompatible with the verified coordinates,
        the program stops instead of silently changing the ROI.

    An explicit command-line --roi x,y,w,h can still be used deliberately for
    diagnostics. Normal experiment runs should not pass --roi.
    """
    save_roi_path = args.save_dir / "roi.json"
    idx = max(0, min(int(args.roi_image_index), len(image_paths) - 1))
    img = cv2.imread(str(image_paths[idx]))
    if img is None:
        raise RuntimeError(f"Could not read ROI image: {image_paths[idx]}")
    img = preprocess(img, use_clahe=not args.no_clahe, resize_scale=1.0)
    ih, iw = img.shape[:2]

    # Explicit CLI override is kept only for deliberate diagnostics.
    if args.roi is not None:
        x, y, w, h = [int(v) for v in args.roi]
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
            raise RuntimeError(
                f"Requested --roi {(x, y, w, h)} is outside source image "
                f"{iw}x{ih}."
            )
        roi = (x, y, w, h)
        save_roi(*roi, save_roi_path)
        print(
            "WARNING: command-line --roi override is active. "
            f"Using ROI {roi} instead of the locked RUN20 ROI."
        )
        return roi

    mode = str(args.roi_mode).strip().lower()

    # ------------------------------------------------------------------
    # TRUE LOCKED ROI: this is the normal/default path.
    # ------------------------------------------------------------------
    if mode == "fixed":
        x = int(FIXED_ROI_X_PX)
        y = int(FIXED_ROI_Y_PX)
        w = int(FIXED_ROI_WIDTH_PX)
        h = int(FIXED_ROI_HEIGHT_PX)

        # Never clamp a locked ROI, because clamping would silently change the
        # horizontal physical coordinate system and invalidate fusion calibration.
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
            raise RuntimeError(
                "Locked RUN20 ROI is incompatible with the current source image.\n"
                f"  source image: {iw} x {ih} px\n"
                f"  locked ROI : x={x}, y={y}, w={w}, h={h}\n"
                "The ROI was NOT moved or resized automatically."
            )

        roi = (x, y, w, h)
        save_roi(*roi, save_roi_path)
        print("=" * 72)
        print("LOCKED CAMERA ROI ACTIVE")
        print(f"  source image : {iw} x {ih} px")
        print(f"  ROI          : X={x}, Y={y}, W={w}, H={h}")
        print(f"  ROI center   : X={x + (w - 1) / 2.0:.1f}, Y={y + (h - 1) / 2.0:.1f}")
        print("  selection GUI: DISABLED")
        print("  saved roi.json is output only; it does not control this fixed mode")
        print("=" * 72)
        return roi

    # ------------------------------------------------------------------
    # The old modes are retained only for optional diagnostics.
    # They are NOT used in the normal fixed-ROI experiment.
    # ------------------------------------------------------------------
    auto_candidates = []
    if args.roi_json is not None:
        auto_candidates.append(args.roi_json)
    auto_candidates.extend([save_roi_path, auto_roi_json(args.img_dir)])

    if mode == "auto":
        for p in auto_candidates:
            if p is None or not Path(p).exists():
                continue
            try:
                x, y, w, h = load_roi(Path(p))
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > iw or y + h > ih:
                    continue
                roi = (x, y, w, h)
                save_roi(*roi, save_roi_path)
                print(f"AUTO ROI loaded: {roi}")
                return roi
            except Exception:
                continue
        print("AUTO ROI not found; switching to manual.")
        mode = "manual"

    roi = select_free_manual_roi(img)
    if roi is None:
        raise RuntimeError("Manual ROI selection cancelled.")
    save_roi(*roi, save_roi_path)
    return roi


def _resize_for_preview(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize only for screen display; never affects ROI/stitch coordinates."""
    h, w = image.shape[:2]
    scale = min(1.0, float(max_width) / max(w, 1), float(max_height) / max(h, 1))
    if scale >= 0.999999:
        return image.copy()
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _build_fixed_roi_preview_canvas(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    frame_index: int,
    frame_count: int,
    filename: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return annotated full-frame preview + enlarged exact ROI crop."""
    x, y, w, h = [int(v) for v in roi]
    full = image.copy()
    if SHOW_PREVIEW_CENTER_GUIDES:
        draw_preview_center_guides(full)

    # Thick black backing + yellow ROI border so it stays visible on any texture.
    cv2.rectangle(full, (x, y), (x + w - 1, y + h - 1), (0, 0, 0), 5, cv2.LINE_AA)
    cv2.rectangle(full, (x, y), (x + w - 1, y + h - 1), (0, 255, 255), 2, cv2.LINE_AA)

    roi_cx = x + (w - 1) / 2.0
    roi_cy = y + (h - 1) / 2.0
    label1 = f"LOCKED ROI  X={x}  Y={y}  W={w}  H={h}"
    label2 = f"frame {frame_index + 1}/{frame_count}: {filename}"
    label3 = f"ROI center=({roi_cx:.1f}, {roi_cy:.1f})"
    y_text = max(28, y - 54)
    for text, yy in ((label1, y_text), (label2, y_text + 24), (label3, y_text + 48)):
        cv2.putText(full, text, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(full, text, (12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)

    crop = crop_roi(image, x, y, w, h)
    zoom = max(1.0, float(ROI_PREVIEW_ZOOM_SCALE))
    zoomed = cv2.resize(
        crop,
        (max(1, int(round(w * zoom))), max(1, int(round(h * zoom)))),
        interpolation=cv2.INTER_NEAREST,
    )
    zh, zw = zoomed.shape[:2]
    zcx, zcy = zw // 2, zh // 2
    cv2.line(zoomed, (zcx, 0), (zcx, zh - 1), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(zoomed, (0, zcy), (zw - 1, zcy), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(
        zoomed,
        f"ROI ZOOM x{zoom:g}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        zoomed,
        f"ROI ZOOM x{zoom:g}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return full, zoomed


def preview_fixed_roi_before_stitch(
    args: argparse.Namespace,
    image_paths: list[Path],
    roi: tuple[int, int, int, int],
) -> None:
    """Pause before stitching so the locked ROI can be visually verified.

    Controls:
      Left / A / P   previous frame
      Right / D / N  next frame
      Home           first frame
      End            last frame
      Enter / Space  accept fixed ROI and start stitching
      Esc / Q        cancel this run

    The ROI cannot be dragged or resized here.  This preserves the fixed RUN20
    physical coordinate system while still allowing visual inspection.
    """
    if not PREVIEW_FIXED_ROI_BEFORE_STITCH:
        return
    if not image_paths:
        raise RuntimeError("No images available for fixed ROI preview.")

    idx = max(0, min(int(args.roi_image_index), len(image_paths) - 1))
    full_window = "RUN20 LOCKED ROI PREVIEW - Enter/Space=start, Esc/Q=cancel"
    zoom_window = "RUN20 LOCKED ROI ZOOM - Left/Right browse frames"
    cv2.namedWindow(full_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(zoom_window, cv2.WINDOW_NORMAL)

    confirmed = False
    confirmed_preview = None
    confirmed_path = None
    try:
        while True:
            path = image_paths[idx]
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"Could not read ROI preview image: {path}")
            img = preprocess(img, use_clahe=not args.no_clahe, resize_scale=1.0)

            ih, iw = img.shape[:2]
            x, y, w, h = [int(v) for v in roi]
            if x < 0 or y < 0 or x + w > iw or y + h > ih:
                raise RuntimeError(
                    f"Locked ROI {roi} is outside preview image {iw}x{ih}: {path}"
                )

            full_annotated, zoomed = _build_fixed_roi_preview_canvas(
                img, roi, idx, len(image_paths), path.name
            )
            full_show = _resize_for_preview(
                full_annotated,
                int(ROI_PREVIEW_MAX_FULL_WIDTH),
                int(ROI_PREVIEW_MAX_FULL_HEIGHT),
            )

            cv2.imshow(full_window, full_show)
            cv2.imshow(zoom_window, zoomed)
            # Keep the zoom window large enough to inspect individual wire ends.
            cv2.resizeWindow(zoom_window, zoomed.shape[1], zoomed.shape[0])

            key = cv2.waitKeyEx(0)
            if key in (13, 10, 32):  # Enter / Space
                confirmed = True
                confirmed_preview = full_annotated
                confirmed_path = path
                break
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (2424832, 65361, ord("a"), ord("A"), ord("p"), ord("P")):
                idx = max(0, idx - 1)
                continue
            if key in (2555904, 65363, ord("d"), ord("D"), ord("n"), ord("N")):
                idx = min(len(image_paths) - 1, idx + 1)
                continue
            if key in (2359296, 65360):  # Home
                idx = 0
                continue
            if key in (2293760, 65367):  # End
                idx = len(image_paths) - 1
                continue
    finally:
        cv2.destroyWindow(full_window)
        cv2.destroyWindow(zoom_window)
        cv2.waitKey(1)

    if not confirmed:
        raise RuntimeError("Locked ROI preview cancelled. Stitching was NOT started.")

    if SAVE_CONFIRMED_ROI_PREVIEW and confirmed_preview is not None:
        preview_path = args.save_dir / "locked_roi_preview_confirmed.png"
        cv2.imwrite(str(preview_path), confirmed_preview)
        print(f"Confirmed ROI preview saved: {preview_path}")
    print("Locked ROI visually confirmed. Starting stitch...")
    if confirmed_path is not None:
        print(f"  confirmed on frame: {confirmed_path.name}")


# =============================================================================
# REAL ENCODER HELPERS
# =============================================================================
def _parse_csv_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _image_index_from_name(path: Path) -> int | None:
    m = re.match(r"img_(\d+)", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def find_encoder_feedback_csv(img_dir: Path, requested: Path | None) -> Path | None:
    if requested is not None:
        p = requested.resolve()
        return p if p.exists() else None
    for p in (
        img_dir.parent / ENCODER_FEEDBACK_FILENAME,
        img_dir / ENCODER_FEEDBACK_FILENAME,
        img_dir.parent.parent / ENCODER_FEEDBACK_FILENAME,
    ):
        if p.exists():
            return p.resolve()
    return None


def load_real_encoder_angles(
    image_paths: list[Path], feedback_csv: Path, direction: int | None = None
) -> tuple[dict[str, float], list[dict]]:
    if not feedback_csv.exists():
        raise FileNotFoundError(feedback_csv)
    image_by_index = {idx: p for p in image_paths if (idx := _image_index_from_name(p)) is not None}
    if not image_by_index:
        raise RuntimeError("Could not parse image indices like img_0000_...")

    samples_by_move_index: dict[int, list[dict]] = {}
    with feedback_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_index", "arbitration_id", "decoded_ok", "position_deg"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError("Encoder feedback CSV missing columns: " + ", ".join(sorted(missing)))
        for csv_row_index, row in enumerate(reader):
            try:
                move_index = int(float(row["image_index"]))
                arbitration_id = int(float(row["arbitration_id"]))
            except (TypeError, ValueError):
                continue
            if not _parse_csv_bool(row.get("decoded_ok", "")) or arbitration_id != ENCODER_REAL_FEEDBACK_ARBITRATION_ID:
                continue
            try:
                position_deg = float(row["position_deg"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(position_deg):
                continue
            try:
                target_deg = float(row.get("target_position_deg", "nan"))
            except (TypeError, ValueError):
                target_deg = float("nan")
            sample = {
                "csv_row_index": csv_row_index,
                "move_image_index": move_index,
                "position_deg": position_deg,
                "target_position_deg": target_deg,
                "phase": row.get("phase", ""),
                "time_s": row.get("time_s", ""),
            }
            samples_by_move_index.setdefault(move_index, []).append(sample)

    if not samples_by_move_index:
        raise RuntimeError(f"No valid real encoder feedback in {feedback_csv}")

    angle_by_filename: dict[str, float] = {}
    selected_rows: list[dict] = []
    for move_index in sorted(samples_by_move_index):
        sample = samples_by_move_index[move_index][0]  # exact run20 policy: earliest valid sample
        image_index = int(move_index + ENCODER_FEEDBACK_INDEX_SHIFT)
        image_path = image_by_index.get(image_index)
        if image_path is None:
            continue
        pos = float(sample["position_deg"])
        angle_by_filename[image_path.name] = pos
        selected_rows.append({
            "image_index": image_index,
            "filename": image_path.name,
            "feedback_move_image_index": move_index,
            "encoder_position_deg_raw": pos,
            "feedback_phase": sample["phase"],
            "feedback_time_s": sample["time_s"],
            "feedback_csv_row_index": sample["csv_row_index"],
        })

    if len(angle_by_filename) < max(6, len(image_paths) // 2):
        raise RuntimeError(f"Too few images mapped from encoder feedback: {len(angle_by_filename)}/{len(image_paths)}")

    # Reject obviously impossible isolated encoder values, but never force 1 degree exactly.
    ordered = [(p, angle_by_filename[p.name]) for p in image_paths if p.name in angle_by_filename]
    raw = np.asarray([v for _, v in ordered], dtype=np.float64)
    unw = np.rad2deg(np.unwrap(np.deg2rad(raw)))
    diffs = np.diff(unw)
    finite = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-9)]
    sign = -1.0 if finite.size and np.median(finite) < 0 else 1.0
    travelled = sign * (unw - unw[0])
    step = np.diff(travelled)
    bad = np.where(
        (~np.isfinite(step))
        | (step <= 0)
        | (np.abs(step - ENCODER_EXPECTED_STEP_DEG) > ENCODER_MAX_SINGLE_STEP_DEVIATION_DEG)
    )[0]
    if bad.size:
        print(f"Warning: {bad.size} encoder step(s) outside nominal sanity window; kept because feedback is the global coordinate.")

    return angle_by_filename, selected_rows


def unwrap_capture_angles(image_paths: list[Path], angles: dict[str, float], fallback_step_deg: float) -> np.ndarray:
    values = []
    missing = False
    for i, path in enumerate(image_paths):
        angle = angles.get(path.name)
        if angle is None:
            missing = True; values.append(float(i) * float(fallback_step_deg))
        else:
            values.append(float(angle))
    if missing:
        print("Warning: some frame angles missing; fallback index angles used.")
    raw = np.asarray(values, dtype=np.float64)
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(raw)))
    travelled = unwrapped - unwrapped[0]
    diffs = np.diff(travelled)
    finite = diffs[np.isfinite(diffs) & (np.abs(diffs) > 1e-9)]
    if finite.size and np.median(finite) < 0:
        travelled = -travelled
    return travelled


def drop_duplicate_360(
    image_paths: list[Path], travelled_deg: np.ndarray, total_rotation_deg: float, enabled: bool
) -> tuple[list[Path], np.ndarray, list[Path]]:
    if not enabled or len(image_paths) < 2:
        return image_paths, travelled_deg, []
    keep, vals, dropped = [], [], []
    for p, angle in zip(image_paths, travelled_deg):
        if angle >= total_rotation_deg - 0.05 or angle < -0.05:
            dropped.append(p)
        else:
            keep.append(p); vals.append(float(angle))
    if not keep:
        return image_paths, travelled_deg, []
    return keep, np.asarray(vals, dtype=np.float64), dropped


def save_encoder_angle_table(
    path: Path,
    image_paths: list[Path],
    angles: dict[str, float],
    travelled: np.ndarray,
    selected_rows: list[dict],
) -> None:
    by_filename = {str(r.get("filename", "")): r for r in selected_rows}
    fields = [
        "image_index", "filename", "encoder_position_deg_raw", "travelled_deg",
        "feedback_move_image_index", "feedback_phase", "feedback_time_s", "feedback_csv_row_index",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (p, tr) in enumerate(zip(image_paths, travelled)):
            src = by_filename.get(p.name, {})
            writer.writerow({
                "image_index": _image_index_from_name(p) if _image_index_from_name(p) is not None else i,
                "filename": p.name,
                "encoder_position_deg_raw": angles.get(p.name, ""),
                "travelled_deg": f"{float(tr):.9f}",
                "feedback_move_image_index": src.get("feedback_move_image_index", ""),
                "feedback_phase": src.get("feedback_phase", ""),
                "feedback_time_s": src.get("feedback_time_s", ""),
                "feedback_csv_row_index": src.get("feedback_csv_row_index", ""),
            })


# =============================================================================
# CYLINDRICAL CORRECTION - INLINED PINJIE BEHAVIOR
# =============================================================================
def get_center_crop_for_matching(band: np.ndarray, center_ratio: float = 0.35) -> np.ndarray:
    h, w = band.shape[:2]
    use_w = max(8, int(round(w * center_ratio))); use_w = min(use_w, w)
    x1 = (w - use_w) // 2
    return band[:, x1:x1 + use_w]


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).reshape(-1); b = b.astype(np.float32).reshape(-1)
    a -= float(np.mean(a)); b -= float(np.mean(b))
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def estimate_vertical_shift(prev_band: np.ndarray, cur_band: np.ndarray, max_shift: int, center_ratio: float = 0.45):
    a = cv2.cvtColor(get_center_crop_for_matching(prev_band, center_ratio), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(get_center_crop_for_matching(cur_band, center_ratio), cv2.COLOR_BGR2GRAY)
    h = min(a.shape[0], b.shape[0]); a = a[:h]; b = b[:h]
    max_shift = max(1, min(int(max_shift), h - 12))
    best_shift, best_score = 0, -1e18
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            aa, bb = a[shift:, :], b[:h - shift, :]
        else:
            aa, bb = a[:h + shift, :], b[-shift:, :]
        if aa.shape[0] < 12: continue
        score = zncc(aa, bb)
        if score > best_score:
            best_shift, best_score = shift, score
    return best_shift, best_score


def estimate_cylinder_radius_from_sequence(
    image_paths: list[Path], angles: dict[str, float], roi: tuple[int, int, int, int],
    use_clahe: bool = True, resize_scale: float = 1.0, center_offset_y: float = 0.0,
):
    if len(image_paths) < 2:
        return None, []
    x, y, w, h = roi
    center_y = y + (h - 1) / 2.0 + float(center_offset_y)
    calibration_h = max(int(CURVATURE_CALIBRATION_HEIGHT), h * 2, 32)
    pair_count = min(int(CURVATURE_CALIBRATION_PAIRS), len(image_paths) - 1)
    pair_indices = np.unique(np.linspace(1, len(image_paths) - 1, pair_count, dtype=np.int32))
    records = []
    for i in pair_indices:
        pp, cp = image_paths[int(i) - 1], image_paths[int(i)]
        pa, ca = angles.get(pp.name), angles.get(cp.name)
        if pa is None or ca is None: continue
        step_rad = np.deg2rad(float(ca) - float(pa))
        if abs(step_rad) < 1e-8: continue
        prev = cv2.imread(str(pp)); cur = cv2.imread(str(cp))
        if prev is None or cur is None: continue
        prev = preprocess(prev, use_clahe, resize_scale); cur = preprocess(cur, use_clahe, resize_scale)
        prev_band = crop_centered_band(prev, x, center_y, w, calibration_h)
        cur_band = crop_centered_band(cur, x, center_y, w, calibration_h)
        uh, uw = min(prev_band.shape[0], cur_band.shape[0]), min(prev_band.shape[1], cur_band.shape[1])
        prev_band, cur_band = prev_band[:uh, :uw], cur_band[:uh, :uw]
        shift, score = estimate_vertical_shift(prev_band, cur_band, max(2, int(round(uh * CURVATURE_MAX_SHIFT_RATIO))), MATCH_CENTER_RATIO)
        radius = abs(float(shift) / step_rad)
        records.append({"pair_index": int(i), "shift_px": int(shift), "score": float(score), "step_deg": float(np.rad2deg(step_rad)), "radius_px": float(radius)})
    valid = [r["radius_px"] for r in records if r["score"] >= CURVATURE_MIN_SCORE and abs(r["shift_px"]) >= 1]
    if not valid:
        return None, records
    center = float(median(valid)); mad = float(median([abs(v - center) for v in valid]))
    if mad > 0:
        valid = [v for v in valid if abs(v - center) <= 3.5 * mad]
    radius = float(median(valid))
    minimum_radius = max(h / 2.0 + 1.0, 10.0)
    if radius < minimum_radius:
        return None, records
    return radius, records


def cylindrical_rectify_band(band: np.ndarray, radius_px: float, center_offset_y: float = 0.0) -> np.ndarray:
    if radius_px is None or radius_px <= 0:
        return band.copy()
    h, w = band.shape[:2]
    if h < 2: return band.copy()
    center_y = (h - 1) / 2.0 + float(center_offset_y)
    top_dy, bottom_dy = -center_y, (h - 1) - center_y
    if max(abs(top_dy), abs(bottom_dy)) >= radius_px:
        raise ValueError(f"ROI reaches outside projected cylinder: half-height={max(abs(top_dy), abs(bottom_dy)):.2f}, radius={radius_px:.2f}")
    theta_top, theta_bottom = np.arcsin(top_dy / radius_px), np.arcsin(bottom_dy / radius_px)
    arc_top, arc_bottom = radius_px * theta_top, radius_px * theta_bottom
    output_h = max(2, int(round(abs(arc_bottom - arc_top))) + 1)
    surface_s = np.linspace(arc_top, arc_bottom, output_h, dtype=np.float32)
    source_y = center_y + radius_px * np.sin(surface_s / radius_px)
    map_x = np.tile(np.arange(w, dtype=np.float32), (output_h, 1))
    map_y = np.tile(source_y.reshape(output_h, 1), (1, w)).astype(np.float32)
    return cv2.remap(band, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def estimate_radius_if_needed(args, image_paths, visual_angles, roi):
    if not args.use_curvature_correction:
        return None, []
    if args.cylinder_radius_px and args.cylinder_radius_px > 0:
        return float(args.cylinder_radius_px), []
    return estimate_cylinder_radius_from_sequence(
        image_paths, visual_angles, roi,
        use_clahe=not args.no_clahe,
        resize_scale=1.0,
        center_offset_y=args.cylinder_center_offset_y,
    )


def write_radius_records(path: Path, records: list[dict]) -> None:
    if not records: return
    keys = ["pair_index", "shift_px", "score", "step_deg", "radius_px"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(records)


# =============================================================================
# EXACT RUN20 PINJIE-STYLE OVERLAP HELPERS
# =============================================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def estimate_overlap_rows_global(prev_band, cur_band, min_overlap, max_overlap, center_ratio):
    prev_match = get_center_crop_for_matching(prev_band, center_ratio)
    cur_match = get_center_crop_for_matching(cur_band, center_ratio)
    prev_gray = cv2.cvtColor(prev_match, cv2.COLOR_BGR2GRAY)
    cur_gray = cv2.cvtColor(cur_match, cv2.COLOR_BGR2GRAY)
    h = min(prev_gray.shape[0], cur_gray.shape[0])
    min_overlap = max(2, min(min_overlap, h - 1))
    max_overlap = max(min_overlap + 1, min(max_overlap, h - 1))
    best_overlap, best_score = min_overlap, -1e18
    for o in range(min_overlap, max_overlap + 1):
        score = zncc(prev_gray[-o:, :], cur_gray[:o, :])
        if score > best_score:
            best_overlap, best_score = o, score
    return best_overlap, best_score


def median_filter_1d(values, window=5):
    if window < 1: return values[:]
    if window % 2 == 0: window += 1
    half = window // 2; out = []
    for i in range(len(values)):
        out.append(int(round(median(values[max(0, i-half):min(len(values), i+half+1)]))))
    return out


def compute_global_default(raw_overlaps, scores, min_o, max_o, trust_threshold):
    good = [o for o, s in zip(raw_overlaps, scores) if s >= trust_threshold and o not in (min_o, max_o)]
    return int(round(median(good if good else raw_overlaps)))


def local_repair_value(raw_overlaps, scores, idx, min_o, max_o, global_default, trust_threshold):
    vals = []
    for j in range(max(0, idx - LOCAL_REPAIR_RADIUS), min(len(raw_overlaps), idx + LOCAL_REPAIR_RADIUS + 1)):
        if scores[j] >= trust_threshold and raw_overlaps[j] not in (min_o, max_o):
            vals.append(raw_overlaps[j])
    return int(round(median(vals))) if vals else global_default


def refine_overlap_sequence(raw_overlaps, scores, min_o, max_o, trust_threshold):
    global_default = compute_global_default(raw_overlaps, scores, min_o, max_o, trust_threshold)
    stage1 = []
    for i, (o, s) in enumerate(zip(raw_overlaps, scores)):
        hit_boundary, low_conf = (o == min_o or o == max_o), s < trust_threshold
        if BOUNDARY_PENALTY and (hit_boundary or low_conf):
            stage1.append(local_repair_value(raw_overlaps, scores, i, min_o, max_o, global_default, trust_threshold))
        else:
            stage1.append(o)
    stage2 = [clamp(int(v), min_o, max_o) for v in median_filter_1d(stage1, MEDIAN_FILTER_WINDOW)]
    return stage1, stage2, global_default


def blend_vertical_overlap(tail_a: np.ndarray, head_b: np.ndarray) -> np.ndarray:
    o = tail_a.shape[0]
    if o <= 1: return head_b.copy()
    a, b = tail_a.astype(np.float32), head_b.astype(np.float32)
    alpha = np.linspace(1.0, 0.0, o, dtype=np.float32).reshape(o, 1, 1)
    return np.clip(a * alpha + b * (1.0 - alpha), 0, 255).astype(np.uint8)


def match_color_to_previous_overlap(cur_band, prev_tail, overlap_rows):
    overlap_rows = min(overlap_rows, prev_tail.shape[0], cur_band.shape[0])
    if overlap_rows < 4: return cur_band
    ref = prev_tail[-overlap_rows:, :].astype(np.float32)
    cur = cur_band[:overlap_rows, :].astype(np.float32)
    out = cur_band.astype(np.float32)
    for c in range(cur_band.shape[2]):
        ref_mean, cur_mean = float(np.mean(ref[:, :, c])), float(np.mean(cur[:, :, c]))
        ref_std, cur_std = float(np.std(ref[:, :, c])), float(np.std(cur[:, :, c]))
        gain = float(np.clip(ref_std / max(cur_std, 1e-6), 0.85, 1.18))
        offset = float(np.clip(ref_mean - cur_mean * gain, -25.0, 25.0))
        out[:, :, c] = out[:, :, c] * gain + offset
    return np.clip(out, 0, 255).astype(np.uint8)


def highpass_gray_for_alignment(band):
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray - cv2.GaussianBlur(gray, (0, 0), 3.0)


def overlap_alignment_score(prev_overlap, cur_overlap, shift_x):
    if prev_overlap.shape[0] < 8 or cur_overlap.shape[0] < 8: return -1e18
    a, b = highpass_gray_for_alignment(prev_overlap), highpass_gray_for_alignment(cur_overlap)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1]); a, b = a[:h, :w], b[:h, :w]
    shift_x = int(shift_x)
    if shift_x > 0:
        if shift_x >= w - 8: return -1e18
        aa, bb = a[:, shift_x:], b[:, :w-shift_x]
    elif shift_x < 0:
        sx = -shift_x
        if sx >= w - 8: return -1e18
        aa, bb = a[:, :w-sx], b[:, sx:]
    else:
        aa, bb = a, b
    return zncc(aa, bb)


def estimate_local_overlap_and_shift(prev_result, cur_band, overlap_rows):
    base = int(overlap_rows); h_prev, h_cur = prev_result.shape[0], cur_band.shape[0]
    min_o = max(8, base - LOCAL_ALIGN_MAX_SHIFT_OVERLAP_ROWS)
    max_o = min(h_prev, h_cur - 1, base + LOCAL_ALIGN_MAX_SHIFT_OVERLAP_ROWS)
    if min_o > max_o:
        return max(1, min(base, h_prev, h_cur - 1)), 0, -1e18
    best_o, best_dx, best_score = base, 0, -1e18
    for o in range(min_o, max_o + 1):
        po, co = prev_result[-o:, :], cur_band[:o, :]
        for dx in range(-LOCAL_ALIGN_MAX_SHIFT_X, LOCAL_ALIGN_MAX_SHIFT_X + 1):
            score = overlap_alignment_score(po, co, dx)
            if score > best_score:
                best_o, best_dx, best_score = o, dx, score
    return max(1, min(int(best_o), h_prev, h_cur - 1)), int(best_dx), float(best_score)


def shift_band_x_replicate(band, shift_x):
    shift_x = int(round(shift_x))
    if shift_x == 0: return band.copy()
    h, w = band.shape[:2]
    m = np.float32([[1, 0, shift_x], [0, 1, 0]])
    return cv2.warpAffine(band, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def stitch_like_pinjie(bands: list[np.ndarray], args):
    if not bands: raise RuntimeError("No bands generated.")
    if len(bands) == 1:
        h0 = bands[0].shape[0]
        return bands[0].copy(), [(0, h0)], [], {
            "global_min_overlap": "", "global_max_overlap": "", "global_default_overlap": "",
            "natural_frame_tops": np.asarray([0.0]),
            "natural_frame_centers": np.asarray([0.5 * (h0 - 1.0)]),
            "actual_overlaps": np.asarray([], dtype=np.int32), "actual_dx": np.asarray([], dtype=np.int32),
            "actual_align_scores": np.asarray([], dtype=np.float64),
        }

    h_band = int(bands[0].shape[0])
    global_min_overlap = max(2, int(round(h_band * GLOBAL_MIN_OVERLAP_RATIO)))
    global_max_overlap = min(h_band - 1, int(round(h_band * GLOBAL_MAX_OVERLAP_RATIO)))
    raw_overlaps, scores = [], []
    for i in range(1, len(bands)):
        o, score = estimate_overlap_rows_global(bands[i-1], bands[i], global_min_overlap, global_max_overlap, MATCH_CENTER_RATIO)
        raw_overlaps.append(int(o)); scores.append(float(score))
    repaired, refined, global_default = refine_overlap_sequence(raw_overlaps, scores, global_min_overlap, global_max_overlap, SCORE_TRUST_THRESHOLD)

    if PINJIE_STYLE_COMPOSE_MODE != "aligned-blend":
        raise RuntimeError("RUN20 standalone intentionally supports the verified aligned-blend compose mode only.")

    result = bands[0].copy()
    natural_tops = np.zeros(len(bands), dtype=np.float64)
    natural_centers = np.zeros(len(bands), dtype=np.float64)
    natural_centers[0] = 0.5 * (bands[0].shape[0] - 1.0)
    actual_overlaps, actual_dx, actual_scores = [], [], []
    source_ranges = [(0, int(bands[0].shape[0]))]

    for i in range(1, len(bands)):
        cur_band = bands[i]
        requested = int(refined[i-1])
        requested = max(1, min(requested, result.shape[0], cur_band.shape[0] - 1))
        aligned_overlap, dx, align_score = estimate_local_overlap_and_shift(result, cur_band, requested)
        aligned_overlap = int(np.clip(aligned_overlap, 1, min(result.shape[0], cur_band.shape[0] - 1)))
        if dx != 0: cur_band = shift_band_x_replicate(cur_band, dx)
        if bool(args.match_color):
            cur_band = match_color_to_previous_overlap(cur_band, result[-aligned_overlap:, :], aligned_overlap)
        frame_top = float(result.shape[0] - aligned_overlap)
        frame_center = frame_top + 0.5 * (cur_band.shape[0] - 1.0)
        composed = blend_vertical_overlap(result[-aligned_overlap:, :], cur_band[:aligned_overlap, :])
        old_rows = int(result.shape[0])
        result = np.vstack([result[:-aligned_overlap, :], composed, cur_band[aligned_overlap:, :]])
        natural_tops[i], natural_centers[i] = frame_top, frame_center
        actual_overlaps.append(aligned_overlap); actual_dx.append(int(dx)); actual_scores.append(float(align_score))
        source_ranges.append((old_rows, int(result.shape[0])))
        if i <= 5 or i % 25 == 0:
            print(f"aligned stitch band_{i}: requested_overlap={requested}, actual_overlap={aligned_overlap}, dx={dx}, score={align_score:.6f}, natural_center={frame_center:.3f}")

    if float(np.max(natural_centers)) > float(result.shape[0] - 1) + 1e-6:
        raise RuntimeError("Internal overlap geometry error: frame center exceeds natural image.")

    records = []
    for i, (raw_o, score, rep_o, ref_o) in enumerate(zip(raw_overlaps, scores, repaired, refined), start=1):
        records.append({
            "pair_index": i - 1,
            "raw_overlap": int(raw_o), "score": float(score), "repaired_overlap": int(rep_o), "refined_overlap": int(ref_o),
            "actual_aligned_overlap": int(actual_overlaps[i-1]), "actual_local_dx": int(actual_dx[i-1]),
            "actual_local_align_score": float(actual_scores[i-1]), "natural_frame_top": float(natural_tops[i]),
            "natural_frame_center": float(natural_centers[i]), "search_min": int(global_min_overlap), "search_max": int(global_max_overlap),
        })
    info = {
        "global_min_overlap": int(global_min_overlap), "global_max_overlap": int(global_max_overlap), "global_default_overlap": int(global_default),
        "natural_frame_tops": natural_tops, "natural_frame_centers": natural_centers,
        "actual_overlaps": np.asarray(actual_overlaps, dtype=np.int32), "actual_dx": np.asarray(actual_dx, dtype=np.int32),
        "actual_align_scores": np.asarray(actual_scores, dtype=np.float64),
    }
    return result, source_ranges, records, info


# =============================================================================
# DIAGNOSTIC AUTO REGISTRATION - DOES NOT MOVE GLOBAL ENCODER CENTERS IN RUN20
# =============================================================================
def _pair_registration_measure(prev_band, cur_band, expected_step: float, args):
    h = min(prev_band.shape[0], cur_band.shape[0])
    w = min(prev_band.shape[1], cur_band.shape[1])
    x_margin = min(max(0, REGISTRATION_X_MARGIN_PX), max(0, w // 3))
    prev = prev_band[:, x_margin:w-x_margin if x_margin else w]
    cur = cur_band[:, x_margin:w-x_margin if x_margin else w]
    best = None; second = -1e18
    for step in range(int(round(expected_step)) - args.registration_search_rows, int(round(expected_step)) + args.registration_search_rows + 1):
        if step <= 0 or step >= h - args.registration_min_overlap_rows: continue
        overlap = h - step
        po, co = prev[-overlap:, :], cur[:overlap, :]
        for dx in range(-args.registration_search_cols, args.registration_search_cols + 1):
            score = overlap_alignment_score(po, co, dx)
            item = (float(score), int(step), int(dx), int(overlap))
            if best is None or score > best[0]:
                if best is not None: second = max(second, best[0])
                best = item
            else:
                second = max(second, score)
    if best is None:
        return None
    score, step, dx, overlap = best
    return {
        "measured_source_step_px": float(step), "measured_dx_px": float(dx), "registration_score": float(score),
        "registration_peak_margin": float(score - second) if np.isfinite(second) else 0.0,
        "source_overlap_rows": int(overlap),
        "hit_search_boundary": bool(abs(step - expected_step) >= args.registration_search_rows - REGISTRATION_BOUNDARY_MARGIN_PX),
    }


def robust_registered_centers(bands, travelled, output_rows_per_deg, args, lock_global_centers_to_encoder=True):
    theory_total = int(round(float(args.total_rotation_deg) * float(output_rows_per_deg)))
    encoder_centers = np.asarray(travelled, dtype=np.float64) * float(output_rows_per_deg)
    records = []
    preliminary_rates = []
    if args.auto_register:
        # First rough source scale comes from median natural source step implied by overlap geometry.
        med_angle = np.median(np.diff(travelled)) if len(travelled) > 1 else 1.0
        rough_source_rate = max(1.0, (bands[0].shape[0] * (1.0 - 0.5*(GLOBAL_MIN_OVERLAP_RATIO+GLOBAL_MAX_OVERLAP_RATIO))) / max(med_angle, 1e-6))
        for i in range(1, len(bands)):
            dtheta = float(travelled[i] - travelled[i-1])
            expected = dtheta * rough_source_rate
            m = _pair_registration_measure(bands[i-1], bands[i], expected, args)
            if m is None:
                m = {"measured_source_step_px": float(expected), "measured_dx_px": 0.0, "registration_score": -1e18, "registration_peak_margin": 0.0, "source_overlap_rows": 0, "hit_search_boundary": True}
            rate = float(m["measured_source_step_px"] / max(dtheta, 1e-9))
            preliminary = (
                m["registration_score"] >= args.registration_min_score
                and not m["hit_search_boundary"]
                and abs(m["measured_source_step_px"] - expected) <= REGISTRATION_MAX_INITIAL_STEP_ERROR_PX
            )
            if preliminary and np.isfinite(rate): preliminary_rates.append(rate)
            records.append({
                "pair_index": i-1, "delta_angle_deg": dtheta,
                "initial_output_scale_prediction_px": float(dtheta * output_rows_per_deg),
                **m, "initial_step_error_px": float(m["measured_source_step_px"] - expected),
                "measured_source_rows_per_deg": rate, "preliminary_valid": bool(preliminary),
            })

    if preliminary_rates:
        rate_median = float(np.median(preliminary_rates))
        rate_mad = float(np.median(np.abs(np.asarray(preliminary_rates) - rate_median)))
        rate_tol = max(1e-9, 3.5 * rate_mad)
    else:
        # Safe diagnostic fallback. Final encoder placement is still exact.
        rate_median = float(output_rows_per_deg)
        rate_mad = 0.0; rate_tol = 0.0
    source_rows_per_deg = rate_median

    for i, rec in enumerate(records, start=1):
        rate = float(rec["measured_source_rows_per_deg"])
        final_valid = bool(rec["preliminary_valid"] and (rate_mad <= 0 or abs(rate-rate_median) <= rate_tol))
        expected_source_step = float(rec["delta_angle_deg"] * source_rows_per_deg)
        rec["final_valid"] = final_valid
        rec["expected_source_step_px"] = expected_source_step
        rec["source_step_error_px"] = float(rec["measured_source_step_px"] - expected_source_step)
        rec["used_output_step_px"] = float(encoder_centers[i] - encoder_centers[i-1])
        rec["center_correction_px"] = 0.0
        rec["global_center_locked_to_encoder"] = bool(lock_global_centers_to_encoder)
        rec["global_source_rows_per_deg"] = source_rows_per_deg
        rec["robust_rate_median"] = rate_median; rec["robust_rate_mad"] = rate_mad; rec["robust_rate_tolerance"] = rate_tol

    return encoder_centers, theory_total, float(output_rows_per_deg), float(source_rows_per_deg), records


def save_seam_registration(path: Path, records: list[dict], image_paths: list[Path]) -> None:
    if not records:
        return
    rows = []
    for i, rec in enumerate(records):
        r = dict(rec)
        r["from_filename"] = image_paths[i].name
        r["to_filename"] = image_paths[i+1].name
        rows.append(r)
    # Match run20's useful column order.
    fields = [
        "pair_index", "from_filename", "to_filename", "delta_angle_deg", "initial_output_scale_prediction_px",
        "measured_source_step_px", "measured_dx_px", "registration_score", "registration_peak_margin", "source_overlap_rows",
        "hit_search_boundary", "initial_step_error_px", "measured_source_rows_per_deg", "preliminary_valid", "final_valid",
        "expected_source_step_px", "source_step_error_px", "used_output_step_px", "center_correction_px",
        "global_center_locked_to_encoder", "global_source_rows_per_deg", "robust_rate_median", "robust_rate_mad", "robust_rate_tolerance",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


# =============================================================================
# RUN20 STAGE-B PIECEWISE REAL-ENCODER Y WARP
# =============================================================================
def encoder_midpoint_output_ranges(encoder_centers: np.ndarray, total_rows: int):
    centers = np.asarray(encoder_centers, dtype=np.float64); n = centers.size
    if n == 0: return []
    seam_after = np.empty(n, dtype=np.float64)
    for i in range(n):
        left = float(centers[i]); right = float(centers[i+1]) if i+1 < n else float(centers[0]) + float(total_rows)
        if right <= left: right += total_rows
        seam_after[i] = 0.5 * (left + right)
    ranges = []
    for i in range(n):
        start = seam_after[i-1] if i > 0 else seam_after[-1] - total_rows
        ranges.append((int(math.ceil(start)), int(math.ceil(seam_after[i]))))
    return ranges


def warp_overlap_result_to_encoder_grid(
    natural_image: np.ndarray,
    natural_centers: np.ndarray,
    travelled_deg: np.ndarray,
    *, output_rows_per_deg: float, total_rotation_deg: float, total_rows: int,
):
    src_anchor = np.asarray(natural_centers, dtype=np.float64)
    angle_anchor = np.asarray(travelled_deg, dtype=np.float64)
    if src_anchor.size != angle_anchor.size:
        raise RuntimeError("natural_centers and encoder travelled angles have different lengths.")
    if src_anchor.size < 2:
        raise RuntimeError("At least 2 encoder/natural anchors are required.")
    if np.any(np.diff(src_anchor) <= 0) or np.any(np.diff(angle_anchor) <= 0):
        raise RuntimeError("Natural and encoder anchors must be strictly increasing.")

    dst_anchor = angle_anchor * float(output_rows_per_deg)
    source_rate_samples = np.diff(src_anchor) / np.maximum(np.diff(angle_anchor), 1e-9)
    natural_rows_per_deg = float(np.median(source_rate_samples[np.isfinite(source_rate_samples)]))
    target_end = float(total_rows)
    remaining_deg = float(total_rotation_deg - angle_anchor[-1])
    source_end = float(src_anchor[-1] + max(0.0, remaining_deg) * natural_rows_per_deg)
    max_source_y = float(natural_image.shape[0] - 1)
    source_end_was_clamped = bool(source_end > max_source_y)
    source_end = float(np.clip(source_end, src_anchor[-1] + 1e-6, max_source_y))

    dst_full = np.concatenate(([0.0], dst_anchor[1:], [target_end]))
    src_full = np.concatenate(([src_anchor[0]], src_anchor[1:], [source_end]))
    unique = np.ones(dst_full.size, dtype=bool); unique[1:] = np.diff(dst_full) > 1e-9
    dst_full, src_full = dst_full[unique], src_full[unique]
    target_y = np.arange(int(total_rows), dtype=np.float64)
    source_y = np.interp(target_y, dst_full, src_full)
    source_y = np.clip(source_y, 0.0, max_source_y)

    h_out, width, channels = int(total_rows), int(natural_image.shape[1]), int(natural_image.shape[2])
    out = np.empty((h_out, width, channels), dtype=np.uint8)
    chunk_rows = max(32, int(ENCODER_OVERLAP_WARP_CHUNK_ROWS))
    for y0 in range(0, h_out, chunk_rows):
        y1 = min(h_out, y0 + chunk_rows); sy = source_y[y0:y1]
        sy0 = np.clip(np.floor(sy).astype(np.int32), 0, natural_image.shape[0]-1)
        sy1 = np.clip(sy0 + 1, 0, natural_image.shape[0]-1)
        alpha = np.clip((sy - sy0).astype(np.float32), 0, 1).reshape(-1, 1, 1)
        a = natural_image[sy0].astype(np.float32); b = natural_image[sy1].astype(np.float32)
        out[y0:y1] = np.clip(a*(1-alpha)+b*alpha, 0, 255).astype(np.uint8)

    natural_at_encoder = np.interp(dst_anchor, dst_full, src_full)
    error = natural_at_encoder - src_anchor
    info = {
        "warp_mode": "overlap_visual_then_piecewise_real_encoder_y_warp",
        "natural_rows_per_deg": natural_rows_per_deg,
        "source_start_anchor": float(src_anchor[0]), "source_end_anchor": float(source_end),
        "source_end_was_clamped": source_end_was_clamped,
        "encoder_anchor_count": int(src_anchor.size),
        "max_abs_anchor_source_error_px": float(np.max(np.abs(error))) if error.size else 0.0,
        "natural_input_rows": int(natural_image.shape[0]), "metric_output_rows": int(total_rows),
    }
    return out, info


# =============================================================================
# OUTPUT HELPERS
# =============================================================================
def save_pinjie_style_overlap(path, records, image_paths, travelled, output_rows_per_deg):
    if not records: return
    rows = []
    for i, rec in enumerate(records):
        r = dict(rec)
        r["from_filename"] = image_paths[i].name; r["to_filename"] = image_paths[i+1].name
        r["from_travelled_deg"] = float(travelled[i]); r["to_travelled_deg"] = float(travelled[i+1])
        dtheta = float(travelled[i+1]-travelled[i])
        r["encoder_delta_deg"] = dtheta; r["encoder_delta_output_px"] = dtheta * float(output_rows_per_deg)
        r["natural_overlap_center_step_px"] = float(rec["natural_frame_center"] - (0.5*(image_paths and 0 or 0))) if False else ""
        r["hybrid_global_position_source"] = "real_encoder"
        r["hybrid_local_seam_source"] = "overlap"
        rows.append(r)
    # Fill exact center steps from sequential natural centers.
    centers = [0.5]
    for rec in records: centers.append(float(rec["natural_frame_center"]))
    # first actual center value isn't inferable from records alone here; use pair steps from tops/overlap approximately.
    for i, r in enumerate(rows):
        if i == 0:
            r["natural_overlap_center_step_px"] = float(records[0]["natural_frame_top"])
        else:
            r["natural_overlap_center_step_px"] = float(records[i]["natural_frame_center"] - records[i-1]["natural_frame_center"])
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)


def find_laser_reference_rows() -> tuple[Path | None, int | None]:
    root = Path(LASER_RESTORED_DIR)
    for name in LASER_HEIGHT_REFERENCE_NAMES:
        p = root / name
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None: return p, int(img.shape[0])
    return None, None


def copy_alias(src: Path, dst: Path) -> None:
    try:
        if src.resolve() != dst.resolve(): shutil.copy2(src, dst)
    except Exception:
        pass


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RUN20 standalone: locked ROI + pinjie aligned-blend visual stitch + real-encoder piecewise Y warp")
    parser.add_argument("--img-dir", type=Path, default=Path(INPUT_IMG_DIR))
    parser.add_argument("--save-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--roi-json", type=Path, default=Path(ROI_JSON_PATH) if ROI_JSON_PATH else None)
    parser.add_argument("--roi", type=parse_roi_text, default=None)
    parser.add_argument("--roi-mode", choices=("manual", "fixed", "auto"), default=ROI_SELECTION_MODE)
    parser.add_argument("--roi-image-index", type=int, default=ROI_SELECT_IMAGE_INDEX)
    parser.add_argument("--wheel-diameter-mm", type=float, default=WHEEL_DIAMETER_MM)
    parser.add_argument("--surface-mm-per-pixel", type=float, default=STITCH_SURFACE_MM_PER_PIXEL)
    parser.add_argument("--fallback-angle-step-deg", type=float, default=FALLBACK_ANGLE_STEP_DEG)
    parser.add_argument("--total-rotation-deg", type=float, default=TOTAL_ROTATION_DEG)
    parser.add_argument("--keep-360-duplicate", dest="drop_360_duplicate", action="store_false", default=DROP_REAL_360_DUPLICATE)
    parser.add_argument("--no-real-encoder", dest="use_real_encoder", action="store_false", default=USE_REAL_ENCODER_FEEDBACK)
    parser.add_argument("--encoder-feedback-csv", type=Path, default=None)
    parser.add_argument("--no-clahe", action="store_true", default=not USE_CLAHE)
    parser.add_argument("--no-curvature-correction", dest="use_curvature_correction", action="store_false", default=USE_CYLINDRICAL_CORRECTION)
    parser.add_argument("--cylinder-radius-px", type=float, default=CYLINDER_RADIUS_PX)
    parser.add_argument("--cylinder-center-offset-y", type=float, default=CYLINDER_CENTER_OFFSET_Y_PX)
    parser.add_argument("--match-color", action="store_true", default=MATCH_COLOR_TO_PREVIOUS)
    parser.add_argument("--auto-register", dest="auto_register", action="store_true", default=AUTO_SEAM_REGISTRATION)
    parser.add_argument("--no-auto-register", dest="auto_register", action="store_false")
    parser.add_argument("--registration-search-rows", type=int, default=REGISTRATION_SEARCH_ROWS)
    parser.add_argument("--registration-search-cols", type=int, default=REGISTRATION_SEARCH_COLS)
    parser.add_argument("--registration-min-overlap-rows", type=int, default=REGISTRATION_MIN_OVERLAP_ROWS)
    parser.add_argument("--registration-min-score", type=float, default=REGISTRATION_MIN_SCORE)
    parser.add_argument("--seam-feather-rows", type=int, default=SEAM_FEATHER_ROWS)
    return parser.parse_args()


# =============================================================================
# MAIN RUN20 PIPELINE
# =============================================================================
def run_angle_stitch(args: argparse.Namespace) -> None:
    img_dir = args.img_dir.resolve(); save_dir = args.save_dir.resolve(); save_dir.mkdir(parents=True, exist_ok=True)
    image_paths = load_images(img_dir, IMAGE_EXT)
    if not image_paths: raise FileNotFoundError(f"No input images found in {img_dir}")

    encoder_feedback_path = None; encoder_source_rows = []
    if args.use_real_encoder:
        encoder_feedback_path = find_encoder_feedback_csv(img_dir, args.encoder_feedback_csv)
    if args.use_real_encoder and encoder_feedback_path is not None:
        print(f"Using REAL encoder feedback: {encoder_feedback_path}")
        angles, encoder_source_rows = load_real_encoder_angles(image_paths, encoder_feedback_path)
    else:
        if args.use_real_encoder:
            print("Warning: camera_motor_feedback.csv not found; falling back to capture angles.")
        angles = load_capture_angles(img_dir)

    # Curvature estimation must follow the original visual/capture angle convention when available.
    capture_angles = load_capture_angles(img_dir)
    visual_angles = capture_angles if sum(p.name in capture_angles for p in image_paths) >= max(6, len(image_paths)//2) else angles

    travelled = unwrap_capture_angles(image_paths, angles, args.fallback_angle_step_deg)
    image_paths, travelled, dropped = drop_duplicate_360(image_paths, travelled, args.total_rotation_deg, args.drop_360_duplicate)
    if dropped: print(f"Dropped {len(dropped)} frame(s) at/after 360 deg.")

    if args.use_real_encoder and encoder_feedback_path is not None and SAVE_ENCODER_ANGLE_TABLE:
        save_encoder_angle_table(save_dir / "encoder_angle_mapping.csv", image_paths, angles, travelled, encoder_source_rows)

    roi = choose_roi(args, image_paths); x, y, w, h = roi
    print(f"ROI: x={x}, y={y}, w={w}, h={h}")

    # Fixed ROI stays unchanged, but stitching does not start until the user
    # visually confirms its position on the current image set.
    preview_fixed_roi_before_stitch(args, image_paths, roi)

    radius_px, radius_records = estimate_radius_if_needed(args, image_paths, visual_angles, roi)
    if radius_px is not None: print(f"Cylindrical correction radius: {radius_px:.3f} px")
    else: print("Cylindrical correction disabled or radius could not be estimated.")
    write_radius_records(save_dir / "cylinder_radius_estimation.csv", radius_records)

    output_rows_per_deg = (math.pi * float(args.wheel_diameter_mm) / float(args.total_rotation_deg)) / float(args.surface_mm_per_pixel)
    debug_dir = save_dir / "angle_debug_strips"
    if SAVE_DEBUG_STRIPS:
        debug_dir.mkdir(parents=True, exist_ok=True)

    # A fixed representative frame is additionally saved at several
    # preprocessing stages for thesis figures. This branch is read-only with
    # respect to the stitching algorithm and therefore does not change RUN20.
    paper_dir = save_dir / PAPER_PREPROCESS_DIRNAME
    if SAVE_PAPER_PREPROCESS_IMAGES:
        paper_dir.mkdir(parents=True, exist_ok=True)
        if image_paths:
            requested_idx = int(PAPER_EXAMPLE_IMAGE_INDEX)
            if requested_idx < 0:
                paper_example_idx = len(image_paths) // 2
            else:
                paper_example_idx = min(requested_idx, len(image_paths) - 1)
        else:
            paper_example_idx = 0
    else:
        paper_example_idx = -1

    bands=[]; raw_shapes=[]
    for i, path in enumerate(image_paths):
        # NOTE: the files in img_dir are already the outputs of
        # undistort_and_metric_rectify.py. Keep this image so that the paper
        # folder contains the geometrically corrected full-frame input.
        img_rectified=cv2.imread(str(path))
        if img_rectified is None:
            raise RuntimeError(f"Could not read image: {path}")

        img=preprocess(img_rectified, use_clahe=not args.no_clahe, resize_scale=1.0)
        band_before_cyl=crop_roi(img,x,y,w,h)
        raw_shapes.append(tuple(int(v) for v in band_before_cyl.shape[:2]))

        if radius_px is not None:
            band=cylindrical_rectify_band(
                band_before_cyl,
                radius_px,
                args.cylinder_center_offset_y,
            )
        else:
            band=band_before_cyl.copy()

        # Save one representative preprocessing sequence for direct use in
        # the thesis: corrected full frame -> CLAHE full frame -> ROI before
        # cylindrical correction -> ROI after cylindrical correction.
        if SAVE_PAPER_PREPROCESS_IMAGES and i == paper_example_idx:
            cv2.imwrite(str(paper_dir / "01_rectified_input.png"), img_rectified)
            cv2.imwrite(str(paper_dir / "02_after_clahe.png"), img)
            cv2.imwrite(str(paper_dir / "03_roi_before_cylindrical.png"), band_before_cyl)
            cv2.imwrite(str(paper_dir / "04_roi_after_cylindrical.png"), band)
            with (paper_dir / "paper_preprocess_info.txt").open("w", encoding="utf-8") as pf:
                pf.write(f"source_file={path.name}\n")
                pf.write(f"image_index={i}\n")
                pf.write(f"encoder_travelled_deg={float(travelled[i]):.9f}\n")
                pf.write(f"roi_x={x}\nroi_y={y}\nroi_w={w}\nroi_h={h}\n")
                pf.write(f"clahe_enabled={not args.no_clahe}\n")
                pf.write(f"cylindrical_correction_enabled={radius_px is not None}\n")
                pf.write(f"cylinder_radius_px={'' if radius_px is None else f'{radius_px:.9f}'}\n")
            print(f"Paper preprocessing examples saved: {paper_dir}")

        if bands and band.shape[:2] != bands[0].shape[:2]:
            band=force_band_size(band,bands[0].shape[0],bands[0].shape[1])
        bands.append(band)
        if SAVE_DEBUG_STRIPS and i < DEBUG_STRIP_COUNT:
            cv2.imwrite(str(debug_dir/f"band_{i:04d}.png"),band)
        if i%20==0 or i==len(image_paths)-1:
            print(f"[{i+1}/{len(image_paths)}] {path.name} | center={float(travelled[i]):.3f} deg, band={band.shape[0]} rows")

    registered_centers,total_rows,metric_rows_per_deg,source_rows_per_deg,registration_records = robust_registered_centers(
        bands, travelled, output_rows_per_deg, args,
        lock_global_centers_to_encoder=bool(LOCK_REAL_ENCODER_GLOBAL_CENTERS and args.use_real_encoder and encoder_feedback_path is not None),
    )
    save_seam_registration(save_dir/"seam_registration.csv",registration_records,image_paths)
    if args.use_real_encoder and encoder_feedback_path is not None:
        print("Placement mode: REAL ENCODER fixed metric grid, global centers LOCKED.")

    natural_result, natural_source_ranges, overlap_records, overlap_info = stitch_like_pinjie(bands,args)
    natural_path=save_dir/"fine_wire_angle_unwrapped_stitched_natural.png"; cv2.imwrite(str(natural_path),natural_result)
    print(f"Pinjie-style visual stitch: natural_rows={natural_result.shape[0]}, metric_target_rows={total_rows}, compose={PINJIE_STYLE_COMPOSE_MODE}")

    encoder_overlap_warp_info={}
    if USE_ENCODER_CONSTRAINED_OVERLAP_WARP and args.use_real_encoder and encoder_feedback_path is not None:
        natural_centers=np.asarray(overlap_info["natural_frame_centers"],dtype=np.float64)
        result,encoder_overlap_warp_info=warp_overlap_result_to_encoder_grid(
            natural_result,natural_centers,travelled,
            output_rows_per_deg=metric_rows_per_deg,total_rotation_deg=float(args.total_rotation_deg),total_rows=int(total_rows),
        )
        source_ranges=encoder_midpoint_output_ranges(registered_centers,total_rows)
        print("Hybrid placement active: OVERLAP controls local seams, REAL ENCODER controls final global Y positions.")
    else:
        if USE_ENCODER_CONSTRAINED_OVERLAP_WARP and ENCODER_OVERLAP_REQUIRE_REAL_FEEDBACK:
            raise RuntimeError("RUN20 encoder-constrained overlap mode requires camera_motor_feedback.csv.")
        result=cv2.resize(natural_result,(natural_result.shape[1],int(total_rows)),interpolation=cv2.INTER_CUBIC)
        source_ranges=natural_source_ranges

    result_path=save_dir/"fine_wire_angle_unwrapped_stitched_laser_height.png"; cv2.imwrite(str(result_path),result)
    # Compatibility aliases commonly consumed downstream.
    legacy_result_path=save_dir/"fine_wire_angle_unwrapped_stitched.png"; copy_alias(result_path,legacy_result_path)
    manual_full_roi_path=save_dir/"fine_wire_angle_unwrapped_stitched_manual_full_roi.png"; copy_alias(result_path,manual_full_roi_path)

    if SAVE_PINJIE_STYLE_OVERLAP_TABLE:
        save_pinjie_style_overlap(save_dir/"pinjie_style_overlap.csv",overlap_records,image_paths,travelled,metric_rows_per_deg)

    laser_ref_path,laser_ref_rows=find_laser_reference_rows()
    info_path=save_dir/"fine_wire_angle_unwrapped_info.csv"
    with info_path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.writer(f)
        for k,v in [
            ("IMG_DIR",str(img_dir)),("ROI_X",x),("ROI_Y",y),("ROI_W",w),("ROI_H",h),("ROI_SELECTION_MODE",args.roi_mode),
            ("FIXED_ROI_X_PX",FIXED_ROI_X_PX),("FIXED_ROI_Y_PX",FIXED_ROI_Y_PX),("FIXED_ROI_WIDTH_PX",FIXED_ROI_WIDTH_PX),("FIXED_ROI_HEIGHT_PX",FIXED_ROI_HEIGHT_PX),("SHOW_PREVIEW_CENTER_GUIDES",SHOW_PREVIEW_CENTER_GUIDES),
            ("WHEEL_DIAMETER_MM",args.wheel_diameter_mm),("STITCH_SURFACE_MM_PER_PIXEL",f"{float(args.surface_mm_per_pixel):.8f}"),
            ("TOTAL_ROTATION_DEG",args.total_rotation_deg),("USE_REAL_ENCODER_FEEDBACK",bool(args.use_real_encoder and encoder_feedback_path is not None)),
            ("ENCODER_FEEDBACK_CSV","" if encoder_feedback_path is None else str(encoder_feedback_path)),
            ("ENCODER_REAL_FEEDBACK_ARBITRATION_ID",ENCODER_REAL_FEEDBACK_ARBITRATION_ID),("ENCODER_FEEDBACK_INDEX_SHIFT",ENCODER_FEEDBACK_INDEX_SHIFT),
            ("LOCK_REAL_ENCODER_GLOBAL_CENTERS",LOCK_REAL_ENCODER_GLOBAL_CENTERS),("ENCODER_SEAM_POLICY","midpoint_between_adjacent_encoder_centers"),
            ("USE_CYLINDRICAL_CORRECTION",radius_px is not None),("CYLINDER_RADIUS_PX","" if radius_px is None else f"{radius_px:.9f}"),
            ("SEAM_FEATHER_ROWS",args.seam_feather_rows),("USE_CONTENT_AWARE_SEAM",USE_CONTENT_AWARE_SEAM),("OVERLAP_KEEP_MODE",OVERLAP_KEEP_MODE),
            ("MATCH_COLOR_TO_PREVIOUS",args.match_color),("USE_PINJIE_STYLE_OVERLAP_STITCH",USE_PINJIE_STYLE_OVERLAP_STITCH),
            ("PINJIE_STYLE_COMPOSE_MODE",PINJIE_STYLE_COMPOSE_MODE),("USE_ENCODER_CONSTRAINED_OVERLAP_WARP",USE_ENCODER_CONSTRAINED_OVERLAP_WARP),
            ("AUTO_SEAM_REGISTRATION",args.auto_register),("REGISTRATION_SEARCH_ROWS",args.registration_search_rows),("REGISTRATION_SEARCH_COLS",args.registration_search_cols),
            ("REGISTRATION_MIN_OVERLAP_ROWS",args.registration_min_overlap_rows),("REGISTRATION_MIN_SCORE",args.registration_min_score),
            ("OUTPUT_ROWS_PER_DEG",f"{metric_rows_per_deg:.9f}"),("SOURCE_CAMERA_ROWS_PER_DEG",f"{source_rows_per_deg:.9f}"),("STITCH_TOTAL_ROWS_THEORY",total_rows),
            ("NATURAL_OUTPUT_ROWS_ACTUAL",int(result.shape[0])),("OUTPUT_WIDTH_PX",int(result.shape[1])),
            ("NATURAL_OUTPUT_PATH",str(natural_path)),("LASER_MATCHED_OUTPUT_PATH",str(result_path)),("DOWNSTREAM_ALIAS_PATH",str(manual_full_roi_path)),
            ("MATCH_LASER_HEIGHT_AFTER_STITCH",MATCH_LASER_HEIGHT_AFTER_STITCH),("LASER_HEIGHT_REFERENCE_PATH","" if laser_ref_path is None else str(laser_ref_path)),
            ("LASER_HEIGHT_REFERENCE_ROWS","" if laser_ref_rows is None else laser_ref_rows),
            ("ANGLE_PLACEMENT_MODE","overlap_visual_stitch_piecewise_warp_to_real_encoder_grid" if encoder_overlap_warp_info else "pinjie_style_overlap_then_metric_height_resize"),
            ("HYBRID_LOCAL_SEAM_SOURCE","overlap" if encoder_overlap_warp_info else ""),("HYBRID_GLOBAL_Y_SOURCE","real_encoder" if encoder_overlap_warp_info else ""),
            ("HYBRID_WARP_MODE",encoder_overlap_warp_info.get("warp_mode","")),
            ("HYBRID_NATURAL_ROWS_PER_DEG",encoder_overlap_warp_info.get("natural_rows_per_deg","")),
            ("HYBRID_ENCODER_ANCHOR_COUNT",encoder_overlap_warp_info.get("encoder_anchor_count","")),
            ("HYBRID_SOURCE_END_WAS_CLAMPED",encoder_overlap_warp_info.get("source_end_was_clamped","")),
        ]:
            writer.writerow([k,v])

    print("Done.")
    print(f"Natural stitch: {natural_path}")
    print(f"Final camera metric stitch: {result_path}")
    print(f"Final size: {result.shape[1]} x {result.shape[0]} px")
    print(f"Camera metric scale: 1 px = {float(args.surface_mm_per_pixel):.8f} mm")
    print(f"Overlap table: {save_dir/'pinjie_style_overlap.csv'}")
    print(f"Registration: {save_dir/'seam_registration.csv'}")
    print(f"Info: {info_path}")
    print(f"ROI: {save_dir/'roi.json'}")
    if SAVE_PAPER_PREPROCESS_IMAGES:
        print(f"Paper preprocessing examples: {save_dir/PAPER_PREPROCESS_DIRNAME}")


def main() -> None:
    args=parse_args(); run_angle_stitch(args)


if __name__ == "__main__":
    main()
