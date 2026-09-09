from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from config import LASER_RESTORED_IMAGE, CAMERA_STITCHED_IMAGE, CALIBRATION_DIR
except Exception:
    LASER_RESTORED_IMAGE = Path(r"D:\project\runs\test1\laser_then_camera\run1\laser\huanyuan_result\laser_height_mm_true_scale_detail_height.png")
    CAMERA_STITCHED_IMAGE = Path(r"D:\project\runs\test1\laser_then_camera\run1\camera\stitched\fine_wire_manual_full_roi_stitched.png")
    CALIBRATION_DIR = Path(r"D:\project\calibration")


# ============================================================
# User settings: you can edit these paths for different data.
# Leave empty to use paths from D:\project\scripts\config.py.
# ============================================================
LASER_IMAGE_PATH = (
    r"D:\project\runs\test1\laser_then_camera\run32"
    r"\laser\huanyuan_result"
    r"\laser_height_mm_true_scale_detail_height.png"
)

CAMERA_IMAGE_PATH = (
    r"D:\project\runs\test1\laser_then_camera\run32"
    r"\camera\stitched"
    r"\fine_wire_angle_unwrapped_stitched_laser_height.png"
)

# Output folder. Each run creates affine_YYYYMMDD_HHMMSS inside this folder.
OUTPUT_ROOT = r""
OUTPUT_RUN_NAME = r""

# Minimum 3 corresponding cone points are needed for affine transform.
# More points are better. Click points in the SAME ORDER in both images.
MIN_POINTS = 3

# RANSAC is useful when one clicked point is slightly wrong.
USE_RANSAC = True
RANSAC_REPROJ_THRESHOLD_PX = 12.0

# ============================================================
# Physical-coordinate calibration settings
# ============================================================
COMMON_MM_PER_PIXEL = 0.05473705

# ============================================================
# Circular Y / 0-360 seam handling
# ============================================================
# X uses millimetres. Y is first converted from image row to 0..360 deg,
# automatically unwrapped across the circular seam, then converted to arc-mm
# only for the affine fit so X/Y residuals have compatible physical units.
#
# IMPORTANT:
# The current stitched camera image and restored laser image are both treated
# as one complete 360-degree circumference, even if their pixel heights differ.
WHEEL_DIAMETER_MM = 150.0
CIRCULAR_PERIOD_DEG = 360.0
AUTO_CIRCULAR_UNWRAP = True
CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM

# For a corresponding pair, choose the equivalent camera angle
# theta + k*360 that is nearest to the laser angle.
# This makes e.g. laser=350 deg and camera=5 deg become camera=365 deg.
UNWRAP_CAMERA_TO_NEAREST_LASER_BRANCH = True

# Camera stitched-image physical coordinate conversion.
# IMPORTANT: the HORIZONTAL CENTER of the stitched camera ROI is X = 0 mm.
# Therefore different ROI widths are allowed, as long as each ROI is really
# centered on the same physical wheel/axial center.
#
# For an image width W and pixel column u:
#     u0 = (W - 1) / 2
#     X_camera_mm = (u - u0) * CAMERA_X_MM_PER_PIXEL
CAMERA_X_MM_PER_PIXEL = COMMON_MM_PER_PIXEL
# Camera Y is NOT taken from a fixed mm/px. It is normalized by current image height to 0..360 deg.

# Laser X conversion: use the real laser_x_axis_mm.csv range whenever possible.
USE_REAL_LASER_X_AXIS = True
LASER_X_AXIS_CSV = r""   # empty = auto-detect beside the restored laser PNG
LASER_X_MM_PER_PIXEL_FALLBACK = COMMON_MM_PER_PIXEL
LASER_X_ORIGIN_MM_FALLBACK = 0.0
# Laser Y is NOT taken from a fixed mm/px. It is normalized by current image height to 0..360 deg.

# X-origin convention used by this calibration:
#   CAMERA: stitched ROI horizontal center = X 0 mm
#   LASER : real laser native X coordinate from laser_x_axis_mm.csv; its
#           physical X=0 is kept as the laser X origin.
# No fixed-shoulder pixel coordinate is required in this version.

# 12 px at 0.05353937 mm/px is about 0.64 mm.
RANSAC_REPROJ_THRESHOLD_MM = 0.65
# ============================================================

# Final saved affine matrix policy:
# 1) RANSAC first identifies valid inliers;
# 2) only those inliers are used to refit the final affine matrix;
# 3) affine_calibration.json and affine_points.csv save the inlier-only result.
REFIT_FINAL_MATRIX_WITH_INLIERS_ONLY = True

# Overlay display. Camera is warped into laser image coordinates.
CAMERA_ALPHA = 0.65
LASER_ALPHA = 0.35

# The laser PNG uses nearly white pixels for invalid/no-data area.
LASER_INVALID_WHITE_THRESHOLD = 245

# Viewer settings.
VIEW_WIDTH = 1200
VIEW_HEIGHT = 850
MIN_WINDOW_WIDTH = 1
# Display images at native pixel scale by default: 1 image pixel = 1 screen pixel.
NATIVE_PIXEL_VIEW = True
ZOOM_FACTOR = 1.25
PAN_FRACTION = 0.22
FAST_PAN_FRACTION = 0.75
POINT_RADIUS = 7
# ============================================================


class ImagePointPicker:
    def __init__(self, title: str, image_bgr: np.ndarray, min_points: int) -> None:
        self.title = title
        self.img = image_bgr
        self.h, self.w = image_bgr.shape[:2]
        self.min_points = min_points
        self.points: list[tuple[float, float]] = []
        # Native-pixel display: do not stretch a narrow image to a wide window.
        # At zoom=1.0, one source image pixel is shown as one screen pixel.
        # Both laser and camera images therefore use the same visual pixel scale.
        self.window_w = max(1, min(VIEW_WIDTH, self.w))
        self.window_h = max(1, min(VIEW_HEIGHT, self.h))
        self.zoom = self._default_zoom()
        self.cx = self.w / 2.0
        self.cy = self._top_center_y()
        self._last_mouse_xy: tuple[int, int] | None = None
        self._needs_refresh = True
        self._finished = False
        self._cancelled = False

    def _default_zoom(self) -> float:
        # Keep native pixel scale by default. The viewer only zooms in, so the
        # image is never squeezed or stretched differently in X and Y.
        return 1.0 if NATIVE_PIXEL_VIEW else max(0.02, min(self.window_w / max(self.w, 1), 1.0))

    def _top_center_y(self) -> float:
        visible_h = max(1.0, self.window_h / max(self.zoom, 1e-9))
        return min(self.h - 1.0, max(0.0, visible_h / 2.0))

    def _bottom_center_y(self) -> float:
        visible_h = max(1.0, self.window_h / max(self.zoom, 1e-9))
        return min(self.h - 1.0, max(0.0, self.h - visible_h / 2.0))

    def _visible_rect(self) -> tuple[int, int, int, int]:
        view_w = max(1, int(round(self.window_w / self.zoom)))
        view_h = max(1, int(round(self.window_h / self.zoom)))
        x0 = int(round(self.cx - view_w / 2))
        y0 = int(round(self.cy - view_h / 2))
        x0 = max(0, min(max(0, self.w - view_w), x0))
        y0 = max(0, min(max(0, self.h - view_h), y0))
        x1 = min(self.w, x0 + view_w)
        y1 = min(self.h, y0 + view_h)
        return x0, y0, x1, y1

    def _screen_to_image(self, sx: int, sy: int) -> tuple[float, float]:
        x0, y0, x1, y1 = self._visible_rect()
        crop_w = max(1, x1 - x0)
        crop_h = max(1, y1 - y0)
        scale_x = crop_w / max(1, self.window_w)
        scale_y = crop_h / max(1, self.window_h)
        x = x0 + sx * scale_x
        y = y0 + sy * scale_y
        return min(self.w - 1, max(0.0, x)), min(self.h - 1, max(0.0, y))

    def _image_to_screen(self, x: float, y: float) -> tuple[int, int] | None:
        x0, y0, x1, y1 = self._visible_rect()
        if x < x0 or x >= x1 or y < y0 or y >= y1:
            return None
        sx = int(round((x - x0) * self.window_w / max(1, x1 - x0)))
        sy = int(round((y - y0) * self.window_h / max(1, y1 - y0)))
        return sx, sy

    def _set_center_from_screen(self, sx: int, sy: int) -> None:
        x, y = self._screen_to_image(sx, sy)
        self.cx = x
        self.cy = y

    def _zoom_at(self, factor: float, sx: int | None = None, sy: int | None = None) -> None:
        if sx is not None and sy is not None:
            before = self._screen_to_image(sx, sy)
        else:
            before = (self.cx, self.cy)
        min_zoom = 1.0 if NATIVE_PIXEL_VIEW else 0.02
        self.zoom = max(min_zoom, min(20.0, self.zoom * factor))
        if sx is not None and sy is not None:
            after = self._screen_to_image(sx, sy)
            self.cx += before[0] - after[0]
            self.cy += before[1] - after[1]
        self._clip_center()
        self._needs_refresh = True

    def _clip_center(self) -> None:
        self.cx = min(self.w - 1, max(0.0, self.cx))
        self.cy = min(self.h - 1, max(0.0, self.cy))

    def _pan(self, dx_frac: float, dy_frac: float) -> None:
        x0, y0, x1, y1 = self._visible_rect()
        self.cx += (x1 - x0) * dx_frac
        self.cy += (y1 - y0) * dy_frac
        self._clip_center()
        self._needs_refresh = True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        self._last_mouse_xy = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = self._screen_to_image(x, y)
            self.points.append((ix, iy))
            self._needs_refresh = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()
                self._needs_refresh = True
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Wheel alone scrolls vertically, which is better for very tall stitched images.
            # Hold Ctrl while using the wheel if you want to zoom around the mouse position.
            ctrl_down = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
            if ctrl_down:
                if flags > 0:
                    self._zoom_at(ZOOM_FACTOR, x, y)
                else:
                    self._zoom_at(1.0 / ZOOM_FACTOR, x, y)
            else:
                if flags > 0:
                    self._pan(0, -PAN_FRACTION)
                else:
                    self._pan(0, PAN_FRACTION)

    def _make_view(self) -> np.ndarray:
        x0, y0, x1, y1 = self._visible_rect()
        crop = self.img[y0:y1, x0:x1]
        interpolation = cv2.INTER_NEAREST if self.zoom >= 1.0 else cv2.INTER_AREA
        view = cv2.resize(crop, (self.window_w, self.window_h), interpolation=interpolation)

        # Center crosshair.
        cv2.line(view, (self.window_w // 2 - 18, self.window_h // 2), (self.window_w // 2 + 18, self.window_h // 2), (0, 255, 0), 1)
        cv2.line(view, (self.window_w // 2, self.window_h // 2 - 18), (self.window_w // 2, self.window_h // 2 + 18), (0, 255, 0), 1)

        for idx, (px, py) in enumerate(self.points, start=1):
            screen = self._image_to_screen(px, py)
            if screen is None:
                continue
            sx, sy = screen
            color = (0, 255, 255) if idx < len(self.points) else (0, 0, 255)
            cv2.circle(view, (sx, sy), POINT_RADIUS, color, 2, cv2.LINE_AA)
            cv2.line(view, (sx - 15, sy), (sx + 15, sy), color, 1, cv2.LINE_AA)
            cv2.line(view, (sx, sy - 15), (sx, sy + 15), color, 1, cv2.LINE_AA)
            cv2.putText(view, str(idx), (sx + 9, sy - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        info1 = f"{self.title} | points={len(self.points)} | image={self.w}x{self.h} | zoom={self.zoom:.3f}"
        info2 = "Native pixels: 1 image px = 1 screen px | Left=add, Right/U=undo, Wheel=scroll, Ctrl+wheel or +/-=zoom, Enter=finish"
        for y, text in [(24, info1), (50, info2)]:
            cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    def run(self) -> np.ndarray:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.window_w, self.window_h)
        cv2.setMouseCallback(self.title, self._on_mouse)
        while True:
            if self._needs_refresh:
                cv2.imshow(self.title, self._make_view())
                self._needs_refresh = False
            key = cv2.waitKeyEx(30)
            if key < 0:
                continue
            if key in (13, 10):
                if len(self.points) >= self.min_points:
                    self._finished = True
                    break
                print(f"Need at least {self.min_points} points. Current: {len(self.points)}")
            elif key == 27:
                self._cancelled = True
                break
            elif key in (ord("u"), ord("U"), 8, 127):
                if self.points:
                    self.points.pop()
                    self._needs_refresh = True
            elif key in (ord("+"), ord("=")):
                sx, sy = self._last_mouse_xy if self._last_mouse_xy else (self.window_w // 2, self.window_h // 2)
                self._zoom_at(ZOOM_FACTOR, sx, sy)
            elif key in (ord("-"), ord("_")):
                sx, sy = self._last_mouse_xy if self._last_mouse_xy else (self.window_w // 2, self.window_h // 2)
                self._zoom_at(1.0 / ZOOM_FACTOR, sx, sy)
            elif key in (ord("a"), ord("A"), 2424832):
                self._pan(-PAN_FRACTION, 0)
            elif key in (ord("d"), ord("D"), 2555904):
                self._pan(PAN_FRACTION, 0)
            elif key in (ord("w"), ord("W"), 2490368):
                self._pan(0, -PAN_FRACTION)
            elif key in (ord("s"), ord("S"), 2621440):
                self._pan(0, PAN_FRACTION)
            elif key in (2162688,):
                self._pan(0, -FAST_PAN_FRACTION)
            elif key in (2228224,):
                self._pan(0, FAST_PAN_FRACTION)
            elif key in (2359296, ord("t"), ord("T")):
                self.cy = self._top_center_y()
                self._needs_refresh = True
            elif key in (2293760, ord("b"), ord("B")):
                self.cy = self._bottom_center_y()
                self._needs_refresh = True
            elif key in (ord("r"), ord("R")):
                self.zoom = self._default_zoom()
                self.cx = self.w / 2.0
                self.cy = self._top_center_y()
                self._needs_refresh = True
            elif key in (ord("c"), ord("C")):
                self.points.clear()
                self._needs_refresh = True
        cv2.destroyWindow(self.title)
        if self._cancelled:
            raise KeyboardInterrupt("Point selection cancelled")
        return np.asarray(self.points, dtype=np.float32)


def read_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".png"
    ok, data = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    data.tofile(str(path))


def resolve_path(text: str, default_path: Path) -> Path:
    value = str(text).strip()
    return Path(value) if value else Path(default_path)


def make_output_dir(root_text: str, run_name: str) -> Path:
    root = Path(root_text) if str(root_text).strip() else Path(CALIBRATION_DIR) / "affine_calibration"
    name = str(run_name).strip() or f"affine_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = root / name
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir



def infer_laser_x_axis_csv(laser_image_path: Path) -> Path | None:
    explicit = str(LASER_X_AXIS_CSV).strip()
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    folder = laser_image_path.parent
    stem = laser_image_path.stem
    for suffix in (
        "_true_scale_detail_height",
        "_true_scale_absolute_height",
        "_detail_height",
        "_absolute_height",
    ):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break

    candidates = [
        folder / f"{stem}_x_axis_mm.csv",
        folder / "laser_height_mm_x_axis_mm.csv",
        folder.parent / "laser_x_axis_mm.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    generic = sorted(folder.glob("*_x_axis_mm.csv"))
    return generic[0] if generic else None


def load_laser_x_mapping(laser_image_path: Path, laser_width_px: int) -> dict:
    info = {
        "source": "fallback_uniform_pixel_scale",
        "x_axis_csv": "",
        "x_min_mm": float(LASER_X_ORIGIN_MM_FALLBACK),
        "x_max_mm": float(
            LASER_X_ORIGIN_MM_FALLBACK
            + max(0, laser_width_px - 1) * LASER_X_MM_PER_PIXEL_FALLBACK
        ),
        "mm_per_pixel": float(LASER_X_MM_PER_PIXEL_FALLBACK),
        "used_real_x_axis": False,
    }

    if not USE_REAL_LASER_X_AXIS:
        return info

    x_csv = infer_laser_x_axis_csv(laser_image_path)
    if x_csv is None:
        print("Warning: laser_x_axis_mm.csv not found; using fallback X scale.")
        return info

    try:
        raw = np.genfromtxt(x_csv, delimiter=",", dtype=np.float64, encoding="utf-8-sig")
        raw = np.asarray(raw, dtype=np.float64).reshape(-1)
        raw = raw[np.isfinite(raw)]
        if raw.size < 2:
            raise ValueError("fewer than 2 finite X values")
        x_min = float(np.min(raw))
        x_max = float(np.max(raw))
        if x_max <= x_min:
            raise ValueError(f"invalid X range {x_min}..{x_max}")
        step = (x_max - x_min) / max(laser_width_px - 1, 1)
        info.update({
            "source": "real_laser_x_axis_range",
            "x_axis_csv": str(x_csv),
            "x_min_mm": x_min,
            "x_max_mm": x_max,
            "mm_per_pixel": float(step),
            "used_real_x_axis": True,
        })
    except Exception as exc:
        print(f"Warning: could not use {x_csv}: {exc}; using fallback X scale.")
    return info


def _pixel_y_to_theta_deg(y_px: np.ndarray, image_height_px: int) -> np.ndarray:
    """Map current image rows to the periodic 0..360 degree coordinate."""
    h = max(1, int(image_height_px))
    y = np.asarray(y_px, dtype=np.float64)
    return y / float(h) * float(CIRCULAR_PERIOD_DEG)


def _theta_deg_to_arc_mm(theta_deg: np.ndarray) -> np.ndarray:
    """Convert an UNWRAPPED angle to signed/unwrapped circumference arc mm."""
    theta = np.asarray(theta_deg, dtype=np.float64)
    return theta / float(CIRCULAR_PERIOD_DEG) * float(CIRCUMFERENCE_MM)


def _arc_mm_to_theta_deg(arc_mm: np.ndarray) -> np.ndarray:
    arc = np.asarray(arc_mm, dtype=np.float64)
    return arc / float(CIRCUMFERENCE_MM) * float(CIRCULAR_PERIOD_DEG)


def camera_pixels_to_x_theta(
    points_px: np.ndarray,
    camera_width_px: int,
    camera_height_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera clicks -> X(mm) and raw periodic theta(deg).

    Camera horizontal ROI center is X=0 mm, so ROI widths may differ between
    runs as long as the ROI center remains at the same physical axial center.
    """
    pts = np.asarray(points_px, dtype=np.float64)
    center_x_px = (float(camera_width_px) - 1.0) / 2.0
    x_mm = (
        pts[:, 0] - center_x_px
    ) * float(CAMERA_X_MM_PER_PIXEL)
    theta_deg = _pixel_y_to_theta_deg(pts[:, 1], camera_height_px)
    return x_mm, theta_deg


def laser_pixels_to_x_theta(
    points_px: np.ndarray,
    laser_x_info: dict,
    laser_height_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Laser clicks -> X(mm) from real X axis and raw periodic theta(deg)."""
    pts = np.asarray(points_px, dtype=np.float64)
    x_mm = (
        float(laser_x_info["x_min_mm"])
        + pts[:, 0] * float(laser_x_info["mm_per_pixel"])
    )
    theta_deg = _pixel_y_to_theta_deg(pts[:, 1], laser_height_px)
    return x_mm, theta_deg


def unwrap_camera_theta_to_laser(
    camera_theta_deg: np.ndarray,
    laser_theta_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose camera theta + k*360 nearest each corresponding laser theta.

    Example:
        laser 350 deg, camera 5 deg -> camera 365 deg, k=+1
        laser  10 deg, camera 355 deg -> camera -5 deg, k=-1
    """
    camera_theta = np.asarray(camera_theta_deg, dtype=np.float64)
    laser_theta = np.asarray(laser_theta_deg, dtype=np.float64)
    if camera_theta.shape != laser_theta.shape:
        raise ValueError("camera/laser theta arrays must have the same shape")

    if not AUTO_CIRCULAR_UNWRAP:
        return camera_theta.copy(), np.zeros(camera_theta.shape, dtype=np.int32)

    period = float(CIRCULAR_PERIOD_DEG)
    branch_k = np.rint(
        (laser_theta - camera_theta) / period
    ).astype(np.int32)
    unwrapped = camera_theta + branch_k.astype(np.float64) * period
    return unwrapped, branch_k


def build_physical_points(
    camera_pts_px: np.ndarray,
    laser_pts_px: np.ndarray,
    *,
    camera_width_px: int,
    camera_height_px: int,
    laser_height_px: int,
    laser_x_info: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert clicked pixels to seam-safe physical coordinates.

    Affine coordinate:
        first component  = X mm
        second component = unwrapped circumferential arc mm

    The raw/unwrapped theta values are also returned for diagnostics and for
    downstream circular modulo handling.
    """
    cam_x_mm, cam_theta_raw = camera_pixels_to_x_theta(
        camera_pts_px,
        camera_width_px,
        camera_height_px,
    )
    laser_x_mm, laser_theta_raw = laser_pixels_to_x_theta(
        laser_pts_px,
        laser_x_info,
        laser_height_px,
    )

    cam_theta_unwrapped, branch_k = unwrap_camera_theta_to_laser(
        cam_theta_raw,
        laser_theta_raw,
    )

    camera_physical = np.column_stack(
        [
            cam_x_mm,
            _theta_deg_to_arc_mm(cam_theta_unwrapped),
        ]
    ).astype(np.float32)

    # Laser is the target branch/reference. Its clicked rows remain in the
    # base 0..360 interval. Predicted outputs are wrapped modulo 360 only when
    # converting back to the final laser image.
    laser_physical = np.column_stack(
        [
            laser_x_mm,
            _theta_deg_to_arc_mm(laser_theta_raw),
        ]
    ).astype(np.float32)

    meta = {
        "camera_theta_raw_deg": cam_theta_raw,
        "camera_theta_unwrapped_deg": cam_theta_unwrapped,
        "camera_wrap_branch_k": branch_k,
        "laser_theta_raw_deg": laser_theta_raw,
    }
    return camera_physical, laser_physical, meta


def camera_px_to_physical_matrix(
    camera_width_px: int,
    camera_height_px: int,
    branch_k: int,
) -> np.ndarray:
    """3x3: one camera pixel branch -> [X_mm, unwrapped_arc_mm]."""
    sx = float(CAMERA_X_MM_PER_PIXEL)
    center_x_px = (float(camera_width_px) - 1.0) / 2.0
    tx = -center_x_px * sx

    sy = float(CIRCUMFERENCE_MM) / max(float(camera_height_px), 1.0)
    ty = float(branch_k) * float(CIRCUMFERENCE_MM)

    return np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def laser_px_to_physical_matrix(
    laser_x_info: dict,
    laser_height_px: int,
) -> np.ndarray:
    """3x3: laser pixel -> [X_mm, base-branch arc_mm]."""
    sx = float(laser_x_info["mm_per_pixel"])
    tx = float(laser_x_info["x_min_mm"])
    sy = float(CIRCUMFERENCE_MM) / max(float(laser_height_px), 1.0)
    return np.array(
        [
            [sx, 0.0, tx],
            [0.0, sy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def physical_affine_to_current_pixel_affine_for_branch(
    matrix_mm: np.ndarray,
    laser_x_info: dict,
    camera_width_px: int,
    camera_height_px: int,
    laser_height_px: int,
    branch_k: int,
) -> np.ndarray:
    """Camera-px(branch k) -> laser-px for CURRENT images, debug only."""
    m = np.eye(3, dtype=np.float64)
    m[:2, :] = np.asarray(matrix_mm, dtype=np.float64)
    cam_px_to_phys = camera_px_to_physical_matrix(
        camera_width_px,
        camera_height_px,
        branch_k,
    )
    laser_phys_to_px = np.linalg.inv(
        laser_px_to_physical_matrix(
            laser_x_info,
            laser_height_px,
        )
    )
    return (laser_phys_to_px @ m @ cam_px_to_phys)[:2, :]


def physical_laser_points_to_wrapped_pixels(
    laser_points_physical: np.ndarray,
    laser_x_info: dict,
    laser_height_px: int,
) -> np.ndarray:
    """[X_mm, arc_mm] -> current laser pixels with circular Y modulo."""
    pts = np.asarray(laser_points_physical, dtype=np.float64)
    x = (
        pts[:, 0] - float(laser_x_info["x_min_mm"])
    ) / float(laser_x_info["mm_per_pixel"])

    theta = _arc_mm_to_theta_deg(pts[:, 1])
    theta_wrapped = np.mod(theta, float(CIRCULAR_PERIOD_DEG))
    y = (
        theta_wrapped / float(CIRCULAR_PERIOD_DEG)
        * float(laser_height_px)
    )
    return np.column_stack([x, y]).astype(np.float32)


def circular_warp_camera_to_laser(
    camera_img: np.ndarray,
    matrix_mm: np.ndarray,
    laser_x_info: dict,
    laser_size_wh: tuple[int, int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Warp three equivalent camera Y branches and wrap them onto laser canvas.

    A single ordinary warpAffine cannot represent the 0/360 discontinuity.
    We therefore warp camera branches k=-1,0,+1 separately. This is used only
    for calibration debug/overlay. The reusable calibration is the physical
    matrix plus circular metadata in the JSON.
    """
    laser_w, laser_h = [int(v) for v in laser_size_wh]
    camera_h, camera_w = camera_img.shape[:2]

    out = np.full((laser_h, laser_w, 3), 255, dtype=np.uint8)
    branch_matrices: dict[int, np.ndarray] = {}

    for branch_k in (-1, 0, 1):
        m_px = physical_affine_to_current_pixel_affine_for_branch(
            matrix_mm,
            laser_x_info,
            camera_w,
            camera_h,
            laser_h,
            branch_k,
        )
        branch_matrices[branch_k] = m_px

        warped = cv2.warpAffine(
            camera_img,
            m_px,
            (laser_w, laser_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Border is white. Current camera imagery is dark, so use non-white
        # pixels to merge the valid part of each circular branch.
        valid = np.any(warped < 250, axis=2)
        out[valid] = warped[valid]

    return out, branch_matrices



def write_physical_points_csv(
    path: Path,
    camera_px: np.ndarray,
    laser_px: np.ndarray,
    camera_physical: np.ndarray,
    laser_physical: np.ndarray,
    predicted_laser_physical: np.ndarray,
    residual_mm: np.ndarray,
    inliers: np.ndarray,
    circular_meta: dict,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "camera_x_px",
            "camera_y_px",
            "laser_x_px",
            "laser_y_px",
            "camera_x_mm",
            "camera_theta_raw_deg",
            "camera_wrap_branch_k",
            "camera_theta_unwrapped_deg",
            "camera_arc_unwrapped_mm",
            "laser_x_mm",
            "laser_theta_raw_deg",
            "laser_arc_mm",
            "predicted_laser_x_mm",
            "predicted_laser_arc_mm",
            "predicted_laser_theta_deg",
            "predicted_laser_theta_wrapped_deg",
            "residual_mm",
            "inlier",
        ])

        mask = np.asarray(inliers).reshape(-1)
        cam_raw = np.asarray(circular_meta["camera_theta_raw_deg"])
        cam_unwrapped = np.asarray(
            circular_meta["camera_theta_unwrapped_deg"]
        )
        branches = np.asarray(circular_meta["camera_wrap_branch_k"])
        laser_raw = np.asarray(circular_meta["laser_theta_raw_deg"])

        pred_theta = _arc_mm_to_theta_deg(
            predicted_laser_physical[:, 1]
        )
        pred_theta_wrapped = np.mod(
            pred_theta,
            float(CIRCULAR_PERIOD_DEG),
        )

        for i, values in enumerate(
            zip(
                camera_px,
                laser_px,
                camera_physical,
                laser_physical,
                predicted_laser_physical,
                residual_mm,
                cam_raw,
                branches,
                cam_unwrapped,
                laser_raw,
            ),
            start=1,
        ):
            cpx, lpx, cphy, lphy, pphy, rmm, c_raw, bk, c_un, l_raw = values
            writer.writerow([
                i,
                f"{cpx[0]:.6f}",
                f"{cpx[1]:.6f}",
                f"{lpx[0]:.6f}",
                f"{lpx[1]:.6f}",
                f"{cphy[0]:.9f}",
                f"{float(c_raw):.9f}",
                int(bk),
                f"{float(c_un):.9f}",
                f"{cphy[1]:.9f}",
                f"{lphy[0]:.9f}",
                f"{float(l_raw):.9f}",
                f"{lphy[1]:.9f}",
                f"{pphy[0]:.9f}",
                f"{pphy[1]:.9f}",
                f"{float(pred_theta[i-1]):.9f}",
                f"{float(pred_theta_wrapped[i-1]):.9f}",
                f"{float(rmm):.9f}",
                int(mask[i - 1]),
            ])



def estimate_affine(camera_pts: np.ndarray, laser_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(camera_pts) != len(laser_pts):
        raise ValueError("camera point count must equal laser point count")
    if len(camera_pts) < 3:
        raise ValueError("at least 3 point pairs are required")

    if len(camera_pts) == 3 and not USE_RANSAC:
        matrix = cv2.getAffineTransform(camera_pts.astype(np.float32), laser_pts.astype(np.float32))
        inliers = np.ones((3, 1), dtype=np.uint8)
        return matrix, inliers

    method = cv2.RANSAC if USE_RANSAC else cv2.LMEDS
    matrix, inliers = cv2.estimateAffine2D(
        camera_pts.astype(np.float32),
        laser_pts.astype(np.float32),
        method=method,
        ransacReprojThreshold=float(RANSAC_REPROJ_THRESHOLD_MM),
        maxIters=5000,
        confidence=0.995,
        refineIters=20,
    )
    if matrix is None:
        raise RuntimeError("cv2.estimateAffine2D failed. Check point order and use at least 3 non-collinear points.")
    if inliers is None:
        inliers = np.ones((len(camera_pts), 1), dtype=np.uint8)
    return matrix, inliers.astype(np.uint8)



def refit_affine_with_inliers(
    camera_pts: np.ndarray,
    laser_pts: np.ndarray,
    inliers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit the final camera->laser affine matrix using RANSAC inliers only."""
    inlier_mask = np.asarray(inliers).reshape(-1).astype(bool)
    if len(inlier_mask) != len(camera_pts):
        raise ValueError("inlier mask length does not match point count")
    if int(np.count_nonzero(inlier_mask)) < 3:
        raise RuntimeError(
            f"Only {int(np.count_nonzero(inlier_mask))} valid inliers; "
            "at least 3 non-collinear inliers are required."
        )

    inlier_camera = np.asarray(camera_pts[inlier_mask], dtype=np.float64)
    inlier_laser = np.asarray(laser_pts[inlier_mask], dtype=np.float64)

    design = np.column_stack(
        [
            inlier_camera[:, 0],
            inlier_camera[:, 1],
            np.ones(len(inlier_camera), dtype=np.float64),
        ]
    )
    rank = int(np.linalg.matrix_rank(design))
    if rank < 3:
        raise RuntimeError(
            "The selected inlier points are collinear or nearly collinear. "
            "A full 2D affine matrix cannot be determined reliably."
        )

    coeff_x, *_ = np.linalg.lstsq(design, inlier_laser[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, inlier_laser[:, 1], rcond=None)
    matrix = np.vstack([coeff_x, coeff_y]).astype(np.float64)

    return matrix, inlier_mask, inlier_camera.astype(np.float32), inlier_laser.astype(np.float32)


def write_inlier_points_csv(
    path: Path,
    original_indices_1based: np.ndarray,
    camera_pts: np.ndarray,
    laser_pts: np.ndarray,
    predicted: np.ndarray,
    residuals: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "inlier_order",
            "original_point_index",
            "camera_x_px",
            "camera_y_px",
            "laser_x_px",
            "laser_y_px",
            "predicted_laser_x_px",
            "predicted_laser_y_px",
            "residual_px",
        ])
        for order, (original_index, c, l, p, r) in enumerate(
            zip(original_indices_1based, camera_pts, laser_pts, predicted, residuals),
            start=1,
        ):
            writer.writerow([
                order,
                int(original_index),
                f"{c[0]:.6f}",
                f"{c[1]:.6f}",
                f"{l[0]:.6f}",
                f"{l[1]:.6f}",
                f"{p[0]:.6f}",
                f"{p[1]:.6f}",
                f"{float(r):.6f}",
            ])


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((len(points), 1), dtype=np.float32)
    homo = np.hstack([points.astype(np.float32), ones])
    return homo @ matrix.T


def laser_valid_mask_from_png(laser_bgr: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(laser_bgr)
    white = (b >= LASER_INVALID_WHITE_THRESHOLD) & (g >= LASER_INVALID_WHITE_THRESHOLD) & (r >= LASER_INVALID_WHITE_THRESHOLD)
    return (~white).astype(np.uint8) * 255


def blend_on_valid(laser_bgr: np.ndarray, warped_camera_bgr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    base = warped_camera_bgr.copy()
    blended = cv2.addWeighted(warped_camera_bgr, CAMERA_ALPHA, laser_bgr, LASER_ALPHA, 0.0)
    mask = valid_mask > 0
    out = base.copy()
    out[mask] = blended[mask]
    return out


def draw_points_on_image(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], prefix: str) -> np.ndarray:
    out = img.copy()
    for idx, (x, y) in enumerate(pts, start=1):
        xi, yi = int(round(x)), int(round(y))
        cv2.circle(out, (xi, yi), 13, color, 2, cv2.LINE_AA)
        cv2.line(out, (xi - 18, yi), (xi + 18, yi), color, 2, cv2.LINE_AA)
        cv2.line(out, (xi, yi - 18), (xi, yi + 18), color, 2, cv2.LINE_AA)
        cv2.putText(out, f"{prefix}{idx}", (xi + 16, yi - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def resize_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target_h / max(1, h)
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_debug_compare(laser_bgr: np.ndarray, warped_camera_bgr: np.ndarray, overlay_bgr: np.ndarray, max_h: int = 2600) -> np.ndarray:
    target_h = min(max_h, laser_bgr.shape[0])
    left = resize_height(laser_bgr, target_h)
    mid = resize_height(warped_camera_bgr, target_h)
    right = resize_height(overlay_bgr, target_h)
    gap = np.full((target_h, 18, 3), 255, dtype=np.uint8)
    return np.hstack([left, gap, mid, gap, right])


def write_points_csv(path: Path, camera_pts: np.ndarray, laser_pts: np.ndarray, predicted: np.ndarray, residuals: np.ndarray, inliers: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "camera_x_px",
            "camera_y_px",
            "laser_x_px",
            "laser_y_px",
            "predicted_laser_x_px",
            "predicted_laser_y_px",
            "residual_px",
            "inlier",
        ])
        for i, (c, l, p, r) in enumerate(zip(camera_pts, laser_pts, predicted, residuals), start=1):
            writer.writerow([i, f"{c[0]:.6f}", f"{c[1]:.6f}", f"{l[0]:.6f}", f"{l[1]:.6f}", f"{p[0]:.6f}", f"{p[1]:.6f}", f"{float(r):.6f}", int(inliers[i - 1, 0])])


def run_calibration(
    laser_path: Path,
    camera_path: Path,
    output_dir: Path,
) -> None:
    print("Reading images...")
    laser_img = read_image(laser_path)
    camera_img = read_image(camera_path)

    laser_h, laser_w = laser_img.shape[:2]
    camera_h, camera_w = camera_img.shape[:2]
    laser_x_info = load_laser_x_mapping(laser_path, laser_w)

    print(f"Laser image : {laser_path}  {laser_w}x{laser_h} px")
    print(f"Camera image: {camera_path}  {camera_w}x{camera_h} px")
    print(f"Wheel diameter: {WHEEL_DIAMETER_MM:.6f} mm")
    print(f"Circumference: {CIRCUMFERENCE_MM:.6f} mm")
    print(
        f"Camera X scale: {CAMERA_X_MM_PER_PIXEL:.9f} mm/px; "
        f"ROI center u0={(camera_w - 1) / 2.0:.3f} -> X=0"
    )
    print(
        "Laser X: "
        f"{laser_x_info['source']}, "
        f"range={float(laser_x_info['x_min_mm']):.6f}.."
        f"{float(laser_x_info['x_max_mm']):.6f} mm, "
        f"step={float(laser_x_info['mm_per_pixel']):.9f} mm/px"
    )
    print(
        "Circular Y: current image height -> 0..360 deg; "
        "camera points automatically unwrap by +/-360 deg when needed."
    )
    print()

    print("Step 1: click cone centers in the LASER height image.")
    print(
        "A laser point may be near the BOTTOM while its corresponding "
        "camera point is near the TOP. That is allowed."
    )
    laser_pts_px = ImagePointPicker(
        "1 Select LASER cone centers",
        laser_img,
        MIN_POINTS,
    ).run()

    print()
    print(f"Laser points selected: {len(laser_pts_px)}")
    print(
        "Step 2: click the SAME cone centers in the CAMERA stitched image, "
        "in the SAME ORDER."
    )
    camera_pts_px = ImagePointPicker(
        "2 Select CAMERA cone centers in same order",
        camera_img,
        len(laser_pts_px),
    ).run()

    if len(camera_pts_px) != len(laser_pts_px):
        raise RuntimeError(
            f"Point count mismatch: laser={len(laser_pts_px)}, "
            f"camera={len(camera_pts_px)}"
        )

    camera_physical, laser_physical, circular_meta = (
        build_physical_points(
            camera_pts_px,
            laser_pts_px,
            camera_width_px=camera_w,
            camera_height_px=camera_h,
            laser_height_px=laser_h,
            laser_x_info=laser_x_info,
        )
    )

    print()
    print("Clicked points converted to seam-safe physical coordinates:")
    for i in range(len(camera_pts_px)):
        cpx = camera_pts_px[i]
        lpx = laser_pts_px[i]
        cphy = camera_physical[i]
        lphy = laser_physical[i]
        c_raw = float(circular_meta["camera_theta_raw_deg"][i])
        c_un = float(
            circular_meta["camera_theta_unwrapped_deg"][i]
        )
        bk = int(circular_meta["camera_wrap_branch_k"][i])
        l_raw = float(circular_meta["laser_theta_raw_deg"][i])

        print(
            f"  P{i+1}: "
            f"camera px=({cpx[0]:.2f},{cpx[1]:.2f}) "
            f"X={cphy[0]:.6f}mm, theta={c_raw:.3f}deg "
            f"-> unwrapped={c_un:.3f}deg (k={bk:+d}); "
            f"laser px=({lpx[0]:.2f},{lpx[1]:.2f}) "
            f"X={lphy[0]:.6f}mm, theta={l_raw:.3f}deg"
        )

    ransac_matrix_mm, inliers = estimate_affine(
        camera_physical,
        laser_physical,
    )
    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_indices_1based = np.flatnonzero(inlier_mask) + 1

    if REFIT_FINAL_MATRIX_WITH_INLIERS_ONLY:
        (
            matrix_mm,
            inlier_mask,
            inlier_camera_physical,
            inlier_laser_physical,
        ) = refit_affine_with_inliers(
            camera_physical,
            laser_physical,
            inliers,
        )
        fit_method = (
            "ransac_select_inliers_then_least_squares_refit_"
            "Xmm_unwrapped_arcmm"
        )
    else:
        matrix_mm = np.asarray(
            ransac_matrix_mm,
            dtype=np.float64,
        )
        inlier_camera_physical = camera_physical[inlier_mask]
        inlier_laser_physical = laser_physical[inlier_mask]
        fit_method = "opencv_ransac_Xmm_unwrapped_arcmm"

    predicted_all_physical = transform_points(
        matrix_mm,
        camera_physical,
    )
    residual_vec_all = (
        predicted_all_physical - laser_physical
    )
    residuals_all_mm = np.sqrt(
        np.sum(residual_vec_all * residual_vec_all, axis=1)
    )
    all_rmse_mm = float(
        np.sqrt(np.mean(residuals_all_mm * residuals_all_mm))
    )
    all_max_error_mm = float(np.max(residuals_all_mm))

    predicted_inlier_physical = transform_points(
        matrix_mm,
        inlier_camera_physical,
    )
    residual_vec_inlier = (
        predicted_inlier_physical - inlier_laser_physical
    )
    residuals_inlier_mm = np.sqrt(
        np.sum(residual_vec_inlier * residual_vec_inlier, axis=1)
    )
    inlier_rmse_mm = float(
        np.sqrt(np.mean(residuals_inlier_mm * residuals_inlier_mm))
    )
    inlier_max_error_mm = float(np.max(residuals_inlier_mm))

    predicted_all_px = physical_laser_points_to_wrapped_pixels(
        predicted_all_physical,
        laser_x_info,
        laser_h,
    )
    pixel_residuals_all = np.sqrt(
        np.sum((predicted_all_px - laser_pts_px) ** 2, axis=1)
    )

    print()
    print(
        "FINAL affine: CAMERA [X_mm, unwrapped_arc_mm] -> "
        "LASER [X_mm, arc_mm]"
    )
    print(matrix_mm)
    print(f"Inliers: {inlier_count}/{len(camera_pts_px)}")
    print(
        f"Inlier RMSE: {inlier_rmse_mm:.6f} mm, "
        f"max error: {inlier_max_error_mm:.6f} mm"
    )
    print(
        f"All-point RMSE: {all_rmse_mm:.6f} mm, "
        f"max error: {all_max_error_mm:.6f} mm"
    )

    # Circular debug warp: use three equivalent camera Y branches.
    warped_camera, branch_matrices = circular_warp_camera_to_laser(
        camera_img,
        matrix_mm,
        laser_x_info,
        (laser_w, laser_h),
    )

    valid_mask = laser_valid_mask_from_png(laser_img)
    overlay = blend_on_valid(
        laser_img,
        warped_camera,
        valid_mask,
    )

    laser_marked = draw_points_on_image(
        laser_img,
        laser_pts_px,
        (0, 0, 255),
        "L",
    )
    warped_marked = draw_points_on_image(
        warped_camera,
        predicted_all_px,
        (0, 255, 255),
        "C",
    )
    overlay_marked = draw_points_on_image(
        overlay,
        laser_pts_px,
        (0, 0, 255),
        "L",
    )
    overlay_marked = draw_points_on_image(
        overlay_marked,
        predicted_all_px,
        (0, 255, 255),
        "C",
    )

    save_image(
        output_dir / "camera_warped_to_laser_circular.png",
        warped_camera,
    )
    save_image(
        output_dir / "laser_camera_affine_overlay_circular.png",
        overlay,
    )
    save_image(
        output_dir / "laser_camera_affine_overlay_with_points.png",
        overlay_marked,
    )
    save_image(
        output_dir / "laser_points_marked.png",
        laser_marked,
    )
    save_image(
        output_dir / "camera_warped_points_marked.png",
        warped_marked,
    )
    save_image(
        output_dir / "affine_debug_compare.png",
        make_debug_compare(
            laser_marked,
            warped_marked,
            overlay_marked,
        ),
    )

    write_points_csv(
        output_dir / "affine_all_points_diagnostic_px.csv",
        camera_pts_px,
        laser_pts_px,
        predicted_all_px,
        pixel_residuals_all,
        inliers,
    )

    write_physical_points_csv(
        output_dir / "affine_points_physical_circular.csv",
        camera_pts_px,
        laser_pts_px,
        camera_physical,
        laser_physical,
        predicted_all_physical,
        residuals_all_mm,
        inliers,
        circular_meta,
    )

    branch_json = {
        str(k): np.asarray(v, dtype=float).tolist()
        for k, v in branch_matrices.items()
    }

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "calibration_coordinate_type": (
            "X_mm_plus_circular_Y_unwrapped_arc_mm"
        ),
        "laser_image": str(laser_path),
        "camera_image": str(camera_path),
        "laser_image_size_px": [int(laser_w), int(laser_h)],
        "camera_image_size_px": [int(camera_w), int(camera_h)],
        "camera_roi_center_x_px": float(
            (camera_w - 1) / 2.0
        ),
        "camera_roi_center_is_x0_mm": True,
        "output_dir": str(output_dir),

        "transform_direction": (
            "camera_Xmm_unwrappedArcMm_to_"
            "laser_Xmm_baseArcMm"
        ),
        "fit_method": fit_method,
        "final_matrix_uses_inliers_only": bool(
            REFIT_FINAL_MATRIX_WITH_INLIERS_ONLY
        ),

        # PRIMARY reusable matrix.
        "camera_physical_to_laser_physical_affine_2x3": (
            matrix_mm.tolist()
        ),
        "camera_mm_to_laser_mm_affine_2x3": (
            matrix_mm.tolist()
        ),
        "affine_matrix_2x3": matrix_mm.tolist(),
        "affine": {
            "camera_physical_to_laser_physical_affine_2x3": (
                matrix_mm.tolist()
            )
        },

        # Debug-only branch-specific pixel matrices. There is intentionally no
        # single ordinary camera-px -> laser-px matrix because circular Y has a
        # 0/360 discontinuity.
        "current_images_debug_camera_branch_px_to_laser_px": (
            branch_json
        ),

        "coordinate_system": {
            "x": {
                "camera_definition": (
                    "stitched_roi_horizontal_center_is_X0_mm"
                ),
                "camera_x_mm_per_pixel": float(
                    CAMERA_X_MM_PER_PIXEL
                ),
                "camera_center_x_px_current_image": float(
                    (camera_w - 1) / 2.0
                ),
                "laser_definition": (
                    "real_laser_x_axis_mm_native_physical_coordinate"
                ),
                "laser_x_source": str(
                    laser_x_info["source"]
                ),
                "laser_x_axis_csv": str(
                    laser_x_info["x_axis_csv"]
                ),
                "laser_x_min_mm": float(
                    laser_x_info["x_min_mm"]
                ),
                "laser_x_max_mm": float(
                    laser_x_info["x_max_mm"]
                ),
            },

            "circular_y": {
                "period_deg": float(CIRCULAR_PERIOD_DEG),
                "wheel_diameter_mm": float(WHEEL_DIAMETER_MM),
                "circumference_mm": float(CIRCUMFERENCE_MM),
                "camera_pixel_to_angle": (
                    "theta_deg = y_px / current_camera_height_px * 360"
                ),
                "laser_pixel_to_angle": (
                    "theta_deg = y_px / current_laser_height_px * 360"
                ),
                "unwrap_rule": (
                    "camera_theta_unwrapped = camera_theta_raw + "
                    "round((laser_theta_raw-camera_theta_raw)/360)*360"
                ),
                "final_wrap_rule": (
                    "theta_final_deg = theta_predicted_deg mod 360"
                ),
                "affine_second_coordinate": (
                    "unwrapped_arc_mm = theta_unwrapped/360*circumference_mm"
                ),
            },
        },

        "camera_points_px": camera_pts_px.tolist(),
        "laser_points_px": laser_pts_px.tolist(),
        "camera_points_physical": camera_physical.tolist(),
        "laser_points_physical": laser_physical.tolist(),
        "camera_theta_raw_deg": np.asarray(
            circular_meta["camera_theta_raw_deg"],
            dtype=float,
        ).tolist(),
        "camera_theta_unwrapped_deg": np.asarray(
            circular_meta["camera_theta_unwrapped_deg"],
            dtype=float,
        ).tolist(),
        "camera_wrap_branch_k": np.asarray(
            circular_meta["camera_wrap_branch_k"],
            dtype=int,
        ).tolist(),
        "laser_theta_raw_deg": np.asarray(
            circular_meta["laser_theta_raw_deg"],
            dtype=float,
        ).tolist(),

        "predicted_laser_points_physical": (
            predicted_all_physical.tolist()
        ),
        "predicted_laser_points_px_wrapped": (
            predicted_all_px.tolist()
        ),

        "residual_error_mm": residuals_all_mm.tolist(),
        "rmse_mm": inlier_rmse_mm,
        "max_error_mm": inlier_max_error_mm,
        "all_point_rmse_mm": all_rmse_mm,
        "all_point_max_error_mm": all_max_error_mm,
        "inlier_count": inlier_count,
        "inlier_indices_1based": (
            inlier_indices_1based.astype(int).tolist()
        ),
        "ransac_inlier_mask_all_points": (
            inliers.reshape(-1).astype(int).tolist()
        ),
        "ransac_reproj_threshold_mm": float(
            RANSAC_REPROJ_THRESHOLD_MM
        ),

        "reuse_conditions": {
            "different_wheel_width_is_allowed": True,
            "different_camera_pixel_width_is_allowed": True,
            "different_camera_pixel_height_is_allowed": True,
            "different_laser_pixel_height_is_allowed": True,
            "camera_and_laser_mounting_must_be_unchanged": True,
            "camera_roi_center_must_stay_at_same_physical_X": True,
            "laser_x_axis_definition_must_be_unchanged": True,
            "wheel_diameter_for_this_arc_mm_calibration_must_match": True,
            "note": (
                "The 0/360 seam is handled automatically. "
                "If wheel diameter changes, use the same theta-unwrapped logic "
                "but either refit in X-mm/theta-deg coordinates or update the "
                "downstream physical conversion consistently."
            ),
        },
    }

    for name in (
        "affine_calibration.json",
        "affine_calibration_physical_circular.json",
    ):
        (output_dir / name).write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    readme = f"""Circular physical affine calibration

Primary transform:
  CAMERA [X_mm, unwrapped_arc_mm] -> LASER [X_mm, base_arc_mm]

X:
  Camera ROI horizontal center = X=0 mm.
  Laser X comes from the real laser_x_axis_mm coordinate.

Circular Y:
  theta_camera_raw = camera_y_px / camera_height * 360
  theta_laser_raw  = laser_y_px  / laser_height  * 360

  For each corresponding pair:
    theta_camera_unwrapped
      = theta_camera_raw
      + round((theta_laser_raw-theta_camera_raw)/360)*360

Example:
  laser point = 350 deg
  camera point = 5 deg
  camera point is automatically changed to 365 deg for fitting.

The affine uses arc-mm rather than raw degrees:
  arc_mm = theta_unwrapped / 360 * circumference_mm

After transform:
  theta_predicted = arc_predicted / circumference * 360
  theta_final = theta_predicted mod 360

Quality:
  inlier RMSE = {inlier_rmse_mm:.6f} mm
  inlier max  = {inlier_max_error_mm:.6f} mm
  all RMSE    = {all_rmse_mm:.6f} mm
  all max     = {all_max_error_mm:.6f} mm

Important:
  A point near the laser bottom and its matching point near the camera top can
  now be clicked normally in the same point order. Do NOT manually move/copy
  the point to the other end of either image.

Downstream fusion:
  The fusion program must use the saved circular metadata and apply modulo 360
  after the physical affine. Do not use one ordinary pixel affine across the
  0/360 seam.
"""
    (output_dir / "readme.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print()
    print("Saved circular physical affine calibration:")
    print(f"  {output_dir / 'affine_calibration.json'}")
    print(
        f"  {output_dir / 'affine_calibration_physical_circular.json'}"
    )
    print(
        f"  {output_dir / 'affine_points_physical_circular.csv'}"
    )
    print(
        f"  {output_dir / 'laser_camera_affine_overlay_with_points.png'}"
    )
    print()
    print(
        "IMPORTANT: downstream fusion must wrap predicted theta with "
        "theta % 360 instead of applying one global pixel affine."
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click corresponding cone centers; use ROI-center X=0 and automatic circular 0/360 Y unwrap; compute physical affine.")
    parser.add_argument("--laser", default=LASER_IMAGE_PATH, help="laser height PNG path")
    parser.add_argument("--camera", default=CAMERA_IMAGE_PATH, help="camera stitched image path")
    parser.add_argument("--output-root", default=OUTPUT_ROOT, help="output root folder")
    parser.add_argument("--run-name", default=OUTPUT_RUN_NAME, help="output run folder name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    laser_path = resolve_path(args.laser, Path(LASER_RESTORED_IMAGE)).resolve()
    camera_path = resolve_path(args.camera, Path(CAMERA_STITCHED_IMAGE)).resolve()
    output_dir = make_output_dir(args.output_root, args.run_name).resolve()

    print("Cone affine calibration - ROI CENTER X=0 + CIRCULAR Y AUTO-UNWRAP")
    print(f"Laser image : {laser_path}")
    print(f"Camera image: {camera_path}")
    print(f"Output dir  : {output_dir}")
    print()
    print("Click rule: same cone order. Top/bottom seam crossings are handled automatically by +/-360 degree unwrap.")
    print("Controls: left click add, right click/U undo, +/- zoom, WASD pan, Enter finish.")
    print()

    run_calibration(laser_path, camera_path, output_dir)


if __name__ == "__main__":
    main()
