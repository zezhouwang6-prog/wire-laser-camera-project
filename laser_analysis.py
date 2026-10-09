from __future__ import annotations

"""
laser_analysis.py
=================

Final convergence / quality-controlled version

Laser-only wire-tip analysis for the brush wheel.

This version is designed specifically for the current project and addresses the
main problems found in the previous result:

1) BOTH X and Y are resampled to the same physical scale:
       1 px = 0.05473705 mm
   Therefore a ~16 mm x ~471.19 mm laser map becomes approximately:
       W ~= 293 px
       H ~= 8609 px
   instead of the old 299 x 8802 grid.

2) Tip segmentation is no longer based on one fixed foreground threshold.
   Workflow:
       residual_mm
       -> NaN-aware smoothing
       -> local maxima (tip seeds)
       -> loose foreground support
       -> marker-controlled watershed
       -> per-seed RELATIVE-HEIGHT region growth/trimming
   The final tip region is therefore defined relative to the height of its own
   peak, which is more suitable for polished wire tips with unequal heights.

3) Touching tips are separated by marker-controlled watershed.

4) 3-D plane fitting requires substantially more valid height points by default
   (20 points), preventing unstable very-large tilt angles from tiny samples.

5) Spatial density is corrected by the ACTUALLY VALID LASER OBSERVATION AREA:
       corrected density = detected tips / valid laser area
   It no longer assumes that every cell has the same usable laser coverage.

6) Circularity is calculated consistently from contour area and contour
   perimeter in physical coordinates.  The old mixed pixel-area/perimeter
   calculation is removed.

7) The program no longer deletes a wire merely because its fitted ellipse falls
   outside one hard geometry range.  Instead it records a `shape_quality` flag.
   This makes it possible to see exactly why a tip is questionable rather than
   silently losing large numbers of tips.

8) Every major processing stage is written to:
       processing_stage_summary.csv
       rejection_reason_summary.csv

9) Small-fragment convergence filter is tightened to 7 pixels based on the latest run.
10) A publication-quality tilt subset requires at least 50 valid height points.
11) Publication-oriented statistics are exported automatically.
12) The program is linked to project config.py.
   Normally you only change:
       RUN_NAME
       COMBINED_RUN_NAME
   in config.py and then run:
       python laser_analysis.py

Input
-----
The program automatically reads:
    config.LASER_HEIGHT_MAP_NPZ

If that name is not present in config.py, it automatically constructs:
    config.LASER_RESTORED_DIR / "laser_height_mm_height_map.npz"

Output
------
The default output folder is:
    current active run / laser_tip_analysis

Main output files
-----------------
wire_tip_geometry.csv
wire_tip_geometry_valid.csv
wire_tip_geometry_shape_ok.csv
wire_tip_instance_map.npz
wire_tip_overlay.png
residual_true_scale.png
spatial_density_grid.csv
spatial_density_heatmap.png
processing_stage_summary.csv
rejection_reason_summary.csv
analysis_summary.csv
analysis_config.json

For before/after grinding comparison
------------------------------------
Use EXACTLY THE SAME parameters for both datasets.

Recommended comparison metrics:
    area_mm2
    major_axis_mm
    minor_axis_mm
    aspect_ratio
    circularity
    height_range_p95_p05_mm
    tilt_deg
    detrended_range_p95_p05_mm
    detrended_rms_mm
    nearest_neighbor_mm
    density_corrected_wires_per_valid_mm2
"""

import argparse
import csv
import json
import math
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


# =============================================================================
# PROJECT PATH LINKAGE
# =============================================================================

try:
    import config as project_config

    CONFIG_LINK_ACTIVE = True

    PROJECT_ROOT = Path(
        getattr(project_config, "PROJECT_ROOT", Path(__file__).resolve().parent.parent)
    )

    RUN_NAME = str(
        getattr(project_config, "RUN_NAME", "")
    )

    COMBINED_RUN_NAME = str(
        getattr(project_config, "COMBINED_RUN_NAME", "")
    )

    USE_COMBINED_ACQUISITION = bool(
        getattr(project_config, "USE_COMBINED_ACQUISITION", False)
    )

    ACTIVE_COMBINED_RUN_DIR = getattr(
        project_config,
        "ACTIVE_COMBINED_RUN_DIR",
        None,
    )

    LASER_RESTORED_DIR = Path(
        getattr(
            project_config,
            "LASER_RESTORED_DIR",
            Path(__file__).resolve().parent / "laser" / "huanyuan_result",
        )
    )

    LASER_HEIGHT_MAP_NPZ = Path(
        getattr(
            project_config,
            "LASER_HEIGHT_MAP_NPZ",
            LASER_RESTORED_DIR / "laser_height_mm_height_map.npz",
        )
    )

    if hasattr(project_config, "LASER_TIP_ANALYSIS_DIR"):
        LASER_TIP_ANALYSIS_DIR = Path(
            project_config.LASER_TIP_ANALYSIS_DIR
        )
    else:
        if USE_COMBINED_ACQUISITION and ACTIVE_COMBINED_RUN_DIR is not None:
            LASER_TIP_ANALYSIS_DIR = (
                Path(ACTIVE_COMBINED_RUN_DIR) / "laser_tip_analysis"
            )
        else:
            run_dir = Path(
                getattr(
                    project_config,
                    "RUN_DIR",
                    Path(__file__).resolve().parent,
                )
            )
            LASER_TIP_ANALYSIS_DIR = run_dir / "laser_tip_analysis"

except Exception as config_exc:
    CONFIG_LINK_ACTIVE = False
    CONFIG_IMPORT_ERROR = repr(config_exc)

    _HERE = Path(__file__).resolve().parent

    PROJECT_ROOT = _HERE.parent
    RUN_NAME = ""
    COMBINED_RUN_NAME = ""
    USE_COMBINED_ACQUISITION = False
    ACTIVE_COMBINED_RUN_DIR = None

    LASER_RESTORED_DIR = _HERE / "laser" / "huanyuan_result"
    LASER_HEIGHT_MAP_NPZ = (
        LASER_RESTORED_DIR / "laser_height_mm_height_map.npz"
    )
    LASER_TIP_ANALYSIS_DIR = _HERE / "laser_tip_analysis"


# =============================================================================
# FIXED PHYSICAL SCALE
# =============================================================================

# IMPORTANT:
# This is the unified physical scale currently used for the project.
# X and Y are BOTH resampled to this value.
UNIFIED_MM_PER_PIXEL = 0.05473705

REFERENCE_WIRE_COUNT = 7088  # reporting only; never forces detections


# =============================================================================
# SEGMENTATION SETTINGS
# =============================================================================

# NaN-aware Gaussian smoothing on the true-scale residual map.
SMOOTH_SIGMA_PX = 0.80

# A seed must be a local maximum above this residual value.
SEED_MIN_RESIDUAL_MM = 0.16

# Local maximum suppression size.
PEAK_MIN_DISTANCE_PX = 5

# Watershed is allowed to explore a loose support region above this absolute
# residual.  This is intentionally lower than the seed threshold.
GROW_ABSOLUTE_FLOOR_MM = 0.012

# After watershed, every basin is trimmed according to its OWN peak height:
#     local threshold = max(
#         GROW_ABSOLUTE_FLOOR_MM,
#         peak_height * GROW_RELATIVE_PEAK_RATIO
#     )
#
# Lower -> larger tip regions.
# Higher -> smaller tip regions concentrated near the top.
GROW_RELATIVE_PEAK_RATIO = 0.32

# Minimum smoothed-valid support created by normalized Gaussian filtering.
SMOOTH_VALID_WEIGHT_MIN = 0.12

# Very small/huge segmented islands are obvious noise and can be rejected
# before geometry calculation.
MIN_SEGMENT_PIXELS = 7
MAX_SEGMENT_PIXELS = 450

# Optional one-pixel closing after adaptive trimming.
APPLY_REGION_CLOSING = True


# =============================================================================
# SHAPE QUALITY LIMITS
# =============================================================================

# IMPORTANT:
# These DO NOT delete the tip.  They only set `shape_quality`.
# This prevents the old 6611 -> 4754 silent loss.

MIN_AREA_MM2 = 0.008
MAX_AREA_MM2 = 1.50

MIN_MAJOR_AXIS_MM = 0.08
MAX_MAJOR_AXIS_MM = 2.40

MIN_MINOR_AXIS_MM = 0.03
MAX_MINOR_AXIS_MM = 1.90

MAX_ASPECT_RATIO = 6.0


# =============================================================================
# 3-D HEIGHT QUALITY
# =============================================================================

# Previous result showed unstable tilt when only 5-10 valid points were used.
# Require at least 20 points for 3-D plane fitting.
MIN_VALID_HEIGHT_POINTS = 20
MIN_VALID_HEIGHT_FRACTION = 0.25

HEIGHT_LOW_PERCENTILE = 5.0
HEIGHT_HIGH_PERCENTILE = 95.0

PLANE_FIT_MAX_ITER = 4
PLANE_FIT_MAD_MULTIPLIER = 3.5
PLANE_FIT_MIN_POINTS = 20

# For publication-quality tilt statistics, use a stricter subset.
TILT_HIGH_QUALITY_MIN_VALID_POINTS = 50


# =============================================================================
# 0 / 360 DEGREE SEAM
# =============================================================================

MERGE_PERIODIC_Y_SEAM = True
PERIODIC_SEAM_BAND_PX = 12

PERIODIC_MAX_X_DISTANCE_MM = 0.65
PERIODIC_MAX_Y_DISTANCE_MM = 0.90


# =============================================================================
# SPATIAL DENSITY
# =============================================================================

DENSITY_X_BINS = 10
DENSITY_Y_BINS = 30

# A grid cell with almost no valid laser area is not suitable for density
# comparison.  It is still written to CSV, but the corrected density is NaN.
MIN_DENSITY_VALID_COVERAGE_FRACTION = 0.10


# =============================================================================
# OUTPUT
# =============================================================================

SAVE_TRUE_SCALE_ARRAYS = True
SAVE_INSTANCE_NPZ = True
SAVE_INSTANCE_UINT16_PNG = True
SAVE_OVERLAY = True
SAVE_RESIDUAL_PREVIEW = True
SAVE_DENSITY_HEATMAP = True

OVERLAY_LINE_WIDTH = 1

# Save publication-oriented summary tables.
SAVE_PAPER_STATISTICS = True


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Laser-only wire-tip segmentation and 3-D geometry analysis."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=LASER_HEIGHT_MAP_NPZ,
        help=(
            "Laser height-map NPZ. Default follows config.py automatically."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=LASER_TIP_ANALYSIS_DIR,
    )

    parser.add_argument(
        "--mm-per-pixel",
        type=float,
        default=UNIFIED_MM_PER_PIXEL,
        help=(
            "Unified X/Y physical sampling. "
            "Current project value is 0.05473705 mm/px."
        ),
    )

    parser.add_argument(
        "--reference-wire-count",
        type=int,
        default=REFERENCE_WIRE_COUNT,
    )

    parser.add_argument(
        "--smooth-sigma-px",
        type=float,
        default=SMOOTH_SIGMA_PX,
    )

    parser.add_argument(
        "--seed-min-residual-mm",
        type=float,
        default=SEED_MIN_RESIDUAL_MM,
    )

    parser.add_argument(
        "--peak-min-distance-px",
        type=int,
        default=PEAK_MIN_DISTANCE_PX,
    )

    parser.add_argument(
        "--grow-absolute-floor-mm",
        type=float,
        default=GROW_ABSOLUTE_FLOOR_MM,
    )

    parser.add_argument(
        "--grow-relative-peak-ratio",
        type=float,
        default=GROW_RELATIVE_PEAK_RATIO,
    )

    parser.add_argument(
        "--min-segment-pixels",
        type=int,
        default=MIN_SEGMENT_PIXELS,
    )

    parser.add_argument(
        "--max-segment-pixels",
        type=int,
        default=MAX_SEGMENT_PIXELS,
    )

    parser.add_argument(
        "--min-valid-height-points",
        type=int,
        default=MIN_VALID_HEIGHT_POINTS,
    )

    parser.add_argument(
        "--min-valid-height-fraction",
        type=float,
        default=MIN_VALID_HEIGHT_FRACTION,
    )

    parser.add_argument(
        "--density-x-bins",
        type=int,
        default=DENSITY_X_BINS,
    )

    parser.add_argument(
        "--density-y-bins",
        type=int,
        default=DENSITY_Y_BINS,
    )

    parser.add_argument(
        "--no-periodic-seam-merge",
        action="store_true",
    )

    return parser.parse_args()


# =============================================================================
# NPZ INPUT
# =============================================================================

def npz_scalar(
    data: np.lib.npyio.NpzFile,
    name: str,
    default: float,
) -> float:
    if name not in data.files:
        return float(default)

    try:
        value = np.asarray(data[name]).reshape(-1)[0]
        return float(value)
    except Exception:
        return float(default)


def load_laser_npz(
    path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict,
]:
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            "Laser height-map file was not found.\n"
            f"Resolved input:\n  {path}\n\n"
            "The analysis program follows config.py.\n"
            "Please check:\n"
            "  RUN_NAME\n"
            "  USE_COMBINED_ACQUISITION\n"
            "  COMBINED_RUN_NAME\n\n"
            "Expected restored file:\n"
            f"  {LASER_HEIGHT_MAP_NPZ}\n"
        )

    data = np.load(
        str(path),
        allow_pickle=True,
    )

    if "height_mm" not in data.files:
        raise KeyError(
            f"{path} does not contain 'height_mm'"
        )

    height = np.asarray(
        data["height_mm"],
        dtype=np.float32,
    )

    if height.ndim != 2:
        raise ValueError(
            f"height_mm must be 2-D, got {height.shape}"
        )

    rows, cols = height.shape

    if "residual_mm" in data.files:
        residual = np.asarray(
            data["residual_mm"],
            dtype=np.float32,
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=RuntimeWarning,
            )
            row_median = np.nanmedian(
                height,
                axis=1,
                keepdims=True,
            )
        residual = height - row_median

    if residual.shape != height.shape:
        raise ValueError(
            "residual_mm and height_mm shapes differ"
        )

    if "x_axis_mm" in data.files:
        x_axis = np.asarray(
            data["x_axis_mm"],
            dtype=np.float64,
        ).reshape(-1)
    else:
        x_axis = np.arange(
            cols,
            dtype=np.float64,
        )

    if "arc_axis_mm" in data.files:
        y_axis = np.asarray(
            data["arc_axis_mm"],
            dtype=np.float64,
        ).reshape(-1)
    else:
        y_axis = np.arange(
            rows,
            dtype=np.float64,
        )

    if x_axis.size != cols:
        raise ValueError(
            f"x_axis_mm has {x_axis.size} values, expected {cols}"
        )

    if y_axis.size != rows:
        raise ValueError(
            f"arc_axis_mm has {y_axis.size} values, expected {rows}"
        )

    if "angle_axis_deg" in data.files:
        angle_axis = np.asarray(
            data["angle_axis_deg"],
            dtype=np.float64,
        ).reshape(-1)
    else:
        y_span = max(
            1e-12,
            float(y_axis[-1] - y_axis[0]),
        )
        angle_axis = (
            (y_axis - y_axis[0])
            / y_span
            * 360.0
        )

    if angle_axis.size != rows:
        angle_axis = np.linspace(
            0.0,
            360.0,
            rows,
            endpoint=False,
            dtype=np.float64,
        )

    metadata = {
        "wheel_diameter_mm": npz_scalar(
            data,
            "wheel_diameter_mm",
            float("nan"),
        ),
        "source_true_scale_mm_per_pixel": npz_scalar(
            data,
            "true_scale_mm_per_pixel",
            float("nan"),
        ),
    }

    data.close()

    return (
        height,
        residual,
        x_axis,
        y_axis,
        angle_axis,
        metadata,
    )


# =============================================================================
# TRUE-SCALE 2-D RESAMPLING
# =============================================================================

def build_uniform_axis(
    axis: np.ndarray,
    step_mm: float,
) -> np.ndarray:
    axis = np.asarray(
        axis,
        dtype=np.float64,
    )

    start = float(axis[0])
    stop = float(axis[-1])

    count = (
        int(
            math.floor(
                (stop - start)
                / float(step_mm)
                + 1e-12
            )
        )
        + 1
    )

    count = max(
        2,
        count,
    )

    target = (
        start
        + np.arange(
            count,
            dtype=np.float64,
        )
        * float(step_mm)
    )

    return target


def aggregate_axis_nanmedian(
    array: np.ndarray,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    step_mm: float,
    axis: int,
) -> np.ndarray:
    """
    Physical bin aggregation using NaN-median.

    axis=1 -> aggregate image columns (X)
    axis=0 -> aggregate image rows    (Y)
    """
    source_axis = np.asarray(
        source_axis,
        dtype=np.float64,
    )
    target_axis = np.asarray(
        target_axis,
        dtype=np.float64,
    )

    edges = np.empty(
        len(target_axis) + 1,
        dtype=np.float64,
    )

    edges[1:-1] = 0.5 * (
        target_axis[:-1]
        + target_axis[1:]
    )

    edges[0] = (
        target_axis[0]
        - 0.5 * float(step_mm)
    )

    edges[-1] = (
        target_axis[-1]
        + 0.5 * float(step_mm)
    )

    bin_id = (
        np.digitize(
            source_axis,
            edges,
        )
        - 1
    )

    if axis == 1:
        out = np.full(
            (
                array.shape[0],
                len(target_axis),
            ),
            np.nan,
            dtype=np.float32,
        )
    elif axis == 0:
        out = np.full(
            (
                len(target_axis),
                array.shape[1],
            ),
            np.nan,
            dtype=np.float32,
        )
    else:
        raise ValueError(
            "axis must be 0 or 1"
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )

        for i in range(
            len(target_axis)
        ):
            src = np.flatnonzero(
                bin_id == i
            )

            if src.size == 0:
                continue

            if axis == 1:
                block = array[:, src]
                out[:, i] = np.nanmedian(
                    block,
                    axis=1,
                )
            else:
                block = array[src, :]
                out[i, :] = np.nanmedian(
                    block,
                    axis=0,
                )

    return out


def resample_to_unified_grid(
    height_raw: np.ndarray,
    residual_raw: np.ndarray,
    x_raw_mm: np.ndarray,
    y_raw_mm: np.ndarray,
    step_mm: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    X and Y are BOTH resampled to the SAME physical step.
    """
    x_target = build_uniform_axis(
        x_raw_mm,
        step_mm,
    )

    y_target = build_uniform_axis(
        y_raw_mm,
        step_mm,
    )

    print(
        f"Unified target grid: "
        f"W={len(x_target)}, H={len(y_target)}, "
        f"dx=dy={step_mm:.8f} mm/px"
    )

    print(
        "  resampling X..."
    )

    height_x = aggregate_axis_nanmedian(
        height_raw,
        x_raw_mm,
        x_target,
        step_mm,
        axis=1,
    )

    residual_x = aggregate_axis_nanmedian(
        residual_raw,
        x_raw_mm,
        x_target,
        step_mm,
        axis=1,
    )

    print(
        "  resampling Y..."
    )

    height_xy = aggregate_axis_nanmedian(
        height_x,
        y_raw_mm,
        y_target,
        step_mm,
        axis=0,
    )

    residual_xy = aggregate_axis_nanmedian(
        residual_x,
        y_raw_mm,
        y_target,
        step_mm,
        axis=0,
    )

    return (
        height_xy,
        residual_xy,
        x_target,
        y_target,
    )


# =============================================================================
# NAN-AWARE SMOOTHING
# =============================================================================

def normalized_gaussian(
    array: np.ndarray,
    sigma_px: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    valid = np.isfinite(
        array
    )

    values = np.where(
        valid,
        array,
        0.0,
    ).astype(
        np.float32
    )

    weights = valid.astype(
        np.float32
    )

    numerator = ndi.gaussian_filter(
        values,
        sigma=float(sigma_px),
        mode="nearest",
    )

    denominator = ndi.gaussian_filter(
        weights,
        sigma=float(sigma_px),
        mode="nearest",
    )

    smoothed = np.full(
        array.shape,
        np.nan,
        dtype=np.float32,
    )

    good = denominator > 1e-8

    smoothed[good] = (
        numerator[good]
        / denominator[good]
    )

    return (
        smoothed,
        denominator,
    )


# =============================================================================
# SEED DETECTION
# =============================================================================

def detect_peak_markers(
    smoothed: np.ndarray,
    valid_weight: np.ndarray,
    seed_min_mm: float,
    peak_min_distance_px: int,
) -> tuple[
    np.ndarray,
    dict[int, float],
]:
    size = max(
        3,
        int(peak_min_distance_px),
    )

    if size % 2 == 0:
        size += 1

    finite = (
        np.isfinite(smoothed)
        & (
            valid_weight
            >= float(
                SMOOTH_VALID_WEIGHT_MIN
            )
        )
    )

    safe = np.where(
        finite,
        smoothed,
        -np.inf,
    )

    local_max = ndi.maximum_filter(
        safe,
        size=(size, size),
        mode="nearest",
    )

    peak_pixels = (
        finite
        & (
            safe
            >= float(seed_min_mm)
        )
        & (
            safe
            >= local_max - 1e-12
        )
    )

    plateau_labels, _ = ndi.label(
        peak_pixels,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    markers = np.zeros(
        smoothed.shape,
        dtype=np.int32,
    )

    seed_peak = {}

    next_id = 1

    objects = ndi.find_objects(
        plateau_labels
    )

    for plateau_id, sl in enumerate(
        objects,
        start=1,
    ):
        if sl is None:
            continue

        local_plateau = (
            plateau_labels[sl]
            == plateau_id
        )

        coords = np.argwhere(
            local_plateau
        )

        if coords.size == 0:
            continue

        values = smoothed[sl][
            local_plateau
        ]

        best = int(
            np.nanargmax(values)
        )

        yy, xx = coords[best]

        gy = (
            sl[0].start
            + int(yy)
        )
        gx = (
            sl[1].start
            + int(xx)
        )

        markers[
            gy,
            gx,
        ] = next_id

        seed_peak[
            next_id
        ] = float(
            smoothed[gy, gx]
        )

        next_id += 1

    return (
        markers,
        seed_peak,
    )


# =============================================================================
# LOOSE WATERSHED SUPPORT
# =============================================================================

def marker_watershed_loose_support(
    smoothed: np.ndarray,
    valid_weight: np.ndarray,
    markers: np.ndarray,
    grow_absolute_floor_mm: float,
) -> np.ndarray:
    support = (
        np.isfinite(smoothed)
        & (
            valid_weight
            >= float(
                SMOOTH_VALID_WEIGHT_MIN
            )
        )
        & (
            smoothed
            >= float(
                grow_absolute_floor_mm
            )
        )
    )

    components, _ = ndi.label(
        support,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    seed_components = np.unique(
        components[
            markers > 0
        ]
    )

    seed_components = seed_components[
        seed_components > 0
    ]

    if seed_components.size == 0:
        return np.zeros_like(
            markers,
            dtype=np.int32,
        )

    usable = np.isin(
        components,
        seed_components,
    )

    finite_values = smoothed[
        usable
    ]

    if finite_values.size == 0:
        return np.zeros_like(
            markers,
            dtype=np.int32,
        )

    high = float(
        np.percentile(
            finite_values,
            99.7,
        )
    )

    high = max(
        high,
        float(
            grow_absolute_floor_mm
        )
        + 1e-8,
    )

    denom = max(
        1e-9,
        high
        - float(
            grow_absolute_floor_mm
        ),
    )

    safe = np.nan_to_num(
        smoothed,
        nan=float(
            grow_absolute_floor_mm
        )
        - 1.0,
    )

    cost = (
        (
            high
            - safe
        )
        / denom
        * 254.0
    )

    cost = np.clip(
        cost,
        0,
        254,
    ).astype(
        np.uint8
    )

    cost[
        ~usable
    ] = 255

    labels = ndi.watershed_ift(
        cost,
        markers.astype(
            np.int32
        ),
    ).astype(
        np.int32
    )

    labels[
        ~usable
    ] = 0

    return labels


# =============================================================================
# ADAPTIVE PEAK-RELATIVE REGION TRIMMING
# =============================================================================

def adaptive_trim_basins(
    watershed_labels: np.ndarray,
    markers: np.ndarray,
    seed_peak: dict[int, float],
    smoothed: np.ndarray,
    grow_absolute_floor_mm: float,
    relative_peak_ratio: float,
) -> np.ndarray:
    """
    Each watershed basin receives its OWN threshold.

    final threshold for seed i:
        max(
            absolute_floor,
            peak_i * relative_peak_ratio
        )

    The connected piece containing the seed is kept.
    """
    out = np.zeros_like(
        watershed_labels,
        dtype=np.int32,
    )

    objects = ndi.find_objects(
        watershed_labels
    )

    seed_positions = {}

    ys, xs = np.where(
        markers > 0
    )

    for y, x in zip(
        ys,
        xs,
    ):
        seed_positions[
            int(
                markers[y, x]
            )
        ] = (
            int(y),
            int(x),
        )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    for label_id, sl in enumerate(
        objects,
        start=1,
    ):
        if sl is None:
            continue

        if label_id not in seed_positions:
            continue

        peak = float(
            seed_peak.get(
                label_id,
                float("nan"),
            )
        )

        if not np.isfinite(
            peak
        ):
            continue

        threshold = max(
            float(
                grow_absolute_floor_mm
            ),
            peak
            * float(
                relative_peak_ratio
            ),
        )

        basin = (
            watershed_labels[sl]
            == label_id
        )

        local_smoothed = smoothed[
            sl
        ]

        candidate = (
            basin
            & np.isfinite(
                local_smoothed
            )
            & (
                local_smoothed
                >= threshold
            )
        )

        if not np.any(
            candidate
        ):
            continue

        if APPLY_REGION_CLOSING:
            candidate_u8 = (
                candidate.astype(
                    np.uint8
                )
                * 255
            )

            candidate_u8 = cv2.morphologyEx(
                candidate_u8,
                cv2.MORPH_CLOSE,
                closing_kernel,
            )

            candidate = (
                candidate_u8
                > 0
            ) & basin

        local_cc, _ = ndi.label(
            candidate,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8,
            ),
        )

        seed_y, seed_x = (
            seed_positions[
                label_id
            ]
        )

        ly = (
            seed_y
            - sl[0].start
        )
        lx = (
            seed_x
            - sl[1].start
        )

        if (
            ly < 0
            or lx < 0
            or ly
            >= local_cc.shape[0]
            or lx
            >= local_cc.shape[1]
        ):
            continue

        selected_cc = int(
            local_cc[
                ly,
                lx,
            ]
        )

        if selected_cc <= 0:
            # Seed may lie on a very narrow pixel removed by closing/threshold.
            # Use the closest candidate component.
            coords = np.argwhere(
                local_cc > 0
            )

            if coords.size == 0:
                continue

            d2 = (
                (
                    coords[:, 0]
                    - ly
                )
                ** 2
                + (
                    coords[:, 1]
                    - lx
                )
                ** 2
            )

            best = int(
                np.argmin(
                    d2
                )
            )

            by, bx = coords[
                best
            ]

            selected_cc = int(
                local_cc[
                    by,
                    bx,
                ]
            )

        keep = (
            local_cc
            == selected_cc
        )

        out[
            sl
        ][
            keep
        ] = label_id

    return out


# =============================================================================
# SIMPLE SEGMENT SIZE FILTER
# =============================================================================

def filter_segment_pixel_count(
    labels: np.ndarray,
    min_pixels: int,
    max_pixels: int,
) -> tuple[
    np.ndarray,
    dict[str, int],
]:
    counts = np.bincount(
        labels.ravel()
    )

    stats = {
        "too_small": 0,
        "too_large": 0,
    }

    keep = np.zeros(
        len(counts),
        dtype=bool,
    )

    for label_id in range(
        1,
        len(counts),
    ):
        count = int(
            counts[label_id]
        )

        if count < int(
            min_pixels
        ):
            stats[
                "too_small"
            ] += 1
            continue

        if count > int(
            max_pixels
        ):
            stats[
                "too_large"
            ] += 1
            continue

        keep[
            label_id
        ] = True

    out = labels.copy()

    if len(keep):
        invalid_pixel = (
            labels > 0
        ) & (
            ~keep[
                labels
            ]
        )

        out[
            invalid_pixel
        ] = 0

    return (
        out,
        stats,
    )


# =============================================================================
# PERIODIC SEAM MERGE
# =============================================================================

def periodic_distance(
    a: float,
    b: float,
    period: float,
) -> float:
    d = abs(
        float(a)
        - float(b)
    )

    return min(
        d,
        max(
            0.0,
            float(period)
            - d,
        ),
    )


def merge_periodic_seam(
    labels: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    step_mm: float,
) -> tuple[
    np.ndarray,
    int,
]:
    if labels.max() <= 0:
        return (
            labels,
            0,
        )

    h = labels.shape[0]

    band = max(
        1,
        min(
            int(
                PERIODIC_SEAM_BAND_PX
            ),
            h // 3,
        ),
    )

    top_ids = np.unique(
        labels[
            :band,
            :
        ]
    )

    bottom_ids = np.unique(
        labels[
            h - band:,
            :
        ]
    )

    top_ids = top_ids[
        top_ids > 0
    ]

    bottom_ids = bottom_ids[
        bottom_ids > 0
    ]

    if (
        top_ids.size == 0
        or bottom_ids.size == 0
    ):
        return (
            labels,
            0,
        )

    circumference = float(
        y_axis_mm[-1]
        - y_axis_mm[0]
        + step_mm
    )

    def center_info(
        label_id: int,
    ):
        yy, xx = np.where(
            labels
            == int(
                label_id
            )
        )

        if len(xx) == 0:
            return None

        return {
            "id": int(
                label_id
            ),
            "x_mm": float(
                np.mean(
                    x_axis_mm[
                        xx
                    ]
                )
            ),
            "y_mm": float(
                np.mean(
                    y_axis_mm[
                        yy
                    ]
                )
            ),
        }

    top = [
        item
        for label_id in top_ids
        if (
            item := center_info(
                int(
                    label_id
                )
            )
        )
        is not None
    ]

    bottom = [
        item
        for label_id in bottom_ids
        if (
            item := center_info(
                int(
                    label_id
                )
            )
        )
        is not None
    ]

    pairs = []

    for a in top:
        for b in bottom:
            dx = abs(
                a[
                    "x_mm"
                ]
                - b[
                    "x_mm"
                ]
            )

            if dx > float(
                PERIODIC_MAX_X_DISTANCE_MM
            ):
                continue

            dy = periodic_distance(
                a[
                    "y_mm"
                ],
                b[
                    "y_mm"
                ],
                circumference,
            )

            if dy > float(
                PERIODIC_MAX_Y_DISTANCE_MM
            ):
                continue

            pairs.append(
                (
                    math.hypot(
                        dx,
                        dy,
                    ),
                    a[
                        "id"
                    ],
                    b[
                        "id"
                    ],
                )
            )

    pairs.sort()

    used_top = set()
    used_bottom = set()
    accepted = []

    for _, top_id, bottom_id in pairs:
        if (
            top_id
            in used_top
            or bottom_id
            in used_bottom
        ):
            continue

        used_top.add(
            top_id
        )

        used_bottom.add(
            bottom_id
        )

        accepted.append(
            (
                top_id,
                bottom_id,
            )
        )

    out = labels.copy()

    for top_id, bottom_id in accepted:
        dst = min(
            int(
                top_id
            ),
            int(
                bottom_id
            ),
        )

        src = max(
            int(
                top_id
            ),
            int(
                bottom_id
            ),
        )

        out[
            out
            == src
        ] = dst

    return (
        out,
        len(
            accepted
        ),
    )


# =============================================================================
# LABEL RENUMBER
# =============================================================================

def renumber_labels(
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    int,
]:
    ids = np.unique(
        labels
    )

    ids = ids[
        ids > 0
    ]

    if ids.size == 0:
        return (
            np.zeros_like(
                labels,
                dtype=np.int32,
            ),
            0,
        )

    lookup = np.zeros(
        int(
            ids[-1]
        )
        + 1,
        dtype=np.int32,
    )

    for new_id, old_id in enumerate(
        ids,
        start=1,
    ):
        lookup[
            int(
                old_id
            )
        ] = int(
            new_id
        )

    return (
        lookup[
            labels
        ].astype(
            np.int32
        ),
        int(
            len(
                ids
            )
        ),
    )


# =============================================================================
# PERIODIC Y UNWRAP
# =============================================================================

def unwrap_region_y(
    y_mm: np.ndarray,
    circumference_mm: float,
) -> np.ndarray:
    y = np.asarray(
        y_mm,
        dtype=np.float64,
    )

    if y.size == 0:
        return y

    if (
        float(
            np.max(
                y
            )
            - np.min(
                y
            )
        )
        <= 0.5
        * float(
            circumference_mm
        )
    ):
        return y

    out = y.copy()

    out[
        out
        < 0.5
        * float(
            circumference_mm
        )
    ] += float(
        circumference_mm
    )

    return out


# =============================================================================
# PHYSICAL CONTOUR GEOMETRY
# =============================================================================

def region_contours(
    mask: np.ndarray,
):
    contours, _ = cv2.findContours(
        mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    return contours


def contour_to_physical(
    contour: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    circumference_mm: float,
) -> np.ndarray:
    pts = contour.reshape(
        -1,
        2,
    )

    x = x_axis_mm[
        pts[
            :,
            0
        ]
    ]

    y = y_axis_mm[
        pts[
            :,
            1
        ]
    ]

    y = unwrap_region_y(
        y,
        circumference_mm,
    )

    return np.column_stack(
        (
            x,
            y,
        )
    ).astype(
        np.float32
    )


def fit_physical_ellipse(
    mask: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    circumference_mm: float,
):
    contours = region_contours(
        mask
    )

    points = []

    for contour in contours:
        if len(
            contour
        ) < 2:
            continue

        physical = contour_to_physical(
            contour,
            x_axis_mm,
            y_axis_mm,
            circumference_mm,
        )

        points.append(
            physical
        )

    if not points:
        return None

    pts = np.vstack(
        points
    )

    if len(
        pts
    ) < 3:
        return None

    cv_pts = pts.reshape(
        -1,
        1,
        2,
    )

    if len(
        pts
    ) >= 5:
        (
            (
                cx,
                cy,
            ),
            (
                axis_a,
                axis_b,
            ),
            angle,
        ) = cv2.fitEllipse(
            cv_pts
        )
    else:
        (
            (
                cx,
                cy,
            ),
            (
                axis_a,
                axis_b,
            ),
            angle,
        ) = cv2.minAreaRect(
            cv_pts
        )

    axis_a = float(
        axis_a
    )

    axis_b = float(
        axis_b
    )

    angle = float(
        angle
    )

    if axis_a >= axis_b:
        major = axis_a
        minor = axis_b
        major_angle = (
            angle
            % 180.0
        )
    else:
        major = axis_b
        minor = axis_a
        major_angle = (
            (
                angle
                + 90.0
            )
            % 180.0
        )

    cy = float(
        cy
    ) % float(
        circumference_mm
    )

    return {
        "ellipse_center_x_mm": float(
            cx
        ),
        "ellipse_center_y_mm": cy,
        "major_axis_mm": float(
            major
        ),
        "minor_axis_mm": float(
            minor
        ),
        "aspect_ratio": float(
            major
            / max(
                minor,
                1e-12,
            )
        ),
        "ellipse_angle_deg": float(
            major_angle
        ),
    }


def contour_area_perimeter_circularity(
    mask: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    circumference_mm: float,
) -> tuple[
    float,
    float,
    float,
]:
    """
    Area and perimeter are calculated from the SAME physical contour geometry.

    This removes the old inconsistency:
        pixel-count area
        + OpenCV contour perimeter
    which could generate circularity > 1.
    """
    contours = region_contours(
        mask
    )

    contour_area_mm2 = 0.0
    perimeter_mm = 0.0

    for contour in contours:
        if len(
            contour
        ) < 3:
            continue

        physical = contour_to_physical(
            contour,
            x_axis_mm,
            y_axis_mm,
            circumference_mm,
        )

        cv_pts = physical.reshape(
            -1,
            1,
            2,
        )

        contour_area_mm2 += abs(
            float(
                cv2.contourArea(
                    cv_pts
                )
            )
        )

        perimeter_mm += float(
            cv2.arcLength(
                cv_pts,
                True,
            )
        )

    if (
        contour_area_mm2 <= 0
        or perimeter_mm <= 0
    ):
        return (
            float(
                contour_area_mm2
            ),
            float(
                perimeter_mm
            ),
            float(
                "nan"
            ),
        )

    raw = (
        4.0
        * math.pi
        * contour_area_mm2
        / (
            perimeter_mm
            * perimeter_mm
        )
    )

    # Discretisation on very tiny contours can still yield a value microscopically
    # above one.  Preserve the raw value for diagnostics but use the physically
    # valid clipped value as circularity.
    circularity = float(
        np.clip(
            raw,
            0.0,
            1.0,
        )
    )

    return (
        float(
            contour_area_mm2
        ),
        float(
            perimeter_mm
        ),
        circularity,
    )


def physical_solidity(
    mask: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    circumference_mm: float,
) -> float:
    contours = region_contours(
        mask
    )

    points = []

    for contour in contours:
        if len(
            contour
        ) < 3:
            continue

        points.append(
            contour_to_physical(
                contour,
                x_axis_mm,
                y_axis_mm,
                circumference_mm,
            )
        )

    if not points:
        return float(
            "nan"
        )

    pts = np.vstack(
        points
    ).reshape(
        -1,
        1,
        2,
    )

    hull = cv2.convexHull(
        pts
    )

    hull_area = abs(
        float(
            cv2.contourArea(
                hull
            )
        )
    )

    contour_area = 0.0

    for contour in contours:
        if len(
            contour
        ) < 3:
            continue

        physical = contour_to_physical(
            contour,
            x_axis_mm,
            y_axis_mm,
            circumference_mm,
        )

        contour_area += abs(
            float(
                cv2.contourArea(
                    physical.reshape(
                        -1,
                        1,
                        2,
                    )
                )
            )
        )

    if hull_area <= 0:
        return float(
            "nan"
        )

    return float(
        np.clip(
            contour_area
            / hull_area,
            0.0,
            1.0,
        )
    )


# =============================================================================
# ROBUST PLANE FIT
# =============================================================================

def robust_plane_fit(
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    z_mm: np.ndarray,
):
    x = np.asarray(
        x_mm,
        dtype=np.float64,
    )

    y = np.asarray(
        y_mm,
        dtype=np.float64,
    )

    z = np.asarray(
        z_mm,
        dtype=np.float64,
    )

    finite = (
        np.isfinite(
            x
        )
        & np.isfinite(
            y
        )
        & np.isfinite(
            z
        )
    )

    x = x[
        finite
    ]

    y = y[
        finite
    ]

    z = z[
        finite
    ]

    if len(
        z
    ) < int(
        PLANE_FIT_MIN_POINTS
    ):
        return None

    x0 = float(
        np.mean(
            x
        )
    )

    y0 = float(
        np.mean(
            y
        )
    )

    xc = (
        x
        - x0
    )

    yc = (
        y
        - y0
    )

    keep = np.ones(
        len(
            z
        ),
        dtype=bool,
    )

    coeff = None

    for _ in range(
        int(
            PLANE_FIT_MAX_ITER
        )
    ):
        if np.count_nonzero(
            keep
        ) < int(
            PLANE_FIT_MIN_POINTS
        ):
            return None

        A = np.column_stack(
            (
                xc[
                    keep
                ],
                yc[
                    keep
                ],
                np.ones(
                    np.count_nonzero(
                        keep
                    ),
                    dtype=np.float64,
                ),
            )
        )

        coeff, *_ = np.linalg.lstsq(
            A,
            z[
                keep
            ],
            rcond=None,
        )

        predicted = (
            coeff[
                0
            ]
            * xc
            + coeff[
                1
            ]
            * yc
            + coeff[
                2
            ]
        )

        residual = (
            z
            - predicted
        )

        center = float(
            np.median(
                residual[
                    keep
                ]
            )
        )

        mad = float(
            np.median(
                np.abs(
                    residual[
                        keep
                    ]
                    - center
                )
            )
        )

        if mad <= 1e-12:
            break

        sigma = (
            1.4826
            * mad
        )

        new_keep = (
            np.abs(
                residual
                - center
            )
            <= float(
                PLANE_FIT_MAD_MULTIPLIER
            )
            * sigma
        )

        if np.array_equal(
            new_keep,
            keep,
        ):
            break

        keep = new_keep

    if (
        coeff is None
        or np.count_nonzero(
            keep
        )
        < int(
            PLANE_FIT_MIN_POINTS
        )
    ):
        return None

    A = np.column_stack(
        (
            xc[
                keep
            ],
            yc[
                keep
            ],
            np.ones(
                np.count_nonzero(
                    keep
                ),
                dtype=np.float64,
            ),
        )
    )

    coeff, *_ = np.linalg.lstsq(
        A,
        z[
            keep
        ],
        rcond=None,
    )

    predicted_all = (
        coeff[
            0
        ]
        * xc
        + coeff[
            1
        ]
        * yc
        + coeff[
            2
        ]
    )

    detrended = (
        z
        - predicted_all
    )

    a = float(
        coeff[
            0
        ]
    )

    b = float(
        coeff[
            1
        ]
    )

    tilt_deg = math.degrees(
        math.atan(
            math.hypot(
                a,
                b,
            )
        )
    )

    azimuth_deg = (
        math.degrees(
            math.atan2(
                b,
                a,
            )
        )
        % 360.0
    )

    return {
        "plane_a_dz_dx": a,
        "plane_b_dz_dy": b,
        "tilt_deg": float(
            tilt_deg
        ),
        "tilt_azimuth_deg": float(
            azimuth_deg
        ),
        "fit_inlier_count": int(
            np.count_nonzero(
                keep
            )
        ),
        "detrended_mm": detrended,
    }


# =============================================================================
# SHAPE QUALITY CLASSIFICATION
# =============================================================================

def classify_shape_quality(
    area_mm2: float,
    ellipse,
) -> tuple[
    str,
    str,
]:
    reasons = []

    if not (
        float(
            MIN_AREA_MM2
        )
        <= area_mm2
        <= float(
            MAX_AREA_MM2
        )
    ):
        reasons.append(
            "area_out_of_range"
        )

    if ellipse is None:
        reasons.append(
            "ellipse_fit_failed"
        )

        return (
            "warning",
            ";".join(
                reasons
            ),
        )

    major = float(
        ellipse[
            "major_axis_mm"
        ]
    )

    minor = float(
        ellipse[
            "minor_axis_mm"
        ]
    )

    aspect = float(
        ellipse[
            "aspect_ratio"
        ]
    )

    if not (
        float(
            MIN_MAJOR_AXIS_MM
        )
        <= major
        <= float(
            MAX_MAJOR_AXIS_MM
        )
    ):
        reasons.append(
            "major_axis_out_of_range"
        )

    if not (
        float(
            MIN_MINOR_AXIS_MM
        )
        <= minor
        <= float(
            MAX_MINOR_AXIS_MM
        )
    ):
        reasons.append(
            "minor_axis_out_of_range"
        )

    if aspect > float(
        MAX_ASPECT_RATIO
    ):
        reasons.append(
            "aspect_ratio_too_large"
        )

    if reasons:
        return (
            "warning",
            ";".join(
                reasons
            ),
        )

    return (
        "ok",
        "",
    )


# =============================================================================
# PER-TIP GEOMETRY
# =============================================================================

def analyze_one_tip(
    wire_id: int,
    labels: np.ndarray,
    height_mm: np.ndarray,
    residual_mm: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    step_mm: float,
    circumference_mm: float,
    min_valid_height_points: int,
    min_valid_height_fraction: float,
) -> dict:
    yy, xx = np.where(
        labels
        == int(
            wire_id
        )
    )

    pixel_count = int(
        len(
            xx
        )
    )

    mask = np.zeros(
        labels.shape,
        dtype=np.uint8,
    )

    mask[
        yy,
        xx,
    ] = 1

    pixel_area_mm2 = (
        float(
            step_mm
        )
        * float(
            step_mm
        )
    )

    area_mm2 = (
        pixel_count
        * pixel_area_mm2
    )

    x_values = x_axis_mm[
        xx
    ]

    y_values = y_axis_mm[
        yy
    ]

    y_unwrapped = unwrap_region_y(
        y_values,
        circumference_mm,
    )

    center_x_mm = float(
        np.mean(
            x_values
        )
    )

    center_y_mm = (
        float(
            np.mean(
                y_unwrapped
            )
        )
        % float(
            circumference_mm
        )
    )

    theta_deg = (
        center_y_mm
        / max(
            float(
                circumference_mm
            ),
            1e-12,
        )
        * 360.0
    )

    ellipse = fit_physical_ellipse(
        mask,
        x_axis_mm,
        y_axis_mm,
        circumference_mm,
    )

    shape_quality, shape_reason = classify_shape_quality(
        area_mm2,
        ellipse,
    )

    (
        contour_area_mm2,
        perimeter_mm,
        circularity,
    ) = contour_area_perimeter_circularity(
        mask,
        x_axis_mm,
        y_axis_mm,
        circumference_mm,
    )

    solidity = physical_solidity(
        mask,
        x_axis_mm,
        y_axis_mm,
        circumference_mm,
    )

    if ellipse is None:
        major_axis_mm = float(
            "nan"
        )
        minor_axis_mm = float(
            "nan"
        )
        aspect_ratio = float(
            "nan"
        )
        ellipse_angle_deg = float(
            "nan"
        )
    else:
        major_axis_mm = float(
            ellipse[
                "major_axis_mm"
            ]
        )
        minor_axis_mm = float(
            ellipse[
                "minor_axis_mm"
            ]
        )
        aspect_ratio = float(
            ellipse[
                "aspect_ratio"
            ]
        )
        ellipse_angle_deg = float(
            ellipse[
                "ellipse_angle_deg"
            ]
        )

    z_all = height_mm[
        yy,
        xx,
    ]

    r_all = residual_mm[
        yy,
        xx,
    ]

    finite_height = np.isfinite(
        z_all
    )

    valid_height_count = int(
        np.count_nonzero(
            finite_height
        )
    )

    valid_height_fraction = float(
        valid_height_count
        / max(
            pixel_count,
            1,
        )
    )

    enough_height = (
        valid_height_count
        >= int(
            min_valid_height_points
        )
        and valid_height_fraction
        >= float(
            min_valid_height_fraction
        )
    )

    height_p05 = float(
        "nan"
    )
    height_median = float(
        "nan"
    )
    height_p95 = float(
        "nan"
    )
    height_range = float(
        "nan"
    )

    residual_median = float(
        "nan"
    )
    residual_p95 = float(
        "nan"
    )

    plane_a = float(
        "nan"
    )
    plane_b = float(
        "nan"
    )
    tilt_deg = float(
        "nan"
    )
    tilt_azimuth_deg = float(
        "nan"
    )
    plane_inliers = 0

    detrended_p05 = float(
        "nan"
    )
    detrended_median = float(
        "nan"
    )
    detrended_p95 = float(
        "nan"
    )
    detrended_range = float(
        "nan"
    )
    detrended_rms = float(
        "nan"
    )

    geometry_quality = (
        "insufficient_valid_height"
    )

    if enough_height:
        z = z_all[
            finite_height
        ].astype(
            np.float64
        )

        xv = x_values[
            finite_height
        ].astype(
            np.float64
        )

        yv = y_unwrapped[
            finite_height
        ].astype(
            np.float64
        )

        height_p05 = float(
            np.percentile(
                z,
                HEIGHT_LOW_PERCENTILE,
            )
        )

        height_median = float(
            np.median(
                z
            )
        )

        height_p95 = float(
            np.percentile(
                z,
                HEIGHT_HIGH_PERCENTILE,
            )
        )

        height_range = (
            height_p95
            - height_p05
        )

        rv_mask = (
            finite_height
            & np.isfinite(
                r_all
            )
        )

        rv = r_all[
            rv_mask
        ]

        if rv.size:
            residual_median = float(
                np.median(
                    rv
                )
            )

            residual_p95 = float(
                np.percentile(
                    rv,
                    HEIGHT_HIGH_PERCENTILE,
                )
            )

        plane = robust_plane_fit(
            xv,
            yv,
            z,
        )

        if plane is None:
            geometry_quality = (
                "plane_fit_failed"
            )
        else:
            geometry_quality = "ok"

            plane_a = float(
                plane[
                    "plane_a_dz_dx"
                ]
            )

            plane_b = float(
                plane[
                    "plane_b_dz_dy"
                ]
            )

            tilt_deg = float(
                plane[
                    "tilt_deg"
                ]
            )

            tilt_azimuth_deg = float(
                plane[
                    "tilt_azimuth_deg"
                ]
            )

            plane_inliers = int(
                plane[
                    "fit_inlier_count"
                ]
            )

            detrended = np.asarray(
                plane[
                    "detrended_mm"
                ],
                dtype=np.float64,
            )

            detrended_p05 = float(
                np.percentile(
                    detrended,
                    HEIGHT_LOW_PERCENTILE,
                )
            )

            detrended_median = float(
                np.median(
                    detrended
                )
            )

            detrended_p95 = float(
                np.percentile(
                    detrended,
                    HEIGHT_HIGH_PERCENTILE,
                )
            )

            detrended_range = (
                detrended_p95
                - detrended_p05
            )

            detrended_rms = float(
                np.sqrt(
                    np.mean(
                        detrended
                        * detrended
                    )
                )
            )

    return {
        "wire_id": int(
            wire_id
        ),

        "center_x_mm": center_x_mm,
        "center_y_mm": center_y_mm,
        "theta_deg": float(
            theta_deg
        ),

        "segment_pixel_count": pixel_count,
        "pixel_area_mm2": float(
            area_mm2
        ),
        "contour_area_mm2": float(
            contour_area_mm2
        ),
        "perimeter_mm": float(
            perimeter_mm
        ),

        "equivalent_diameter_mm": float(
            2.0
            * math.sqrt(
                max(
                    area_mm2,
                    0.0,
                )
                / math.pi
            )
        ),

        "major_axis_mm": major_axis_mm,
        "minor_axis_mm": minor_axis_mm,
        "aspect_ratio": aspect_ratio,
        "ellipse_angle_deg": ellipse_angle_deg,

        "circularity": float(
            circularity
        ),
        "solidity": float(
            solidity
        ),

        "shape_quality": shape_quality,
        "shape_warning_reason": shape_reason,

        "valid_height_count": valid_height_count,
        "valid_height_fraction": valid_height_fraction,
        "geometry_quality": geometry_quality,

        "height_p05_mm": height_p05,
        "height_median_mm": height_median,
        "height_p95_mm": height_p95,
        "height_range_p95_p05_mm": height_range,

        "residual_median_mm": residual_median,
        "residual_p95_mm": residual_p95,

        "plane_a_dz_dx": plane_a,
        "plane_b_dz_dy": plane_b,
        "tilt_deg": tilt_deg,
        "tilt_azimuth_deg": tilt_azimuth_deg,
        "plane_fit_inlier_count": plane_inliers,

        "detrended_p05_mm": detrended_p05,
        "detrended_median_mm": detrended_median,
        "detrended_p95_mm": detrended_p95,
        "detrended_range_p95_p05_mm": detrended_range,
        "detrended_rms_mm": detrended_rms,

        "nearest_neighbor_mm": float(
            "nan"
        ),
    }


# =============================================================================
# PERIODIC NEAREST NEIGHBOUR
# =============================================================================

def attach_periodic_nearest_neighbor(
    records: list[dict],
    circumference_mm: float,
) -> None:
    if len(
        records
    ) < 2:
        return

    points = np.asarray(
        [
            (
                float(
                    record[
                        "center_x_mm"
                    ]
                ),
                float(
                    record[
                        "center_y_mm"
                    ]
                ),
            )
            for record in records
        ],
        dtype=np.float64,
    )

    n = len(
        points
    )

    L = float(
        circumference_mm
    )

    augmented = np.vstack(
        (
            points
            + np.array(
                [
                    0.0,
                    -L,
                ]
            ),
            points,
            points
            + np.array(
                [
                    0.0,
                    L,
                ]
            ),
        )
    )

    source_id = np.tile(
        np.arange(
            n,
            dtype=np.int32,
        ),
        3,
    )

    tree = cKDTree(
        augmented
    )

    k = min(
        10,
        len(
            augmented
        ),
    )

    distances, indices = tree.query(
        points,
        k=k,
    )

    if k == 1:
        distances = distances[
            :,
            None,
        ]

        indices = indices[
            :,
            None,
        ]

    for i in range(
        n
    ):
        best = float(
            "nan"
        )

        for distance, index in zip(
            np.atleast_1d(
                distances[
                    i
                ]
            ),
            np.atleast_1d(
                indices[
                    i
                ]
            ),
        ):
            if int(
                source_id[
                    int(
                        index
                    )
                ]
            ) == i:
                continue

            best = float(
                distance
            )

            break

        records[
            i
        ][
            "nearest_neighbor_mm"
        ] = best


# =============================================================================
# CSV FIELDS
# =============================================================================

GEOMETRY_FIELDS = [
    "wire_id",

    "center_x_mm",
    "center_y_mm",
    "theta_deg",

    "segment_pixel_count",
    "pixel_area_mm2",
    "contour_area_mm2",
    "perimeter_mm",
    "equivalent_diameter_mm",

    "major_axis_mm",
    "minor_axis_mm",
    "aspect_ratio",
    "ellipse_angle_deg",
    "circularity",
    "solidity",

    "shape_quality",
    "shape_warning_reason",

    "valid_height_count",
    "valid_height_fraction",
    "geometry_quality",

    "height_p05_mm",
    "height_median_mm",
    "height_p95_mm",
    "height_range_p95_p05_mm",

    "residual_median_mm",
    "residual_p95_mm",

    "plane_a_dz_dx",
    "plane_b_dz_dy",
    "tilt_deg",
    "tilt_azimuth_deg",
    "plane_fit_inlier_count",

    "detrended_p05_mm",
    "detrended_median_mm",
    "detrended_p95_mm",
    "detrended_range_p95_p05_mm",
    "detrended_rms_mm",

    "nearest_neighbor_mm",
]


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def save_records_csv(
    path: Path,
    records: list[dict],
) -> None:
    with Path(
        path
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=GEOMETRY_FIELDS,
            extrasaction="ignore",
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                record
            )


def residual_preview_u8(
    residual_mm: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(
        residual_mm
    )

    if not np.any(
        finite
    ):
        return np.zeros(
            residual_mm.shape,
            dtype=np.uint8,
        )

    values = residual_mm[
        finite
    ]

    low = float(
        np.percentile(
            values,
            2.0,
        )
    )

    high = float(
        np.percentile(
            values,
            98.5,
        )
    )

    if high <= low + 1e-12:
        high = (
            low
            + 1.0
        )

    gray = (
        (
            np.nan_to_num(
                residual_mm,
                nan=low,
            )
            - low
        )
        / (
            high
            - low
        )
        * 255.0
    )

    return np.clip(
        gray,
        0,
        255,
    ).astype(
        np.uint8
    )


def save_residual_preview(
    path: Path,
    residual_mm: np.ndarray,
) -> None:
    gray = residual_preview_u8(
        residual_mm
    )

    color = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )

    cv2.imwrite(
        str(
            path
        ),
        color,
    )


def save_overlay(
    path: Path,
    residual_mm: np.ndarray,
    labels: np.ndarray,
    records: list[dict],
) -> None:
    base = cv2.applyColorMap(
        residual_preview_u8(
            residual_mm
        ),
        cv2.COLORMAP_TURBO,
    )

    record_map = {
        int(
            record[
                "wire_id"
            ]
        ): record
        for record in records
    }

    for wire_id, record in record_map.items():
        mask = (
            labels
            == int(
                wire_id
            )
        ).astype(
            np.uint8
        )

        contours = region_contours(
            mask
        )

        if (
            record[
                "shape_quality"
            ]
            != "ok"
        ):
            color = (
                255,
                0,
                255,
            )  # magenta: shape warning

        elif (
            record[
                "geometry_quality"
            ]
            == "ok"
        ):
            color = (
                255,
                255,
                255,
            )  # white: valid 3-D geometry

        else:
            color = (
                0,
                255,
                255,
            )  # yellow: insufficient / failed 3-D geometry

        if contours:
            cv2.drawContours(
                base,
                contours,
                -1,
                color,
                int(
                    OVERLAY_LINE_WIDTH
                ),
                cv2.LINE_AA,
            )

    cv2.imwrite(
        str(
            path
        ),
        base,
    )


# =============================================================================
# VALID-AREA-CORRECTED DENSITY
# =============================================================================

def build_density_grid(
    records: list[dict],
    valid_observation_mask: np.ndarray,
    x_axis_mm: np.ndarray,
    y_axis_mm: np.ndarray,
    step_mm: float,
    x_bins: int,
    y_bins: int,
):
    x_bins = max(
        1,
        int(
            x_bins
        ),
    )

    y_bins = max(
        1,
        int(
            y_bins
        ),
    )

    x_edges = np.linspace(
        float(
            x_axis_mm[
                0
            ]
        ),
        float(
            x_axis_mm[
                -1
            ]
            + step_mm
        ),
        x_bins + 1,
    )

    y_edges = np.linspace(
        float(
            y_axis_mm[
                0
            ]
        ),
        float(
            y_axis_mm[
                -1
            ]
            + step_mm
        ),
        y_bins + 1,
    )

    if records:
        center_x = np.asarray(
            [
                float(
                    record[
                        "center_x_mm"
                    ]
                )
                for record in records
            ],
            dtype=np.float64,
        )

        center_y = np.asarray(
            [
                float(
                    record[
                        "center_y_mm"
                    ]
                )
                for record in records
            ],
            dtype=np.float64,
        )

        counts, _, _ = np.histogram2d(
            center_y,
            center_x,
            bins=(
                y_edges,
                x_edges,
            ),
        )
    else:
        counts = np.zeros(
            (
                y_bins,
                x_bins,
            ),
            dtype=np.float64,
        )

    # ---------------------------------------------------------
    # Calculate valid laser area cell by cell from the true map.
    # ---------------------------------------------------------
    x_index = np.digitize(
        x_axis_mm,
        x_edges,
    ) - 1

    y_index = np.digitize(
        y_axis_mm,
        y_edges,
    ) - 1

    valid_pixel_counts = np.zeros(
        (
            y_bins,
            x_bins,
        ),
        dtype=np.int64,
    )

    total_pixel_counts = np.zeros(
        (
            y_bins,
            x_bins,
        ),
        dtype=np.int64,
    )

    for iy in range(
        y_bins
    ):
        row_ids = np.flatnonzero(
            y_index
            == iy
        )

        if row_ids.size == 0:
            continue

        for ix in range(
            x_bins
        ):
            col_ids = np.flatnonzero(
                x_index
                == ix
            )

            if col_ids.size == 0:
                continue

            block = valid_observation_mask[
                np.ix_(
                    row_ids,
                    col_ids,
                )
            ]

            total_pixel_counts[
                iy,
                ix,
            ] = int(
                block.size
            )

            valid_pixel_counts[
                iy,
                ix,
            ] = int(
                np.count_nonzero(
                    block
                )
            )

    pixel_area_mm2 = (
        float(
            step_mm
        )
        * float(
            step_mm
        )
    )

    valid_area_mm2 = (
        valid_pixel_counts.astype(
            np.float64
        )
        * pixel_area_mm2
    )

    nominal_area_mm2 = (
        total_pixel_counts.astype(
            np.float64
        )
        * pixel_area_mm2
    )

    coverage = np.divide(
        valid_area_mm2,
        nominal_area_mm2,
        out=np.zeros_like(
            valid_area_mm2,
        ),
        where=(
            nominal_area_mm2
            > 0
        ),
    )

    corrected_density = np.full(
        counts.shape,
        np.nan,
        dtype=np.float64,
    )

    usable = (
        valid_area_mm2
        > 0
    ) & (
        coverage
        >= float(
            MIN_DENSITY_VALID_COVERAGE_FRACTION
        )
    )

    corrected_density[
        usable
    ] = (
        counts[
            usable
        ]
        / valid_area_mm2[
            usable
        ]
    )

    raw_nominal_density = np.divide(
        counts,
        nominal_area_mm2,
        out=np.full_like(
            counts,
            np.nan,
            dtype=np.float64,
        ),
        where=(
            nominal_area_mm2
            > 0
        ),
    )

    return {
        "counts": counts,
        "valid_pixel_counts": valid_pixel_counts,
        "total_pixel_counts": total_pixel_counts,
        "valid_area_mm2": valid_area_mm2,
        "nominal_area_mm2": nominal_area_mm2,
        "valid_coverage_fraction": coverage,
        "density_corrected": corrected_density,
        "density_raw_nominal": raw_nominal_density,
        "x_edges": x_edges,
        "y_edges": y_edges,
    }


def save_density_csv(
    path: Path,
    density: dict,
) -> None:
    fields = [
        "y_bin",
        "x_bin",
        "x_min_mm",
        "x_max_mm",
        "y_min_mm",
        "y_max_mm",

        "wire_count",

        "valid_pixel_count",
        "total_pixel_count",

        "valid_area_mm2",
        "nominal_area_mm2",
        "valid_coverage_fraction",

        "density_corrected_wires_per_valid_mm2",
        "density_raw_wires_per_nominal_mm2",
    ]

    counts = density[
        "counts"
    ]

    with Path(
        path
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for iy in range(
            counts.shape[
                0
            ]
        ):
            for ix in range(
                counts.shape[
                    1
                ]
            ):
                writer.writerow(
                    {
                        "y_bin": iy,
                        "x_bin": ix,

                        "x_min_mm": float(
                            density[
                                "x_edges"
                            ][
                                ix
                            ]
                        ),
                        "x_max_mm": float(
                            density[
                                "x_edges"
                            ][
                                ix
                                + 1
                            ]
                        ),
                        "y_min_mm": float(
                            density[
                                "y_edges"
                            ][
                                iy
                            ]
                        ),
                        "y_max_mm": float(
                            density[
                                "y_edges"
                            ][
                                iy
                                + 1
                            ]
                        ),

                        "wire_count": int(
                            counts[
                                iy,
                                ix,
                            ]
                        ),

                        "valid_pixel_count": int(
                            density[
                                "valid_pixel_counts"
                            ][
                                iy,
                                ix,
                            ]
                        ),

                        "total_pixel_count": int(
                            density[
                                "total_pixel_counts"
                            ][
                                iy,
                                ix,
                            ]
                        ),

                        "valid_area_mm2": float(
                            density[
                                "valid_area_mm2"
                            ][
                                iy,
                                ix,
                            ]
                        ),

                        "nominal_area_mm2": float(
                            density[
                                "nominal_area_mm2"
                            ][
                                iy,
                                ix,
                            ]
                        ),

                        "valid_coverage_fraction": float(
                            density[
                                "valid_coverage_fraction"
                            ][
                                iy,
                                ix,
                            ]
                        ),

                        "density_corrected_wires_per_valid_mm2": float(
                            density[
                                "density_corrected"
                            ][
                                iy,
                                ix,
                            ]
                        ),

                        "density_raw_wires_per_nominal_mm2": float(
                            density[
                                "density_raw_nominal"
                            ][
                                iy,
                                ix,
                            ]
                        ),
                    }
                )


def save_density_heatmap(
    path: Path,
    corrected_density: np.ndarray,
) -> None:
    values = np.asarray(
        corrected_density,
        dtype=np.float32,
    )

    finite = np.isfinite(
        values
    )

    gray = np.zeros(
        values.shape,
        dtype=np.uint8,
    )

    if np.any(
        finite
    ):
        high = float(
            np.percentile(
                values[
                    finite
                ],
                98.0,
            )
        )

        if high > 1e-12:
            gray[
                finite
            ] = np.clip(
                values[
                    finite
                ]
                / high
                * 255.0,
                0,
                255,
            ).astype(
                np.uint8
            )

    color = cv2.applyColorMap(
        gray,
        cv2.COLORMAP_TURBO,
    )

    color[
        ~finite
    ] = (
        0,
        0,
        0,
    )

    scale = 16

    color = cv2.resize(
        color,
        (
            color.shape[
                1
            ]
            * scale,
            color.shape[
                0
            ]
            * scale,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    cv2.imwrite(
        str(
            path
        ),
        color,
    )


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def finite_stats(
    records: list[dict],
    field: str,
) -> dict:
    values = []

    for record in records:
        if field not in record:
            continue

        try:
            value = float(
                record[
                    field
                ]
            )
        except Exception:
            continue

        if np.isfinite(
            value
        ):
            values.append(
                value
            )

    if not values:
        return {
            "count": 0,
            "mean": float(
                "nan"
            ),
            "std": float(
                "nan"
            ),
            "median": float(
                "nan"
            ),
            "p05": float(
                "nan"
            ),
            "p95": float(
                "nan"
            ),
        }

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "count": int(
            len(
                values
            )
        ),
        "mean": float(
            np.mean(
                values
            )
        ),
        "std": float(
            np.std(
                values
            )
        ),
        "median": float(
            np.median(
                values
            )
        ),
        "p05": float(
            np.percentile(
                values,
                5.0,
            )
        ),
        "p95": float(
            np.percentile(
                values,
                95.0,
            )
        ),
    }


def save_metric_value_csv(
    path: Path,
    rows: list[dict],
) -> None:
    with Path(
        path
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "value",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )



# =============================================================================
# PUBLICATION-ORIENTED STATISTICS
# =============================================================================

def save_group_statistics_csv(
    path: Path,
    groups: dict[str, list[dict]],
) -> None:
    """
    Save comparable statistics for the main geometry/distribution metrics.

    The same file structure can be produced for both the unpolished and
    polished datasets, which makes before/after comparison straightforward.
    """
    metrics = [
        "pixel_area_mm2",
        "contour_area_mm2",
        "major_axis_mm",
        "minor_axis_mm",
        "aspect_ratio",
        "circularity",
        "height_range_p95_p05_mm",
        "tilt_deg",
        "detrended_range_p95_p05_mm",
        "detrended_rms_mm",
        "nearest_neighbor_mm",
    ]

    fields = [
        "group",
        "metric",
        "count",
        "mean",
        "std",
        "median",
        "p05",
        "p25",
        "p75",
        "p95",
        "iqr",
    ]

    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for group_name, records in groups.items():
            for metric in metrics:
                values = []

                for record in records:
                    try:
                        value = float(record.get(metric, float("nan")))
                    except Exception:
                        continue

                    if np.isfinite(value):
                        values.append(value)

                if values:
                    values = np.asarray(
                        values,
                        dtype=np.float64,
                    )

                    p25 = float(np.percentile(values, 25.0))
                    p75 = float(np.percentile(values, 75.0))

                    row = {
                        "group": group_name,
                        "metric": metric,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "p05": float(np.percentile(values, 5.0)),
                        "p25": p25,
                        "p75": p75,
                        "p95": float(np.percentile(values, 95.0)),
                        "iqr": float(p75 - p25),
                    }
                else:
                    row = {
                        "group": group_name,
                        "metric": metric,
                        "count": 0,
                        "mean": float("nan"),
                        "std": float("nan"),
                        "median": float("nan"),
                        "p05": float("nan"),
                        "p25": float("nan"),
                        "p75": float("nan"),
                        "p95": float("nan"),
                        "iqr": float("nan"),
                    }

                writer.writerow(row)


def save_tilt_quality_summary_csv(
    path: Path,
    records: list[dict],
) -> None:
    """
    Quantify how tilt changes with the number of valid laser height samples.

    This directly guards against the previously observed bias where tips with
    only a few valid points produced unrealistically large tilt angles.
    """
    bins = [
        (20, 29, "20-29"),
        (30, 49, "30-49"),
        (50, 99, "50-99"),
        (100, 199, "100-199"),
        (200, None, ">=200"),
    ]

    fields = [
        "valid_height_count_group",
        "tip_count",
        "tilt_count",
        "tilt_mean_deg",
        "tilt_median_deg",
        "tilt_p05_deg",
        "tilt_p95_deg",
    ]

    with Path(path).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for lower, upper, label in bins:
            subset = []

            for record in records:
                count = int(record.get("valid_height_count", 0))

                if count < lower:
                    continue

                if upper is not None and count > upper:
                    continue

                subset.append(record)

            tilts = np.asarray(
                [
                    float(record["tilt_deg"])
                    for record in subset
                    if np.isfinite(float(record.get("tilt_deg", float("nan"))))
                ],
                dtype=np.float64,
            )

            if tilts.size:
                writer.writerow(
                    {
                        "valid_height_count_group": label,
                        "tip_count": len(subset),
                        "tilt_count": int(tilts.size),
                        "tilt_mean_deg": float(np.mean(tilts)),
                        "tilt_median_deg": float(np.median(tilts)),
                        "tilt_p05_deg": float(np.percentile(tilts, 5.0)),
                        "tilt_p95_deg": float(np.percentile(tilts, 95.0)),
                    }
                )
            else:
                writer.writerow(
                    {
                        "valid_height_count_group": label,
                        "tip_count": len(subset),
                        "tilt_count": 0,
                        "tilt_mean_deg": float("nan"),
                        "tilt_median_deg": float("nan"),
                        "tilt_p05_deg": float("nan"),
                        "tilt_p95_deg": float("nan"),
                    }
                )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    input_path = Path(
        args.input
    ).resolve()

    out_dir = Path(
        args.out_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "="
        * 78
    )
    print(
        "LASER-ONLY WIRE TIP GEOMETRY ANALYSIS V2"
    )
    print(
        "="
        * 78
    )

    if CONFIG_LINK_ACTIVE:
        print(
            "Path source: config.py"
        )
        print(
            f"  RUN_NAME                 = {RUN_NAME}"
        )
        print(
            f"  USE_COMBINED_ACQUISITION = {USE_COMBINED_ACQUISITION}"
        )
        print(
            f"  COMBINED_RUN_NAME        = {COMBINED_RUN_NAME}"
        )

        if ACTIVE_COMBINED_RUN_DIR is not None:
            print(
                f"  ACTIVE_COMBINED_RUN_DIR  = {ACTIVE_COMBINED_RUN_DIR}"
            )

    else:
        print(
            "WARNING: config.py linkage is not active"
        )
        print(
            f"  config import error = {CONFIG_IMPORT_ERROR}"
        )

    print(
        f"Input NPZ: {input_path}"
    )
    print(
        f"Output   : {out_dir}"
    )
    print(
        f"Unified scale: {args.mm_per_pixel:.8f} mm/px"
    )
    print(
        ""
    )

    (
        height_raw,
        residual_raw,
        x_raw_mm,
        y_raw_mm,
        angle_raw_deg,
        npz_metadata,
    ) = load_laser_npz(
        input_path
    )

    raw_valid_fraction = float(
        np.isfinite(
            height_raw
        ).mean()
    )

    print(
        f"Raw map: "
        f"W={height_raw.shape[1]}, "
        f"H={height_raw.shape[0]}"
    )

    print(
        f"Raw valid height fraction: "
        f"{raw_valid_fraction:.4f}"
    )

    print(
        f"Raw physical X: "
        f"{x_raw_mm[0]:.6f} .. {x_raw_mm[-1]:.6f} mm"
    )

    print(
        f"Raw physical Y: "
        f"{y_raw_mm[0]:.6f} .. {y_raw_mm[-1]:.6f} mm"
    )

    (
        height_mm,
        residual_mm,
        x_axis_mm,
        y_axis_mm,
    ) = resample_to_unified_grid(
        height_raw,
        residual_raw,
        x_raw_mm,
        y_raw_mm,
        float(
            args.mm_per_pixel
        ),
    )

    true_valid_fraction = float(
        np.isfinite(
            height_mm
        ).mean()
    )

    step_mm = float(
        args.mm_per_pixel
    )

    circumference_mm = float(
        y_axis_mm[
            -1
        ]
        - y_axis_mm[
            0
        ]
        + step_mm
    )

    print(
        f"Unified grid actual: "
        f"W={height_mm.shape[1]}, "
        f"H={height_mm.shape[0]}"
    )

    print(
        f"Unified valid fraction: "
        f"{true_valid_fraction:.4f}"
    )

    print(
        f"Unified circumference represented: "
        f"{circumference_mm:.6f} mm"
    )

    # -------------------------------------------------------------------------
    # Save unified arrays
    # -------------------------------------------------------------------------
    if SAVE_TRUE_SCALE_ARRAYS:
        np.savez_compressed(
            out_dir
            / "laser_true_scale_arrays.npz",

            height_mm=height_mm.astype(
                np.float32
            ),

            residual_mm=residual_mm.astype(
                np.float32
            ),

            x_axis_mm=x_axis_mm.astype(
                np.float32
            ),

            arc_axis_mm=y_axis_mm.astype(
                np.float32
            ),

            mm_per_pixel=np.float32(
                step_mm
            ),

            circumference_mm=np.float32(
                circumference_mm
            ),
        )

    if SAVE_RESIDUAL_PREVIEW:
        save_residual_preview(
            out_dir
            / "residual_true_scale.png",
            residual_mm,
        )

    # -------------------------------------------------------------------------
    # Smooth
    # -------------------------------------------------------------------------
    print(
        ""
    )
    print(
        "1/7 NaN-aware smoothing..."
    )

    (
        smoothed,
        valid_weight,
    ) = normalized_gaussian(
        residual_mm,
        float(
            args.smooth_sigma_px
        ),
    )

    # -------------------------------------------------------------------------
    # Seeds
    # -------------------------------------------------------------------------
    print(
        "2/7 Local peak seeds..."
    )

    (
        markers,
        seed_peak,
    ) = detect_peak_markers(
        smoothed,
        valid_weight,
        float(
            args.seed_min_residual_mm
        ),
        int(
            args.peak_min_distance_px
        ),
    )

    seed_count = int(
        markers.max()
    )

    print(
        f"  seeds = {seed_count}"
    )

    # -------------------------------------------------------------------------
    # Watershed
    # -------------------------------------------------------------------------
    print(
        "3/7 Loose marker watershed..."
    )

    watershed_labels = marker_watershed_loose_support(
        smoothed,
        valid_weight,
        markers,
        float(
            args.grow_absolute_floor_mm
        ),
    )

    watershed_count = int(
        np.unique(
            watershed_labels[
                watershed_labels
                > 0
            ]
        ).size
    )

    print(
        f"  watershed basins = {watershed_count}"
    )

    # -------------------------------------------------------------------------
    # Adaptive trim
    # -------------------------------------------------------------------------
    print(
        "4/7 Peak-relative adaptive region growth/trimming..."
    )

    adaptive_labels = adaptive_trim_basins(
        watershed_labels,
        markers,
        seed_peak,
        smoothed,
        float(
            args.grow_absolute_floor_mm
        ),
        float(
            args.grow_relative_peak_ratio
        ),
    )

    adaptive_count = int(
        np.unique(
            adaptive_labels[
                adaptive_labels
                > 0
            ]
        ).size
    )

    print(
        f"  adaptive regions = {adaptive_count}"
    )

    # -------------------------------------------------------------------------
    # Pixel count filter
    # -------------------------------------------------------------------------
    (
        labels,
        pixel_filter_stats,
    ) = filter_segment_pixel_count(
        adaptive_labels,
        int(
            args.min_segment_pixels
        ),
        int(
            args.max_segment_pixels
        ),
    )

    pixel_filtered_count = int(
        np.unique(
            labels[
                labels
                > 0
            ]
        ).size
    )

    print(
        f"  after pixel-count filter = {pixel_filtered_count}"
    )

    # -------------------------------------------------------------------------
    # Periodic seam
    # -------------------------------------------------------------------------
    seam_merge_count = 0

    if (
        MERGE_PERIODIC_Y_SEAM
        and not args.no_periodic_seam_merge
    ):
        (
            labels,
            seam_merge_count,
        ) = merge_periodic_seam(
            labels,
            x_axis_mm,
            y_axis_mm,
            step_mm,
        )

    labels, label_count = renumber_labels(
        labels
    )

    print(
        f"  periodic seam merges = {seam_merge_count}"
    )

    print(
        f"  final segmented regions = {label_count}"
    )

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------
    print(
        "5/7 Per-tip 2-D / 3-D geometry..."
    )

    records = []

    for wire_id in range(
        1,
        label_count
        + 1,
    ):
        record = analyze_one_tip(
            wire_id,
            labels,
            height_mm,
            residual_mm,
            x_axis_mm,
            y_axis_mm,
            step_mm,
            circumference_mm,
            int(
                args.min_valid_height_points
            ),
            float(
                args.min_valid_height_fraction
            ),
        )

        records.append(
            record
        )

        if wire_id % 1000 == 0:
            print(
                f"  {wire_id}/{label_count}"
            )

    attach_periodic_nearest_neighbor(
        records,
        circumference_mm,
    )

    shape_ok_records = [
        record
        for record in records
        if record[
            "shape_quality"
        ] == "ok"
    ]

    geometry_ok_records = [
        record
        for record in records
        if record[
            "geometry_quality"
        ] == "ok"
    ]

    strict_valid_records = [
        record
        for record in records
        if (
            record[
                "shape_quality"
            ]
            == "ok"
            and record[
                "geometry_quality"
            ]
            == "ok"
        )
    ]

    # Publication-quality tilt subset:
    # shape OK + 3-D geometry OK + at least 50 valid height points.
    tilt_high_quality_records = [
        record
        for record in strict_valid_records
        if int(
            record.get(
                "valid_height_count",
                0,
            )
        )
        >= int(
            TILT_HIGH_QUALITY_MIN_VALID_POINTS
        )
    ]

    # -------------------------------------------------------------------------
    # Spatial density
    # -------------------------------------------------------------------------
    print(
        "6/7 Valid-area-corrected spatial density..."
    )

    valid_observation_mask = np.isfinite(
        height_mm
    )

    # For spatial distribution use shape-OK tip centers.
    density = build_density_grid(
        shape_ok_records,
        valid_observation_mask,
        x_axis_mm,
        y_axis_mm,
        step_mm,
        int(
            args.density_x_bins
        ),
        int(
            args.density_y_bins
        ),
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    print(
        "7/7 Saving outputs..."
    )

    save_records_csv(
        out_dir
        / "wire_tip_geometry.csv",
        records,
    )

    save_records_csv(
        out_dir
        / "wire_tip_geometry_shape_ok.csv",
        shape_ok_records,
    )

    save_records_csv(
        out_dir
        / "wire_tip_geometry_valid.csv",
        strict_valid_records,
    )

    save_records_csv(
        out_dir
        / "wire_tip_geometry_tilt_high_quality.csv",
        tilt_high_quality_records,
    )

    if SAVE_INSTANCE_NPZ:
        np.savez_compressed(
            out_dir
            / "wire_tip_instance_map.npz",

            instance_map=labels.astype(
                np.int32
            ),

            wire_count=np.int32(
                len(
                    records
                )
            ),

            x_axis_mm=x_axis_mm.astype(
                np.float32
            ),

            arc_axis_mm=y_axis_mm.astype(
                np.float32
            ),

            mm_per_pixel=np.float32(
                step_mm
            ),

            circumference_mm=np.float32(
                circumference_mm
            ),
        )

    if (
        SAVE_INSTANCE_UINT16_PNG
        and label_count
        <= np.iinfo(
            np.uint16
        ).max
    ):
        cv2.imwrite(
            str(
                out_dir
                / "wire_tip_instance_id_uint16.png"
            ),
            labels.astype(
                np.uint16
            ),
        )

    if SAVE_OVERLAY:
        save_overlay(
            out_dir
            / "wire_tip_overlay.png",
            residual_mm,
            labels,
            records,
        )

    save_density_csv(
        out_dir
        / "spatial_density_grid.csv",
        density,
    )

    if SAVE_DENSITY_HEATMAP:
        save_density_heatmap(
            out_dir
            / "spatial_density_heatmap.png",
            density[
                "density_corrected"
            ],
        )

    # -------------------------------------------------------------------------
    # Processing-stage summary
    # -------------------------------------------------------------------------
    stage_rows = [
        {
            "metric": "seed_count",
            "value": seed_count,
        },
        {
            "metric": "watershed_basin_count",
            "value": watershed_count,
        },
        {
            "metric": "adaptive_region_count",
            "value": adaptive_count,
        },
        {
            "metric": "pixel_filter_too_small_removed",
            "value": pixel_filter_stats[
                "too_small"
            ],
        },
        {
            "metric": "pixel_filter_too_large_removed",
            "value": pixel_filter_stats[
                "too_large"
            ],
        },
        {
            "metric": "after_pixel_filter_count",
            "value": pixel_filtered_count,
        },
        {
            "metric": "periodic_seam_merged_pairs",
            "value": seam_merge_count,
        },
        {
            "metric": "final_segment_count",
            "value": len(
                records
            ),
        },
        {
            "metric": "shape_ok_count",
            "value": len(
                shape_ok_records
            ),
        },
        {
            "metric": "geometry_3d_ok_count",
            "value": len(
                geometry_ok_records
            ),
        },
        {
            "metric": "strict_shape_and_3d_ok_count",
            "value": len(
                strict_valid_records
            ),
        },
        {
            "metric": "tilt_high_quality_count",
            "value": len(
                tilt_high_quality_records
            ),
        },
        {
            "metric": "tilt_high_quality_min_valid_points",
            "value": int(
                TILT_HIGH_QUALITY_MIN_VALID_POINTS
            ),
        },
        {
            "metric": "reference_wire_count",
            "value": int(
                args.reference_wire_count
            ),
        },
        {
            "metric": "reference_is_reporting_only",
            "value": 1,
        },
    ]

    save_metric_value_csv(
        out_dir
        / "processing_stage_summary.csv",
        stage_rows,
    )

    # -------------------------------------------------------------------------
    # Rejection / warning reason summary
    # -------------------------------------------------------------------------
    reason_counter = Counter()

    for record in records:
        if record[
            "shape_quality"
        ] != "ok":
            reasons = (
                record[
                    "shape_warning_reason"
                ].split(
                    ";"
                )
            )

            for reason in reasons:
                if reason:
                    reason_counter[
                        "shape:"
                        + reason
                    ] += 1

        if record[
            "geometry_quality"
        ] != "ok":
            reason_counter[
                "3d:"
                + record[
                    "geometry_quality"
                ]
            ] += 1

    rejection_rows = [
        {
            "metric": reason,
            "value": count,
        }
        for reason, count in sorted(
            reason_counter.items()
        )
    ]

    save_metric_value_csv(
        out_dir
        / "rejection_reason_summary.csv",
        rejection_rows,
    )

    # -------------------------------------------------------------------------
    # Analysis summary
    # -------------------------------------------------------------------------
    summary_rows = []

    def add_summary(
        metric,
        value,
    ):
        summary_rows.append(
            {
                "metric": metric,
                "value": value,
            }
        )

    add_summary(
        "raw_width_px",
        int(
            height_raw.shape[
                1
            ]
        ),
    )

    add_summary(
        "raw_height_px",
        int(
            height_raw.shape[
                0
            ]
        ),
    )

    add_summary(
        "raw_valid_fraction",
        raw_valid_fraction,
    )

    add_summary(
        "unified_width_px",
        int(
            height_mm.shape[
                1
            ]
        ),
    )

    add_summary(
        "unified_height_px",
        int(
            height_mm.shape[
                0
            ]
        ),
    )

    add_summary(
        "unified_mm_per_pixel",
        step_mm,
    )

    add_summary(
        "unified_valid_fraction",
        true_valid_fraction,
    )

    add_summary(
        "circumference_mm",
        circumference_mm,
    )

    add_summary(
        "final_segment_count",
        len(
            records
        ),
    )

    add_summary(
        "shape_ok_count",
        len(
            shape_ok_records
        ),
    )

    add_summary(
        "3d_geometry_ok_count",
        len(
            geometry_ok_records
        ),
    )

    add_summary(
        "strict_valid_count",
        len(
            strict_valid_records
        ),
    )

    if int(
        args.reference_wire_count
    ) > 0:
        add_summary(
            "final_segment_count_vs_reference_fraction",
            len(
                records
            )
            / int(
                args.reference_wire_count
            ),
        )

        add_summary(
            "shape_ok_count_vs_reference_fraction",
            len(
                shape_ok_records
            )
            / int(
                args.reference_wire_count
            ),
        )

    key_metrics = [
        "pixel_area_mm2",
        "contour_area_mm2",
        "major_axis_mm",
        "minor_axis_mm",
        "aspect_ratio",
        "circularity",
        "height_range_p95_p05_mm",
        "tilt_deg",
        "detrended_range_p95_p05_mm",
        "detrended_rms_mm",
        "nearest_neighbor_mm",
    ]

    for field in key_metrics:
        if field in {
            "height_range_p95_p05_mm",
            "tilt_deg",
            "detrended_range_p95_p05_mm",
            "detrended_rms_mm",
        }:
            source = geometry_ok_records
        elif field in {
            "major_axis_mm",
            "minor_axis_mm",
            "aspect_ratio",
            "circularity",
        }:
            source = shape_ok_records
        else:
            source = records

        stats = finite_stats(
            source,
            field,
        )

        for stat_name, value in stats.items():
            add_summary(
                f"{field}_{stat_name}",
                value,
            )

    finite_density = density[
        "density_corrected"
    ][
        np.isfinite(
            density[
                "density_corrected"
            ]
        )
    ]

    if finite_density.size:
        add_summary(
            "corrected_density_mean_wires_per_valid_mm2",
            float(
                np.mean(
                    finite_density
                )
            ),
        )

        add_summary(
            "corrected_density_median_wires_per_valid_mm2",
            float(
                np.median(
                    finite_density
                )
            ),
        )

    save_metric_value_csv(
        out_dir
        / "analysis_summary.csv",
        summary_rows,
    )

    # -------------------------------------------------------------------------
    # Publication-oriented statistics
    # -------------------------------------------------------------------------
    if SAVE_PAPER_STATISTICS:
        save_group_statistics_csv(
            out_dir
            / "paper_geometry_statistics.csv",
            {
                "all_segments": records,
                "shape_ok": shape_ok_records,
                "strict_shape_and_3d_ok": strict_valid_records,
                "tilt_high_quality": tilt_high_quality_records,
            },
        )

        save_tilt_quality_summary_csv(
            out_dir
            / "tilt_quality_by_valid_point_count.csv",
            geometry_ok_records,
        )

    # -------------------------------------------------------------------------
    # Config JSON
    # -------------------------------------------------------------------------
    analysis_config = {
        "input": str(
            input_path
        ),

        "output_dir": str(
            out_dir
        ),

        "path_linked_to_project_config": bool(
            CONFIG_LINK_ACTIVE
        ),

        "run_name": RUN_NAME,
        "combined_run_name": COMBINED_RUN_NAME,

        "unified_mm_per_pixel": float(
            step_mm
        ),

        "expected_unified_grid": {
            "width_px": int(
                height_mm.shape[
                    1
                ]
            ),
            "height_px": int(
                height_mm.shape[
                    0
                ]
            ),
        },

        "smooth_sigma_px": float(
            args.smooth_sigma_px
        ),

        "seed_min_residual_mm": float(
            args.seed_min_residual_mm
        ),

        "peak_min_distance_px": int(
            args.peak_min_distance_px
        ),

        "grow_absolute_floor_mm": float(
            args.grow_absolute_floor_mm
        ),

        "grow_relative_peak_ratio": float(
            args.grow_relative_peak_ratio
        ),

        "min_segment_pixels": int(
            args.min_segment_pixels
        ),

        "max_segment_pixels": int(
            args.max_segment_pixels
        ),

        "min_valid_height_points": int(
            args.min_valid_height_points
        ),

        "min_valid_height_fraction": float(
            args.min_valid_height_fraction
        ),

        "plane_fit_min_points": int(
            PLANE_FIT_MIN_POINTS
        ),

        "tilt_high_quality_min_valid_points": int(
            TILT_HIGH_QUALITY_MIN_VALID_POINTS
        ),

        "shape_quality_limits": {
            "min_area_mm2": MIN_AREA_MM2,
            "max_area_mm2": MAX_AREA_MM2,
            "min_major_axis_mm": MIN_MAJOR_AXIS_MM,
            "max_major_axis_mm": MAX_MAJOR_AXIS_MM,
            "min_minor_axis_mm": MIN_MINOR_AXIS_MM,
            "max_minor_axis_mm": MAX_MINOR_AXIS_MM,
            "max_aspect_ratio": MAX_ASPECT_RATIO,
        },

        "density": {
            "x_bins": int(
                args.density_x_bins
            ),
            "y_bins": int(
                args.density_y_bins
            ),
            "valid_area_corrected": True,
            "minimum_valid_coverage_fraction": (
                MIN_DENSITY_VALID_COVERAGE_FRACTION
            ),
        },

        "reference_wire_count": int(
            args.reference_wire_count
        ),

        "reference_is_reporting_only": True,
    }

    (
        out_dir
        / "analysis_config.json"
    ).write_text(
        json.dumps(
            analysis_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Final console report
    # -------------------------------------------------------------------------
    print(
        ""
    )
    print(
        "="
        * 78
    )
    print(
        "DONE"
    )
    print(
        "="
        * 78
    )

    print(
        f"Unified grid: "
        f"{height_mm.shape[1]} x {height_mm.shape[0]} "
        f"(W x H)"
    )

    print(
        f"Scale: "
        f"{step_mm:.8f} mm/px in BOTH X and Y"
    )

    print(
        f"Seeds: {seed_count}"
    )

    print(
        f"Final segmented tips: {len(records)}"
    )

    print(
        f"Shape-OK tips: {len(shape_ok_records)}"
    )

    print(
        f"3-D geometry-OK tips: {len(geometry_ok_records)}"
    )

    print(
        f"Strict shape+3D valid tips: {len(strict_valid_records)}"
    )

    print(
        f"High-quality tilt tips "
        f"(valid height points >= {TILT_HIGH_QUALITY_MIN_VALID_POINTS}): "
        f"{len(tilt_high_quality_records)}"
    )

    print(
        f"Reference physical count: "
        f"{int(args.reference_wire_count)} "
        f"(REPORTING ONLY)"
    )

    print(
        ""
    )

    print(
        "Overlay colors:"
    )

    print(
        "  WHITE   = shape OK + 3-D geometry OK"
    )

    print(
        "  YELLOW  = shape OK but insufficient/failed 3-D fit"
    )

    print(
        "  MAGENTA = shape warning (NOT silently deleted)"
    )

    print(
        ""
    )

    print(
        "For the polished/unpolished comparison, use the SAME analysis_config.json "
        "parameters for both datasets."
    )

    print(
        "Recommended publication files:"
    )
    print(
        "  2-D tip shape        : wire_tip_geometry_shape_ok.csv"
    )
    print(
        "  3-D tip geometry     : wire_tip_geometry_valid.csv"
    )
    print(
        "  high-quality tilt    : wire_tip_geometry_tilt_high_quality.csv"
    )
    print(
        "  paper statistics     : paper_geometry_statistics.csv"
    )
    print(
        "  tilt quality check   : tilt_quality_by_valid_point_count.csv"
    )


if __name__ == "__main__":
    main()
