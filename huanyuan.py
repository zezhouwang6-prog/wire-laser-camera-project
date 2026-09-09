from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from config import LASER_RECORD_DIR, LASER_RESTORED_DIR

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ===================== USER SETTINGS =====================
# V2: accepts a shorter sync_index.csv and discards laser rows without encoder mapping.
# Change this path whenever you want to restore another data set.
# It can be a record folder, for example:
#   r"D:\project\MotorLaser\record10"
# Or it can be one height CSV file, for example:
#   r"D:\project\MotorLaser\record10\laser_height_mm.csv"
INPUT_PATH = str(LASER_RECORD_DIR)

# Leave empty to save results to "<input folder>\huanyuan_result".
# Or set your own folder, for example:
#   r"D:\project\MotorLaser\my_result"
OUTPUT_DIR = str(LASER_RESTORED_DIR)

# Output rows for the 0-360 degree height map.
#
# NEW encoder-physical-Y behavior:
#   0 = automatically use the TRUE physical circumference grid:
#
#       round(pi * WHEEL_DIAMETER_MM / TRUE_SCALE_MM_PER_PIXEL)
#
#   With the current values:
#       D = 150 mm
#       TRUE_SCALE_MM_PER_PIXEL = 0.05473705 mm/px
#   the result is 8609 rows.
#
# A positive value still overrides the output row count, but the rows are STILL
# assigned from the real encoder travelled angle rather than profile order.
ANGLE_BINS = 0

# Merge this many laser X columns into one image column.
# Use 1 when you do not want horizontal compression.
X_GROUP = 1

# Physical parameters used to write coordinate axes for restoration.
WHEEL_DIAMETER_MM = 150.0
ROTATION_DEG = 360.0
LASER_HZ = 500.0
LASER_WIDTH_MM = 16.0

# Also save full-revolution images on the unified true-scale raster.
# The current formal fusion scale is 0.05473705 mm/px.
# The old *_true_scale_detail_height.png filename is kept so downstream
# scripts continue to find the same restored laser image name.
SAVE_TRUE_SCALE_PNG = True
TRUE_SCALE_MM_PER_PIXEL = 0.05473705
TRUE_SCALE_PIXELS_PER_MM = 1.0 / TRUE_SCALE_MM_PER_PIXEL

# Extra lower-resolution scale for visual fusion with the natural camera stitch.
# 0.0504 mm/px is the empirical natural camera stitching scale.
SAVE_NATURAL_FUSION_SCALE_PNG = False
NATURAL_FUSION_MM_PER_PIXEL = 0.0504
NATURAL_FUSION_PIXELS_PER_MM = 1.0 / NATURAL_FUSION_MM_PER_PIXEL

# The laser may start before the motor actually moves. When sync_index.csv is
# present, automatically skip those stationary laser profiles.
AUTO_TRIM_STATIONARY_START = True
MANUAL_TRIM_START_ROWS = 0
START_MOVE_MIN_DEG = 0.10

# Automatically remove laser profiles collected after the motor has stopped.
# The final END_STATIONARY_WINDOW_ROWS encoder samples are treated as stationary
# when their total angle span is no larger than END_STATIONARY_MAX_SPAN_DEG.
# At 500 Hz, 100 rows correspond to 0.2 s. The current normal rotation is much
# larger than 0.05 deg in 0.2 s, while stopped duplicate rows are nearly constant.
AUTO_TRIM_STATIONARY_END = True
END_STATIONARY_WINDOW_ROWS = 100
END_STATIONARY_MAX_SPAN_DEG = 0.05
END_STATIONARY_MIN_ROWS = 50
END_KEEP_BUFFER_ROWS = 3

# Fallback when sync_index.csv cannot provide usable encoder angles.
#
# In that case the old code used every remaining CSV row as 0-360 degrees.
# If the laser continues collecting after the wheel has already stopped, the
# repeated stationary profiles are therefore stretched into the end of the
# restored circumference and appear as a long tail.
#
# This fallback detects a long, nearly unchanged suffix directly from the laser
# height profiles and removes it before row-index 0-360 mapping.
AUTO_TRIM_DUPLICATE_HEIGHT_TAIL = True

# Sample every Nth laser X value only for the duplicate-tail detector. This
# keeps the detector fast and memory-light; the actual restoration still uses
# all configured X data.
HEIGHT_TAIL_SAMPLE_COLUMN_STEP = 20

# Median absolute height change between adjacent profiles below this value is
# considered stationary. The uploaded result has a moving median near 0.008 mm
# and a stationary tail near 0-0.0008 mm, so 0.0015 mm cleanly separates them.
HEIGHT_TAIL_DIFF_THRESHOLD_MM = 0.0015

# Use a rolling median so a few noisy rows do not break the stationary suffix.
HEIGHT_TAIL_WINDOW_ROWS = 50

# Require at least this many stationary rows before trimming. At 500 Hz,
# 250 rows = 0.5 s, so normal small local texture similarities are not removed.
HEIGHT_TAIL_MIN_ROWS = 250

# Keep a few rows after the detected moving/stationary transition.
HEIGHT_TAIL_KEEP_BUFFER_ROWS = 3

# Minimum number of mutually valid sampled X points needed to compare a pair
# of adjacent profiles.
HEIGHT_TAIL_MIN_VALID_SAMPLES = 20

# Extra trimming after the first detected motor movement.
# Keep this at 0 for complete restoration. Increase it only when you
# intentionally want to discard the startup movement section.
EXTRA_TRIM_START_SECONDS = 0.0
EXTRA_TRIM_START_ROWS = 0

# With sync_index.csv, unfold by travelled angle from the first moving profile.
# This makes the first moving profile become 0 deg in the restored image.
USE_TRAVELLED_ANGLE_FROM_SYNC = True

# ---------------------------------------------------------------------
# REAL ENCODER -> PHYSICAL Y GRID
# ---------------------------------------------------------------------
# True = every retained laser profile is placed DIRECTLY onto the physical
# 0..360-degree output row determined by the real motor encoder angle.
#
# The old automatic mode did this:
#     valid profile order -> 0,1,2,... -> later resize to 8802 rows
#
# The new mode does this:
#     real travelled encoder degree -> physical row in the 8802-row circle
#
# Therefore non-uniform motor speed is preserved instead of being stretched
# uniformly over the circumference.
USE_REAL_ENCODER_DIRECT_PHYSICAL_Y = True

# When the direct physical-Y mode is enabled, silently falling back to row
# index would destroy the purpose of this correction.  Keep this True for
# formal fusion data.  If sync_index.csv / encoder feedback is unavailable,
# the program stops with a clear error.
REQUIRE_REAL_ENCODER_FOR_PHYSICAL_Y = True

# Use the same metric Y scale as the final true-scale image.
DIRECT_ENCODER_Y_MM_PER_PIXEL = TRUE_SCALE_MM_PER_PIXEL

# A measured 360-degree endpoint is the same physical seam as 0 degrees.
# The physical image uses rows [0, H-1] to represent [0, 360) and therefore
# maps an exact/near-360 sample to the final row rather than creating H+1 rows.
DIRECT_ENCODER_CLIP_TO_OPEN_360 = True

# Save a row-by-row diagnostic table:
# output row, physical angle, arc length, and number of laser profiles assigned.
SAVE_DIRECT_ENCODER_Y_GRID_CSV = True

# Angle source used from sync_index.csv:
#   "auto"     - prefer encoder feedback unless it is stuck while fusion/theory
#                continues to move.
#   "feedback" - force feedback_position_deg.
#   "fusion"   - force fusion_position_deg.
#   "theory"   - force theory_position_deg.
#   "row_index"- ignore sync angles and spread retained CSV rows uniformly.
SYNC_ANGLE_SOURCE = "auto"

# Some runs keep writing sync rows after the encoder feedback has reached the
# final angle. Do not switch away from feedback by default: theory/fusion may
# keep advancing even while the measured height profiles are already repeated.
# Use --sync-angle-source fusion/theory only when you have verified feedback is
# wrong and the laser profiles are still truly moving.
AUTO_FALLBACK_FROM_STUCK_FEEDBACK = False
FEEDBACK_STUCK_WINDOW_ROWS = 100
FEEDBACK_STUCK_MAX_SPAN_DEG = 0.05
FALLBACK_MOVING_MIN_SPAN_DEG = 0.25

# Remove very low false returns. These usually appear as dark purple points in
# the detail image. Increase this value if real valleys are removed.
FILTER_LOW_OUTLIERS_BY_ROW = True
LOW_OUTLIER_BELOW_ROW_MEDIAN_MM = 1.0

# =========================================================

INVALID_BELOW_MM = -90.0


def load_font(size: int):
    for path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def find_height_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(path)

    preferred = path / "laser_height_mm.csv"
    if preferred.exists():
        return preferred

    ignored = {
        "events.csv",
        "laser_index.csv",
        "laser_summary.csv",
        "laser_x_axis_mm.csv",
        "metadata.csv",
        "motor_feedback.csv",
        "sync_index.csv",
    }
    candidates = [
        p for p in path.glob("*.csv")
        if p.name.lower() not in ignored and p.stat().st_size > 1024 * 1024
    ]
    if not candidates:
        raise FileNotFoundError(f"No large height CSV found in {path}")
    return max(candidates, key=lambda p: p.stat().st_size)


def angular_delta_deg(values: np.ndarray, start_value: float) -> np.ndarray:
    return (values - start_value + 180.0) % 360.0 - 180.0


def final_angle_span_deg(values: np.ndarray, window_rows: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    window = int(max(2, window_rows))
    tail = finite_values[-min(window, finite_values.size):]
    return float(np.nanmax(tail) - np.nanmin(tail))


def choose_sync_angle_field(
    data: np.ndarray,
    names: tuple[str, ...],
) -> tuple[str, str, dict[str, float | int | str]]:
    requested = str(SYNC_ANGLE_SOURCE).strip().lower()
    aliases = {
        "feedback": "feedback_position_deg",
        "encoder": "feedback_position_deg",
        "fusion": "fusion_position_deg",
        "theory": "theory_position_deg",
    }

    info: dict[str, float | int | str] = {
        "sync_angle_source_setting": requested,
        "sync_angle_source_reason": "",
        "feedback_final_span_deg": float("nan"),
        "fallback_final_span_deg": float("nan"),
        "fallback_angle_field": "",
    }

    if requested != "auto":
        field = aliases.get(requested, requested)
        if field not in names:
            raise ValueError(f"sync_index.csv has no requested angle column: {field}")
        info["sync_angle_source_reason"] = "forced_by_setting"
        return field, "forced_by_setting", info

    fallback_field = ""
    for candidate in ("fusion_position_deg", "theory_position_deg"):
        if candidate in names:
            fallback_field = candidate
            break
    info["fallback_angle_field"] = fallback_field

    if "feedback_position_deg" in names:
        feedback = np.asarray(data["feedback_position_deg"], dtype=np.float64)
        feedback_span = final_angle_span_deg(feedback, FEEDBACK_STUCK_WINDOW_ROWS)
        info["feedback_final_span_deg"] = feedback_span

        if AUTO_FALLBACK_FROM_STUCK_FEEDBACK and fallback_field:
            fallback = np.asarray(data[fallback_field], dtype=np.float64)
            fallback_span = final_angle_span_deg(fallback, FEEDBACK_STUCK_WINDOW_ROWS)
            info["fallback_final_span_deg"] = fallback_span

            if (
                np.isfinite(feedback_span)
                and np.isfinite(fallback_span)
                and feedback_span <= float(FEEDBACK_STUCK_MAX_SPAN_DEG)
                and fallback_span >= float(FALLBACK_MOVING_MIN_SPAN_DEG)
            ):
                reason = "feedback_stuck_fallback_to_" + fallback_field
                info["sync_angle_source_reason"] = reason
                return fallback_field, reason, info

        info["sync_angle_source_reason"] = "auto_feedback"
        return "feedback_position_deg", "auto_feedback", info

    if fallback_field:
        reason = "auto_" + fallback_field
        info["sync_angle_source_reason"] = reason
        return fallback_field, reason, info

    raise ValueError(
        "sync_index.csv has no feedback_position_deg, fusion_position_deg, or theory_position_deg column"
    )


def extra_start_trim_rows() -> int:
    by_time = int(round(max(0.0, float(EXTRA_TRIM_START_SECONDS)) * LASER_HZ))
    by_rows = int(max(0, EXTRA_TRIM_START_ROWS))
    return by_time + by_rows


def detect_stationary_end_row(
    travelled: np.ndarray,
    finite: np.ndarray,
    start_row: int,
) -> tuple[int, int, float]:
    """Return exclusive end row after removing a stationary encoder tail.

    The function only trims when the final encoder samples form a sufficiently
    long, nearly constant-angle plateau. Therefore normal data is kept when the
    motor is still moving at the end of acquisition.
    """
    row_count = int(travelled.size)
    if not AUTO_TRIM_STATIONARY_END or row_count <= start_row:
        return row_count, 0, float("nan")

    finite_indices = np.flatnonzero(finite & (np.arange(row_count) >= start_row))
    if finite_indices.size == 0:
        return row_count, 0, float("nan")

    # Ignore trailing NaN encoder entries first.
    last_finite_exclusive = int(finite_indices[-1]) + 1
    window = int(max(2, END_STATIONARY_WINDOW_ROWS))
    min_rows = int(max(2, END_STATIONARY_MIN_ROWS))
    if last_finite_exclusive - start_row < min_rows:
        return row_count, 0, float("nan")

    final_window_start = max(start_row, last_finite_exclusive - window)
    final_values = travelled[final_window_start:last_finite_exclusive]
    final_values = final_values[np.isfinite(final_values)]
    if final_values.size < min_rows:
        return row_count, 0, float("nan")

    final_span = float(np.nanmax(final_values) - np.nanmin(final_values))
    if final_span > float(END_STATIONARY_MAX_SPAN_DEG):
        # The encoder is still moving at the end, so do not trim anything.
        return row_count, 0, final_span

    # Walk backward one row at a time to locate the beginning of the final
    # constant-angle plateau. A moving row entering the window makes its angle
    # span exceed the stationary threshold and stops the backward search.
    plateau_start = final_window_start
    while plateau_start > start_row:
        test_start = plateau_start - 1
        test_end = min(last_finite_exclusive, test_start + window)
        values = travelled[test_start:test_end]
        values = values[np.isfinite(values)]
        if values.size < min_rows:
            break
        span = float(np.nanmax(values) - np.nanmin(values))
        if span > float(END_STATIONARY_MAX_SPAN_DEG):
            break
        plateau_start = test_start

    end_row = int(min(row_count, plateau_start + max(0, END_KEEP_BUFFER_ROWS)))
    # Do not report trimming unless there is a meaningful stationary tail.
    stationary_rows = int(last_finite_exclusive - plateau_start)
    if stationary_rows < min_rows:
        return row_count, 0, final_span

    trimmed_rows = int(max(0, row_count - end_row))
    return end_row, trimmed_rows, final_span



def detect_duplicate_height_tail_from_csv(
    csv_path: Path,
    row_count: int,
    start_row: int,
) -> tuple[int, int, dict[str, float | int | str]]:
    """Detect a long stationary/repeated laser-profile suffix.

    This is only a fallback for data sets where sync_index.csv cannot provide a
    usable encoder-angle mapping. It never changes the normal encoder-based
    stationary-end detector.

    Returns
    -------
    end_row_exclusive:
        First row excluded from restoration.
    trimmed_rows:
        Number of rows removed from the end.
    info:
        Diagnostic values written to the summary.
    """
    info: dict[str, float | int | str] = {
        "height_tail_detector_used": "False",
        "height_tail_detected": "False",
        "height_tail_diff_threshold_mm": float(HEIGHT_TAIL_DIFF_THRESHOLD_MM),
        "height_tail_window_rows": int(HEIGHT_TAIL_WINDOW_ROWS),
        "height_tail_min_rows": int(HEIGHT_TAIL_MIN_ROWS),
        "height_tail_sample_column_step": int(HEIGHT_TAIL_SAMPLE_COLUMN_STEP),
        "height_tail_detected_start_row": "",
        "height_tail_final_window_median_diff_mm": float("nan"),
        "height_tail_moving_reference_median_diff_mm": float("nan"),
    }

    if (
        not AUTO_TRIM_DUPLICATE_HEIGHT_TAIL
        or row_count <= 1
        or row_count - start_row < int(HEIGHT_TAIL_MIN_ROWS)
    ):
        return int(row_count), 0, info

    info["height_tail_detector_used"] = "True"

    step = max(1, int(HEIGHT_TAIL_SAMPLE_COLUMN_STEP))
    signatures: list[np.ndarray] = []

    print("Scanning laser profiles for repeated stationary tail...")
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for line in f:
            profile = np.fromstring(line.strip(), sep=",", dtype=np.float32)
            if profile.size == 0:
                continue

            sampled = profile[::step].astype(np.float32, copy=False)
            sampled = np.where(sampled <= INVALID_BELOW_MM, np.nan, sampled)
            signatures.append(sampled.copy())

    if len(signatures) < 2:
        return int(row_count), 0, info

    # Protect against a malformed CSV count mismatch.
    usable_rows = min(int(row_count), len(signatures))
    signatures = signatures[:usable_rows]

    # Different rows should normally have the same X count. If one malformed
    # row differs, compare only the common prefix.
    diffs = np.full(usable_rows - 1, np.nan, dtype=np.float32)
    min_valid = max(4, int(HEIGHT_TAIL_MIN_VALID_SAMPLES))

    for i in range(usable_rows - 1):
        a = signatures[i]
        b = signatures[i + 1]
        common = min(a.size, b.size)
        if common <= 0:
            continue
        a = a[:common]
        b = b[:common]
        valid = np.isfinite(a) & np.isfinite(b)
        if int(np.count_nonzero(valid)) < min_valid:
            continue
        diffs[i] = float(np.nanmedian(np.abs(a[valid] - b[valid])))

    window = max(5, int(HEIGHT_TAIL_WINDOW_ROWS))
    if diffs.size < window:
        return int(row_count), 0, info

    # Diagnostic moving-reference level from the middle portion of the run.
    ref_lo = max(int(start_row), int(round(0.10 * usable_rows)))
    ref_hi = max(ref_lo + 1, int(round(0.80 * usable_rows)))
    moving_reference = diffs[ref_lo:min(ref_hi, diffs.size)]
    moving_reference = moving_reference[np.isfinite(moving_reference)]
    if moving_reference.size:
        info["height_tail_moving_reference_median_diff_mm"] = float(
            np.nanmedian(moving_reference)
        )

    final_window = diffs[max(0, diffs.size - window):]
    final_window = final_window[np.isfinite(final_window)]
    if final_window.size < max(3, window // 2):
        return int(row_count), 0, info

    final_median = float(np.nanmedian(final_window))
    info["height_tail_final_window_median_diff_mm"] = final_median

    threshold = float(HEIGHT_TAIL_DIFF_THRESHOLD_MM)
    if not np.isfinite(final_median) or final_median > threshold:
        print(
            "Height-tail detector: final profiles are still changing "
            f"(median adjacent diff={final_median:.6f} mm); no end trim."
        )
        return int(row_count), 0, info

    # Forward rolling median of adjacent-profile changes. Find the first point
    # in the final low-change region that remains stationary to the end.
    rolling = np.full(diffs.size - window + 1, np.nan, dtype=np.float32)
    for j in range(rolling.size):
        values = diffs[j:j + window]
        values = values[np.isfinite(values)]
        if values.size >= max(3, window // 2):
            rolling[j] = float(np.nanmedian(values))

    finite_roll = np.isfinite(rolling)
    low_roll = finite_roll & (rolling <= threshold)

    # Search backward from the end for the last non-stationary rolling window.
    last_bad = -1
    for j in range(rolling.size - 1, -1, -1):
        if not low_roll[j]:
            last_bad = j
            break

    if last_bad < 0:
        plateau_start = max(int(start_row), 0)
    else:
        # The first fully low-change window after the last bad window.
        plateau_start = int(last_bad + 1)

    plateau_start = max(int(start_row), min(plateau_start, usable_rows - 1))
    stationary_rows = int(usable_rows - plateau_start)

    if stationary_rows < int(HEIGHT_TAIL_MIN_ROWS):
        print(
            "Height-tail detector: low-change suffix is too short "
            f"({stationary_rows} rows); no end trim."
        )
        return int(row_count), 0, info

    end_row = int(
        min(
            usable_rows,
            plateau_start + max(0, int(HEIGHT_TAIL_KEEP_BUFFER_ROWS)),
        )
    )
    trimmed_rows = int(max(0, row_count - end_row))

    info["height_tail_detected"] = "True"
    info["height_tail_detected_start_row"] = int(plateau_start)

    print(
        "Height-tail detector: duplicate stationary suffix detected. "
        f"start_row={plateau_start}, end_row_exclusive={end_row}, "
        f"removed={trimmed_rows} rows "
        f"(~{trimmed_rows / max(LASER_HZ, 1e-9):.3f} s), "
        f"final_median_diff={final_median:.6f} mm."
    )
    return end_row, trimmed_rows, info



def direct_encoder_physical_row_count(angle_bins: int) -> int:
    """Return the output Y row count for direct real-encoder mapping.

    angle_bins > 0:
        explicit user override.

    angle_bins <= 0:
        exact physical circumference grid at DIRECT_ENCODER_Y_MM_PER_PIXEL.
    """
    if int(angle_bins) > 0:
        return int(angle_bins)

    circumference_mm = math.pi * float(WHEEL_DIAMETER_MM)
    rows = int(
        round(
            circumference_mm
            / max(float(DIRECT_ENCODER_Y_MM_PER_PIXEL), 1e-12)
        )
    )
    return max(1, rows)


def unwrap_finite_encoder_angles(angles: np.ndarray) -> np.ndarray:
    """Unwrap only finite encoder samples and preserve their original row indices.

    np.unwrap() applied directly to an array containing NaN can contaminate the
    sequence after the first missing value.  This helper keeps shorter or sparse
    sync_index.csv files safe.
    """
    values = np.asarray(angles, dtype=np.float64)
    result = np.full(values.shape, np.nan, dtype=np.float64)

    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return result

    finite_values = values[finite_idx]
    unwrapped_values = np.rad2deg(
        np.unwrap(
            np.deg2rad(finite_values)
        )
    )
    result[finite_idx] = unwrapped_values
    return result


def direct_encoder_angle_to_bins(
    travelled_deg: np.ndarray,
    valid: np.ndarray,
    output_rows: int,
) -> np.ndarray:
    """Map real travelled motor angle directly onto the physical 0..360 Y grid."""
    travelled = np.asarray(travelled_deg, dtype=np.float64)
    valid_mask = np.asarray(valid, dtype=bool)

    bins = np.full(travelled.shape, -1, dtype=np.int32)
    if int(output_rows) <= 0 or not np.any(valid_mask):
        return bins

    # Physical output rows represent [0, 360), not a duplicated 360-degree row.
    # Tiny negative/over-360 values admitted by the trimming tolerance are
    # clamped only at the seam; all interior non-uniform encoder motion remains.
    if DIRECT_ENCODER_CLIP_TO_OPEN_360:
        upper = np.nextafter(float(ROTATION_DEG), 0.0)
    else:
        upper = float(ROTATION_DEG)

    angle = np.clip(
        travelled[valid_mask],
        0.0,
        upper,
    )

    mapped = np.floor(
        angle
        / max(float(ROTATION_DEG), 1e-12)
        * int(output_rows)
    ).astype(np.int64)

    mapped = np.clip(
        mapped,
        0,
        int(output_rows) - 1,
    )
    bins[valid_mask] = mapped.astype(np.int32)
    return bins



def read_angles(
    record_dir: Path,
    row_count: int,
    angle_bins: int,
    height_csv_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Build laser-profile -> output-row mapping.

    In the new default mode, every valid laser profile is assigned directly by
    REAL travelled encoder angle:

        y = floor(theta_travelled / 360 * H)

    where H is normally the true physical circumference row count (8609 for
    D=150 mm and 0.05473705 mm/px).

    This removes the old profile-order stretching that could make 0/360 align
    while the middle of the revolution drifted.
    """
    auto_angle_bins = int(angle_bins) <= 0
    sync_path = record_dir / "sync_index.csv"
    use_row_index_setting = (
        str(SYNC_ANGLE_SOURCE).strip().lower() == "row_index"
    )

    if sync_path.exists() and not use_row_index_setting:
        try:
            data = np.genfromtxt(
                sync_path,
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8-sig",
            )

            data = np.atleast_1d(data)
            names = data.dtype.names or ()

            angle_field, angle_reason, angle_source_info = choose_sync_angle_field(
                data,
                names,
            )

            sync_row_count = int(data.shape[0])
            if sync_row_count <= 0:
                raise RuntimeError("sync_index.csv has no data rows.")

            if "laser_line" not in names:
                raise RuntimeError(
                    "sync_index.csv has no laser_line column."
                )

            # ----------------------------------------------------------
            # Rebuild one encoder angle for each numeric height CSV row.
            # A shorter sync file is allowed; unmatched height rows stay NaN
            # and will simply be excluded.
            # ----------------------------------------------------------
            angles = np.full(
                int(row_count),
                np.nan,
                dtype=np.float64,
            )
            sync_angles = np.asarray(
                data[angle_field],
                dtype=np.float64,
            )
            sync_lines_raw = np.asarray(
                data["laser_line"]
            )

            mapped_sync_rows = 0
            sync_line_mode = "sequential_order"

            try:
                sync_lines_float = np.asarray(
                    sync_lines_raw,
                    dtype=np.float64,
                )
                line_ok = (
                    np.isfinite(sync_lines_float)
                    & np.isfinite(sync_angles)
                )

                sync_lines = np.rint(
                    sync_lines_float[line_ok]
                ).astype(np.int64)
                angle_values = sync_angles[line_ok]

                # Detect uncommon 1-based laser_line indexing.
                if (
                    sync_lines.size
                    and np.min(sync_lines) >= 1
                    and np.max(sync_lines) <= row_count
                    and not np.any(sync_lines == 0)
                ):
                    sync_lines = sync_lines - 1
                    sync_line_mode = "laser_line_1_based"
                else:
                    sync_line_mode = "laser_line_0_based"

                in_range = (
                    (sync_lines >= 0)
                    & (sync_lines < row_count)
                    & np.isfinite(angle_values)
                )

                if (
                    np.count_nonzero(in_range)
                    >= max(
                        1,
                        min(sync_row_count, row_count) // 2,
                    )
                ):
                    angles[sync_lines[in_range]] = angle_values[in_range]
                    mapped_sync_rows = int(
                        np.count_nonzero(
                            np.isfinite(angles)
                        )
                    )
                else:
                    raise ValueError(
                        "too few usable laser_line indices"
                    )

            except Exception:
                usable_sync_rows = min(
                    row_count,
                    sync_row_count,
                )
                angles[:usable_sync_rows] = sync_angles[:usable_sync_rows]
                mapped_sync_rows = int(
                    np.count_nonzero(
                        np.isfinite(angles)
                    )
                )
                sync_line_mode = "sequential_order"

            finite = np.isfinite(angles)

            if not np.any(finite):
                raise RuntimeError(
                    f"No finite {angle_field} values could be mapped to laser rows."
                )

            print(
                f"Sync rows: file={sync_row_count}, "
                f"mapped={mapped_sync_rows}, "
                f"laser_rows={row_count}, "
                f"unmatched_laser_rows={row_count - mapped_sync_rows}, "
                f"mode={sync_line_mode}, "
                f"angle={angle_field}, "
                f"reason={angle_reason}"
            )

            # ----------------------------------------------------------
            # Stationary-start trimming.
            # ----------------------------------------------------------
            start_row = int(
                max(
                    0,
                    MANUAL_TRIM_START_ROWS,
                )
            )
            auto_start_row = 0

            if AUTO_TRIM_STATIONARY_START:
                first_valid = int(
                    np.flatnonzero(finite)[0]
                )
                delta_from_first = np.abs(
                    angular_delta_deg(
                        angles,
                        float(angles[first_valid]),
                    )
                )

                moving = np.flatnonzero(
                    finite
                    & (
                        delta_from_first
                        >= float(START_MOVE_MIN_DEG)
                    )
                )
                if moving.size:
                    auto_start_row = int(
                        moving[0]
                    )
                    start_row = max(
                        start_row,
                        auto_start_row,
                    )

            base_start_row = int(
                max(
                    0,
                    min(
                        row_count,
                        start_row,
                    ),
                )
            )

            extra_rows = extra_start_trim_rows()
            start_row = int(
                max(
                    0,
                    min(
                        row_count,
                        base_start_row + extra_rows,
                    ),
                )
            )

            info: dict[str, float | int | str] = {
                "angle_source": "sync_index.csv:" + angle_field,
                "sync_angle_source_setting": angle_source_info.get(
                    "sync_angle_source_setting"
                ),
                "sync_angle_source_reason": angle_source_info.get(
                    "sync_angle_source_reason"
                ),
                "feedback_final_span_deg": angle_source_info.get(
                    "feedback_final_span_deg"
                ),
                "fallback_angle_field": angle_source_info.get(
                    "fallback_angle_field"
                ),
                "fallback_final_span_deg": angle_source_info.get(
                    "fallback_final_span_deg"
                ),
                "sync_file_rows": int(sync_row_count),
                "mapped_sync_rows": int(mapped_sync_rows),
                "unmatched_laser_rows": int(
                    row_count - mapped_sync_rows
                ),
                "sync_line_mode": sync_line_mode,
                "auto_trim_start_rows": int(auto_start_row),
                "manual_trim_start_rows": int(
                    MANUAL_TRIM_START_ROWS
                ),
                "base_trim_start_rows": int(
                    base_start_row
                ),
                "extra_trim_start_seconds": float(
                    EXTRA_TRIM_START_SECONDS
                ),
                "extra_trim_start_rows": int(
                    extra_rows
                ),
                "used_trim_start_rows": int(
                    start_row
                ),
                "angle_mapping": "not_resolved_yet",
                "motion_direction": "unknown",
                "kept_rows": 0,
                "direct_encoder_physical_y": str(
                    USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
                ),
                "direct_encoder_y_mm_per_pixel": float(
                    DIRECT_ENCODER_Y_MM_PER_PIXEL
                ),
            }

            if start_row >= row_count:
                info["requested_angle_bins"] = int(angle_bins)
                info["resolved_angle_bins"] = 0
                return (
                    np.full(
                        row_count,
                        -1,
                        dtype=np.int32,
                    ),
                    info,
                )

            # Use first finite encoder sample AT/AFTER the requested trim as
            # the exact travelled-angle zero reference.
            start_candidates = np.flatnonzero(
                finite
                & (
                    np.arange(row_count)
                    >= start_row
                )
            )
            if start_candidates.size == 0:
                raise RuntimeError(
                    "No finite encoder sample remains after start trimming."
                )

            start_ref_row = int(
                start_candidates[0]
            )
            info["encoder_zero_reference_row"] = int(
                start_ref_row
            )
            info["encoder_zero_reference_deg_raw"] = float(
                angles[start_ref_row]
            )

            # ----------------------------------------------------------
            # REAL travelled angle.
            # ----------------------------------------------------------
            if USE_TRAVELLED_ANGLE_FROM_SYNC:
                unwrapped = unwrap_finite_encoder_angles(
                    angles
                )

                rel = (
                    unwrapped
                    - float(
                        unwrapped[start_ref_row]
                    )
                )

                tail = rel[start_ref_row:]
                tail = tail[
                    np.isfinite(tail)
                ]
                end_delta = (
                    float(
                        np.nanmedian(
                            tail[
                                -min(
                                    500,
                                    tail.size,
                                ):
                            ]
                        )
                    )
                    if tail.size
                    else 0.0
                )

                direction = (
                    -1.0
                    if end_delta < 0.0
                    else 1.0
                )
                travelled = rel * direction

                # Remove a final stationary encoder plateau exactly as in the
                # uploaded program.
                end_row, trimmed_end_rows, final_stationary_span_deg = (
                    detect_stationary_end_row(
                        travelled,
                        finite,
                        start_ref_row,
                    )
                )

                valid = (
                    finite
                    & (
                        np.arange(row_count)
                        >= start_ref_row
                    )
                    & (
                        np.arange(row_count)
                        < end_row
                    )
                    & (
                        travelled
                        >= -float(
                            START_MOVE_MIN_DEG
                        )
                    )
                    & (
                        travelled
                        <= float(
                            ROTATION_DEG
                        )
                        + float(
                            START_MOVE_MIN_DEG
                        )
                    )
                )

                valid_indices = np.flatnonzero(
                    valid
                )
                kept_rows = int(
                    valid_indices.size
                )

                if kept_rows <= 0:
                    raise RuntimeError(
                        "No real-encoder moving laser profiles remain "
                        "after start/end trimming."
                    )

                # ======================================================
                # NEW CORE LOGIC
                # ======================================================
                if USE_REAL_ENCODER_DIRECT_PHYSICAL_Y:
                    resolved_bins = (
                        direct_encoder_physical_row_count(
                            angle_bins
                        )
                    )

                    bins = direct_encoder_angle_to_bins(
                        travelled,
                        valid,
                        resolved_bins,
                    )

                    valid_travelled = travelled[
                        valid
                    ]
                    valid_target_rows = bins[
                        valid
                    ]

                    info["angle_mapping"] = (
                        "real_encoder_travelled_deg_direct_to_physical_y_grid"
                    )
                    info["direct_encoder_target_rows"] = int(
                        resolved_bins
                    )
                    info["direct_encoder_target_angle_step_deg"] = float(
                        ROTATION_DEG
                        / max(
                            resolved_bins,
                            1,
                        )
                    )
                    info["direct_encoder_target_arc_step_mm"] = float(
                        math.pi
                        * WHEEL_DIAMETER_MM
                        / max(
                            resolved_bins,
                            1,
                        )
                    )
                    info["encoder_travelled_min_deg"] = float(
                        np.nanmin(
                            valid_travelled
                        )
                    )
                    info["encoder_travelled_max_deg"] = float(
                        np.nanmax(
                            valid_travelled
                        )
                    )
                    info["encoder_travelled_span_deg"] = float(
                        np.nanmax(
                            valid_travelled
                        )
                        - np.nanmin(
                            valid_travelled
                        )
                    )
                    info["direct_encoder_first_occupied_row"] = int(
                        np.min(
                            valid_target_rows
                        )
                    )
                    info["direct_encoder_last_occupied_row"] = int(
                        np.max(
                            valid_target_rows
                        )
                    )

                else:
                    # Original behavior retained as an explicit comparison mode.
                    bins = np.full(
                        row_count,
                        -1,
                        dtype=np.int32,
                    )

                    if auto_angle_bins:
                        bins[valid_indices] = np.arange(
                            kept_rows,
                            dtype=np.int32,
                        )
                        resolved_bins = kept_rows
                        info["angle_mapping"] = (
                            "valid_profile_order_0_to_360_OLD_MODE"
                        )
                    else:
                        resolved_bins = int(
                            angle_bins
                        )
                        bin_values = np.floor(
                            np.clip(
                                travelled,
                                0.0,
                                ROTATION_DEG,
                            )
                            / ROTATION_DEG
                            * resolved_bins
                        ).astype(
                            np.int32
                        )
                        bins[valid] = np.clip(
                            bin_values[valid],
                            0,
                            resolved_bins - 1,
                        )
                        info["angle_mapping"] = (
                            "travelled_angle_to_fixed_bins_OLD_MODE"
                        )

                info["motion_direction"] = (
                    "negative"
                    if direction < 0
                    else "positive"
                )
                info["auto_trim_stationary_end"] = str(
                    AUTO_TRIM_STATIONARY_END
                )
                info["used_end_row_exclusive"] = int(
                    end_row
                )
                info["trimmed_stationary_end_rows"] = int(
                    trimmed_end_rows
                )
                info["final_stationary_span_deg"] = float(
                    final_stationary_span_deg
                )
                info["kept_rows"] = int(
                    kept_rows
                )
                info["requested_angle_bins"] = int(
                    angle_bins
                )
                info["resolved_angle_bins"] = int(
                    resolved_bins
                )
                return bins, info

            # ----------------------------------------------------------
            # Absolute/modulo mode retained only for compatibility.
            # Direct physical-Y formal use should keep
            # USE_TRAVELLED_ANGLE_FROM_SYNC=True.
            # ----------------------------------------------------------
            valid = (
                finite
                & (
                    np.arange(row_count)
                    >= start_ref_row
                )
            )
            valid_indices = np.flatnonzero(
                valid
            )
            kept_rows = int(
                valid_indices.size
            )

            if kept_rows <= 0:
                raise RuntimeError(
                    "No valid encoder rows remain."
                )

            if USE_REAL_ENCODER_DIRECT_PHYSICAL_Y:
                resolved_bins = (
                    direct_encoder_physical_row_count(
                        angle_bins
                    )
                )
                modulo_angle = np.mod(
                    angles,
                    ROTATION_DEG,
                )
                bins = np.full(
                    row_count,
                    -1,
                    dtype=np.int32,
                )
                mapped = np.floor(
                    modulo_angle[valid]
                    / ROTATION_DEG
                    * resolved_bins
                ).astype(
                    np.int32
                )
                bins[valid] = np.clip(
                    mapped,
                    0,
                    resolved_bins - 1,
                )
                info["angle_mapping"] = (
                    "real_encoder_absolute_modulo_direct_to_physical_y_grid"
                )
            else:
                bins = np.full(
                    row_count,
                    -1,
                    dtype=np.int32,
                )
                if auto_angle_bins:
                    bins[valid_indices] = np.arange(
                        kept_rows,
                        dtype=np.int32,
                    )
                    resolved_bins = kept_rows
                    info["angle_mapping"] = (
                        "valid_profile_order_0_to_360_OLD_MODE"
                    )
                else:
                    resolved_bins = int(
                        angle_bins
                    )
                    bin_values = np.floor(
                        (
                            angles
                            % 360.0
                        )
                        / 360.0
                        * resolved_bins
                    ).astype(
                        np.int32
                    )
                    bins[valid] = np.clip(
                        bin_values[valid],
                        0,
                        resolved_bins - 1,
                    )
                    info["angle_mapping"] = (
                        "absolute_modulo_to_fixed_bins_OLD_MODE"
                    )

            info["kept_rows"] = int(
                kept_rows
            )
            info["requested_angle_bins"] = int(
                angle_bins
            )
            info["resolved_angle_bins"] = int(
                resolved_bins
            )
            return bins, info

        except Exception as exc:
            if (
                USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
                and REQUIRE_REAL_ENCODER_FOR_PHYSICAL_Y
            ):
                raise RuntimeError(
                    "Real-encoder physical-Y restoration failed. "
                    "The program will NOT silently fall back to profile order / "
                    "row index because that would reintroduce the middle-of-circle "
                    "Y distortion.\n"
                    f"sync file: {sync_path}\n"
                    f"reason: {exc}"
                ) from exc

            print(
                f"Warning: could not use sync_index.csv: {exc}"
            )

    # ==============================================================
    # Fallback / deliberately forced row-index mode.
    # ==============================================================
    if (
        USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
        and REQUIRE_REAL_ENCODER_FOR_PHYSICAL_Y
        and not use_row_index_setting
    ):
        raise RuntimeError(
            "USE_REAL_ENCODER_DIRECT_PHYSICAL_Y=True but no usable "
            "sync_index.csv real encoder mapping was found at:\n"
            f"  {sync_path}\n"
            "Formal physical-Y restoration has stopped instead of stretching "
            "profile order over 360 degrees."
        )

    if use_row_index_setting:
        print(
            "SYNC_ANGLE_SOURCE=row_index; using row index by explicit request."
        )
    else:
        print(
            "No usable sync_index.csv found; using legacy row-index fallback."
        )

    base_start_row = int(
        max(
            0,
            min(
                row_count,
                MANUAL_TRIM_START_ROWS,
            ),
        )
    )
    extra_rows = extra_start_trim_rows()
    start_row = int(
        max(
            0,
            min(
                row_count,
                base_start_row + extra_rows,
            ),
        )
    )

    end_row = int(
        row_count
    )
    trimmed_duplicate_tail_rows = 0
    height_tail_info: dict[str, float | int | str] = {
        "height_tail_detector_used": "False",
        "height_tail_detected": "False",
    }

    if (
        height_csv_path is not None
        and AUTO_TRIM_DUPLICATE_HEIGHT_TAIL
    ):
        (
            end_row,
            trimmed_duplicate_tail_rows,
            height_tail_info,
        ) = detect_duplicate_height_tail_from_csv(
            height_csv_path,
            row_count,
            start_row,
        )

    end_row = int(
        max(
            start_row,
            min(
                row_count,
                end_row,
            ),
        )
    )

    bins = np.full(
        row_count,
        -1,
        dtype=np.int32,
    )
    kept = max(
        0,
        end_row - start_row,
    )

    if (
        USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
        and use_row_index_setting
    ):
        # Explicit diagnostic-only row-index mode on the same physical output
        # height.  This is intentionally labelled as NOT encoder-correct.
        resolved_bins = direct_encoder_physical_row_count(
            angle_bins
        )
        if kept > 0:
            mapped = np.floor(
                np.arange(
                    kept,
                    dtype=np.float64,
                )
                / max(
                    kept,
                    1,
                )
                * resolved_bins
            ).astype(
                np.int32
            )
            bins[start_row:end_row] = np.clip(
                mapped,
                0,
                resolved_bins - 1,
            )
        angle_mapping = (
            "EXPLICIT_ROW_INDEX_UNIFORM_TO_PHYSICAL_GRID_NOT_REAL_ENCODER"
        )
    else:
        resolved_bins = (
            kept
            if auto_angle_bins
            else int(
                angle_bins
            )
        )

        if kept > 0:
            if auto_angle_bins:
                bins[start_row:end_row] = np.arange(
                    kept,
                    dtype=np.int32,
                )
            else:
                bin_values = np.floor(
                    np.arange(
                        kept,
                        dtype=np.float64,
                    )
                    / max(
                        kept,
                        1,
                    )
                    * resolved_bins
                ).astype(
                    np.int32
                )
                bins[start_row:end_row] = np.clip(
                    bin_values,
                    0,
                    resolved_bins - 1,
                )

        angle_mapping = (
            "valid_profile_order_0_to_360"
            if auto_angle_bins
            else "row_index_0_to_360"
        )

    info = {
        "angle_source": "row_index_fallback",
        "sync_angle_source_setting": str(
            SYNC_ANGLE_SOURCE
        ).strip().lower(),
        "sync_angle_source_reason": (
            "forced_row_index"
            if use_row_index_setting
            else "sync_unusable_row_index_fallback"
        ),
        "feedback_final_span_deg": float(
            "nan"
        ),
        "fallback_angle_field": "",
        "fallback_final_span_deg": float(
            "nan"
        ),
        "auto_trim_start_rows": 0,
        "manual_trim_start_rows": int(
            MANUAL_TRIM_START_ROWS
        ),
        "base_trim_start_rows": int(
            base_start_row
        ),
        "extra_trim_start_seconds": float(
            EXTRA_TRIM_START_SECONDS
        ),
        "extra_trim_start_rows": int(
            extra_rows
        ),
        "used_trim_start_rows": int(
            start_row
        ),
        "auto_trim_stationary_end": str(
            AUTO_TRIM_STATIONARY_END
        ),
        "used_end_row_exclusive": int(
            end_row
        ),
        "trimmed_stationary_end_rows": int(
            trimmed_duplicate_tail_rows
        ),
        "trimmed_duplicate_height_tail_rows": int(
            trimmed_duplicate_tail_rows
        ),
        "angle_mapping": angle_mapping,
        "motion_direction": "unknown",
        "kept_rows": int(
            max(
                0,
                kept,
            )
        ),
        "requested_angle_bins": int(
            angle_bins
        ),
        "resolved_angle_bins": int(
            max(
                0,
                resolved_bins,
            )
        ),
        "direct_encoder_physical_y": str(
            USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
        ),
        "direct_encoder_y_mm_per_pixel": float(
            DIRECT_ENCODER_Y_MM_PER_PIXEL
        ),
    }

    info.update(
        height_tail_info
    )
    return bins, info

def count_numeric_rows(csv_path: Path) -> int:
    count = 0
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for line in f:
            arr = np.fromstring(line.strip(), sep=",", dtype=np.float32)
            if arr.size:
                count += 1
    return count


def reduce_profile(profile: np.ndarray, x_group: int) -> np.ndarray:
    usable = profile.size // x_group * x_group
    if usable <= 0:
        raise ValueError("CSV row has too few columns.")
    values = profile[:usable].astype(np.float32, copy=False)
    values = np.where(values <= INVALID_BELOW_MM, np.nan, values)
    grouped = values.reshape(-1, x_group)
    with np.errstate(all="ignore"):
        reduced = np.nanmax(grouped, axis=1)
    return reduced.astype(np.float32, copy=False)


def filter_low_outliers(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    if not FILTER_LOW_OUTLIERS_BY_ROW:
        return matrix, 0

    filtered = matrix.copy()
    with np.errstate(all="ignore"):
        row_med = np.nanmedian(filtered, axis=1)
    too_low = np.isfinite(filtered) & np.isfinite(row_med[:, None])
    too_low &= (filtered < (row_med[:, None] - LOW_OUTLIER_BELOW_ROW_MEDIAN_MM))
    removed = int(np.count_nonzero(too_low))
    filtered[too_low] = np.nan
    return filtered, removed


def build_height_map(csv_path: Path, angle_bins: int, x_group: int) -> tuple[np.ndarray, dict[str, float | int | str]]:
    print(f"Counting rows: {csv_path}")
    row_count = count_numeric_rows(csv_path)
    if row_count <= 0:
        raise RuntimeError("Height CSV has no numeric rows.")
    print(f"Numeric rows: {row_count}")

    bins, build_info = read_angles(csv_path.parent, row_count, angle_bins, csv_path)
    resolved_angle_bins = int(build_info.get("resolved_angle_bins", angle_bins))
    if resolved_angle_bins <= 0:
        raise RuntimeError("No valid moving laser rows remain after trimming.")
    if bool(USE_REAL_ENCODER_DIRECT_PHYSICAL_Y):
        output_mode_text = (
            "real-encoder physical circumference grid"
            if int(angle_bins) <= 0
            else "real-encoder user-fixed grid"
        )
    else:
        output_mode_text = (
            "automatic valid-row count"
            if int(angle_bins) <= 0
            else "fixed"
        )
    print(
        f"Output rows: {resolved_angle_bins} ({output_mode_text})"
    )
    print(f"Y mapping: {build_info.get('angle_mapping')}")
    if build_info.get("encoder_travelled_span_deg") is not None:
        print(
            "Real encoder travelled span: "
            f"{build_info.get('encoder_travelled_min_deg')} .. "
            f"{build_info.get('encoder_travelled_max_deg')} deg "
            f"(span={build_info.get('encoder_travelled_span_deg')} deg)"
        )
    skipped_rows = int(np.count_nonzero(bins < 0))
    if skipped_rows:
        print(f"Skipping {skipped_rows} rows outside valid moving 0-360 deg range.")
    print(
        "Start trim rows: "
        f"auto={build_info.get('auto_trim_start_rows')}, "
        f"manual={build_info.get('manual_trim_start_rows')}, "
        f"extra={build_info.get('extra_trim_start_rows')}, "
        f"used={build_info.get('used_trim_start_rows')}"
    )
    trimmed_end_rows = int(build_info.get("trimmed_stationary_end_rows", 0) or 0)
    if trimmed_end_rows > 0:
        print(
            "End stationary trim: "
            f"removed={trimmed_end_rows} rows, "
            f"end_row_exclusive={build_info.get('used_end_row_exclusive')}, "
            f"final_span_deg={build_info.get('final_stationary_span_deg')}"
        )
    elif AUTO_TRIM_STATIONARY_END or AUTO_TRIM_DUPLICATE_HEIGHT_TAIL:
        print("End stationary trim: no stationary duplicate tail detected.")

    out = None
    row_index = 0
    used_rows = 0
    row_hit_counts = np.zeros(
        resolved_angle_bins,
        dtype=np.int32,
    )
    print("Reading height CSV and building image matrix...")
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for line in f:
            profile = np.fromstring(line.strip(), sep=",", dtype=np.float32)
            if profile.size == 0:
                continue
            y = int(bins[row_index])
            if y < 0:
                row_index += 1
                continue
            if y >= resolved_angle_bins:
                raise RuntimeError(
                    f"Mapped output row {y} exceeds resolved row count "
                    f"{resolved_angle_bins} at source profile {row_index}."
                )
            row_hit_counts[y] += 1
            reduced = reduce_profile(profile, x_group)
            if out is None:
                out = np.full((resolved_angle_bins, reduced.size), -np.inf, dtype=np.float32)
            valid = np.isfinite(reduced)
            if np.any(valid):
                current = out[y, valid]
                out[y, valid] = np.maximum(current, reduced[valid])
            used_rows += 1
            row_index += 1
            if row_index % 2000 == 0:
                print(f"  processed {row_index}/{row_count} rows")

    if out is None:
        raise RuntimeError("Could not build height map.")
    out[out == -np.inf] = np.nan
    out, removed_low_points = filter_low_outliers(out)
    if removed_low_points:
        print(f"Filtered low false-return points: {removed_low_points}")
    build_info["numeric_rows"] = int(row_count)
    build_info["requested_angle_bins"] = int(angle_bins)
    build_info["resolved_angle_bins"] = int(resolved_angle_bins)
    build_info["used_csv_rows"] = int(used_rows)
    build_info["skipped_csv_rows"] = int(skipped_rows)
    build_info["filter_low_outliers_by_row"] = str(FILTER_LOW_OUTLIERS_BY_ROW)
    build_info["low_outlier_below_row_median_mm"] = float(LOW_OUTLIER_BELOW_ROW_MEDIAN_MM)
    build_info["removed_low_outlier_points"] = int(removed_low_points)

    occupied = row_hit_counts > 0
    build_info["encoder_grid_occupied_rows"] = int(
        np.count_nonzero(occupied)
    )
    build_info["encoder_grid_empty_rows"] = int(
        resolved_angle_bins - np.count_nonzero(occupied)
    )
    build_info["encoder_grid_coverage_fraction"] = float(
        np.mean(occupied)
    )
    build_info["encoder_grid_min_profiles_per_occupied_row"] = int(
        np.min(row_hit_counts[occupied])
    ) if np.any(occupied) else 0
    build_info["encoder_grid_max_profiles_per_row"] = int(
        np.max(row_hit_counts)
    ) if row_hit_counts.size else 0
    build_info["encoder_grid_mean_profiles_per_occupied_row"] = float(
        np.mean(row_hit_counts[occupied])
    ) if np.any(occupied) else 0.0

    # Private in-memory diagnostic array; save_outputs writes it to CSV.
    build_info["_encoder_row_hit_counts"] = row_hit_counts

    print(
        "Encoder physical-Y grid coverage: "
        f"{build_info['encoder_grid_occupied_rows']}/{resolved_angle_bins} rows "
        f"({build_info['encoder_grid_coverage_fraction'] * 100.0:.2f}%), "
        f"empty={build_info['encoder_grid_empty_rows']}, "
        f"profiles/occupied-row mean="
        f"{build_info['encoder_grid_mean_profiles_per_occupied_row']:.3f}"
    )

    return out, build_info


def colorize(values: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    finite = np.isfinite(t)
    t = np.where(finite, t, 0.0)
    anchors = np.array(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=np.float32,
    )
    pos = t * (len(anchors) - 1)
    idx = np.minimum(pos.astype(np.int32), len(anchors) - 2)
    frac = (pos - idx)[..., None]
    rgb = anchors[idx] * (1.0 - frac) + anchors[idx + 1] * frac
    rgb[~finite] = [250, 250, 250]
    return rgb.astype(np.uint8)


def build_physical_axes(row_count: int, col_count: int):
    """Build axes for the RAW numeric height matrix.

    IMPORTANT
    ---------
    The raw LJ-X height matrix may contain thousands of X samples (for example
    3200 columns).  Those raw columns span the nominal laser field 0..16 mm and
    therefore keep their own native sampling interval:

        native_step = 16 mm / (raw_col_count - 1)

    This raw axis is NOT the same thing as the later 292-pixel true-scale PNG
    axis.  Keeping the two axes separate avoids the old inconsistency where the
    affine calibration used 0..16 mm on a 292-pixel image and therefore derived
    0.05498282 mm/px instead of the unified 0.05473705 mm/px.
    """
    if col_count <= 1:
        x_axis_mm = np.zeros(col_count, dtype=np.float32)
    else:
        x_axis_mm = np.linspace(
            0.0,
            LASER_WIDTH_MM,
            col_count,
            dtype=np.float32,
        )

    angle_axis_deg = (
        np.arange(row_count, dtype=np.float32)
        * (ROTATION_DEG / max(row_count, 1))
    )
    arc_axis_mm = (
        angle_axis_deg
        / ROTATION_DEG
        * (math.pi * WHEEL_DIAMETER_MM)
    )
    return (
        x_axis_mm,
        angle_axis_deg.astype(np.float32),
        arc_axis_mm.astype(np.float32),
    )


def true_scale_width_px(mm_per_pixel: float = TRUE_SCALE_MM_PER_PIXEL) -> int:
    """Return the true-scale PNG width used by save_true_scale_png()."""
    scale = max(float(mm_per_pixel), 1e-12)
    return max(
        1,
        int(round(float(LASER_WIDTH_MM) / scale)),
    )


def build_true_scale_x_axis_mm(
    width_px: int,
    mm_per_pixel: float = TRUE_SCALE_MM_PER_PIXEL,
) -> np.ndarray:
    """Build the X coordinate of TRUE-SCALE PNG pixel centers.

    The formal fusion convention is now:

        X_true_scale[col] = col * 0.05473705 mm

    Therefore, for a 292-pixel restored laser image:
        col 0   -> 0.00000000 mm
        col 291 -> 15.92848155 mm

    The nominal laser field remains 16 mm, but the discrete 292-pixel true-scale
    raster uses the exact unified pixel pitch.  This is the axis that must be
    supplied to the affine calibration program.
    """
    width = max(0, int(width_px))
    return (
        np.arange(width, dtype=np.float64)
        * float(mm_per_pixel)
    ).astype(np.float32)


def save_png(matrix: np.ndarray, out_path: Path, title: str, low: float, high: float):
    rgb = colorize(matrix, low, high)
    image = Image.fromarray(rgb, "RGB")

    margin_l, margin_t, margin_b = 86, 36, 64
    canvas = Image.new("RGB", (image.width + margin_l + 16, image.height + margin_t + margin_b), "white")
    canvas.paste(image, (margin_l, margin_t))
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    small = load_font(13)

    draw.text((margin_l, 8), title, fill=(20, 20, 20), font=font)
    draw.rectangle((margin_l, margin_t, margin_l + image.width - 1, margin_t + image.height - 1), outline=(60, 60, 60))

    for value in np.linspace(0.0, LASER_WIDTH_MM, 5):
        x = margin_l + int(round(value / LASER_WIDTH_MM * max(image.width - 1, 1)))
        draw.line((x, margin_t + image.height, x, margin_t + image.height + 5), fill=(60, 60, 60))
        draw.text((x - 10, margin_t + image.height + 8), f"{value:g}", fill=(20, 20, 20), font=small)

    for value in np.linspace(0.0, ROTATION_DEG, 5):
        y = margin_t + int(round(value / ROTATION_DEG * max(image.height - 1, 1)))
        draw.line((margin_l - 5, y, margin_l, y), fill=(60, 60, 60))
        draw.text((36, y - 7), f"{value:.0f}", fill=(20, 20, 20), font=small)

    draw.text((margin_l + 70, margin_t + image.height + 34), f"Laser X width: 0-{LASER_WIDTH_MM:g} mm", fill=(20, 20, 20), font=small)
    draw.text((8, margin_t + image.height // 2), "angle deg", fill=(20, 20, 20), font=small)
    draw.text((margin_l + image.width - 250, margin_t + image.height + 34), f"color: {low:.3f} to {high:.3f} mm", fill=(20, 20, 20), font=small)

    canvas.save(out_path)


def save_true_scale_png(matrix: np.ndarray, out_path: Path, low: float, high: float, pixels_per_mm: float):
    circumference_mm = math.pi * WHEEL_DIAMETER_MM
    mm_per_pixel = 1.0 / max(float(pixels_per_mm), 1e-12)
    width_px = true_scale_width_px(mm_per_pixel)
    height_px = max(1, int(round(circumference_mm * pixels_per_mm)))

    rgb = colorize(matrix, low, high)
    image = Image.fromarray(rgb, "RGB")
    if image.size != (width_px, height_px):
        image = image.resize((width_px, height_px), Image.Resampling.NEAREST)
    image.save(out_path)
    return width_px, height_px



def save_direct_encoder_y_grid_csv(
    out_dir: Path,
    prefix: str,
    row_count: int,
    build_info: dict,
) -> Path | None:
    """Save the actual physical Y grid used by the restored laser map."""
    if (
        not SAVE_DIRECT_ENCODER_Y_GRID_CSV
        or not USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
    ):
        return None

    hit_counts = build_info.get(
        "_encoder_row_hit_counts"
    )
    if hit_counts is None:
        return None

    hit_counts = np.asarray(
        hit_counts,
        dtype=np.int64,
    )
    if hit_counts.size != int(row_count):
        return None

    circumference_mm = (
        math.pi
        * float(
            WHEEL_DIAMETER_MM
        )
    )

    out_path = (
        out_dir
        / f"{prefix}_encoder_physical_y_grid.csv"
    )

    with out_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(
            f
        )
        writer.writerow(
            [
                "output_row",
                "angle_deg",
                "arc_mm",
                "profile_hit_count",
                "occupied",
            ]
        )

        for row in range(
            int(row_count)
        ):
            angle = (
                row
                / max(
                    float(row_count),
                    1.0,
                )
                * float(
                    ROTATION_DEG
                )
            )
            arc = (
                angle
                / max(
                    float(
                        ROTATION_DEG
                    ),
                    1e-12,
                )
                * circumference_mm
            )

            hits = int(
                hit_counts[
                    row
                ]
            )

            writer.writerow(
                [
                    row,
                    f"{angle:.9f}",
                    f"{arc:.9f}",
                    hits,
                    int(
                        hits > 0
                    ),
                ]
            )

    return out_path



def save_outputs(matrix: np.ndarray, out_dir: Path, prefix: str, build_info: dict[str, float | int | str]):
    out_dir.mkdir(parents=True, exist_ok=True)
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise RuntimeError("No valid height values after filtering invalid points.")

    # RAW numeric matrix axes.
    native_x_axis_mm, angle_axis_deg, arc_axis_mm = build_physical_axes(
        matrix.shape[0],
        matrix.shape[1],
    )
    circumference_mm = math.pi * WHEEL_DIAMETER_MM

    # TRUE-SCALE PNG X axis used by affine calibration / visual fusion.
    true_scale_width = true_scale_width_px(TRUE_SCALE_MM_PER_PIXEL)
    true_scale_x_axis_mm = build_true_scale_x_axis_mm(
        true_scale_width,
        TRUE_SCALE_MM_PER_PIXEL,
    )

    if true_scale_x_axis_mm.size > 1:
        resolved_true_scale_step = float(
            true_scale_x_axis_mm[1] - true_scale_x_axis_mm[0]
        )
        if not np.isclose(
            resolved_true_scale_step,
            float(TRUE_SCALE_MM_PER_PIXEL),
            rtol=0.0,
            atol=1e-7,
        ):
            raise RuntimeError(
                "True-scale X-axis consistency check failed: "
                f"axis step={resolved_true_scale_step:.9f} mm/px, "
                f"configured={TRUE_SCALE_MM_PER_PIXEL:.9f} mm/px."
            )

    # Compatibility name intentionally points to the TRUE-SCALE PNG axis.
    # The affine calibration auto-detects <prefix>_x_axis_mm.csv beside
    # *_true_scale_detail_height.png, so this file MUST correspond to that PNG.
    x_axis_path = out_dir / f"{prefix}_x_axis_mm.csv"

    # Preserve the physical axis of the raw numeric height matrix separately.
    native_x_axis_path = out_dir / f"{prefix}_native_x_axis_mm.csv"

    angle_axis_path = out_dir / f"{prefix}_angle_axis_deg.csv"
    arc_axis_path = out_dir / f"{prefix}_arc_axis_mm.csv"

    np.savetxt(
        x_axis_path,
        true_scale_x_axis_mm,
        delimiter=",",
        header="x_mm",
        comments="",
    )
    np.savetxt(
        native_x_axis_path,
        native_x_axis_mm,
        delimiter=",",
        header="x_mm",
        comments="",
    )
    np.savetxt(
        angle_axis_path,
        angle_axis_deg,
        delimiter=",",
        header="angle_deg",
        comments="",
    )
    np.savetxt(
        arc_axis_path,
        arc_axis_mm,
        delimiter=",",
        header="arc_mm",
        comments="",
    )

    encoder_y_grid_csv = save_direct_encoder_y_grid_csv(
        out_dir,
        prefix,
        matrix.shape[0],
        build_info,
    )

    abs_low, abs_high = np.nanpercentile(matrix, [1.0, 99.0])
    abs_png = out_dir / f"{prefix}_absolute_height.png"
    save_png(matrix, abs_png, "Absolute height map", float(abs_low), float(abs_high))
    true_abs_png = None
    true_abs_size = None
    natural_abs_png = None
    natural_abs_size = None
    if SAVE_TRUE_SCALE_PNG:
        true_abs_png = out_dir / f"{prefix}_true_scale_absolute_height.png"
        true_abs_size = save_true_scale_png(
            matrix,
            true_abs_png,
            float(abs_low),
            float(abs_high),
            TRUE_SCALE_PIXELS_PER_MM,
        )
    if SAVE_NATURAL_FUSION_SCALE_PNG:
        natural_abs_png = out_dir / f"{prefix}_true_scale_0p0504mm_absolute_height.png"
        natural_abs_size = save_true_scale_png(
            matrix,
            natural_abs_png,
            float(abs_low),
            float(abs_high),
            NATURAL_FUSION_PIXELS_PER_MM,
        )

    with np.errstate(all="ignore"):
        row_med = np.nanmedian(matrix, axis=1)
    residual = matrix - row_med[:, None]
    res_low, res_high = np.nanpercentile(residual, [2.0, 98.0])
    detail_png = out_dir / f"{prefix}_detail_height.png"
    save_png(residual, detail_png, "Detail height map, row median removed", float(res_low), float(res_high))
    true_detail_png = None
    true_detail_size = None
    natural_detail_png = None
    natural_detail_size = None
    if SAVE_TRUE_SCALE_PNG:
        true_detail_png = out_dir / f"{prefix}_true_scale_detail_height.png"
        true_detail_size = save_true_scale_png(
            residual,
            true_detail_png,
            float(res_low),
            float(res_high),
            TRUE_SCALE_PIXELS_PER_MM,
        )
    if SAVE_NATURAL_FUSION_SCALE_PNG:
        natural_detail_png = out_dir / f"{prefix}_true_scale_0p0504mm_detail_height.png"
        natural_detail_size = save_true_scale_png(
            residual,
            natural_detail_png,
            float(res_low),
            float(res_high),
            NATURAL_FUSION_PIXELS_PER_MM,
        )

    npz_path = out_dir / f"{prefix}_height_map.npz"
    np.savez_compressed(
        npz_path,
        height_mm=matrix,
        residual_mm=residual,
        # Raw numeric matrix axis (same column count as height_mm).
        x_axis_mm=native_x_axis_mm,

        # True-scale restored PNG pixel-center axis (same width as the
        # *_true_scale_* PNG; this is the affine/fusion X coordinate).
        true_scale_x_axis_mm=true_scale_x_axis_mm,

        angle_axis_deg=angle_axis_deg,
        arc_axis_mm=arc_axis_mm,
        wheel_diameter_mm=np.float32(WHEEL_DIAMETER_MM),
        rotation_deg=np.float32(ROTATION_DEG),
        laser_hz=np.float32(LASER_HZ),
        laser_width_mm=np.float32(LASER_WIDTH_MM),
        true_scale_mm_per_pixel=np.float32(TRUE_SCALE_MM_PER_PIXEL),
        true_scale_width_px=np.int32(true_scale_width),
        true_scale_x_min_mm=np.float32(
            true_scale_x_axis_mm[0] if true_scale_x_axis_mm.size else 0.0
        ),
        true_scale_x_max_mm=np.float32(
            true_scale_x_axis_mm[-1] if true_scale_x_axis_mm.size else 0.0
        ),
        native_matrix_x_min_mm=np.float32(
            native_x_axis_mm[0] if native_x_axis_mm.size else 0.0
        ),
        native_matrix_x_max_mm=np.float32(
            native_x_axis_mm[-1] if native_x_axis_mm.size else 0.0
        ),
        x_axis_csv_role=np.array(
            "true_scale_png_pixel_center_axis_for_affine_and_fusion"
        ),
        natural_fusion_mm_per_pixel=np.float32(NATURAL_FUSION_MM_PER_PIXEL),
    )

    summary_path = out_dir / f"{prefix}_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"shape_rows_cols={matrix.shape[0]},{matrix.shape[1]}",
                f"valid_fraction={float(np.mean(finite)):.6f}",
                f"wheel_diameter_mm={WHEEL_DIAMETER_MM:.6f}",
                f"wheel_circumference_mm={circumference_mm:.6f}",
                f"rotation_deg={ROTATION_DEG:.6f}",
                f"laser_hz={LASER_HZ:.6f}",
                f"laser_width_mm={LASER_WIDTH_MM:.6f}",
                f"true_scale_mm_per_pixel={TRUE_SCALE_MM_PER_PIXEL:.9f}",
                f"true_scale_pixels_per_mm={TRUE_SCALE_PIXELS_PER_MM:.6f}",
                f"natural_fusion_mm_per_pixel={NATURAL_FUSION_MM_PER_PIXEL:.9f}",
                f"natural_fusion_pixels_per_mm={NATURAL_FUSION_PIXELS_PER_MM:.6f}",
                f"angle_source={build_info.get('angle_source')}",
                f"sync_angle_source_setting={build_info.get('sync_angle_source_setting')}",
                f"sync_angle_source_reason={build_info.get('sync_angle_source_reason')}",
                f"feedback_stuck_window_rows={FEEDBACK_STUCK_WINDOW_ROWS}",
                f"feedback_stuck_max_span_deg={FEEDBACK_STUCK_MAX_SPAN_DEG:.6f}",
                f"fallback_moving_min_span_deg={FALLBACK_MOVING_MIN_SPAN_DEG:.6f}",
                f"feedback_final_span_deg={build_info.get('feedback_final_span_deg')}",
                f"fallback_angle_field={build_info.get('fallback_angle_field')}",
                f"fallback_final_span_deg={build_info.get('fallback_final_span_deg')}",
                f"sync_file_rows={build_info.get('sync_file_rows')}",
                f"mapped_sync_rows={build_info.get('mapped_sync_rows')}",
                f"unmatched_laser_rows={build_info.get('unmatched_laser_rows')}",
                f"sync_line_mode={build_info.get('sync_line_mode')}",
                f"angle_mapping={build_info.get('angle_mapping')}",
                f"direct_encoder_physical_y={USE_REAL_ENCODER_DIRECT_PHYSICAL_Y}",
                f"require_real_encoder_for_physical_y={REQUIRE_REAL_ENCODER_FOR_PHYSICAL_Y}",
                f"direct_encoder_y_mm_per_pixel={DIRECT_ENCODER_Y_MM_PER_PIXEL:.9f}",
                f"direct_encoder_target_rows={build_info.get('direct_encoder_target_rows')}",
                f"direct_encoder_target_angle_step_deg={build_info.get('direct_encoder_target_angle_step_deg')}",
                f"direct_encoder_target_arc_step_mm={build_info.get('direct_encoder_target_arc_step_mm')}",
                f"encoder_zero_reference_row={build_info.get('encoder_zero_reference_row')}",
                f"encoder_zero_reference_deg_raw={build_info.get('encoder_zero_reference_deg_raw')}",
                f"encoder_travelled_min_deg={build_info.get('encoder_travelled_min_deg')}",
                f"encoder_travelled_max_deg={build_info.get('encoder_travelled_max_deg')}",
                f"encoder_travelled_span_deg={build_info.get('encoder_travelled_span_deg')}",
                f"direct_encoder_first_occupied_row={build_info.get('direct_encoder_first_occupied_row')}",
                f"direct_encoder_last_occupied_row={build_info.get('direct_encoder_last_occupied_row')}",
                f"encoder_grid_occupied_rows={build_info.get('encoder_grid_occupied_rows')}",
                f"encoder_grid_empty_rows={build_info.get('encoder_grid_empty_rows')}",
                f"encoder_grid_coverage_fraction={build_info.get('encoder_grid_coverage_fraction')}",
                f"encoder_grid_min_profiles_per_occupied_row={build_info.get('encoder_grid_min_profiles_per_occupied_row')}",
                f"encoder_grid_max_profiles_per_row={build_info.get('encoder_grid_max_profiles_per_row')}",
                f"encoder_grid_mean_profiles_per_occupied_row={build_info.get('encoder_grid_mean_profiles_per_occupied_row')}",
                f"encoder_physical_y_grid_csv={encoder_y_grid_csv}",
                f"motion_direction={build_info.get('motion_direction')}",
                f"numeric_rows={build_info.get('numeric_rows')}",
                f"requested_angle_bins={build_info.get('requested_angle_bins')}",
                f"resolved_angle_bins={build_info.get('resolved_angle_bins')}",
                f"auto_trim_stationary_start={AUTO_TRIM_STATIONARY_START}",
                f"start_move_min_deg={START_MOVE_MIN_DEG:.6f}",
                f"auto_trim_start_rows={build_info.get('auto_trim_start_rows')}",
                f"manual_trim_start_rows={build_info.get('manual_trim_start_rows')}",
                f"base_trim_start_rows={build_info.get('base_trim_start_rows')}",
                f"extra_trim_start_seconds={build_info.get('extra_trim_start_seconds')}",
                f"extra_trim_start_rows={build_info.get('extra_trim_start_rows')}",
                f"used_trim_start_rows={build_info.get('used_trim_start_rows')}",
                f"auto_trim_stationary_end={build_info.get('auto_trim_stationary_end')}",
                f"end_stationary_window_rows={END_STATIONARY_WINDOW_ROWS}",
                f"end_stationary_max_span_deg={END_STATIONARY_MAX_SPAN_DEG:.6f}",
                f"end_stationary_min_rows={END_STATIONARY_MIN_ROWS}",
                f"end_keep_buffer_rows={END_KEEP_BUFFER_ROWS}",
                f"auto_trim_duplicate_height_tail={AUTO_TRIM_DUPLICATE_HEIGHT_TAIL}",
                f"height_tail_sample_column_step={HEIGHT_TAIL_SAMPLE_COLUMN_STEP}",
                f"height_tail_diff_threshold_mm={HEIGHT_TAIL_DIFF_THRESHOLD_MM:.9f}",
                f"height_tail_window_rows={HEIGHT_TAIL_WINDOW_ROWS}",
                f"height_tail_min_rows={HEIGHT_TAIL_MIN_ROWS}",
                f"height_tail_keep_buffer_rows={HEIGHT_TAIL_KEEP_BUFFER_ROWS}",
                f"height_tail_detector_used={build_info.get('height_tail_detector_used')}",
                f"height_tail_detected={build_info.get('height_tail_detected')}",
                f"height_tail_detected_start_row={build_info.get('height_tail_detected_start_row')}",
                f"height_tail_final_window_median_diff_mm={build_info.get('height_tail_final_window_median_diff_mm')}",
                f"height_tail_moving_reference_median_diff_mm={build_info.get('height_tail_moving_reference_median_diff_mm')}",
                f"trimmed_duplicate_height_tail_rows={build_info.get('trimmed_duplicate_height_tail_rows')}",
                f"used_end_row_exclusive={build_info.get('used_end_row_exclusive')}",
                f"trimmed_stationary_end_rows={build_info.get('trimmed_stationary_end_rows')}",
                f"final_stationary_span_deg={build_info.get('final_stationary_span_deg')}",
                f"used_csv_rows={build_info.get('used_csv_rows')}",
                f"skipped_csv_rows={build_info.get('skipped_csv_rows')}",
                f"filter_low_outliers_by_row={FILTER_LOW_OUTLIERS_BY_ROW}",
                f"low_outlier_below_row_median_mm={LOW_OUTLIER_BELOW_ROW_MEDIAN_MM:.6f}",
                f"removed_low_outlier_points={build_info.get('removed_low_outlier_points')}",
                # The compatibility x_axis CSV now corresponds to the
                # TRUE-SCALE PNG, not the raw 3200-column numeric matrix.
                f"x_axis_csv_role=true_scale_png_pixel_center_axis_for_affine_and_fusion",
                f"x_axis_range_mm={float(true_scale_x_axis_mm[0]) if true_scale_x_axis_mm.size else 0.0:.6f},{float(true_scale_x_axis_mm[-1]) if true_scale_x_axis_mm.size else 0.0:.6f}",
                f"x_axis_step_mm={TRUE_SCALE_MM_PER_PIXEL:.9f}",
                f"true_scale_x_axis_width_px={int(true_scale_width)}",
                f"true_scale_x_axis_center_span_mm={float(true_scale_x_axis_mm[-1] - true_scale_x_axis_mm[0]) if true_scale_x_axis_mm.size > 1 else 0.0:.9f}",
                f"true_scale_raster_footprint_mm={float(true_scale_width * TRUE_SCALE_MM_PER_PIXEL):.9f}",
                f"nominal_laser_width_mm={LASER_WIDTH_MM:.9f}",
                f"native_matrix_x_axis_range_mm={float(native_x_axis_mm[0]) if native_x_axis_mm.size else 0.0:.6f},{float(native_x_axis_mm[-1]) if native_x_axis_mm.size else 0.0:.6f}",
                f"native_matrix_x_axis_step_mm={float(native_x_axis_mm[1] - native_x_axis_mm[0]) if native_x_axis_mm.size > 1 else 0.0:.9f}",
                f"angle_step_deg={float(ROTATION_DEG / max(matrix.shape[0], 1)):.9f}",
                f"arc_step_mm={float(circumference_mm / max(matrix.shape[0], 1)):.9f}",
                f"absolute_color_mm={abs_low:.6f},{abs_high:.6f}",
                f"detail_color_mm={res_low:.6f},{res_high:.6f}",
                f"absolute_png={abs_png}",
                f"detail_png={detail_png}",
                f"true_scale_absolute_png={true_abs_png}",
                f"true_scale_absolute_size_px={true_abs_size}",
                f"true_scale_detail_png={true_detail_png}",
                f"true_scale_detail_size_px={true_detail_size}",
                f"natural_fusion_absolute_png={natural_abs_png}",
                f"natural_fusion_absolute_size_px={natural_abs_size}",
                f"natural_fusion_detail_png={natural_detail_png}",
                f"natural_fusion_detail_size_px={natural_detail_size}",
                f"npz={npz_path}",
                f"x_axis_csv={x_axis_path}",
                f"native_x_axis_csv={native_x_axis_path}",
                f"angle_axis_csv={angle_axis_path}",
                f"arc_axis_csv={arc_axis_path}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved: {abs_png}")
    print(f"Saved: {detail_png}")
    if true_abs_png is not None:
        print(f"Saved: {true_abs_png}")
    if true_detail_png is not None:
        print(f"Saved: {true_detail_png}")
    if natural_abs_png is not None:
        print(f"Saved: {natural_abs_png}")
    if natural_detail_png is not None:
        print(f"Saved: {natural_detail_png}")
    print(f"Saved: {npz_path}")
    print(
        f"Saved: {x_axis_path} "
        f"(TRUE-SCALE PNG X axis for affine/fusion; "
        f"{true_scale_width} cols, step={TRUE_SCALE_MM_PER_PIXEL:.9f} mm/px)"
    )
    print(
        f"Saved: {native_x_axis_path} "
        f"(RAW numeric matrix X axis; {matrix.shape[1]} cols over "
        f"0..{LASER_WIDTH_MM:g} mm)"
    )
    print(f"Saved: {angle_axis_path}")
    print(f"Saved: {arc_axis_path}")
    if encoder_y_grid_csv is not None:
        print(f"Saved: {encoder_y_grid_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Restore LJ-X laser height CSV to height-map PNG images."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=INPUT_PATH,
        help="Height CSV path or record folder. Default: INPUT_PATH in this file.",
    )
    parser.add_argument(
        "--out",
        default=OUTPUT_DIR,
        help="Output folder. Default: OUTPUT_DIR in this file; empty means <record folder>/huanyuan_result.",
    )
    parser.add_argument(
        "--angle-bins",
        type=int,
        default=ANGLE_BINS,
        help=(
            "Output rows for one 360 degree map. "
            "With real-encoder physical-Y enabled, 0 means the physical "
            "circumference grid at DIRECT_ENCODER_Y_MM_PER_PIXEL "
            "(currently about 8609 rows); a positive value overrides the "
            "row count but still uses real encoder angle for placement. "
            f"Default: {ANGLE_BINS}"
        ),
    )
    parser.add_argument(
        "--x-group",
        type=int,
        default=X_GROUP,
        help=f"Merge this many laser X columns into one output column. Default: {X_GROUP}",
    )
    parser.add_argument(
        "--sync-angle-source",
        choices=("auto", "feedback", "fusion", "theory", "row_index"),
        default=SYNC_ANGLE_SOURCE,
        help=(
            "Angle source from sync_index.csv. Default auto keeps encoder feedback "
            "unless you explicitly choose fusion/theory/row_index."
        ),
    )
    return parser.parse_args()


def main():
    global SYNC_ANGLE_SOURCE
    args = parse_args()
    SYNC_ANGLE_SOURCE = args.sync_angle_source
    csv_path = find_height_csv(Path(args.input).resolve())
    out_setting = str(args.out).strip() if args.out is not None else ""
    out_dir = Path(out_setting).resolve() if out_setting else csv_path.parent / "huanyuan_result"
    prefix = csv_path.stem

    print(f"Height CSV: {csv_path}")
    print(f"Output folder: {out_dir}")
    angle_bins_text = (
        (
            f"physical real-encoder grid "
            f"({direct_encoder_physical_row_count(args.angle_bins)} rows)"
        )
        if (
            int(args.angle_bins) <= 0
            and USE_REAL_ENCODER_DIRECT_PHYSICAL_Y
        )
        else (
            "automatic(valid moving rows)"
            if int(args.angle_bins) <= 0
            else str(args.angle_bins)
        )
    )
    print(f"angle_bins={angle_bins_text}, x_group={args.x_group}")

    matrix, build_info = build_height_map(csv_path, args.angle_bins, args.x_group)
    save_outputs(matrix, out_dir, prefix, build_info)
    print("Done.")


if __name__ == "__main__":
    main()
