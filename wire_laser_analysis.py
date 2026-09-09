from __future__ import annotations

"""
丝端面 - 激光高度对应分析程序

目标
----
把相机拼接图分割得到的每根丝端面实例，与融合程序生成的
mapped_height_mm.npz 逐像素对应，并输出每根丝的：

- 全图中心坐标
- 物理 X 位置
- 周向角度 theta
- 直径
- 端面 mask 像素数
- 有效激光像素数
- 激光覆盖率
- height_mm 统计
- residual_mm 统计
- 3D 建模可用等级

路径管理
--------
本程序默认只读取项目的 config.py：

    SEGMENTATION_DIR
    FUSION_DIR
    CAMERA_STITCHED_IMAGE
    ACTIVE_COMBINED_RUN_DIR / RUN_DIR

因此以后换实验时，只需要在 config.py 里修改：
    RUN_NAME
    COMBINED_RUN_NAME
等路径设置，本程序本身不用再改路径。

默认输入
--------
SEGMENTATION_DIR / "wire_instance_map.npz"
SEGMENTATION_DIR / "wire_full_predictions.csv"
FUSION_DIR       / "mapped_height_mm.npz"

默认输出
--------
当前 run 目录 / "wire_laser_analysis"/
    wire_laser_correspondence.csv
    wire_laser_correspondence.npz
    wire_laser_summary.json
    wire_laser_summary.txt
    wire_laser_ready_overlay.png
    wire_laser_coverage_histogram.png
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    import config as project_config
except Exception as exc:
    raise RuntimeError(
        "无法导入项目路径程序 config.py。\n"
        "请把本程序放到 D:\\project\\scripts 目录，确保同目录或 Python 路径中存在 config.py。\n"
        f"原始错误: {exc}"
    ) from exc


# =============================================================================
# PATHS: all derived from config.py
# =============================================================================

SEGMENTATION_DIR = Path(project_config.SEGMENTATION_DIR)
FUSION_DIR = Path(project_config.FUSION_DIR)
CAMERA_STITCHED_IMAGE = Path(project_config.CAMERA_STITCHED_IMAGE)

DEFAULT_INSTANCE_MAP_NPZ = SEGMENTATION_DIR / "wire_instance_map.npz"
DEFAULT_PREDICTIONS_CSV = SEGMENTATION_DIR / "wire_full_predictions.csv"
DEFAULT_MAPPED_HEIGHT_NPZ = FUSION_DIR / "mapped_height_mm.npz"

if bool(getattr(project_config, "USE_COMBINED_ACQUISITION", False)):
    DEFAULT_ANALYSIS_DIR = (
        Path(project_config.ACTIVE_COMBINED_RUN_DIR)
        / "wire_laser_analysis"
    )
else:
    DEFAULT_ANALYSIS_DIR = (
        Path(project_config.RUN_DIR)
        / "wire_laser_analysis"
    )


# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# 第一版 3D 建模建议阈值。
READY_DIRECT_MIN_COVERAGE = 0.80
PARTIAL_MIN_COVERAGE = 0.30

# 有效像素过少时，即使覆盖率偶然较高，也不建议直接建模。
READY_DIRECT_MIN_VALID_PIXELS = 20
PARTIAL_MIN_VALID_PIXELS = 10

# residual 离群点诊断：
# |x - median| > MAD_SIGMA * 1.4826 * MAD
MAD_SIGMA = 4.0

# 若 MAD 接近 0，避免把浮点噪声误判为离群。
MAD_MIN_SCALE_MM = 1e-4

# 可视化。
SAVE_STATUS_OVERLAY = True
SAVE_COVERAGE_HISTOGRAM = True
OVERLAY_ALPHA = 0.42

# 当前程序不会修改原始 height_mm，也不会插值补全缺失激光数据。


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze correspondence between full-image wire instances and "
            "mapped laser height_mm on the same camera coordinate grid."
        )
    )
    parser.add_argument(
        "--instance-map",
        type=Path,
        default=DEFAULT_INSTANCE_MAP_NPZ,
        help="Default: config.SEGMENTATION_DIR / wire_instance_map.npz",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=DEFAULT_PREDICTIONS_CSV,
        help="Default: config.SEGMENTATION_DIR / wire_full_predictions.csv",
    )
    parser.add_argument(
        "--mapped-height",
        type=Path,
        default=DEFAULT_MAPPED_HEIGHT_NPZ,
        help="Default: config.FUSION_DIR / mapped_height_mm.npz",
    )
    parser.add_argument(
        "--camera-image",
        type=Path,
        default=CAMERA_STITCHED_IMAGE,
        help="Used only for status overlay.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
    )
    parser.add_argument(
        "--ready-coverage",
        type=float,
        default=READY_DIRECT_MIN_COVERAGE,
    )
    parser.add_argument(
        "--partial-coverage",
        type=float,
        default=PARTIAL_MIN_COVERAGE,
    )
    parser.add_argument(
        "--ready-min-valid",
        type=int,
        default=READY_DIRECT_MIN_VALID_PIXELS,
    )
    parser.add_argument(
        "--partial-min-valid",
        type=int,
        default=PARTIAL_MIN_VALID_PIXELS,
    )
    parser.add_argument(
        "--mad-sigma",
        type=float,
        default=MAD_SIGMA,
    )
    return parser.parse_args()


# =============================================================================
# LOADERS
# =============================================================================

def _scalar_from_npz(data, key, default=None):
    if key not in data:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    try:
        return value.reshape(-1)[0].item()
    except Exception:
        return value.reshape(-1)[0]


def load_instance_map(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到丝实例图: {path}\n"
            "请先运行全图坐标版丝端面分割程序。"
        )

    with np.load(path, allow_pickle=False) as data:
        if "instance_map" not in data:
            raise KeyError(
                f"{path} 中不存在 instance_map"
            )

        instance_map = np.asarray(
            data["instance_map"],
            dtype=np.int32,
        )

        meta = {
            "image_width_px": _scalar_from_npz(
                data,
                "image_width_px",
                instance_map.shape[1],
            ),
            "image_height_px": _scalar_from_npz(
                data,
                "image_height_px",
                instance_map.shape[0],
            ),
            "wire_count": _scalar_from_npz(
                data,
                "wire_count",
                int(instance_map.max()),
            ),
            "pixel_size_mm": _scalar_from_npz(
                data,
                "pixel_size_mm",
                None,
            ),
            "camera_center_x_px": _scalar_from_npz(
                data,
                "camera_center_x_px",
                None,
            ),
        }

    if instance_map.ndim != 2:
        raise ValueError(
            f"instance_map 必须是二维，当前 shape={instance_map.shape}"
        )

    if np.any(instance_map < 0):
        raise ValueError("instance_map 中存在负 ID。")

    return instance_map, meta


def load_mapped_height(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到融合后的真实高度文件: {path}\n"
            "请先运行带 mapped_height_mm.npz 输出的融合程序。"
        )

    with np.load(path, allow_pickle=False) as data:
        if "height_mm" not in data:
            raise KeyError(
                f"{path} 中不存在 height_mm"
            )
        if "valid_mask" not in data:
            raise KeyError(
                f"{path} 中不存在 valid_mask"
            )

        height_mm = np.asarray(
            data["height_mm"],
            dtype=np.float32,
        )
        valid_mask = np.asarray(
            data["valid_mask"],
            dtype=np.uint8,
        ).astype(bool)

        residual_mm = (
            np.asarray(
                data["residual_mm"],
                dtype=np.float32,
            )
            if "residual_mm" in data
            else None
        )

        camera_x_axis_mm = (
            np.asarray(
                data["camera_x_axis_mm"],
                dtype=np.float32,
            )
            if "camera_x_axis_mm" in data
            else None
        )

        angle_axis_deg = (
            np.asarray(
                data["angle_axis_deg"],
                dtype=np.float32,
            )
            if "angle_axis_deg" in data
            else None
        )

        meta = {
            "camera_width_px": _scalar_from_npz(
                data,
                "camera_width_px",
                height_mm.shape[1],
            ),
            "camera_height_px": _scalar_from_npz(
                data,
                "camera_height_px",
                height_mm.shape[0],
            ),
            "camera_zero_x_px": _scalar_from_npz(
                data,
                "camera_zero_x_px",
                None,
            ),
            "camera_x_mm_per_pixel": _scalar_from_npz(
                data,
                "camera_x_mm_per_pixel",
                None,
            ),
            "horizontal_x_scale": _scalar_from_npz(
                data,
                "horizontal_x_scale",
                None,
            ),
            "laser_left_on_camera_px": _scalar_from_npz(
                data,
                "laser_left_on_camera_px",
                None,
            ),
        }

    if height_mm.ndim != 2:
        raise ValueError(
            f"height_mm 必须是二维，当前 shape={height_mm.shape}"
        )
    if valid_mask.shape != height_mm.shape:
        raise ValueError(
            "valid_mask 与 height_mm shape 不一致: "
            f"{valid_mask.shape} vs {height_mm.shape}"
        )
    if residual_mm is not None and residual_mm.shape != height_mm.shape:
        raise ValueError(
            "residual_mm 与 height_mm shape 不一致: "
            f"{residual_mm.shape} vs {height_mm.shape}"
        )

    # 最终有效高度一定同时满足 valid_mask 和 finite(height_mm)。
    valid_mask &= np.isfinite(height_mm)

    return (
        height_mm,
        valid_mask,
        residual_mm,
        camera_x_axis_mm,
        angle_axis_deg,
        meta,
    )


def load_predictions_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到分割结果 CSV: {path}"
        )

    rows = {}
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                wire_id = int(row["id"])
            except Exception:
                continue
            rows[wire_id] = row
    return rows


# =============================================================================
# VALIDATION
# =============================================================================

def validate_coordinate_grids(
    instance_map,
    instance_meta,
    height_mm,
    height_meta,
    camera_x_axis_mm,
    angle_axis_deg,
):
    errors = []
    warnings = []

    if instance_map.shape != height_mm.shape:
        errors.append(
            "instance_map 与 mapped height 尺寸不一致："
            f"{instance_map.shape} vs {height_mm.shape}"
        )

    h, w = instance_map.shape

    iw = instance_meta.get("image_width_px")
    ih = instance_meta.get("image_height_px")
    hw = height_meta.get("camera_width_px")
    hh = height_meta.get("camera_height_px")

    if iw is not None and int(iw) != w:
        warnings.append(
            f"instance NPZ image_width_px={iw}, 实际 map width={w}"
        )
    if ih is not None and int(ih) != h:
        warnings.append(
            f"instance NPZ image_height_px={ih}, 实际 map height={h}"
        )
    if hw is not None and int(hw) != w:
        errors.append(
            f"mapped height camera_width_px={hw}, instance width={w}"
        )
    if hh is not None and int(hh) != h:
        errors.append(
            f"mapped height camera_height_px={hh}, instance height={h}"
        )

    if camera_x_axis_mm is not None and camera_x_axis_mm.shape != (w,):
        errors.append(
            "camera_x_axis_mm 长度与图像宽度不一致："
            f"{camera_x_axis_mm.shape} vs ({w},)"
        )
    if angle_axis_deg is not None and angle_axis_deg.shape != (h,):
        errors.append(
            "angle_axis_deg 长度与图像高度不一致："
            f"{angle_axis_deg.shape} vs ({h},)"
        )

    instance_center = instance_meta.get("camera_center_x_px")
    mapped_center = height_meta.get("camera_zero_x_px")
    if (
        instance_center is not None
        and mapped_center is not None
        and abs(float(instance_center) - float(mapped_center)) > 1e-3
    ):
        errors.append(
            "相机 X=0 原点不一致："
            f"instance={instance_center}, mapped_height={mapped_center}"
        )

    instance_mmpp = instance_meta.get("pixel_size_mm")
    mapped_mmpp = height_meta.get("camera_x_mm_per_pixel")
    if (
        instance_mmpp is not None
        and mapped_mmpp is not None
        and abs(float(instance_mmpp) - float(mapped_mmpp)) > 1e-6
    ):
        errors.append(
            "相机 X 像素比例不一致："
            f"instance={instance_mmpp}, mapped_height={mapped_mmpp}"
        )

    x_scale = height_meta.get("horizontal_x_scale")
    if x_scale is not None and abs(float(x_scale) - 1.0) > 1e-6:
        warnings.append(
            f"mapped height horizontal_x_scale={x_scale}，不是 1.0。"
        )

    if errors:
        message = "\n".join(f"  - {x}" for x in errors)
        raise RuntimeError(
            "相机丝实例与 mapped_height_mm 不能直接逐像素对应：\n"
            + message
        )

    return warnings


# =============================================================================
# FAST PER-WIRE GROUP STATISTICS
# =============================================================================

def _group_stats_by_id(
    ids: np.ndarray,
    values: np.ndarray,
    max_id: int,
    mad_sigma: float,
):
    """
    ids and values are already restricted to finite valid samples.

    Sort once by wire ID, then calculate robust statistics per group.
    """
    result = {
        "count": np.zeros(max_id + 1, dtype=np.int64),
        "mean": np.full(max_id + 1, np.nan, dtype=np.float64),
        "median": np.full(max_id + 1, np.nan, dtype=np.float64),
        "min": np.full(max_id + 1, np.nan, dtype=np.float64),
        "max": np.full(max_id + 1, np.nan, dtype=np.float64),
        "std": np.full(max_id + 1, np.nan, dtype=np.float64),
        "p05": np.full(max_id + 1, np.nan, dtype=np.float64),
        "p95": np.full(max_id + 1, np.nan, dtype=np.float64),
        "mad": np.full(max_id + 1, np.nan, dtype=np.float64),
        "outlier_count": np.zeros(max_id + 1, dtype=np.int64),
        "outlier_fraction": np.full(max_id + 1, np.nan, dtype=np.float64),
    }

    if ids.size == 0:
        return result

    order = np.argsort(ids, kind="stable")
    ids_sorted = ids[order]
    values_sorted = values[order].astype(
        np.float64,
        copy=False,
    )

    unique_ids, starts, counts = np.unique(
        ids_sorted,
        return_index=True,
        return_counts=True,
    )

    for wire_id, start, count in zip(
        unique_ids.tolist(),
        starts.tolist(),
        counts.tolist(),
    ):
        if wire_id <= 0 or wire_id > max_id:
            continue

        vals = values_sorted[start : start + count]
        if vals.size == 0:
            continue

        median = float(np.median(vals))
        abs_dev = np.abs(vals - median)
        mad = float(np.median(abs_dev))
        robust_sigma = 1.4826 * mad

        if robust_sigma < MAD_MIN_SCALE_MM:
            outlier = np.zeros(vals.shape, dtype=bool)
        else:
            outlier = (
                abs_dev
                > float(mad_sigma) * robust_sigma
            )

        result["count"][wire_id] = int(vals.size)
        result["mean"][wire_id] = float(np.mean(vals))
        result["median"][wire_id] = median
        result["min"][wire_id] = float(np.min(vals))
        result["max"][wire_id] = float(np.max(vals))
        result["std"][wire_id] = float(np.std(vals))
        result["p05"][wire_id] = float(np.percentile(vals, 5.0))
        result["p95"][wire_id] = float(np.percentile(vals, 95.0))
        result["mad"][wire_id] = mad
        result["outlier_count"][wire_id] = int(np.count_nonzero(outlier))
        result["outlier_fraction"][wire_id] = float(np.mean(outlier))

    return result


def compute_all_wire_statistics(
    instance_map,
    height_mm,
    valid_mask,
    residual_mm,
    max_id,
    mad_sigma,
):
    ids_all = instance_map.ravel()

    wire_pixels = ids_all > 0
    wire_ids = ids_all[wire_pixels]

    total_counts = np.bincount(
        wire_ids,
        minlength=max_id + 1,
    ).astype(np.int64)

    h_flat = height_mm.ravel()[wire_pixels]
    valid_h = (
        valid_mask.ravel()[wire_pixels]
        & np.isfinite(h_flat)
    )

    valid_ids = wire_ids[valid_h]
    valid_h_values = h_flat[valid_h]

    height_stats = _group_stats_by_id(
        valid_ids,
        valid_h_values,
        max_id,
        mad_sigma,
    )

    valid_counts = height_stats["count"]

    coverage = np.zeros(max_id + 1, dtype=np.float64)
    nonzero = total_counts > 0
    coverage[nonzero] = (
        valid_counts[nonzero]
        / total_counts[nonzero]
    )

    residual_stats = None
    if residual_mm is not None:
        r_flat = residual_mm.ravel()[wire_pixels]
        valid_r = valid_h & np.isfinite(r_flat)
        residual_stats = _group_stats_by_id(
            wire_ids[valid_r],
            r_flat[valid_r],
            max_id,
            mad_sigma,
        )

    return {
        "total_counts": total_counts,
        "valid_counts": valid_counts,
        "coverage": coverage,
        "height": height_stats,
        "residual": residual_stats,
    }


# =============================================================================
# 3D READINESS
# =============================================================================

READY_CODE = {
    "NO_LASER": 0,
    "LOW_COVERAGE": 1,
    "PARTIAL": 2,
    "READY_DIRECT": 3,
}


def classify_3d_ready(
    coverage,
    valid_pixels,
    ready_coverage,
    partial_coverage,
    ready_min_valid,
    partial_min_valid,
):
    if valid_pixels <= 0:
        return "NO_LASER"

    if (
        coverage >= ready_coverage
        and valid_pixels >= ready_min_valid
    ):
        return "READY_DIRECT"

    if (
        coverage >= partial_coverage
        and valid_pixels >= partial_min_valid
    ):
        return "PARTIAL"

    return "LOW_COVERAGE"


# =============================================================================
# CSV / NPZ OUTPUT
# =============================================================================

def _float_or_nan(row, key):
    try:
        value = float(row.get(key, ""))
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _nearest_axis_value(axis, coord):
    if axis is None or not np.isfinite(coord):
        return np.nan
    idx = int(np.clip(round(float(coord)), 0, len(axis) - 1))
    return float(axis[idx])


def build_wire_rows(
    predictions,
    stats,
    camera_x_axis_mm,
    angle_axis_deg,
    max_id,
    ready_coverage,
    partial_coverage,
    ready_min_valid,
    partial_min_valid,
):
    rows = []

    for wire_id in range(1, max_id + 1):
        src = predictions.get(wire_id, {})

        gx = _float_or_nan(
            src,
            "global_center_x_px",
        )
        if not np.isfinite(gx):
            gx = _float_or_nan(src, "center_x")

        gy = _float_or_nan(
            src,
            "global_center_y_px",
        )
        if not np.isfinite(gy):
            gy = _float_or_nan(src, "center_y")

        x_mm_csv = _float_or_nan(src, "center_x_mm")
        theta_csv = _float_or_nan(src, "theta_deg")
        diameter_mm = _float_or_nan(src, "diameter_mm")
        diameter_px = _float_or_nan(src, "diameter_px")
        confidence = _float_or_nan(src, "confidence")

        x_mm_axis = _nearest_axis_value(
            camera_x_axis_mm,
            gx,
        )
        theta_axis = _nearest_axis_value(
            angle_axis_deg,
            gy,
        )

        total_px = int(stats["total_counts"][wire_id])
        valid_px = int(stats["valid_counts"][wire_id])
        coverage = float(stats["coverage"][wire_id])

        status = classify_3d_ready(
            coverage,
            valid_px,
            ready_coverage,
            partial_coverage,
            ready_min_valid,
            partial_min_valid,
        )

        h = stats["height"]
        r = stats["residual"]

        row = {
            "wire_id": wire_id,

            "global_center_x_px": gx,
            "global_center_y_px": gy,

            "center_x_mm_csv": x_mm_csv,
            "center_x_mm_mapped_axis": x_mm_axis,
            "center_x_mm_difference": (
                x_mm_csv - x_mm_axis
                if np.isfinite(x_mm_csv) and np.isfinite(x_mm_axis)
                else np.nan
            ),

            "theta_deg_csv": theta_csv,
            "theta_deg_mapped_axis": theta_axis,
            "theta_deg_difference": (
                theta_csv - theta_axis
                if np.isfinite(theta_csv) and np.isfinite(theta_axis)
                else np.nan
            ),

            "diameter_px": diameter_px,
            "diameter_mm": diameter_mm,
            "confidence": confidence,

            "mask_pixel_count": total_px,
            "laser_valid_pixel_count": valid_px,
            "laser_coverage_ratio": coverage,
            "laser_coverage_percent": coverage * 100.0,

            "height_mean_mm": h["mean"][wire_id],
            "height_median_mm": h["median"][wire_id],
            "height_min_mm": h["min"][wire_id],
            "height_max_mm": h["max"][wire_id],
            "height_std_mm": h["std"][wire_id],
            "height_p05_mm": h["p05"][wire_id],
            "height_p95_mm": h["p95"][wire_id],
            "height_p05_to_p95_mm": (
                h["p95"][wire_id] - h["p05"][wire_id]
                if np.isfinite(h["p95"][wire_id])
                and np.isfinite(h["p05"][wire_id])
                else np.nan
            ),
            "height_mad_mm": h["mad"][wire_id],
            "height_outlier_count_mad": int(
                h["outlier_count"][wire_id]
            ),
            "height_outlier_fraction_mad": h[
                "outlier_fraction"
            ][wire_id],

            "3d_status": status,
            "3d_status_code": READY_CODE[status],
        }

        if r is not None:
            row.update(
                {
                    "residual_valid_pixel_count": int(
                        r["count"][wire_id]
                    ),
                    "residual_mean_mm": r["mean"][wire_id],
                    "residual_median_mm": r["median"][wire_id],
                    "residual_min_mm": r["min"][wire_id],
                    "residual_max_mm": r["max"][wire_id],
                    "residual_std_mm": r["std"][wire_id],
                    "residual_p05_mm": r["p05"][wire_id],
                    "residual_p95_mm": r["p95"][wire_id],
                    "residual_p05_to_p95_mm": (
                        r["p95"][wire_id] - r["p05"][wire_id]
                        if np.isfinite(r["p95"][wire_id])
                        and np.isfinite(r["p05"][wire_id])
                        else np.nan
                    ),
                    "residual_mad_mm": r["mad"][wire_id],
                    "residual_outlier_count_mad": int(
                        r["outlier_count"][wire_id]
                    ),
                    "residual_outlier_fraction_mad": r[
                        "outlier_fraction"
                    ][wire_id],
                }
            )
        else:
            row.update(
                {
                    "residual_valid_pixel_count": 0,
                    "residual_mean_mm": np.nan,
                    "residual_median_mm": np.nan,
                    "residual_min_mm": np.nan,
                    "residual_max_mm": np.nan,
                    "residual_std_mm": np.nan,
                    "residual_p05_mm": np.nan,
                    "residual_p95_mm": np.nan,
                    "residual_p05_to_p95_mm": np.nan,
                    "residual_mad_mm": np.nan,
                    "residual_outlier_count_mad": 0,
                    "residual_outlier_fraction_mad": np.nan,
                }
            )

        rows.append(row)

    return rows


def save_rows_csv(path: Path, rows):
    if not rows:
        return

    fields = list(rows[0].keys())
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)


def save_rows_npz(path: Path, rows):
    if not rows:
        return

    def arr(key, dtype=np.float32):
        values = []
        for row in rows:
            value = row.get(key, np.nan)
            try:
                values.append(float(value))
            except Exception:
                values.append(np.nan)
        return np.asarray(values, dtype=dtype)

    status_strings = np.asarray(
        [row["3d_status"] for row in rows]
    )

    np.savez_compressed(
        path,
        wire_id=np.asarray(
            [row["wire_id"] for row in rows],
            dtype=np.int32,
        ),
        global_center_x_px=arr("global_center_x_px"),
        global_center_y_px=arr("global_center_y_px"),
        center_x_mm=arr("center_x_mm_mapped_axis"),
        theta_deg=arr("theta_deg_mapped_axis"),
        diameter_mm=arr("diameter_mm"),
        confidence=arr("confidence"),
        mask_pixel_count=arr(
            "mask_pixel_count",
            np.int32,
        ),
        laser_valid_pixel_count=arr(
            "laser_valid_pixel_count",
            np.int32,
        ),
        laser_coverage_ratio=arr(
            "laser_coverage_ratio"
        ),
        height_mean_mm=arr("height_mean_mm"),
        height_median_mm=arr("height_median_mm"),
        height_min_mm=arr("height_min_mm"),
        height_max_mm=arr("height_max_mm"),
        height_std_mm=arr("height_std_mm"),
        height_p05_mm=arr("height_p05_mm"),
        height_p95_mm=arr("height_p95_mm"),
        residual_mean_mm=arr("residual_mean_mm"),
        residual_median_mm=arr("residual_median_mm"),
        residual_min_mm=arr("residual_min_mm"),
        residual_max_mm=arr("residual_max_mm"),
        residual_std_mm=arr("residual_std_mm"),
        residual_p05_mm=arr("residual_p05_mm"),
        residual_p95_mm=arr("residual_p95_mm"),
        residual_outlier_fraction_mad=arr(
            "residual_outlier_fraction_mad"
        ),
        status_code=np.asarray(
            [row["3d_status_code"] for row in rows],
            dtype=np.int8,
        ),
        status=status_strings,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    rows,
    instance_map,
    height_mm,
    valid_mask,
    instance_meta,
    height_meta,
    warnings,
    args,
):
    status_counts = {
        name: 0
        for name in READY_CODE
    }
    for row in rows:
        status_counts[row["3d_status"]] += 1

    total_wires = len(rows)

    coverage = np.asarray(
        [
            row["laser_coverage_ratio"]
            for row in rows
        ],
        dtype=np.float64,
    )

    mask_pixels = int(
        np.count_nonzero(instance_map > 0)
    )
    wire_valid_pixels = int(
        np.count_nonzero(
            (instance_map > 0)
            & valid_mask
            & np.isfinite(height_mm)
        )
    )

    coverage_bins = [
        ("100%", 1.0, 1.0),
        ("95%-<100%", 0.95, 1.0),
        ("90%-<95%", 0.90, 0.95),
        ("80%-<90%", 0.80, 0.90),
        ("50%-<80%", 0.50, 0.80),
        ("30%-<50%", 0.30, 0.50),
        (">0%-<30%", np.nextafter(0.0, 1.0), 0.30),
        ("0%", 0.0, 0.0),
    ]

    distribution = {}
    for label, lo, hi in coverage_bins:
        if label == "100%":
            count = int(np.count_nonzero(np.isclose(coverage, 1.0)))
        elif label == "0%":
            count = int(np.count_nonzero(coverage == 0.0))
        elif label == "95%-<100%":
            count = int(
                np.count_nonzero(
                    (coverage >= lo)
                    & (coverage < hi)
                )
            )
        else:
            count = int(
                np.count_nonzero(
                    (coverage >= lo)
                    & (coverage < hi)
                )
            )
        distribution[label] = {
            "count": count,
            "fraction": (
                count / total_wires
                if total_wires
                else 0.0
            ),
        }

    summary = {
        "input_files": {
            "instance_map_npz": str(args.instance_map),
            "predictions_csv": str(args.predictions_csv),
            "mapped_height_npz": str(args.mapped_height),
            "camera_image": str(args.camera_image),
        },
        "coordinate_validation": {
            "shape_rows_cols": [
                int(instance_map.shape[0]),
                int(instance_map.shape[1]),
            ],
            "instance_camera_center_x_px": instance_meta.get(
                "camera_center_x_px"
            ),
            "mapped_camera_zero_x_px": height_meta.get(
                "camera_zero_x_px"
            ),
            "instance_pixel_size_mm": instance_meta.get(
                "pixel_size_mm"
            ),
            "mapped_camera_x_mm_per_pixel": height_meta.get(
                "camera_x_mm_per_pixel"
            ),
            "horizontal_x_scale": height_meta.get(
                "horizontal_x_scale"
            ),
            "warnings": warnings,
        },
        "wire_count": int(total_wires),
        "wire_mask_pixels_total": mask_pixels,
        "wire_laser_valid_pixels_total": wire_valid_pixels,
        "overall_wire_pixel_laser_coverage_ratio": (
            wire_valid_pixels / mask_pixels
            if mask_pixels
            else 0.0
        ),
        "coverage_distribution": distribution,
        "3d_thresholds": {
            "ready_direct_min_coverage": float(args.ready_coverage),
            "partial_min_coverage": float(args.partial_coverage),
            "ready_direct_min_valid_pixels": int(args.ready_min_valid),
            "partial_min_valid_pixels": int(args.partial_min_valid),
        },
        "3d_status_counts": {
            name: {
                "count": int(count),
                "fraction": (
                    count / total_wires
                    if total_wires
                    else 0.0
                ),
            }
            for name, count in status_counts.items()
        },
        "coverage_statistics": {
            "min": float(np.min(coverage)) if coverage.size else None,
            "mean": float(np.mean(coverage)) if coverage.size else None,
            "median": float(np.median(coverage)) if coverage.size else None,
            "p05": float(np.percentile(coverage, 5)) if coverage.size else None,
            "p25": float(np.percentile(coverage, 25)) if coverage.size else None,
            "p75": float(np.percentile(coverage, 75)) if coverage.size else None,
            "p95": float(np.percentile(coverage, 95)) if coverage.size else None,
            "max": float(np.max(coverage)) if coverage.size else None,
        },
    }

    return summary


def save_summary_txt(path: Path, summary):
    statuses = summary["3d_status_counts"]
    dist = summary["coverage_distribution"]

    lines = [
        "丝端面 - 激光高度对应分析",
        "=" * 60,
        "",
        f"shape = {summary['coordinate_validation']['shape_rows_cols']}",
        f"wire_count = {summary['wire_count']}",
        (
            "overall wire-pixel laser coverage = "
            f"{summary['overall_wire_pixel_laser_coverage_ratio']:.4%}"
        ),
        "",
        "3D 建模等级：",
        (
            "  READY_DIRECT = "
            f"{statuses['READY_DIRECT']['count']} "
            f"({statuses['READY_DIRECT']['fraction']:.2%})"
        ),
        (
            "  PARTIAL      = "
            f"{statuses['PARTIAL']['count']} "
            f"({statuses['PARTIAL']['fraction']:.2%})"
        ),
        (
            "  LOW_COVERAGE = "
            f"{statuses['LOW_COVERAGE']['count']} "
            f"({statuses['LOW_COVERAGE']['fraction']:.2%})"
        ),
        (
            "  NO_LASER     = "
            f"{statuses['NO_LASER']['count']} "
            f"({statuses['NO_LASER']['fraction']:.2%})"
        ),
        "",
        "覆盖率分布：",
    ]

    for label, item in dist.items():
        lines.append(
            f"  {label:10s}: "
            f"{item['count']:5d} "
            f"({item['fraction']:.2%})"
        )

    lines.extend(
        [
            "",
            "坐标检查：",
            (
                "  instance camera center x = "
                f"{summary['coordinate_validation']['instance_camera_center_x_px']}"
            ),
            (
                "  mapped height camera zero x = "
                f"{summary['coordinate_validation']['mapped_camera_zero_x_px']}"
            ),
            (
                "  instance mm/px = "
                f"{summary['coordinate_validation']['instance_pixel_size_mm']}"
            ),
            (
                "  mapped height mm/px = "
                f"{summary['coordinate_validation']['mapped_camera_x_mm_per_pixel']}"
            ),
            (
                "  horizontal x scale = "
                f"{summary['coordinate_validation']['horizontal_x_scale']}"
            ),
        ]
    )

    warnings = summary["coordinate_validation"]["warnings"]
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(
            f"  - {w}"
            for w in warnings
        )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_status_overlay(
    camera_path: Path,
    instance_map: np.ndarray,
    rows,
    out_path: Path,
):
    if not camera_path.exists():
        print(
            f"WARNING: camera image not found, skip overlay: {camera_path}"
        )
        return False

    img = cv2.imread(str(camera_path))
    if img is None:
        print(
            f"WARNING: cannot read camera image, skip overlay: {camera_path}"
        )
        return False

    if img.shape[:2] != instance_map.shape:
        print(
            "WARNING: camera image shape differs from instance map; "
            "skip status overlay. "
            f"{img.shape[:2]} vs {instance_map.shape}"
        )
        return False

    # BGR:
    # READY_DIRECT green, PARTIAL yellow,
    # LOW_COVERAGE orange, NO_LASER red.
    colors = {
        "READY_DIRECT": (0, 220, 0),
        "PARTIAL": (0, 220, 220),
        "LOW_COVERAGE": (0, 130, 255),
        "NO_LASER": (0, 0, 255),
    }

    overlay = img.copy()

    for row in rows:
        wire_id = int(row["wire_id"])
        mask = instance_map == wire_id
        if not np.any(mask):
            continue

        color = np.asarray(
            colors[row["3d_status"]],
            dtype=np.float32,
        )
        base = img[mask].astype(np.float32)

        blend = (
            (1.0 - float(OVERLAY_ALPHA)) * base
            + float(OVERLAY_ALPHA) * color[None, :]
        )
        overlay[mask] = np.clip(
            blend,
            0,
            255,
        ).astype(np.uint8)

    cv2.imwrite(str(out_path), overlay)
    return True


def save_coverage_histogram(rows, out_path: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"WARNING: matplotlib unavailable, skip histogram: {exc}"
        )
        return False

    coverage = np.asarray(
        [
            row["laser_coverage_ratio"] * 100.0
            for row in rows
        ],
        dtype=np.float64,
    )

    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111)
    ax.hist(
        coverage,
        bins=np.linspace(0, 100, 21),
    )
    ax.set_xlabel("Laser coverage of each wire end-face (%)")
    ax.set_ylabel("Wire count")
    ax.set_title("Wire - laser coverage distribution")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    args.instance_map = args.instance_map.resolve()
    args.predictions_csv = args.predictions_csv.resolve()
    args.mapped_height = args.mapped_height.resolve()
    args.camera_image = args.camera_image.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("丝端面 - 激光高度对应分析")
    print("=" * 72)
    print("Paths are derived from config.py by default:")
    print(f"  instance map : {args.instance_map}")
    print(f"  predictions  : {args.predictions_csv}")
    print(f"  mapped height: {args.mapped_height}")
    print(f"  camera image : {args.camera_image}")
    print(f"  output       : {args.out_dir}")
    print("")

    instance_map, instance_meta = load_instance_map(
        args.instance_map
    )
    predictions = load_predictions_csv(
        args.predictions_csv
    )

    (
        height_mm,
        valid_mask,
        residual_mm,
        camera_x_axis_mm,
        angle_axis_deg,
        height_meta,
    ) = load_mapped_height(
        args.mapped_height
    )

    warnings = validate_coordinate_grids(
        instance_map,
        instance_meta,
        height_mm,
        height_meta,
        camera_x_axis_mm,
        angle_axis_deg,
    )

    max_id = int(instance_map.max())
    ids_present = np.unique(instance_map)
    ids_present = ids_present[ids_present > 0]

    if ids_present.size != max_id:
        warnings.append(
            f"instance ID 1..{max_id} 中存在缺号；"
            f"实际 unique wire IDs={ids_present.size}"
        )

    missing_csv_ids = [
        i
        for i in range(1, max_id + 1)
        if i not in predictions
    ]
    if missing_csv_ids:
        warnings.append(
            "wire_full_predictions.csv 缺少 "
            f"{len(missing_csv_ids)} 个 instance ID。"
        )

    print("Coordinate validation: OK")
    print(
        f"  common grid = "
        f"{instance_map.shape[0]} rows x {instance_map.shape[1]} cols"
    )
    print(
        f"  wire IDs = {ids_present.size}, max ID={max_id}"
    )
    print(
        f"  mapped valid pixels = "
        f"{np.count_nonzero(valid_mask):,}"
    )
    print(
        f"  residual_mm = "
        f"{'YES' if residual_mm is not None else 'NO'}"
    )

    for warning in warnings:
        print(f"WARNING: {warning}")

    print("")
    print("Computing per-wire statistics...")

    stats = compute_all_wire_statistics(
        instance_map,
        height_mm,
        valid_mask,
        residual_mm,
        max_id,
        args.mad_sigma,
    )

    rows = build_wire_rows(
        predictions,
        stats,
        camera_x_axis_mm,
        angle_axis_deg,
        max_id,
        args.ready_coverage,
        args.partial_coverage,
        args.ready_min_valid,
        args.partial_min_valid,
    )

    summary = build_summary(
        rows,
        instance_map,
        height_mm,
        valid_mask,
        instance_meta,
        height_meta,
        warnings,
        args,
    )

    csv_path = (
        args.out_dir
        / "wire_laser_correspondence.csv"
    )
    npz_path = (
        args.out_dir
        / "wire_laser_correspondence.npz"
    )
    summary_json = (
        args.out_dir
        / "wire_laser_summary.json"
    )
    summary_txt = (
        args.out_dir
        / "wire_laser_summary.txt"
    )
    overlay_path = (
        args.out_dir
        / "wire_laser_ready_overlay.png"
    )
    histogram_path = (
        args.out_dir
        / "wire_laser_coverage_histogram.png"
    )

    save_rows_csv(csv_path, rows)
    save_rows_npz(npz_path, rows)

    summary_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_summary_txt(
        summary_txt,
        summary,
    )

    if SAVE_STATUS_OVERLAY:
        save_status_overlay(
            args.camera_image,
            instance_map,
            rows,
            overlay_path,
        )

    if SAVE_COVERAGE_HISTOGRAM:
        save_coverage_histogram(
            rows,
            histogram_path,
        )

    statuses = summary["3d_status_counts"]

    print("")
    print("=" * 72)
    print("Analysis complete")
    print("=" * 72)
    print(f"Total wires: {summary['wire_count']}")
    print(
        "Overall wire-pixel laser coverage: "
        f"{summary['overall_wire_pixel_laser_coverage_ratio']:.2%}"
    )
    print("")
    print(
        f"READY_DIRECT : {statuses['READY_DIRECT']['count']} "
        f"({statuses['READY_DIRECT']['fraction']:.2%})"
    )
    print(
        f"PARTIAL      : {statuses['PARTIAL']['count']} "
        f"({statuses['PARTIAL']['fraction']:.2%})"
    )
    print(
        f"LOW_COVERAGE : {statuses['LOW_COVERAGE']['count']} "
        f"({statuses['LOW_COVERAGE']['fraction']:.2%})"
    )
    print(
        f"NO_LASER     : {statuses['NO_LASER']['count']} "
        f"({statuses['NO_LASER']['fraction']:.2%})"
    )
    print("")
    print("Outputs:")
    print(f"  CSV     : {csv_path}")
    print(f"  NPZ     : {npz_path}")
    print(f"  Summary : {summary_json}")
    print(f"  Summary : {summary_txt}")
    if overlay_path.exists():
        print(f"  Overlay : {overlay_path}")
    if histogram_path.exists():
        print(f"  Histogram: {histogram_path}")
    print("")
    print(
        "For the next 3-D program, use "
        "wire_laser_correspondence.npz + wire_instance_map.npz "
        "+ mapped_height_mm.npz."
    )


if __name__ == "__main__":
    main()
