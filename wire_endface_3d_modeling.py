from __future__ import annotations

"""
标准圆丝端面 3D V5：全部丝建模 + CloudCompare 单丝信息点
================================================

目标
----
最终每根丝端面强制为“标准平面圆”，但圆内部不再使用单一颜色。

三种信息明确分工：

    YOLO 分割 / 已有圆拟合：
        -> wire_id
        -> 圆心
        -> radius_px / diameter_px

    相机拼接 RGB：
        -> 圆形端面内部的真实颜色 / 明暗 / 纹理

    mapped_height_mm.npz：
        -> 每根丝真实径向高度 H

最终：
        有激光高度的丝：
            R_wire = 75 + H_wire

        无激光高度的丝：
            仍然保留相机标准圆端面，
            但其径向位置固定为轮子基准半径：
                R_wire = 75 mm

这样全部 7088 根丝都会进入最终 3D，
同时不会伪造“无激光”的高度值。

重要变化
--------
上一版 true_rgb PLY 只读取了每根丝“圆心一个像素”的 RGB，
然后把整个圆涂成同一种颜色。

本版改为：

    标准圆内的每一个 3D 网格顶点
        -> 映射回该丝在相机拼接图中的局部位置
        -> 采样相机 RGB
        -> 得到真正的端面纹理变化

同时：
    原 YOLO mask 内的位置：
        直接使用真实相机 RGB

    强制圆后新增、但原 YOLO mask 外的位置：
        不直接读取相邻背景/旁边钢丝
        而是复制“当前 wire_id 原 mask 内最近像素”的 RGB

这样可以避免标准圆扩大后把邻居颜色错误贴到当前丝端面。

360°周期
-------
图像纵向 y 为圆周方向：

    y=0
    和
    y=H-1

物理上相邻。

所有以下过程都支持周期：
    - 圆形 owner map
    - 原 mask 局部 RGB patch
    - 跨 0°/360° 的端面纹理
    - 圆心 theta
    - 3D 圆盘

3D 几何
-------
每根丝生成真正的“平面圆盘”，不是把圆形像素弯曲贴到圆柱上。

假设不考虑丝倾斜：
    圆盘法向 = 轮子局部径向

中心：
    C = [Xc, R*cos(theta), R*sin(theta)]

局部圆盘基向量：
    ex = [1, 0, 0]
    et = [0, -sin(theta), cos(theta)]

圆盘局部坐标：
    u = 轴向
    v = 周向切向

3D：
    P = C + u*ex + v*et

其中：
    u, v 的单位均为 mm

相机纹理
--------
采用极坐标密集圆盘网格：

    中心
    + 多个 radial rings
    + 多个 angular segments

每个顶点都对应圆内一个亚像素相机坐标。
通过填补后的“当前丝 RGB patch”进行双线性采样。

因此：
    - 几何边缘仍是平滑标准圆
    - 颜色不是单色
    - 可以看到端面内部相机真实明暗纹理

默认输入
--------
config.SEGMENTATION_DIR / "wire_full_predictions.csv"
config.SEGMENTATION_DIR / "wire_instance_map.npz"
config.FUSION_DIR       / "mapped_height_mm.npz"
config.CAMERA_STITCHED_IMAGE

默认输出
--------
当前 run / wire_circle_3d_camera_texture/

    wire_instance_map_circle.npz
    wire_instance_map_circle_preview.png
    wire_circle_overlay.png

    wire_circle_laser_table.csv
    wire_circle_summary.json
    wire_circle_summary.txt

    wire_circle_endfaces_camera_texture_3d.npz

    wire_circle_endfaces_camera_rgb.ply
        -> 真实几何 + 相机端面 RGB 纹理
        -> 这是观察端面颜色/纹理的主要文件

    wire_circle_endfaces_height_color.ply
        -> 同样真实几何
        -> 颜色改为按 H 的伪彩色
        -> 只用于观察丝与丝之间的高度变化

单丝信息输出
------------
V5 新增两份关键文件：

    wire_centers_info.ply
        每根丝仅一个中心点。
        PLY 内保存多个自定义 scalar properties：
            wire_id
            diameter_mm
            height_mm
            center_radius_mm
            theta_deg
            laser_coverage_pct
            confidence
            status_code
            height_source_code
            center_x_px
            center_y_px

        推荐在 CloudCompare 中和完整端面 mesh 一起打开。
        把 point size 调大到 3~5，即可把每个中心点当成“信息探针”。

    wire_information.csv
        每根丝一行，保存完整参数及 3D 中心坐标。
        wire_id 是贯穿 YOLO、激光和 3D 的唯一索引。

状态编码：
    status_code:
        0 = READY_DIRECT
        1 = PARTIAL
        2 = LOW_COVERAGE
        3 = NO_LASER

    height_source_code:
        0 = REAL_LASER_HEIGHT
        1 = NO_LASER_BASELINE_RADIUS

依赖
----
numpy
opencv-python
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import config as project_config
except Exception as exc:
    raise RuntimeError(
        "无法导入项目 config.py。\n"
        "请把本程序放到 D:\\project\\scripts，并确保 config.py 可被导入。\n"
        f"原始错误: {exc}"
    ) from exc


# =============================================================================
# PATHS
# =============================================================================

SEGMENTATION_DIR = Path(project_config.SEGMENTATION_DIR)
FUSION_DIR = Path(project_config.FUSION_DIR)
CAMERA_STITCHED_IMAGE = Path(project_config.CAMERA_STITCHED_IMAGE)

DEFAULT_CIRCLE_CSV = (
    SEGMENTATION_DIR
    / "wire_full_predictions.csv"
)

DEFAULT_SOURCE_INSTANCE_MAP = (
    SEGMENTATION_DIR
    / "wire_instance_map.npz"
)

DEFAULT_MAPPED_HEIGHT = (
    FUSION_DIR
    / "mapped_height_mm.npz"
)

if bool(
    getattr(
        project_config,
        "USE_COMBINED_ACQUISITION",
        False,
    )
):
    DEFAULT_RUN_DIR = Path(
        project_config.ACTIVE_COMBINED_RUN_DIR
    )
else:
    DEFAULT_RUN_DIR = Path(
        project_config.RUN_DIR
    )

DEFAULT_OUT_DIR = (
    DEFAULT_RUN_DIR
    / "wire_circle_3d_camera_texture_all_wires_info"
)


# =============================================================================
# PHYSICAL SETTINGS
# =============================================================================

# Wheel
WHEEL_RADIUS_MM = 75.0

# KEYENCE LJ-X8060 reference
LASER_REFERENCE_DISTANCE_MM = 64.0

# H=0 is the 64 mm reference plane.
H_REF_MM = 0.0

# Positive H = surface closer to laser head = radial position increases.
HEIGHT_SIGN = +1.0


# =============================================================================
# HEIGHT SETTINGS
# =============================================================================

MAD_SIGMA = 4.0
MAD_MIN_SCALE_MM = 1e-4

READY_DIRECT_MIN_COVERAGE = 0.80
READY_DIRECT_MIN_VALID_PIXELS = 20

PARTIAL_MIN_COVERAGE = 0.30
PARTIAL_MIN_VALID_PIXELS = 10

# Default 3-D build:
# approximately READY_DIRECT + PARTIAL
DEFAULT_MIN_BUILD_COVERAGE = 0.30
DEFAULT_MIN_BUILD_VALID = 10

# Height is preferably sampled from original YOLO mask.
DEFAULT_HEIGHT_SAMPLING_MODE = "source-mask"


# =============================================================================
# CAMERA TEXTURE / CIRCLE MESH SETTINGS
# =============================================================================

# Number of angular points on every ring.
# 40 gives a smooth enough circle without making the mesh unnecessarily huge.
DEFAULT_ANGULAR_SEGMENTS = 40

# Radial ring count is automatically based on radius_px:
#
#     rings ~= ceil(radius_px * RADIAL_RING_DENSITY)
#
# so texture sampling density roughly follows the camera pixel resolution.
RADIAL_RING_DENSITY = 1.00

MIN_RADIAL_RINGS = 4
MAX_RADIAL_RINGS = 14

# Extra pixels around local RGB patch.
TEXTURE_PATCH_MARGIN_PX = 2

# Camera texture for new circle area:
#
# "nearest-wire":
#     recommended; pixels outside original YOLO mask borrow RGB from nearest
#     pixel belonging to the same wire.
#
# "raw-camera":
#     use original camera image everywhere inside fitted circle; this may bring
#     neighbor/background colors into newly added circle regions.
DEFAULT_TEXTURE_FILL_MODE = "nearest-wire"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def scalar_from_npz(
    data,
    key,
    default=None,
):
    if key not in data:
        return default

    a = np.asarray(
        data[key]
    )

    if a.size != 1:
        return default

    try:
        return (
            a.reshape(-1)[0].item()
        )
    except Exception:
        return a.reshape(-1)[0]


def safe_float(
    row: dict,
    *keys,
    default=np.nan,
):
    for key in keys:
        if key not in row:
            continue

        try:
            v = float(
                row[key]
            )

            if np.isfinite(v):
                return v
        except Exception:
            pass

    return float(
        default
    )


def robust_filter_mad(
    values: np.ndarray,
    sigma: float,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return (
            values,
            np.zeros(
                0,
                dtype=bool,
            ),
            np.nan,
            np.nan,
        )

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    robust_sigma = (
        1.4826 * mad
    )

    if robust_sigma < MAD_MIN_SCALE_MM:
        keep = np.ones(
            values.shape,
            dtype=bool,
        )
    else:
        keep = (
            np.abs(
                values - median
            )
            <= float(sigma)
            * robust_sigma
        )

    return (
        values[keep],
        keep,
        median,
        mad,
    )


def classify_status(
    coverage: float,
    valid_pixels: int,
):
    if valid_pixels <= 0:
        return "NO_LASER"

    if (
        coverage
        >= READY_DIRECT_MIN_COVERAGE
        and valid_pixels
        >= READY_DIRECT_MIN_VALID_PIXELS
    ):
        return "READY_DIRECT"

    if (
        coverage
        >= PARTIAL_MIN_COVERAGE
        and valid_pixels
        >= PARTIAL_MIN_VALID_PIXELS
    ):
        return "PARTIAL"

    return "LOW_COVERAGE"


def interpolate_axis(
    axis: np.ndarray | None,
    coord: float,
    fallback: float,
):
    if (
        axis is None
        or len(axis) == 0
        or not np.isfinite(coord)
    ):
        return float(
            fallback
        )

    xp = np.arange(
        len(axis),
        dtype=np.float64,
    )

    return float(
        np.interp(
            float(coord),
            xp,
            np.asarray(
                axis,
                dtype=np.float64,
            ),
        )
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build standard circular wire end-faces with "
            "per-vertex camera RGB texture and per-wire laser height."
        )
    )

    p.add_argument(
        "--circle-csv",
        type=Path,
        default=DEFAULT_CIRCLE_CSV,
    )

    p.add_argument(
        "--source-instance-map",
        type=Path,
        default=DEFAULT_SOURCE_INSTANCE_MAP,
    )

    p.add_argument(
        "--mapped-height",
        type=Path,
        default=DEFAULT_MAPPED_HEIGHT,
    )

    p.add_argument(
        "--camera-image",
        type=Path,
        default=CAMERA_STITCHED_IMAGE,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    p.add_argument(
        "--wheel-radius-mm",
        type=float,
        default=WHEEL_RADIUS_MM,
    )

    p.add_argument(
        "--mad-sigma",
        type=float,
        default=MAD_SIGMA,
    )

    p.add_argument(
        "--height-sampling-mode",
        choices=[
            "source-mask",
            "circle-map",
            "intersection",
        ],
        default=DEFAULT_HEIGHT_SAMPLING_MODE,
    )

    p.add_argument(
        "--min-build-coverage",
        type=float,
        default=DEFAULT_MIN_BUILD_COVERAGE,
    )

    p.add_argument(
        "--min-build-valid",
        type=int,
        default=DEFAULT_MIN_BUILD_VALID,
    )

    p.add_argument(
        "--angular-segments",
        type=int,
        default=DEFAULT_ANGULAR_SEGMENTS,
        help=(
            "Angular vertices on each disk ring. "
            "Default 40."
        ),
    )

    p.add_argument(
        "--radial-ring-density",
        type=float,
        default=RADIAL_RING_DENSITY,
        help=(
            "radial rings ~= radius_px * density. "
            "Default 1.0."
        ),
    )

    p.add_argument(
        "--texture-fill-mode",
        choices=[
            "nearest-wire",
            "raw-camera",
        ],
        default=DEFAULT_TEXTURE_FILL_MODE,
    )

    return p.parse_args()


# =============================================================================
# LOAD YOLO CIRCLE PARAMETERS
# =============================================================================

def load_circle_csv(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 YOLO 圆结果 CSV: {path}"
        )

    circles = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            try:
                wire_id = int(
                    row["id"]
                )
            except Exception:
                continue

            cx = safe_float(
                row,
                "global_center_x_px",
                "center_x",
            )

            cy = safe_float(
                row,
                "global_center_y_px",
                "center_y",
            )

            radius_px = safe_float(
                row,
                "radius_px",
            )

            diameter_px = safe_float(
                row,
                "diameter_px",
            )

            if not np.isfinite(
                radius_px
            ):
                if np.isfinite(
                    diameter_px
                ):
                    radius_px = (
                        diameter_px
                        * 0.5
                    )

            if not np.isfinite(
                diameter_px
            ):
                if np.isfinite(
                    radius_px
                ):
                    diameter_px = (
                        radius_px
                        * 2.0
                    )

            if (
                not np.isfinite(cx)
                or not np.isfinite(cy)
                or not np.isfinite(radius_px)
                or radius_px <= 0
            ):
                print(
                    "WARNING: skip invalid circle "
                    f"wire_id={wire_id}: "
                    f"cx={cx}, cy={cy}, "
                    f"radius={radius_px}"
                )
                continue

            circles.append(
                {
                    "wire_id": wire_id,

                    "center_x_px": float(
                        cx
                    ),

                    "center_y_px": float(
                        cy
                    ),

                    "radius_px": float(
                        radius_px
                    ),

                    "diameter_px": float(
                        diameter_px
                    ),

                    "diameter_mm_csv": safe_float(
                        row,
                        "diameter_mm",
                    ),

                    "confidence": safe_float(
                        row,
                        "confidence",
                        default=0.0,
                    ),

                    "seam_merged": int(
                        round(
                            safe_float(
                                row,
                                "seam_merged",
                                default=0.0,
                            )
                        )
                    ),
                }
            )

    if not circles:
        raise RuntimeError(
            f"{path} 中没有读取到有效圆参数。"
        )

    circles.sort(
        key=lambda x: x["wire_id"]
    )

    return circles


# =============================================================================
# LOAD MAPS
# =============================================================================

def load_source_instance_map(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到原始 YOLO instance map: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if "instance_map" not in data:
            raise KeyError(
                f"{path} 中不存在 instance_map"
            )

        instance_map = np.asarray(
            data["instance_map"],
            dtype=np.int32,
        )

        meta = {
            "pixel_size_mm":
                scalar_from_npz(
                    data,
                    "pixel_size_mm",
                    None,
                ),

            "camera_center_x_px":
                scalar_from_npz(
                    data,
                    "camera_center_x_px",
                    None,
                ),

            "wire_count":
                scalar_from_npz(
                    data,
                    "wire_count",
                    int(
                        instance_map.max()
                    ),
                ),
        }

    return (
        instance_map,
        meta,
    )


def load_mapped_height(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 mapped_height_mm.npz: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if (
            "height_mm" not in data
            or "valid_mask" not in data
        ):
            raise KeyError(
                f"{path} 必须包含 "
                "height_mm 和 valid_mask"
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
            "camera_zero_x_px":
                scalar_from_npz(
                    data,
                    "camera_zero_x_px",
                    None,
                ),

            "camera_x_mm_per_pixel":
                scalar_from_npz(
                    data,
                    "camera_x_mm_per_pixel",
                    None,
                ),
        }

    valid_mask &= np.isfinite(
        height_mm
    )

    return (
        height_mm,
        valid_mask,
        residual_mm,
        camera_x_axis_mm,
        angle_axis_deg,
        meta,
    )


# =============================================================================
# PERIODIC STANDARD-CIRCLE OWNER MAP
# =============================================================================

def build_circle_owner_map(
    shape: Tuple[int, int],
    circles: List[dict],
):
    """
    Create one unique wire_id per image pixel.

    If fitted circles overlap:
        owner = smaller normalized center distance.

    This owner map is only for:
        - preview
        - optional laser sampling

    Final 3-D disks are always generated COMPLETE
    and are not clipped by this owner map.
    """
    H, W = shape

    owner = np.zeros(
        (H, W),
        dtype=np.int32,
    )

    best_score = np.full(
        (H, W),
        np.inf,
        dtype=np.float32,
    )

    conflict_events = 0
    seam_circle_count = 0

    for idx, circle in enumerate(
        circles,
        start=1,
    ):
        wire_id = int(
            circle["wire_id"]
        )

        cx = float(
            circle["center_x_px"]
        )

        cy = (
            float(
                circle["center_y_px"]
            )
            % H
        )

        r = float(
            circle["radius_px"]
        )

        if (
            cy - r < 0
            or cy + r >= H
        ):
            seam_circle_count += 1

        x0 = max(
            0,
            int(
                math.floor(
                    cx - r
                )
            ),
        )

        x1 = min(
            W - 1,
            int(
                math.ceil(
                    cx + r
                )
            ),
        )

        yu0 = int(
            math.floor(
                cy - r
            )
        )

        yu1 = int(
            math.ceil(
                cy + r
            )
        )

        xs = np.arange(
            x0,
            x1 + 1,
            dtype=np.int32,
        )

        for yu in range(
            yu0,
            yu1 + 1,
        ):
            y = int(
                yu % H
            )

            dy = (
                (
                    float(y)
                    - cy
                    + H * 0.5
                )
                % H
                - H * 0.5
            )

            dx = (
                xs.astype(
                    np.float64
                )
                - cx
            )

            d2 = (
                dx * dx
                + dy * dy
            )

            inside = (
                d2
                <= r * r
            )

            if not np.any(
                inside
            ):
                continue

            xx = xs[
                inside
            ]

            score = (
                np.sqrt(
                    d2[inside]
                )
                / max(
                    r,
                    1e-9,
                )
            ).astype(
                np.float32
            )

            old_ids = owner[
                y,
                xx,
            ]

            conflict_events += int(
                np.count_nonzero(
                    (old_ids != 0)
                    & (
                        old_ids
                        != wire_id
                    )
                )
            )

            old_score = best_score[
                y,
                xx,
            ]

            take = (
                score
                < old_score
            )

            if np.any(
                take
            ):
                xxx = xx[
                    take
                ]

                owner[
                    y,
                    xxx,
                ] = wire_id

                best_score[
                    y,
                    xxx,
                ] = score[
                    take
                ]

        if (
            idx % 1000 == 0
            or idx == len(circles)
        ):
            print(
                "  circle owner map "
                f"{idx}/{len(circles)}"
            )

    return (
        owner,
        conflict_events,
        seam_circle_count,
    )


# =============================================================================
# PER-WIRE LASER HEIGHT
# =============================================================================

def compute_wire_height_table(
    circles,
    source_instance_map,
    circle_map,
    height_mm,
    valid_mask,
    residual_mm,
    mode: str,
    mad_sigma: float,
):
    rows = []

    for idx, circle in enumerate(
        circles,
        start=1,
    ):
        wire_id = int(
            circle["wire_id"]
        )

        if mode == "source-mask":

            sample_mask = (
                source_instance_map
                == wire_id
            )

        elif mode == "circle-map":

            sample_mask = (
                circle_map
                == wire_id
            )

        elif mode == "intersection":

            sample_mask = (
                (
                    source_instance_map
                    == wire_id
                )
                & (
                    circle_map
                    == wire_id
                )
            )

        else:
            raise ValueError(
                mode
            )

        sample_px = int(
            np.count_nonzero(
                sample_mask
            )
        )

        raw_valid = (
            sample_mask
            & valid_mask
            & np.isfinite(
                height_mm
            )
        )

        raw_values = (
            height_mm[
                raw_valid
            ]
        )

        raw_valid_count = int(
            raw_values.size
        )

        coverage = (
            raw_valid_count
            / sample_px
            if sample_px > 0
            else 0.0
        )

        (
            filtered,
            _,
            raw_median,
            raw_mad,
        ) = robust_filter_mad(
            raw_values,
            mad_sigma,
        )

        filtered_count = int(
            filtered.size
        )

        if filtered_count:

            H_base = float(
                np.median(
                    filtered
                )
            )

            H_mean = float(
                np.mean(
                    filtered
                )
            )

            H_std = float(
                np.std(
                    filtered
                )
            )

            H_min = float(
                np.min(
                    filtered
                )
            )

            H_max = float(
                np.max(
                    filtered
                )
            )

            H_p05 = float(
                np.percentile(
                    filtered,
                    5.0,
                )
            )

            H_p95 = float(
                np.percentile(
                    filtered,
                    95.0,
                )
            )

        else:

            H_base = np.nan
            H_mean = np.nan
            H_std = np.nan
            H_min = np.nan
            H_max = np.nan
            H_p05 = np.nan
            H_p95 = np.nan

        residual_median = np.nan

        if residual_mm is not None:

            rv_mask = (
                raw_valid
                & np.isfinite(
                    residual_mm
                )
            )

            rv = residual_mm[
                rv_mask
            ]

            if rv.size:
                residual_median = float(
                    np.median(
                        rv
                    )
                )

        status = classify_status(
            coverage,
            raw_valid_count,
        )

        rows.append(
            {
                **circle,

                "sample_mask_pixel_count":
                    sample_px,

                "laser_raw_valid_count":
                    raw_valid_count,

                "laser_filtered_valid_count":
                    filtered_count,

                "laser_coverage_ratio":
                    float(
                        coverage
                    ),

                "laser_coverage_percent":
                    float(
                        coverage
                        * 100.0
                    ),

                "height_raw_median_mm":
                    raw_median,

                "height_raw_mad_mm":
                    raw_mad,

                "height_base_mm":
                    H_base,

                "height_mean_mm":
                    H_mean,

                "height_std_mm":
                    H_std,

                "height_min_mm":
                    H_min,

                "height_max_mm":
                    H_max,

                "height_p05_mm":
                    H_p05,

                "height_p95_mm":
                    H_p95,

                "residual_median_mm":
                    residual_median,

                "3d_status":
                    status,
            }
        )

        if (
            idx % 1000 == 0
            or idx == len(circles)
        ):
            print(
                "  laser statistics "
                f"{idx}/{len(circles)}"
            )

    return rows


# =============================================================================
# CAMERA RGB TEXTURE PATCH
# =============================================================================

def build_periodic_wire_texture_patch(
    image_rgb: np.ndarray,
    source_instance_map: np.ndarray,
    wire_id: int,
    cx: float,
    cy: float,
    radius_px: float,
    fill_mode: str,
):
    """
    Build a small local patch around one fitted circle.

    Rows are constructed in an UNWRAPPED local coordinate:
        y_unwrapped = floor(cy-r-margin) ... ceil(cy+r+margin)

    Actual image row:
        y = y_unwrapped % H

    Therefore the patch is continuous even if the wire crosses 0°/360°.

    Returns
    -------
    patch_rgb_filled : Hpatch x Wpatch x 3 uint8
    original_local_mask : bool
    x0 : global x offset
    yu0 : unwrapped global y offset

    When fill_mode == nearest-wire:
        every pixel outside current wire's original mask is assigned the RGB
        of the nearest original-mask pixel of THE SAME wire.
    """
    H, W = source_instance_map.shape

    margin = int(
        TEXTURE_PATCH_MARGIN_PX
    )

    r = float(
        radius_px
    )

    x0 = max(
        0,
        int(
            math.floor(
                cx - r - margin
            )
        ),
    )

    x1 = min(
        W - 1,
        int(
            math.ceil(
                cx + r + margin
            )
        ),
    )

    yu0 = int(
        math.floor(
            cy - r - margin
        )
    )

    yu1 = int(
        math.ceil(
            cy + r + margin
        )
    )

    x_indices = np.arange(
        x0,
        x1 + 1,
        dtype=np.int32,
    )

    y_unwrapped = np.arange(
        yu0,
        yu1 + 1,
        dtype=np.int64,
    )

    y_indices = (
        y_unwrapped
        % H
    ).astype(
        np.int32
    )

    patch_rgb = (
        image_rgb[
            y_indices[:, None],
            x_indices[None, :],
        ]
        .copy()
    )

    patch_ids = (
        source_instance_map[
            y_indices[:, None],
            x_indices[None, :],
        ]
    )

    original_local_mask = (
        patch_ids
        == int(wire_id)
    )

    if fill_mode == "raw-camera":
        return (
            patch_rgb,
            original_local_mask,
            x0,
            yu0,
        )

    if fill_mode != "nearest-wire":
        raise ValueError(
            fill_mode
        )

    if not np.any(
        original_local_mask
    ):
        # Should be very rare. No same-wire source pixels exist in patch.
        # Fall back to raw camera rather than inventing a color.
        return (
            patch_rgb,
            original_local_mask,
            x0,
            yu0,
        )

    # OpenCV distanceTransformWithLabels:
    # source pixels (same wire) are zero.
    # Every other pixel gets the label of the nearest zero pixel.
    dt_src = np.where(
        original_local_mask,
        0,
        1,
    ).astype(
        np.uint8
    )

    _, labels = cv2.distanceTransformWithLabels(
        dt_src,
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    max_label = int(
        labels.max()
    )

    if max_label <= 0:
        return (
            patch_rgb,
            original_local_mask,
            x0,
            yu0,
        )

    # label -> RGB of its corresponding zero/source pixel
    label_rgb = np.zeros(
        (
            max_label + 1,
            3,
        ),
        dtype=np.uint8,
    )

    sy, sx = np.nonzero(
        original_local_mask
    )

    source_labels = labels[
        sy,
        sx,
    ]

    source_colors = patch_rgb[
        sy,
        sx,
    ]

    # For DIST_LABEL_PIXEL every source/zero pixel normally owns one label.
    # Assignment is safe even if a label appears more than once.
    label_rgb[
        source_labels
    ] = source_colors

    filled = label_rgb[
        labels
    ]

    # Keep the exact original source colors.
    filled[
        original_local_mask
    ] = patch_rgb[
        original_local_mask
    ]

    return (
        filled,
        original_local_mask,
        x0,
        yu0,
    )


def sample_texture_patch_bilinear(
    patch_rgb: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
):
    """
    Bilinear RGB sampling using cv2.remap.

    map_x / map_y are patch coordinates in pixel units.
    """
    mx = np.asarray(
        map_x,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    my = np.asarray(
        map_y,
        dtype=np.float32,
    ).reshape(
        1,
        -1,
    )

    sampled = cv2.remap(
        patch_rgb,
        mx,
        my,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return sampled.reshape(
        -1,
        3,
    ).astype(
        np.uint8
    )


# =============================================================================
# POLAR DISK MESH
# =============================================================================

def build_local_polar_disk_mesh(
    radius_px: float,
    angular_segments: int,
    radial_ring_density: float,
):
    """
    Build a smooth standard circle in local PIXEL coordinates.

    Vertex 0 = center.
    Then fixed angular segments for every radial ring.

    Returns
    -------
    uv_px : N x 2
        [u_px, v_px]

    faces : M x 3
        local vertex indices
    """
    radius_px = float(
        radius_px
    )

    angular_segments = max(
        16,
        int(
            angular_segments
        ),
    )

    rings = int(
        math.ceil(
            radius_px
            * float(
                radial_ring_density
            )
        )
    )

    rings = int(
        np.clip(
            rings,
            MIN_RADIAL_RINGS,
            MAX_RADIAL_RINGS,
        )
    )

    vertices_uv = [
        [0.0, 0.0]
    ]

    angles = (
        np.arange(
            angular_segments,
            dtype=np.float64,
        )
        * (
            2.0
            * math.pi
            / angular_segments
        )
    )

    cos_a = np.cos(
        angles
    )

    sin_a = np.sin(
        angles
    )

    for ring in range(
        1,
        rings + 1,
    ):
        rr = (
            radius_px
            * ring
            / rings
        )

        u = (
            rr
            * cos_a
        )

        v = (
            rr
            * sin_a
        )

        ring_uv = np.column_stack(
            [
                u,
                v,
            ]
        )

        vertices_uv.extend(
            ring_uv.tolist()
        )

    uv_px = np.asarray(
        vertices_uv,
        dtype=np.float32,
    )

    face_arrays = []

    # Center -> first ring
    first_ring_start = 1

    f0 = np.empty(
        (
            angular_segments,
            3,
        ),
        dtype=np.int32,
    )

    for k in range(
        angular_segments
    ):
        # V3: reverse winding so the disk normal points OUTWARD (+radial).
        f0[
            k
        ] = [
            0,
            first_ring_start
            + (
                (k + 1)
                % angular_segments
            ),
            first_ring_start + k,
        ]

    face_arrays.append(
        f0
    )

    # Ring -> ring
    for ring in range(
        1,
        rings,
    ):
        inner_start = (
            1
            + (
                ring - 1
            )
            * angular_segments
        )

        outer_start = (
            1
            + ring
            * angular_segments
        )

        f = np.empty(
            (
                angular_segments * 2,
                3,
            ),
            dtype=np.int32,
        )

        pos = 0

        for k in range(
            angular_segments
        ):
            kn = (
                k + 1
            ) % angular_segments

            i0 = (
                inner_start
                + k
            )

            i1 = (
                inner_start
                + kn
            )

            o0 = (
                outer_start
                + k
            )

            o1 = (
                outer_start
                + kn
            )

            # V3: reverse both triangles so normals point OUTWARD (+radial).
            f[
                pos
            ] = [
                i0,
                o1,
                o0,
            ]

            f[
                pos + 1
            ] = [
                i0,
                i1,
                o1,
            ]

            pos += 2

        face_arrays.append(
            f
        )

    faces = np.concatenate(
        face_arrays,
        axis=0,
    )

    return (
        uv_px,
        faces,
        rings,
    )


# =============================================================================
# HEIGHT COLOR
# =============================================================================

def wire_height_colors_all_wires(
    rows,
):
    """
    Per-wire color for the height-colored PLY.

    - Wires WITH laser height:
        colored by actual H using TURBO colormap
    - Wires WITHOUT laser height:
        colored magenta, so they are explicitly visible as "NO_LASER"
    """
    n = len(rows)
    colors = np.full(
        (n, 3),
        180,
        dtype=np.uint8,
    )

    hvals = np.asarray(
        [
            float(row["height_base_mm"])
            for row in rows
        ],
        dtype=np.float64,
    )

    finite = np.isfinite(
        hvals
    )

    if np.any(finite):
        lo = float(
            np.percentile(
                hvals[finite],
                5.0,
            )
        )
        hi = float(
            np.percentile(
                hvals[finite],
                95.0,
            )
        )

        if hi <= lo + 1e-9:
            hi = lo + 1.0

        norm = np.clip(
            (hvals - lo) / (hi - lo),
            0.0,
            1.0,
        )

        u8 = np.round(
            norm * 255.0
        ).astype(np.uint8)

        bgr = cv2.applyColorMap(
            u8.reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        ).reshape(-1, 3)

        rgb = bgr[:, ::-1]
        colors[finite] = rgb[finite]

    # Explicit "NO_LASER" color: magenta.
    colors[~finite] = np.array(
        [255, 0, 255],
        dtype=np.uint8,
    )

    return colors


def wire_status_colors(
    rows,
):
    """
    Separate status-color PLY to make data quality obvious.

    READY_DIRECT -> green
    PARTIAL      -> yellow
    LOW_COVERAGE -> orange
    NO_LASER     -> magenta
    """
    lut = {
        "READY_DIRECT": np.array([0, 255, 0], dtype=np.uint8),
        "PARTIAL": np.array([255, 255, 0], dtype=np.uint8),
        "LOW_COVERAGE": np.array([255, 165, 0], dtype=np.uint8),
        "NO_LASER": np.array([255, 0, 255], dtype=np.uint8),
    }

    out = np.zeros((len(rows), 3), dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i] = lut.get(
            row["3d_status"],
            np.array([180, 180, 180], dtype=np.uint8),
        )
    return out


# =============================================================================
# BUILD 3D TEXTURED CIRCULAR DISKS
# =============================================================================

def build_textured_circular_disks(
    rows,
    source_instance_map,
    image_rgb,
    image_height: int,
    mm_per_pixel: float,
    camera_center_x_px: float,
    camera_x_axis_mm,
    wheel_radius_mm: float,
    min_build_coverage: float,
    min_build_valid: int,
    angular_segments: int,
    radial_ring_density: float,
    texture_fill_mode: str,
):
    """
    Build ALL wires.

    Rules:
      - if a wire has finite laser H:
            use real H
      - if a wire has no laser H:
            still build its standard circular end-face
            at baseline wheel radius (R = wheel_radius_mm)

    Therefore every wire in the CSV enters the final 3-D output.
    """
    eligible = list(rows)

    h_colors_per_wire = wire_height_colors_all_wires(
        eligible
    )

    status_colors_per_wire = wire_status_colors(
        eligible
    )

    vertex_blocks = []
    face_blocks = []
    camera_rgb_blocks = []
    height_rgb_blocks = []
    status_rgb_blocks = []
    wire_id_blocks = []

    camera_x_blocks = []
    camera_y_blocks = []
    texture_source_blocks = []
    normal_blocks = []
    height_source_kind_blocks = []

    built_wire_ids = []
    built_wire_center_xyz = []
    built_wire_face_radius_mm = []
    built_wire_height_mm = []
    built_wire_center_radius_mm = []
    built_wire_radial_rings = []
    built_wire_use_real_height = []

    total_camera_direct = 0
    total_texture_filled = 0

    vertex_offset = 0

    for idx, row in enumerate(
        eligible,
        start=1,
    ):
        wire_id = int(
            row["wire_id"]
        )

        cx = float(
            row["center_x_px"]
        )

        cy = (
            float(
                row["center_y_px"]
            )
            % image_height
        )

        radius_px = float(
            row["radius_px"]
        )

        H_wire = float(
            row["height_base_mm"]
        )

        use_real_height = bool(
            np.isfinite(H_wire)
        )

        # If no laser exists, keep the circular end-face at baseline R.
        if use_real_height:
            R_wire = (
                float(wheel_radius_mm)
                + HEIGHT_SIGN
                * (H_wire - H_REF_MM)
            )
        else:
            R_wire = float(
                wheel_radius_mm
            )

        fallback_x_mm = (
            (cx - camera_center_x_px)
            * mm_per_pixel
        )

        Xc = interpolate_axis(
            camera_x_axis_mm,
            cx,
            fallback_x_mm,
        )

        theta = (
            cy
            / float(image_height)
            * 2.0
            * math.pi
        )

        ct = math.cos(theta)
        st = math.sin(theta)

        center = np.array(
            [
                Xc,
                R_wire * ct,
                R_wire * st,
            ],
            dtype=np.float64,
        )

        e_x = np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float64,
        )

        e_t = np.array(
            [0.0, -st, ct],
            dtype=np.float64,
        )

        (
            uv_px,
            local_faces,
            radial_rings,
        ) = build_local_polar_disk_mesh(
            radius_px,
            angular_segments,
            radial_ring_density,
        )

        u_mm = (
            uv_px[:, 0].astype(np.float64)
            * mm_per_pixel
        )

        v_mm = (
            uv_px[:, 1].astype(np.float64)
            * mm_per_pixel
        )

        verts = (
            center[None, :]
            + u_mm[:, None] * e_x[None, :]
            + v_mm[:, None] * e_t[None, :]
        ).astype(np.float32)

        # Explicit outward normal
        local_normals = np.repeat(
            np.asarray(
                [[0.0, ct, st]],
                dtype=np.float32,
            ),
            len(verts),
            axis=0,
        )

        camera_x = (
            cx + uv_px[:, 0]
        ).astype(np.float32)

        camera_y_unwrapped = (
            cy + uv_px[:, 1]
        ).astype(np.float32)

        camera_y_periodic = np.mod(
            camera_y_unwrapped,
            image_height,
        ).astype(np.float32)

        (
            texture_patch,
            original_local_mask,
            x0,
            yu0,
        ) = build_periodic_wire_texture_patch(
            image_rgb,
            source_instance_map,
            wire_id,
            cx,
            cy,
            radius_px,
            texture_fill_mode,
        )

        patch_x = camera_x - float(x0)
        patch_y = camera_y_unwrapped - float(yu0)

        camera_rgb = sample_texture_patch_bilinear(
            texture_patch,
            patch_x,
            patch_y,
        )

        xi = np.clip(
            np.round(patch_x).astype(np.int32),
            0,
            original_local_mask.shape[1] - 1,
        )

        yi = np.clip(
            np.round(patch_y).astype(np.int32),
            0,
            original_local_mask.shape[0] - 1,
        )

        direct = original_local_mask[
            yi,
            xi,
        ]

        texture_source_kind = np.where(
            direct,
            0,
            1,
        ).astype(np.uint8)

        total_camera_direct += int(
            np.count_nonzero(direct)
        )
        total_texture_filled += int(
            np.count_nonzero(~direct)
        )

        h_rgb = np.repeat(
            h_colors_per_wire[idx - 1][None, :],
            len(verts),
            axis=0,
        ).astype(np.uint8)

        s_rgb = np.repeat(
            status_colors_per_wire[idx - 1][None, :],
            len(verts),
            axis=0,
        ).astype(np.uint8)

        height_source_kind = np.full(
            len(verts),
            0 if use_real_height else 1,
            dtype=np.uint8,
        )

        faces = (
            local_faces + vertex_offset
        ).astype(np.int32)

        vertex_blocks.append(verts)
        face_blocks.append(faces)
        camera_rgb_blocks.append(camera_rgb)
        height_rgb_blocks.append(h_rgb)
        status_rgb_blocks.append(s_rgb)
        wire_id_blocks.append(
            np.full(
                len(verts),
                wire_id,
                dtype=np.int32,
            )
        )
        camera_x_blocks.append(camera_x)
        camera_y_blocks.append(camera_y_periodic)
        texture_source_blocks.append(texture_source_kind)
        normal_blocks.append(local_normals)
        height_source_kind_blocks.append(height_source_kind)

        built_wire_ids.append(wire_id)
        built_wire_center_xyz.append(center)
        built_wire_face_radius_mm.append(
            radius_px * mm_per_pixel
        )
        built_wire_height_mm.append(
            H_wire if use_real_height else np.nan
        )
        built_wire_center_radius_mm.append(R_wire)
        built_wire_radial_rings.append(radial_rings)
        built_wire_use_real_height.append(int(use_real_height))

        vertex_offset += len(verts)

        if idx % 500 == 0 or idx == len(eligible):
            print(
                "  textured 3D disks "
                f"{idx}/{len(eligible)}"
            )

    vertices = np.concatenate(
        vertex_blocks,
        axis=0,
    ).astype(np.float32)

    faces = np.concatenate(
        face_blocks,
        axis=0,
    ).astype(np.int32)

    camera_rgb = np.concatenate(
        camera_rgb_blocks,
        axis=0,
    ).astype(np.uint8)

    height_rgb = np.concatenate(
        height_rgb_blocks,
        axis=0,
    ).astype(np.uint8)

    status_rgb = np.concatenate(
        status_rgb_blocks,
        axis=0,
    ).astype(np.uint8)

    vertex_wire_id = np.concatenate(
        wire_id_blocks,
        axis=0,
    ).astype(np.int32)

    vertex_camera_x_px = np.concatenate(
        camera_x_blocks,
        axis=0,
    ).astype(np.float32)

    vertex_camera_y_px = np.concatenate(
        camera_y_blocks,
        axis=0,
    ).astype(np.float32)

    vertex_texture_source_kind = np.concatenate(
        texture_source_blocks,
        axis=0,
    ).astype(np.uint8)

    vertex_normals = np.concatenate(
        normal_blocks,
        axis=0,
    ).astype(np.float32)

    vertex_height_source_kind = np.concatenate(
        height_source_kind_blocks,
        axis=0,
    ).astype(np.uint8)

    return {
        "eligible_rows":
            eligible,

        "vertices":
            vertices,

        "faces":
            faces,

        "camera_rgb":
            camera_rgb,

        "height_rgb":
            height_rgb,

        "status_rgb":
            status_rgb,

        "vertex_wire_id":
            vertex_wire_id,

        "vertex_camera_x_px":
            vertex_camera_x_px,

        "vertex_camera_y_px":
            vertex_camera_y_px,

        "vertex_texture_source_kind":
            vertex_texture_source_kind,

        "vertex_normals":
            vertex_normals,

        # 0 = real laser H
        # 1 = no laser H, baseline R = wheel_radius
        "vertex_height_source_kind":
            vertex_height_source_kind,

        "built_wire_id":
            np.asarray(
                built_wire_ids,
                dtype=np.int32,
            ),

        "built_wire_center_xyz":
            np.asarray(
                built_wire_center_xyz,
                dtype=np.float32,
            ),

        "built_wire_face_radius_mm":
            np.asarray(
                built_wire_face_radius_mm,
                dtype=np.float32,
            ),

        "built_wire_height_mm":
            np.asarray(
                built_wire_height_mm,
                dtype=np.float32,
            ),

        "built_wire_center_radius_mm":
            np.asarray(
                built_wire_center_radius_mm,
                dtype=np.float32,
            ),

        "built_wire_radial_rings":
            np.asarray(
                built_wire_radial_rings,
                dtype=np.int16,
            ),

        "built_wire_use_real_height":
            np.asarray(
                built_wire_use_real_height,
                dtype=np.uint8,
            ),

        "camera_direct_vertex_count":
            int(total_camera_direct),

        "texture_filled_vertex_count":
            int(total_texture_filled),
    }


# =============================================================================
# PLY
# =============================================================================

def write_binary_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    rgb: np.ndarray,
    wire_ids: np.ndarray,
    normals: np.ndarray,
):
    """
    Binary PLY with explicit outward normals.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vertices = np.asarray(
        vertices,
        dtype=np.float32,
    )

    faces = np.asarray(
        faces,
        dtype=np.int32,
    )

    rgb = np.asarray(
        rgb,
        dtype=np.uint8,
    )

    wire_ids = np.asarray(
        wire_ids,
        dtype=np.int32,
    )

    normals = np.asarray(
        normals,
        dtype=np.float32,
    )

    if (
        len(vertices) != len(rgb)
        or len(vertices) != len(wire_ids)
        or len(vertices) != len(normals)
    ):
        raise ValueError(
            "PLY 顶点/颜色/wire_id/normals 数量不一致。"
        )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "property int wire_id\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode(
        "ascii"
    )

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("wire_id", "<i4"),
        ]
    )

    vo = np.empty(
        len(vertices),
        dtype=vertex_dtype,
    )

    vo["x"] = vertices[:, 0]
    vo["y"] = vertices[:, 1]
    vo["z"] = vertices[:, 2]

    vo["nx"] = normals[:, 0]
    vo["ny"] = normals[:, 1]
    vo["nz"] = normals[:, 2]

    vo["red"] = rgb[:, 0]
    vo["green"] = rgb[:, 1]
    vo["blue"] = rgb[:, 2]

    vo["wire_id"] = wire_ids

    face_dtype = np.dtype(
        [
            ("n", "u1"),
            (
                "idx",
                "<i4",
                (3,),
            ),
        ]
    )

    fo = np.empty(
        len(faces),
        dtype=face_dtype,
    )

    fo["n"] = 3
    fo["idx"] = faces

    with path.open(
        "wb"
    ) as f:
        f.write(header)
        vo.tofile(f)
        fo.tofile(f)


# =============================================================================
# 2D VISUALIZATIONS
# =============================================================================

def draw_circle_overlay(
    image_bgr,
    circles,
    out_path: Path,
):
    H, _ = image_bgr.shape[:2]

    out = image_bgr.copy()

    for circle in circles:

        cx = int(
            round(
                float(
                    circle["center_x_px"]
                )
            )
        )

        cy = (
            int(
                round(
                    float(
                        circle["center_y_px"]
                    )
                )
            )
            % H
        )

        r = max(
            1,
            int(
                round(
                    float(
                        circle["radius_px"]
                    )
                )
            ),
        )

        for yy in (
            cy,
            cy - H,
            cy + H,
        ):
            cv2.circle(
                out,
                (
                    cx,
                    yy,
                ),
                r,
                (
                    0,
                    0,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(
        str(
            out_path
        ),
        out,
    )


def deterministic_bgr(
    wire_id: int,
):
    rng = np.random.default_rng(
        int(
            wire_id
        )
        * 10007
        + 17
    )

    c = rng.integers(
        70,
        256,
        size=3,
        dtype=np.uint8,
    )

    return tuple(
        int(
            x
        )
        for x in c
    )


def save_circle_map_preview(
    image_bgr,
    circle_map,
    out_path: Path,
):
    H, W = circle_map.shape

    color = np.zeros(
        (
            H,
            W,
            3,
        ),
        dtype=np.uint8,
    )

    ids = np.unique(
        circle_map
    )

    ids = ids[
        ids > 0
    ]

    for wire_id in ids:
        color[
            circle_map
            == int(
                wire_id
            )
        ] = deterministic_bgr(
            int(
                wire_id
            )
        )

    out = image_bgr.copy()

    mask = (
        circle_map > 0
    )

    if np.any(
        mask
    ):
        blend = cv2.addWeighted(
            image_bgr,
            0.50,
            color,
            0.50,
            0.0,
        )

        out[
            mask
        ] = blend[
            mask
        ]

    cv2.imwrite(
        str(
            out_path
        ),
        out,
    )



# =============================================================================
# CLOUDCOMPARE WIRE-CENTER INFORMATION POINTS
# =============================================================================

STATUS_CODE_MAP = {
    "READY_DIRECT": 0,
    "PARTIAL": 1,
    "LOW_COVERAGE": 2,
    "NO_LASER": 3,
}

STATUS_RGB_MAP = {
    "READY_DIRECT": (0, 255, 0),       # green
    "PARTIAL": (255, 255, 0),          # yellow
    "LOW_COVERAGE": (255, 165, 0),     # orange
    "NO_LASER": (255, 0, 255),         # magenta
}


def build_wire_information_records(
    rows,
    mesh,
    mm_per_pixel: float,
    image_height: int,
    wheel_radius_mm: float,
):
    """
    Build one information record per wire.

    IMPORTANT:
    mesh["built_wire_center_xyz"] is aligned 1:1 with rows because V4/V5
    builds ALL wires in row order.
    """
    center_xyz = np.asarray(
        mesh["built_wire_center_xyz"],
        dtype=np.float64,
    )

    use_real_height = np.asarray(
        mesh["built_wire_use_real_height"],
        dtype=np.uint8,
    )

    if len(center_xyz) != len(rows):
        raise RuntimeError(
            "wire information alignment error: "
            f"center_xyz={len(center_xyz)}, rows={len(rows)}"
        )

    records = []

    for i, row in enumerate(rows):
        wire_id = int(row["wire_id"])

        cx = float(row["center_x_px"])
        cy = float(row["center_y_px"]) % image_height

        radius_px = float(row["radius_px"])
        diameter_px = float(row["diameter_px"])

        radius_mm = radius_px * mm_per_pixel
        diameter_mm = diameter_px * mm_per_pixel

        theta_deg = cy / image_height * 360.0

        H_wire = float(row["height_base_mm"])
        real_height = bool(use_real_height[i])

        if real_height and np.isfinite(H_wire):
            center_radius_mm = (
                wheel_radius_mm
                + HEIGHT_SIGN * (H_wire - H_REF_MM)
            )
            height_for_display = H_wire
            height_source = "REAL_LASER_HEIGHT"
            height_source_code = 0
        else:
            center_radius_mm = float(wheel_radius_mm)
            height_for_display = np.nan
            height_source = "NO_LASER_BASELINE_RADIUS"
            height_source_code = 1

        status = str(row["3d_status"])
        status_code = int(
            STATUS_CODE_MAP.get(status, 99)
        )

        x3d, y3d, z3d = (
            float(center_xyz[i, 0]),
            float(center_xyz[i, 1]),
            float(center_xyz[i, 2]),
        )

        records.append(
            {
                "wire_id": wire_id,

                "x_mm": x3d,
                "y_mm": y3d,
                "z_mm": z3d,

                "center_x_px": cx,
                "center_y_px": cy,
                "theta_deg": theta_deg,

                "radius_px": radius_px,
                "diameter_px": diameter_px,
                "radius_mm": radius_mm,
                "diameter_mm": diameter_mm,

                "confidence": float(row["confidence"]),
                "seam_merged": int(row["seam_merged"]),

                "sample_mask_pixel_count":
                    int(row["sample_mask_pixel_count"]),

                "laser_raw_valid_count":
                    int(row["laser_raw_valid_count"]),

                "laser_filtered_valid_count":
                    int(row["laser_filtered_valid_count"]),

                "laser_coverage_ratio":
                    float(row["laser_coverage_ratio"]),

                "laser_coverage_percent":
                    float(row["laser_coverage_percent"]),

                "height_raw_median_mm":
                    float(row["height_raw_median_mm"]),

                "height_raw_mad_mm":
                    float(row["height_raw_mad_mm"]),

                "height_mm":
                    height_for_display,

                "height_mean_mm":
                    float(row["height_mean_mm"]),

                "height_std_mm":
                    float(row["height_std_mm"]),

                "height_min_mm":
                    float(row["height_min_mm"]),

                "height_max_mm":
                    float(row["height_max_mm"]),

                "height_p05_mm":
                    float(row["height_p05_mm"]),

                "height_p95_mm":
                    float(row["height_p95_mm"]),

                "residual_median_mm":
                    float(row["residual_median_mm"]),

                "wheel_radius_base_mm":
                    float(wheel_radius_mm),

                "center_radius_mm":
                    float(center_radius_mm),

                "status":
                    status,

                "status_code":
                    status_code,

                "height_source":
                    height_source,

                "height_source_code":
                    int(height_source_code),

                "has_real_laser_height":
                    int(real_height),
            }
        )

    return records


def write_wire_information_csv(
    path: Path,
    records,
):
    fields = [
        "wire_id",

        "x_mm",
        "y_mm",
        "z_mm",

        "center_x_px",
        "center_y_px",
        "theta_deg",

        "radius_px",
        "diameter_px",
        "radius_mm",
        "diameter_mm",

        "confidence",
        "seam_merged",

        "sample_mask_pixel_count",
        "laser_raw_valid_count",
        "laser_filtered_valid_count",
        "laser_coverage_ratio",
        "laser_coverage_percent",

        "height_raw_median_mm",
        "height_raw_mad_mm",

        "height_mm",
        "height_mean_mm",
        "height_std_mm",
        "height_min_mm",
        "height_max_mm",
        "height_p05_mm",
        "height_p95_mm",

        "residual_median_mm",

        "wheel_radius_base_mm",
        "center_radius_mm",

        "status",
        "status_code",

        "height_source",
        "height_source_code",
        "has_real_laser_height",
    ]

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
        writer.writerows(records)


def write_wire_centers_info_ply(
    path: Path,
    records,
):
    """
    Write 1 point per wire with CloudCompare-friendly scalar fields.

    Numeric information is stored directly as PLY vertex properties.

    CloudCompare workflow:
      1. Open this file together with the full wire mesh.
      2. Select wire_centers_info.
      3. Increase point size to 3~5.
      4. Choose a scalar field, e.g.:
            wire_id
            diameter_mm
            height_mm
            laser_coverage_pct
            status_code
      5. Use point picking / properties to inspect one center point.

    All scalar fields are float32 for broad PLY/CloudCompare compatibility.
    wire_id <= 7088 is exactly representable as float32.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = len(records)

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),

            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),

            ("wire_id", "<f4"),

            ("diameter_mm", "<f4"),
            ("radius_mm", "<f4"),
            ("height_mm", "<f4"),
            ("center_radius_mm", "<f4"),
            ("theta_deg", "<f4"),

            ("laser_coverage_pct", "<f4"),
            ("laser_valid_points", "<f4"),
            ("confidence", "<f4"),

            ("status_code", "<f4"),
            ("height_source_code", "<f4"),

            ("center_x_px", "<f4"),
            ("center_y_px", "<f4"),
        ]
    )

    out = np.empty(
        n,
        dtype=dtype,
    )

    for i, rec in enumerate(records):
        out["x"][i] = rec["x_mm"]
        out["y"][i] = rec["y_mm"]
        out["z"][i] = rec["z_mm"]

        rgb = STATUS_RGB_MAP.get(
            rec["status"],
            (180, 180, 180),
        )

        out["red"][i] = rgb[0]
        out["green"][i] = rgb[1]
        out["blue"][i] = rgb[2]

        out["wire_id"][i] = float(
            rec["wire_id"]
        )

        out["diameter_mm"][i] = float(
            rec["diameter_mm"]
        )

        out["radius_mm"][i] = float(
            rec["radius_mm"]
        )

        out["height_mm"][i] = float(
            rec["height_mm"]
        )

        out["center_radius_mm"][i] = float(
            rec["center_radius_mm"]
        )

        out["theta_deg"][i] = float(
            rec["theta_deg"]
        )

        out["laser_coverage_pct"][i] = float(
            rec["laser_coverage_percent"]
        )

        out["laser_valid_points"][i] = float(
            rec["laser_filtered_valid_count"]
        )

        out["confidence"][i] = float(
            rec["confidence"]
        )

        out["status_code"][i] = float(
            rec["status_code"]
        )

        out["height_source_code"][i] = float(
            rec["height_source_code"]
        )

        out["center_x_px"][i] = float(
            rec["center_x_px"]
        )

        out["center_y_px"][i] = float(
            rec["center_y_px"]
        )

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"

        "property float x\n"
        "property float y\n"
        "property float z\n"

        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"

        "property float wire_id\n"

        "property float diameter_mm\n"
        "property float radius_mm\n"
        "property float height_mm\n"
        "property float center_radius_mm\n"
        "property float theta_deg\n"

        "property float laser_coverage_pct\n"
        "property float laser_valid_points\n"
        "property float confidence\n"

        "property float status_code\n"
        "property float height_source_code\n"

        "property float center_x_px\n"
        "property float center_y_px\n"

        "end_header\n"
    ).encode("ascii")

    with path.open("wb") as f:
        f.write(header)
        out.tofile(f)


def write_wire_centers_info_readme(
    path: Path,
):
    text = """CloudCompare 单丝信息点使用说明
================================

文件：
    wire_centers_info.ply

每个点 = 一根丝的 3D 端面中心。
wire_id 与 YOLO、激光对应表、3D mesh 中的 wire_id 完全一致。

推荐：
1. 同时打开：
       wire_circle_endfaces_camera_rgb_all_wires.ply
       wire_centers_info.ply

2. 在 DB Tree 中选择 wire_centers_info。

3. 把 Point size 调到 3~5。

4. Scalar Fields 可以切换：
       wire_id
       diameter_mm
       radius_mm
       height_mm
       center_radius_mm
       theta_deg
       laser_coverage_pct
       laser_valid_points
       confidence
       status_code
       height_source_code
       center_x_px
       center_y_px

状态编码：
    0 = READY_DIRECT
    1 = PARTIAL
    2 = LOW_COVERAGE
    3 = NO_LASER

高度来源编码：
    0 = REAL_LASER_HEIGHT
    1 = NO_LASER_BASELINE_RADIUS

中心点 RGB 状态颜色：
    绿色   = READY_DIRECT
    黄色   = PARTIAL
    橙色   = LOW_COVERAGE
    洋红色 = NO_LASER

NO_LASER 的 height_mm 为 NaN。
其 3D center_radius_mm 仍为 75 mm 基准半径。
"""
    path.write_text(
        text,
        encoding="utf-8",
    )


# =============================================================================
# CSV TABLE
# =============================================================================

def save_wire_table(
    path: Path,
    rows,
    mm_per_pixel: float,
    image_height: int,
    wheel_radius_mm: float,
    min_build_coverage: float,
    min_build_valid: int,
):
    fields = [
        "wire_id",

        "center_x_px",
        "center_y_px",
        "theta_deg",

        "radius_px",
        "diameter_px",
        "radius_mm",
        "diameter_mm",

        "confidence",
        "seam_merged",

        "sample_mask_pixel_count",
        "laser_raw_valid_count",
        "laser_filtered_valid_count",
        "laser_coverage_ratio",
        "laser_coverage_percent",

        "height_raw_median_mm",
        "height_raw_mad_mm",

        "height_base_mm",
        "height_mean_mm",
        "height_std_mm",
        "height_min_mm",
        "height_max_mm",
        "height_p05_mm",
        "height_p95_mm",

        "residual_median_mm",

        "wheel_radius_base_mm",
        "wire_center_radius_mm",
        "height_used_for_3d_mm",
        "3d_height_source",
        "build_3d",

        "3d_status",
    ]

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

        for row in rows:
            H_wire = float(
                row["height_base_mm"]
            )

            use_real_height = bool(
                np.isfinite(H_wire)
            )

            if use_real_height:
                R_wire = (
                    wheel_radius_mm
                    + HEIGHT_SIGN
                    * (H_wire - H_REF_MM)
                )
                height_used = H_wire
                height_source = "REAL_LASER_HEIGHT"
            else:
                R_wire = float(
                    wheel_radius_mm
                )
                height_used = np.nan
                height_source = "NO_LASER_BASELINE_RADIUS"

            cy = (
                float(row["center_y_px"])
                % image_height
            )

            diameter_px = float(
                row["diameter_px"]
            )
            diameter_mm = diameter_px * mm_per_pixel

            writer.writerow(
                {
                    "wire_id": row["wire_id"],

                    "center_x_px": row["center_x_px"],
                    "center_y_px": cy,
                    "theta_deg": cy / image_height * 360.0,

                    "radius_px": row["radius_px"],
                    "diameter_px": diameter_px,
                    "radius_mm": diameter_mm * 0.5,
                    "diameter_mm": diameter_mm,

                    "confidence": row["confidence"],
                    "seam_merged": row["seam_merged"],

                    "sample_mask_pixel_count": row["sample_mask_pixel_count"],
                    "laser_raw_valid_count": row["laser_raw_valid_count"],
                    "laser_filtered_valid_count": row["laser_filtered_valid_count"],
                    "laser_coverage_ratio": row["laser_coverage_ratio"],
                    "laser_coverage_percent": row["laser_coverage_percent"],

                    "height_raw_median_mm": row["height_raw_median_mm"],
                    "height_raw_mad_mm": row["height_raw_mad_mm"],

                    "height_base_mm": H_wire,
                    "height_mean_mm": row["height_mean_mm"],
                    "height_std_mm": row["height_std_mm"],
                    "height_min_mm": row["height_min_mm"],
                    "height_max_mm": row["height_max_mm"],
                    "height_p05_mm": row["height_p05_mm"],
                    "height_p95_mm": row["height_p95_mm"],

                    "residual_median_mm": row["residual_median_mm"],

                    "wheel_radius_base_mm": wheel_radius_mm,
                    "wire_center_radius_mm": R_wire,
                    "height_used_for_3d_mm": height_used,
                    "3d_height_source": height_source,
                    "build_3d": 1,

                    "3d_status": row["3d_status"],
                }
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    args.circle_csv = (
        args.circle_csv.resolve()
    )

    args.source_instance_map = (
        args.source_instance_map.resolve()
    )

    args.mapped_height = (
        args.mapped_height.resolve()
    )

    args.camera_image = (
        args.camera_image.resolve()
    )

    args.out_dir = (
        args.out_dir.resolve()
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 82
    )

    print(
        "标准圆丝端面 3D V2："
        "相机 RGB 纹理 + 激光高度"
    )

    print(
        "=" * 82
    )

    print(
        f"circle CSV          : "
        f"{args.circle_csv}"
    )

    print(
        f"source instance map : "
        f"{args.source_instance_map}"
    )

    print(
        f"mapped height       : "
        f"{args.mapped_height}"
    )

    print(
        f"camera image        : "
        f"{args.camera_image}"
    )

    print(
        f"output              : "
        f"{args.out_dir}"
    )

    print()

    circles = load_circle_csv(
        args.circle_csv
    )

    (
        source_instance_map,
        instance_meta,
    ) = load_source_instance_map(
        args.source_instance_map
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

    if (
        source_instance_map.shape
        != height_mm.shape
    ):
        raise RuntimeError(
            "source instance map 与 "
            "mapped_height_mm 尺寸不一致："
            f"{source_instance_map.shape} "
            f"vs {height_mm.shape}"
        )

    H, W = (
        height_mm.shape
    )

    image_bgr = cv2.imread(
        str(
            args.camera_image
        ),
        cv2.IMREAD_COLOR,
    )

    if image_bgr is None:
        raise FileNotFoundError(
            "这一版必须读取相机拼接图，"
            f"但无法读取：{args.camera_image}"
        )

    if image_bgr.shape[:2] != (
        H,
        W,
    ):
        raise RuntimeError(
            "相机图尺寸与 mapped height "
            "尺寸不一致："
            f"{image_bgr.shape[:2]} "
            f"vs {(H, W)}"
        )

    # OpenCV BGR -> RGB
    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    mm_per_pixel = (
        height_meta.get(
            "camera_x_mm_per_pixel"
        )
    )

    if mm_per_pixel is None:
        mm_per_pixel = (
            instance_meta.get(
                "pixel_size_mm"
            )
        )

    if mm_per_pixel is None:
        mm_per_pixel = getattr(
            project_config,
            "SURFACE_FUSION_MM_PER_PIXEL",
            None,
        )

    if mm_per_pixel is None:
        raise RuntimeError(
            "无法得到 mm/px。"
        )

    mm_per_pixel = float(
        mm_per_pixel
    )

    camera_center_x_px = (
        height_meta.get(
            "camera_zero_x_px"
        )
    )

    if camera_center_x_px is None:
        camera_center_x_px = (
            instance_meta.get(
                "camera_center_x_px"
            )
        )

    if camera_center_x_px is None:
        camera_center_x_px = (
            W - 1
        ) / 2.0

    camera_center_x_px = float(
        camera_center_x_px
    )

    print(
        f"grid                 : "
        f"{H} x {W}"
    )

    print(
        f"circles              : "
        f"{len(circles)}"
    )

    print(
        f"pixel size           : "
        f"{mm_per_pixel:.9f} mm/px"
    )

    print(
        f"camera X=0           : "
        f"{camera_center_x_px:.3f} px"
    )

    print(
        f"wheel radius         : "
        f"{args.wheel_radius_mm:.6f} mm"
    )

    print(
        "height formula       : "
        f"R = {args.wheel_radius_mm:.6f} + H"
    )

    print(
        "camera texture fill  : "
        f"{args.texture_fill_mode}"
    )

    print()

    # -------------------------------------------------------------------------
    # Standard circle map
    # -------------------------------------------------------------------------

    print(
        "Building periodic standard-circle owner map..."
    )

    (
        circle_map,
        conflict_events,
        seam_circle_count,
    ) = build_circle_owner_map(
        (
            H,
            W,
        ),
        circles,
    )

    circle_map_path = (
        args.out_dir
        / "wire_instance_map_circle.npz"
    )

    np.savez_compressed(
        circle_map_path,

        instance_map=circle_map.astype(
            np.int32
        ),

        image_width_px=np.int32(
            W
        ),

        image_height_px=np.int32(
            H
        ),

        wire_count=np.int32(
            len(
                circles
            )
        ),

        pixel_size_mm=np.float64(
            mm_per_pixel
        ),

        camera_center_x_px=np.float64(
            camera_center_x_px
        ),

        periodic_y=np.uint8(
            1
        ),

        geometry=np.array(
            "STANDARD_CIRCLE_FROM_EXISTING_YOLO_EQUAL_AREA_FIT"
        ),

        owner_rule=np.array(
            "minimum_normalized_center_distance"
        ),

        source_circle_csv=np.array(
            str(
                args.circle_csv
            )
        ),

        source_instance_map=np.array(
            str(
                args.source_instance_map
            )
        ),
    )

    draw_circle_overlay(
        image_bgr,
        circles,
        args.out_dir
        / "wire_circle_overlay.png",
    )

    save_circle_map_preview(
        image_bgr,
        circle_map,
        args.out_dir
        / "wire_instance_map_circle_preview.png",
    )

    # -------------------------------------------------------------------------
    # Laser height
    # -------------------------------------------------------------------------

    print()

    print(
        "Computing per-wire laser height "
        f"using mode={args.height_sampling_mode}..."
    )

    rows = compute_wire_height_table(
        circles,
        source_instance_map,
        circle_map,
        height_mm,
        valid_mask,
        residual_mm,
        args.height_sampling_mode,
        args.mad_sigma,
    )

    table_path = (
        args.out_dir
        / "wire_circle_laser_table.csv"
    )

    save_wire_table(
        table_path,
        rows,
        mm_per_pixel,
        H,
        args.wheel_radius_mm,
        args.min_build_coverage,
        args.min_build_valid,
    )

    # -------------------------------------------------------------------------
    # 3D camera-textured circles
    # -------------------------------------------------------------------------

    print()

    print(
        "Building camera-textured smooth circular 3D disks..."
    )

    mesh = build_textured_circular_disks(
        rows=rows,

        source_instance_map=
            source_instance_map,

        image_rgb=
            image_rgb,

        image_height=
            H,

        mm_per_pixel=
            mm_per_pixel,

        camera_center_x_px=
            camera_center_x_px,

        camera_x_axis_mm=
            camera_x_axis_mm,

        wheel_radius_mm=
            args.wheel_radius_mm,

        min_build_coverage=
            args.min_build_coverage,

        min_build_valid=
            args.min_build_valid,

        angular_segments=
            args.angular_segments,

        radial_ring_density=
            args.radial_ring_density,

        texture_fill_mode=
            args.texture_fill_mode,
    )

    vertices = mesh[
        "vertices"
    ]

    faces = mesh[
        "faces"
    ]

    camera_rgb = mesh[
        "camera_rgb"
    ]

    height_rgb = mesh[
        "height_rgb"
    ]

    vertex_wire_id = mesh[
        "vertex_wire_id"
    ]

    eligible_rows = mesh[
        "eligible_rows"
    ]

    # -------------------------------------------------------------------------
    # NPZ
    # -------------------------------------------------------------------------

    npz_3d = (
        args.out_dir
        / "wire_circle_endfaces_camera_texture_3d_all_wires.npz"
    )

    np.savez_compressed(
        npz_3d,

        vertices_mm=
            vertices,

        faces=
            faces,

        vertex_wire_id=
            vertex_wire_id,

        vertex_camera_rgb=
            camera_rgb,

        vertex_height_rgb=
            height_rgb,

        vertex_camera_x_px=
            mesh[
                "vertex_camera_x_px"
            ],

        vertex_camera_y_px=
            mesh[
                "vertex_camera_y_px"
            ],

        # 0 = direct original YOLO-mask texture
        # 1 = color filled from nearest same-wire source pixel
        vertex_texture_source_kind=
            mesh[
                "vertex_texture_source_kind"
            ],

        vertex_normals_outward=
            mesh[
                "vertex_normals"
            ],

        vertex_height_source_kind=
            mesh[
                "vertex_height_source_kind"
            ],

        vertex_status_rgb=
            mesh[
                "status_rgb"
            ],

        built_wire_id=
            mesh[
                "built_wire_id"
            ],

        built_wire_center_xyz_mm=
            mesh[
                "built_wire_center_xyz"
            ],

        built_wire_face_radius_mm=
            mesh[
                "built_wire_face_radius_mm"
            ],

        built_wire_height_mm=
            mesh[
                "built_wire_height_mm"
            ],

        built_wire_center_radius_mm=
            mesh[
                "built_wire_center_radius_mm"
            ],

        built_wire_radial_rings=
            mesh[
                "built_wire_radial_rings"
            ],

        built_wire_use_real_height=
            mesh[
                "built_wire_use_real_height"
            ],

        wheel_radius_mm=np.float64(
            args.wheel_radius_mm
        ),

        laser_reference_distance_mm=
            np.float64(
                LASER_REFERENCE_DISTANCE_MM
            ),

        h_ref_mm=np.float64(
            H_REF_MM
        ),

        height_sign=np.float64(
            HEIGHT_SIGN
        ),

        mm_per_pixel=np.float64(
            mm_per_pixel
        ),

        camera_center_x_px=np.float64(
            camera_center_x_px
        ),

        image_height_px=np.int32(
            H
        ),

        image_width_px=np.int32(
            W
        ),

        periodic_y=np.uint8(
            1
        ),

        angular_segments=np.int32(
            max(
                16,
                int(
                    args.angular_segments
                ),
            )
        ),

        radial_ring_density=np.float64(
            args.radial_ring_density
        ),

        texture_fill_mode=np.array(
            args.texture_fill_mode
        ),

        disk_geometry=np.array(
            "FLAT_STANDARD_CIRCLE_OUTWARD_RADIAL_NORMAL_CAMERA_TEXTURED"
        ),
    )

    # -------------------------------------------------------------------------
    # PLYs
    # -------------------------------------------------------------------------

    camera_ply = (
        args.out_dir
        / "wire_circle_endfaces_camera_rgb_all_wires.ply"
    )

    height_ply = (
        args.out_dir
        / "wire_circle_endfaces_height_color_all_wires.ply"
    )

    status_ply = (
        args.out_dir
        / "wire_circle_endfaces_status_color_all_wires.ply"
    )

    write_binary_ply(
        camera_ply,
        vertices,
        faces,
        camera_rgb,
        vertex_wire_id,
        mesh["vertex_normals"],
    )

    write_binary_ply(
        height_ply,
        vertices,
        faces,
        height_rgb,
        vertex_wire_id,
        mesh["vertex_normals"],
    )

    write_binary_ply(
        status_ply,
        vertices,
        faces,
        mesh["status_rgb"],
        vertex_wire_id,
        mesh["vertex_normals"],
    )

    # -------------------------------------------------------------------------
    # One center information point per wire
    # -------------------------------------------------------------------------

    wire_info_records = build_wire_information_records(
        rows=rows,
        mesh=mesh,
        mm_per_pixel=mm_per_pixel,
        image_height=H,
        wheel_radius_mm=args.wheel_radius_mm,
    )

    wire_info_csv = (
        args.out_dir
        / "wire_information.csv"
    )

    wire_centers_ply = (
        args.out_dir
        / "wire_centers_info.ply"
    )

    wire_centers_readme = (
        args.out_dir
        / "wire_centers_info_README.txt"
    )

    write_wire_information_csv(
        wire_info_csv,
        wire_info_records,
    )

    write_wire_centers_info_ply(
        wire_centers_ply,
        wire_info_records,
    )

    write_wire_centers_info_readme(
        wire_centers_readme,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    status_counts = {
        "READY_DIRECT": 0,
        "PARTIAL": 0,
        "LOW_COVERAGE": 0,
        "NO_LASER": 0,
    }

    for row in rows:
        status_counts[
            row["3d_status"]
        ] += 1

    height_values = np.asarray(
        [
            float(
                row["height_base_mm"]
            )
            for row in rows
            if np.isfinite(
                float(
                    row["height_base_mm"]
                )
            )
        ],
        dtype=np.float64,
    )

    direct_count = int(
        mesh[
            "camera_direct_vertex_count"
        ]
    )

    filled_count = int(
        mesh[
            "texture_filled_vertex_count"
        ]
    )

    total_texture_vertices = (
        direct_count
        + filled_count
    )

    built_real_height_count = int(
        np.count_nonzero(
            mesh["built_wire_use_real_height"]
        )
    )
    built_no_laser_count = int(
        len(eligible_rows) - built_real_height_count
    )

    summary = {
        "input": {
            "circle_csv":
                str(
                    args.circle_csv
                ),

            "source_instance_map":
                str(
                    args.source_instance_map
                ),

            "mapped_height":
                str(
                    args.mapped_height
                ),

            "camera_image":
                str(
                    args.camera_image
                ),
        },

        "grid": {
            "height_px":
                H,

            "width_px":
                W,

            "pixel_size_mm":
                mm_per_pixel,

            "camera_center_x_px":
                camera_center_x_px,

            "periodic_y":
                True,
        },

        "circle_geometry": {
            "source":
                "existing YOLO equal-area circle parameters",

            "circle_count":
                len(
                    circles
                ),

            "seam_crossing_circle_count":
                int(
                    seam_circle_count
                ),

            "owner_map_conflict_pixel_events":
                int(
                    conflict_events
                ),

            "3d_shape":
                "complete smooth flat standard circle",

            "3d_overlap_policy":
                "do not clip independent 3D disks",
        },

        "orientation": {
            "triangle_winding": "OUTWARD",
            "vertex_normal_formula": "[0, cos(theta), sin(theta)]",
            "normal_direction": "wheel exterior",
        },

        "camera_texture": {
            "mode":
                "per-vertex RGB sampled from camera end-face",

            "fill_mode":
                args.texture_fill_mode,

            "direct_original_mask_vertex_count":
                direct_count,

            "nearest_same_wire_filled_vertex_count":
                filled_count,

            "direct_fraction":
                (
                    direct_count
                    / total_texture_vertices
                    if total_texture_vertices
                    else 0.0
                ),

            "filled_fraction":
                (
                    filled_count
                    / total_texture_vertices
                    if total_texture_vertices
                    else 0.0
                ),

            "angular_segments":
                max(
                    16,
                    int(
                        args.angular_segments
                    ),
                ),

            "radial_ring_density":
                float(
                    args.radial_ring_density
                ),
        },

        "laser": {
            "sampling_mode":
                args.height_sampling_mode,

            "mad_sigma":
                float(
                    args.mad_sigma
                ),

            "H_ref_mm":
                H_REF_MM,

            "laser_reference_distance_mm":
                LASER_REFERENCE_DISTANCE_MM,

            "height_sign":
                HEIGHT_SIGN,
        },

        "wheel": {
            "wheel_radius_mm":
                float(
                    args.wheel_radius_mm
                ),

            "formula":
                "R_wire = wheel_radius_mm + H_wire",
        },

        "3d_build": {
            "policy":
                "ALL_WIRES_BUILT",

            "real_laser_height_wires":
                built_real_height_count,

            "no_laser_baseline_wires":
                built_no_laser_count,

            "built_wire_count":
                int(
                    len(
                        eligible_rows
                    )
                ),

            "built_wire_fraction":
                (
                    len(
                        eligible_rows
                    )
                    / len(
                        circles
                    )
                    if circles
                    else 0.0
                ),

            "vertex_count":
                int(
                    len(
                        vertices
                    )
                ),

            "face_count":
                int(
                    len(
                        faces
                    )
                ),
        },

        "status_counts":
            status_counts,

        "height_base_statistics_mm": {
            "count":
                int(
                    height_values.size
                ),

            "min":
                (
                    float(
                        np.min(
                            height_values
                        )
                    )
                    if height_values.size
                    else None
                ),

            "median":
                (
                    float(
                        np.median(
                            height_values
                        )
                    )
                    if height_values.size
                    else None
                ),

            "mean":
                (
                    float(
                        np.mean(
                            height_values
                        )
                    )
                    if height_values.size
                    else None
                ),

            "max":
                (
                    float(
                        np.max(
                            height_values
                        )
                    )
                    if height_values.size
                    else None
                ),

            "p05":
                (
                    float(
                        np.percentile(
                            height_values,
                            5.0,
                        )
                    )
                    if height_values.size
                    else None
                ),

            "p95":
                (
                    float(
                        np.percentile(
                            height_values,
                            95.0,
                        )
                    )
                    if height_values.size
                    else None
                ),
        },

        "outputs": {
            "circle_instance_map":
                str(
                    circle_map_path
                ),

            "wire_laser_table":
                str(
                    table_path
                ),

            "3d_npz":
                str(
                    npz_3d
                ),

            "camera_rgb_ply":
                str(
                    camera_ply
                ),

            "height_color_ply":
                str(
                    height_ply
                ),

            "status_color_ply":
                str(
                    status_ply
                ),

            "wire_centers_info_ply":
                str(
                    wire_centers_ply
                ),

            "wire_information_csv":
                str(
                    wire_info_csv
                ),

            "wire_centers_info_readme":
                str(
                    wire_centers_readme
                ),
        },
    }

    summary_json = (
        args.out_dir
        / "wire_circle_summary.json"
    )

    summary_json.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_txt = (
        args.out_dir
        / "wire_circle_summary.txt"
    )

    lines = [
        "标准圆丝端面 3D V5：全部丝建模 + CloudCompare 单丝信息点",
        "=" * 72,
        "",

        f"grid = {H} x {W}",
        f"circle count = {len(circles)}",

        (
            "seam-crossing circles = "
            f"{seam_circle_count}"
        ),

        (
            "circle-map conflict events = "
            f"{conflict_events}"
        ),

        "",

        (
            "pixel scale = "
            f"{mm_per_pixel:.9f} mm/px"
        ),

        (
            "wheel radius = "
            f"{args.wheel_radius_mm:.6f} mm"
        ),

        "height formula = R = wheel_radius + H",

        "",

        (
            "height sampling mode = "
            f"{args.height_sampling_mode}"
        ),

        (
            "texture fill mode = "
            f"{args.texture_fill_mode}"
        ),

        (
            "camera texture direct vertices = "
            f"{direct_count:,}"
        ),

        (
            "camera texture nearest-wire filled vertices = "
            f"{filled_count:,}"
        ),

        (
            "camera texture filled fraction = "
            f"{filled_count / total_texture_vertices:.2%}"
            if total_texture_vertices
            else "camera texture filled fraction = N/A"
        ),

        "",

        "3D status:",

        (
            "  READY_DIRECT = "
            f"{status_counts['READY_DIRECT']}"
        ),

        (
            "  PARTIAL      = "
            f"{status_counts['PARTIAL']}"
        ),

        (
            "  LOW_COVERAGE = "
            f"{status_counts['LOW_COVERAGE']}"
        ),

        (
            "  NO_LASER     = "
            f"{status_counts['NO_LASER']}"
        ),

        "",

        (
            "built wires = "
            f"{len(eligible_rows)} / {len(circles)}"
        ),

        (
            "vertices = "
            f"{len(vertices):,}"
        ),

        (
            "faces = "
            f"{len(faces):,}"
        ),

        "",

        "FILES TO VIEW:",

        "  1) wire_circle_endfaces_camera_rgb_all_wires.ply",
        "     -> standard-circle geometry + camera end-face RGB texture",

        "  2) wire_circle_endfaces_height_color_all_wires.ply",
        "     -> same true geometry + real H pseudo-color",
        "        NO_LASER wires are MAGENTA",

        "  3) wire_circle_endfaces_status_color_all_wires.ply",
        "     -> green=READY_DIRECT, yellow=PARTIAL, orange=LOW_COVERAGE, magenta=NO_LASER",

        "  4) wire_centers_info.ply",
        "     -> one point per wire; CloudCompare scalar fields for wire-level inspection",

        "  5) wire_information.csv",
        "     -> one row per wire; exact complete wire information",

        "",

        "IMPORTANT:",
        "  - camera_rgb.ply is no longer one-color-per-wire.",
        "  - every mesh vertex samples camera texture.",
        "  - new circle region outside original YOLO mask uses nearest SAME-wire RGB.",
        "  - geometry is NOT height exaggerated.",
        "  - every wire is built into 3-D.",
        "  - wires with laser use REAL H.",
        "  - wires without laser stay at baseline wheel radius and are marked as NO_LASER.",
        "  - every 3-D disk is a complete flat circle.",
        "  - y is periodic across 0/360 degrees.",
    ]

    summary_txt.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print()

    print(
        "=" * 82
    )

    print(
        "Complete"
    )

    print(
        "=" * 82
    )

    print(
        f"Circles              : "
        f"{len(circles)}"
    )

    print(
        f"Built 3D wires       : "
        f"{len(eligible_rows)} / {len(circles)}"
    )

    print(
        f"  real laser height  : "
        f"{built_real_height_count}"
    )

    print(
        f"  no laser baseline  : "
        f"{built_no_laser_count}"
    )

    print(
        f"Vertices             : "
        f"{len(vertices):,}"
    )

    print(
        f"Faces                : "
        f"{len(faces):,}"
    )

    print()

    print(
        "Camera texture:"
    )

    print(
        "  direct original-mask vertices : "
        f"{direct_count:,}"
    )

    print(
        "  nearest-wire filled vertices  : "
        f"{filled_count:,}"
    )

    if total_texture_vertices:
        print(
            "  filled fraction               : "
            f"{filled_count / total_texture_vertices:.2%}"
        )

    print()

    print(
        "Outputs:"
    )

    print(
        f"  circle map   : "
        f"{circle_map_path}"
    )

    print(
        "  circle overlay: "
        f"{args.out_dir / 'wire_circle_overlay.png'}"
    )

    print(
        "  circle preview: "
        f"{args.out_dir / 'wire_instance_map_circle_preview.png'}"
    )

    print(
        f"  laser table  : "
        f"{table_path}"
    )

    print(
        f"  3D NPZ       : "
        f"{npz_3d}"
    )

    print(
        f"  CAMERA RGB   : "
        f"{camera_ply}"
    )

    print(
        f"  HEIGHT COLOR : "
        f"{height_ply}"
    )

    print(
        f"  STATUS COLOR : "
        f"{status_ply}"
    )

    print(
        f"  WIRE CENTERS : "
        f"{wire_centers_ply}"
    )

    print(
        f"  WIRE INFO CSV: "
        f"{wire_info_csv}"
    )

    print(
        f"  summary      : "
        f"{summary_json}"
    )

    print(
        f"  summary      : "
        f"{summary_txt}"
    )

    print()

    print(
        "CloudCompare 现在先打开："
    )

    print(
        "  wire_circle_endfaces_camera_rgb.ply"
    )

    print(
        "这份是标准圆端面 + 相机真实 RGB 纹理 + 外向法向。"
    )

    print()

    print(
        "如果要在 CloudCompare 中逐根查看信息，同时打开："
    )

    print(
        "  wire_centers_info.ply"
    )

    print(
        "每一个中心点对应一个 wire_id，并带直径、高度、覆盖率、状态等 Scalar Fields。"
    )

    print()

    print(
        "如果要专门观察各根丝高度（NO_LASER=洋红色），再打开："
    )

    print(
        "  wire_circle_endfaces_height_color.ply"
    )


if __name__ == "__main__":
    main()
