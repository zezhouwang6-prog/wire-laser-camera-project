from __future__ import annotations

"""
单丝 3D -> 原始数据追溯检查器
===========================

用途
----
在 CloudCompare 中通过 wire_centers_info.ply 找到一个 wire_id 后，
用本程序查看该丝的完整来源信息。

例如：
    python wire_info_viewer.py --wire-id 3265

自动读取：
    config.py

默认读取：
    当前 run / wire_circle_3d_camera_texture_all_wires_info / wire_information.csv
    config.CAMERA_STITCHED_IMAGE
    config.SEGMENTATION_DIR / wire_instance_map.npz
    config.FUSION_DIR / mapped_height_mm.npz

输出：
    current_run/
      wire_circle_3d_camera_texture_all_wires_info/
        wire_inspection/
          wire_003265/
            wire_003265_camera.png
            wire_003265_mask_circle.png
            wire_003265_laser.png
            wire_003265_summary.png
            wire_003265_info.json
            wire_003265_info.txt

图像说明
--------
camera:
    原相机纹理
    红色 = 原 YOLO mask 轮廓
    绿色 = 强制标准圆
    黄色十字 = 圆心

mask_circle:
    白色 = 原 YOLO instance mask
    绿色 = 标准圆区域
    黄色 = 二者重合
    红十字 = 圆心

laser:
    mapped_height_mm 的局部高度伪彩图
    黑色 = 无有效激光
    白色轮廓 = 当前 wire 原 mask
    绿色圆 = 标准圆

summary:
    四联图 + 该 wire 的主要参数

360°周期
-------
Y 方向按周期截取，因此 y≈0 或 y≈H-1 的丝不会被截断。
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
        "无法导入项目 config.py。\n"
        "请把本程序放到 D:\\project\\scripts。\n"
        f"原始错误: {exc}"
    ) from exc


# =============================================================================
# DEFAULT PATHS
# =============================================================================

SEGMENTATION_DIR = Path(
    project_config.SEGMENTATION_DIR
)

FUSION_DIR = Path(
    project_config.FUSION_DIR
)

CAMERA_STITCHED_IMAGE = Path(
    project_config.CAMERA_STITCHED_IMAGE
)

if bool(
    getattr(
        project_config,
        "USE_COMBINED_ACQUISITION",
        False,
    )
):
    RUN_DIR = Path(
        project_config.ACTIVE_COMBINED_RUN_DIR
    )
else:
    RUN_DIR = Path(
        project_config.RUN_DIR
    )

MODEL_DIR = (
    RUN_DIR
    / "wire_circle_3d_camera_texture_all_wires_info"
)

DEFAULT_INFO_CSV = (
    MODEL_DIR
    / "wire_information.csv"
)

DEFAULT_INSTANCE_MAP = (
    SEGMENTATION_DIR
    / "wire_instance_map.npz"
)

DEFAULT_MAPPED_HEIGHT = (
    FUSION_DIR
    / "mapped_height_mm.npz"
)

DEFAULT_INSPECTION_DIR = (
    MODEL_DIR
    / "wire_inspection"
)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Inspect one wire by wire_id: camera, YOLO mask, "
            "fitted circle, mapped laser height, and all numeric information."
        )
    )

    p.add_argument(
        "--wire-id",
        type=int,
        required=True,
    )

    p.add_argument(
        "--info-csv",
        type=Path,
        default=DEFAULT_INFO_CSV,
    )

    p.add_argument(
        "--camera-image",
        type=Path,
        default=CAMERA_STITCHED_IMAGE,
    )

    p.add_argument(
        "--instance-map",
        type=Path,
        default=DEFAULT_INSTANCE_MAP,
    )

    p.add_argument(
        "--mapped-height",
        type=Path,
        default=DEFAULT_MAPPED_HEIGHT,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_INSPECTION_DIR,
    )

    p.add_argument(
        "--crop-half-size",
        type=int,
        default=24,
        help=(
            "Half crop size in pixels. "
            "24 => approximately 49x49 local source pixels."
        ),
    )

    p.add_argument(
        "--panel-size",
        type=int,
        default=420,
        help="Display size of each summary panel.",
    )

    return p.parse_args()


# =============================================================================
# DATA LOADERS
# =============================================================================

def read_wire_record(
    csv_path: Path,
    wire_id: int,
):
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 wire_information.csv: {csv_path}\n"
            "请先运行 V5 主建模程序。"
        )

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                rid = int(
                    float(
                        row["wire_id"]
                    )
                )
            except Exception:
                continue

            if rid == int(wire_id):
                return row

    raise KeyError(
        f"wire_id={wire_id} 不存在于 {csv_path}"
    )


def to_float(
    row,
    key,
    default=np.nan,
):
    try:
        return float(
            row[key]
        )
    except Exception:
        return float(
            default
        )


def to_int(
    row,
    key,
    default=0,
):
    try:
        return int(
            round(
                float(
                    row[key]
                )
            )
        )
    except Exception:
        return int(
            default
        )


def load_instance_map(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        if "instance_map" not in data:
            raise KeyError(
                f"{path} 不包含 instance_map"
            )

        return np.asarray(
            data["instance_map"],
            dtype=np.int32,
        )


def load_height(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            path
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
                f"{path} 缺少 height_mm / valid_mask"
            )

        h = np.asarray(
            data["height_mm"],
            dtype=np.float32,
        )

        valid = np.asarray(
            data["valid_mask"],
            dtype=np.uint8,
        ).astype(bool)

    valid &= np.isfinite(h)

    return h, valid


# =============================================================================
# PERIODIC CROPPING
# =============================================================================

def periodic_crop(
    arr: np.ndarray,
    cx: float,
    cy: float,
    half: int,
):
    """
    Crop around (cx, cy), wrapping only the Y direction.

    Returns:
        crop
        x0
        y_unwrapped0
    """
    H, W = arr.shape[:2]

    half = max(
        3,
        int(half),
    )

    x0 = max(
        0,
        int(
            math.floor(
                cx - half
            )
        ),
    )

    x1 = min(
        W - 1,
        int(
            math.ceil(
                cx + half
            )
        ),
    )

    yu0 = int(
        math.floor(
            cy - half
        )
    )

    yu1 = int(
        math.ceil(
            cy + half
        )
    )

    xs = np.arange(
        x0,
        x1 + 1,
        dtype=np.int32,
    )

    y_unwrapped = np.arange(
        yu0,
        yu1 + 1,
        dtype=np.int64,
    )

    ys = (
        y_unwrapped
        % H
    ).astype(
        np.int32
    )

    if arr.ndim == 2:
        crop = arr[
            ys[:, None],
            xs[None, :],
        ]
    else:
        crop = arr[
            ys[:, None],
            xs[None, :],
            :,
        ]

    return (
        crop.copy(),
        x0,
        yu0,
    )


def fitted_circle_mask(
    shape,
    center_local_x: float,
    center_local_y: float,
    radius_px: float,
):
    h, w = shape[:2]

    yy, xx = np.indices(
        (h, w),
        dtype=np.float32,
    )

    d2 = (
        (xx - center_local_x) ** 2
        + (yy - center_local_y) ** 2
    )

    return (
        d2 <= radius_px ** 2
    )


# =============================================================================
# DRAWING
# =============================================================================

def contour_from_mask(
    mask: np.ndarray,
):
    contours, _ = cv2.findContours(
        mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    return contours


def draw_camera_panel(
    camera_crop,
    wire_mask,
    circle_mask,
    center_xy,
):
    out = camera_crop.copy()

    mask_contours = contour_from_mask(
        wire_mask
    )

    circle_contours = contour_from_mask(
        circle_mask
    )

    cv2.drawContours(
        out,
        mask_contours,
        -1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.drawContours(
        out,
        circle_contours,
        -1,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    cx, cy = center_xy

    cv2.drawMarker(
        out,
        (
            int(round(cx)),
            int(round(cy)),
        ),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=7,
        thickness=1,
    )

    return out


def draw_mask_panel(
    wire_mask,
    circle_mask,
    center_xy,
):
    h, w = wire_mask.shape

    out = np.zeros(
        (h, w, 3),
        dtype=np.uint8,
    )

    # Circle only = green
    only_circle = (
        circle_mask
        & ~wire_mask
    )
    out[
        only_circle
    ] = (
        0,
        180,
        0,
    )

    # Original YOLO only = white
    only_mask = (
        wire_mask
        & ~circle_mask
    )
    out[
        only_mask
    ] = (
        255,
        255,
        255,
    )

    # Intersection = yellow
    both = (
        wire_mask
        & circle_mask
    )
    out[
        both
    ] = (
        0,
        255,
        255,
    )

    cx, cy = center_xy

    cv2.drawMarker(
        out,
        (
            int(round(cx)),
            int(round(cy)),
        ),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=7,
        thickness=1,
    )

    return out


def draw_laser_panel(
    height_crop,
    valid_crop,
    wire_mask,
    circle_mask,
):
    values = height_crop[
        valid_crop
    ]

    if values.size:
        lo = float(
            np.percentile(
                values,
                5,
            )
        )
        hi = float(
            np.percentile(
                values,
                95,
            )
        )

        if hi <= lo + 1e-9:
            hi = lo + 1.0

        norm = np.clip(
            (
                height_crop
                - lo
            )
            / (
                hi - lo
            ),
            0.0,
            1.0,
        )

        u8 = np.round(
            norm * 255.0
        ).astype(
            np.uint8
        )

        out = cv2.applyColorMap(
            u8,
            cv2.COLORMAP_TURBO,
        )

        out[
            ~valid_crop
        ] = (
            0,
            0,
            0,
        )
    else:
        out = np.zeros(
            (
                height_crop.shape[0],
                height_crop.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

    cv2.drawContours(
        out,
        contour_from_mask(
            wire_mask
        ),
        -1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.drawContours(
        out,
        contour_from_mask(
            circle_mask
        ),
        -1,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return out


def resize_panel(
    image,
    size: int,
    nearest=False,
):
    interpolation = (
        cv2.INTER_NEAREST
        if nearest
        else cv2.INTER_CUBIC
    )

    return cv2.resize(
        image,
        (
            size,
            size,
        ),
        interpolation=interpolation,
    )


def add_title(
    image,
    title: str,
):
    out = image.copy()

    cv2.rectangle(
        out,
        (0, 0),
        (
            out.shape[1] - 1,
            30,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        out,
        title,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def build_text_panel(
    record,
    size: int,
):
    panel = np.full(
        (
            size,
            size,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    wire_id = to_int(
        record,
        "wire_id",
    )

    height = to_float(
        record,
        "height_mm",
    )

    lines = [
        f"wire_id: {wire_id}",
        "",
        f"XYZ mm: {to_float(record,'x_mm'):.3f},",
        f"        {to_float(record,'y_mm'):.3f},",
        f"        {to_float(record,'z_mm'):.3f}",
        "",
        f"center px: ({to_float(record,'center_x_px'):.2f},",
        f"            {to_float(record,'center_y_px'):.2f})",
        f"theta: {to_float(record,'theta_deg'):.3f} deg",
        "",
        f"diameter: {to_float(record,'diameter_mm'):.4f} mm",
        f"radius:   {to_float(record,'radius_mm'):.4f} mm",
        "",
        (
            f"height H: {height:.4f} mm"
            if np.isfinite(height)
            else "height H: NaN (NO LASER)"
        ),
        f"center R: {to_float(record,'center_radius_mm'):.4f} mm",
        "",
        f"coverage: {to_float(record,'laser_coverage_percent'):.2f} %",
        f"valid pts: {to_int(record,'laser_filtered_valid_count')}",
        f"confidence: {to_float(record,'confidence'):.4f}",
        "",
        f"status: {record.get('status','')}",
        f"height source:",
        f"  {record.get('height_source','')}",
    ]

    y = 26

    for line in lines:
        cv2.putText(
            panel,
            line,
            (
                12,
                y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        y += 21

        if y > size - 10:
            break

    return panel


# =============================================================================
# SAVE INFO
# =============================================================================

def json_safe_value(
    v,
):
    if isinstance(
        v,
        (
            np.integer,
            int,
        ),
    ):
        return int(v)

    if isinstance(
        v,
        (
            np.floating,
            float,
        ),
    ):
        x = float(v)
        if not np.isfinite(x):
            return None
        return x

    return v


def normalized_record(
    row,
):
    out = {}

    for k, v in row.items():
        if k in {
            "status",
            "height_source",
        }:
            out[k] = v
            continue

        try:
            fv = float(v)

            if np.isfinite(fv):
                if (
                    abs(
                        fv - round(fv)
                    )
                    < 1e-12
                    and k in {
                        "wire_id",
                        "seam_merged",
                        "sample_mask_pixel_count",
                        "laser_raw_valid_count",
                        "laser_filtered_valid_count",
                        "status_code",
                        "height_source_code",
                        "has_real_laser_height",
                    }
                ):
                    out[k] = int(
                        round(fv)
                    )
                else:
                    out[k] = fv
            else:
                out[k] = None

        except Exception:
            out[k] = v

    return out


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    args.info_csv = (
        args.info_csv.resolve()
    )
    args.camera_image = (
        args.camera_image.resolve()
    )
    args.instance_map = (
        args.instance_map.resolve()
    )
    args.mapped_height = (
        args.mapped_height.resolve()
    )
    args.out_dir = (
        args.out_dir.resolve()
    )

    record = read_wire_record(
        args.info_csv,
        args.wire_id,
    )

    image = cv2.imread(
        str(
            args.camera_image
        ),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"无法读取相机图: {args.camera_image}"
        )

    instance_map = load_instance_map(
        args.instance_map
    )

    height_mm, valid_mask = load_height(
        args.mapped_height
    )

    if (
        image.shape[:2]
        != instance_map.shape
        or instance_map.shape
        != height_mm.shape
    ):
        raise RuntimeError(
            "camera / instance_map / mapped_height 尺寸不一致："
            f"{image.shape[:2]}, "
            f"{instance_map.shape}, "
            f"{height_mm.shape}"
        )

    H, W = instance_map.shape

    cx = to_float(
        record,
        "center_x_px",
    )

    cy = (
        to_float(
            record,
            "center_y_px",
        )
        % H
    )

    radius_px = to_float(
        record,
        "radius_px",
    )

    wire_id = int(
        args.wire_id
    )

    half = max(
        int(
            args.crop_half_size
        ),
        int(
            math.ceil(
                radius_px + 8
            )
        ),
    )

    camera_crop, x0, yu0 = periodic_crop(
        image,
        cx,
        cy,
        half,
    )

    id_crop, x0_id, yu0_id = periodic_crop(
        instance_map,
        cx,
        cy,
        half,
    )

    height_crop, x0_h, yu0_h = periodic_crop(
        height_mm,
        cx,
        cy,
        half,
    )

    valid_crop, x0_v, yu0_v = periodic_crop(
        valid_mask,
        cx,
        cy,
        half,
    )

    if not (
        x0 == x0_id == x0_h == x0_v
        and yu0 == yu0_id == yu0_h == yu0_v
    ):
        raise RuntimeError(
            "periodic crop offsets mismatch"
        )

    center_local_x = (
        cx - x0
    )

    center_local_y = (
        cy - yu0
    )

    wire_mask = (
        id_crop
        == wire_id
    )

    circle_mask = fitted_circle_mask(
        id_crop.shape,
        center_local_x,
        center_local_y,
        radius_px,
    )

    camera_panel = draw_camera_panel(
        camera_crop,
        wire_mask,
        circle_mask,
        (
            center_local_x,
            center_local_y,
        ),
    )

    mask_panel = draw_mask_panel(
        wire_mask,
        circle_mask,
        (
            center_local_x,
            center_local_y,
        ),
    )

    laser_panel = draw_laser_panel(
        height_crop,
        valid_crop,
        wire_mask,
        circle_mask,
    )

    panel_size = max(
        260,
        int(
            args.panel_size
        ),
    )

    camera_big = add_title(
        resize_panel(
            camera_panel,
            panel_size,
        ),
        "Camera: RED=YOLO, GREEN=Circle",
    )

    mask_big = add_title(
        resize_panel(
            mask_panel,
            panel_size,
            nearest=True,
        ),
        "Mask: WHITE=YOLO, GREEN=Circle",
    )

    laser_big = add_title(
        resize_panel(
            laser_panel,
            panel_size,
        ),
        "Laser mapped height",
    )

    text_big = add_title(
        build_text_panel(
            record,
            panel_size,
        ),
        "Wire information",
    )

    top = np.hstack(
        [
            camera_big,
            mask_big,
        ]
    )

    bottom = np.hstack(
        [
            laser_big,
            text_big,
        ]
    )

    summary = np.vstack(
        [
            top,
            bottom,
        ]
    )

    wire_dir = (
        args.out_dir
        / f"wire_{wire_id:06d}"
    )

    wire_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera_path = (
        wire_dir
        / f"wire_{wire_id:06d}_camera.png"
    )

    mask_path = (
        wire_dir
        / f"wire_{wire_id:06d}_mask_circle.png"
    )

    laser_path = (
        wire_dir
        / f"wire_{wire_id:06d}_laser.png"
    )

    summary_path = (
        wire_dir
        / f"wire_{wire_id:06d}_summary.png"
    )

    json_path = (
        wire_dir
        / f"wire_{wire_id:06d}_info.json"
    )

    txt_path = (
        wire_dir
        / f"wire_{wire_id:06d}_info.txt"
    )

    cv2.imwrite(
        str(camera_path),
        camera_panel,
    )

    cv2.imwrite(
        str(mask_path),
        mask_panel,
    )

    cv2.imwrite(
        str(laser_path),
        laser_panel,
    )

    cv2.imwrite(
        str(summary_path),
        summary,
    )

    record_norm = normalized_record(
        record
    )

    record_norm["inspection"] = {
        "wire_id": wire_id,
        "periodic_y": True,
        "image_height_px": H,
        "image_width_px": W,
        "crop_half_size_px": half,
        "crop_x0_px": x0,
        "crop_y_unwrapped0_px": yu0,
        "source_camera": str(
            args.camera_image
        ),
        "source_instance_map": str(
            args.instance_map
        ),
        "source_mapped_height": str(
            args.mapped_height
        ),
    }

    json_path.write_text(
        json.dumps(
            record_norm,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"wire_id = {wire_id}",
        "=" * 60,
    ]

    for key, value in record_norm.items():
        if key == "inspection":
            continue
        lines.append(
            f"{key} = {value}"
        )

    txt_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"wire_id = {wire_id}")
    print("=" * 72)

    print(
        f"status         : {record.get('status','')}"
    )

    print(
        f"height source  : {record.get('height_source','')}"
    )

    h = to_float(
        record,
        "height_mm",
    )

    print(
        "height H (mm)  : "
        + (
            f"{h:.6f}"
            if np.isfinite(h)
            else "NaN"
        )
    )

    print(
        f"diameter (mm)  : "
        f"{to_float(record,'diameter_mm'):.6f}"
    )

    print(
        f"coverage (%)   : "
        f"{to_float(record,'laser_coverage_percent'):.3f}"
    )

    print(
        f"confidence     : "
        f"{to_float(record,'confidence'):.6f}"
    )

    print(
        f"3D center (mm) : "
        f"({to_float(record,'x_mm'):.4f}, "
        f"{to_float(record,'y_mm'):.4f}, "
        f"{to_float(record,'z_mm'):.4f})"
    )

    print()
    print("Outputs:")
    print(f"  {summary_path}")
    print(f"  {camera_path}")
    print(f"  {mask_path}")
    print(f"  {laser_path}")
    print(f"  {json_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()
