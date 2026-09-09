from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from config import YOLO_MODEL_PATH, CAMERA_STITCHED_IMAGE, SEGMENTATION_DIR

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = YOLO_MODEL_PATH
DEFAULT_IMAGE = CAMERA_STITCHED_IMAGE
DEFAULT_OUT_DIR = SEGMENTATION_DIR

# =============================================================================
# Existing detection / measurement settings
# =============================================================================
TILE_SIZE = 320
OVERLAP = 0.50
CONF = 0.25
EDGE_CONF = 0.5
EDGE_MARGIN_RATIO = 0.10
IOU = 0.80
MAX_DET = 1000
IMGSZ = 320

# Camera stitched-image metric scale.
PIXEL_SIZE_MM = 0.05473705

MERGE_CENTER_DISTANCE_PX = 10.0
MERGE_CENTER_RADIUS_RATIO = 0.65
MERGE_OVERLAP_IOU = 0.35

MIN_DIAMETER_PX = 10
MAX_DIAMETER_PX = 20.0
RADIUS_SCALE = 1.0

# =============================================================================
# NEW: full-image / 3-D preparation settings
# =============================================================================

# The stitched image is a 360-degree unwrapped surface:
#     image X -> wheel axial direction
#     image Y -> circumference / rotation direction
#
# Therefore top and bottom are periodic neighbors.
ENABLE_PERIODIC_Y_SEAM_MERGE = True

# Search band near y=0 and y=H for the same wire split by the 0/360 seam.
# The effective band is automatically enlarged if the wire diameter is larger.
PERIODIC_SEAM_BAND_PX = 30.0

# If a wire is cut by the 0/360 seam, each visible half may be smaller than the
# normal minimum diameter. Keep small seam pieces temporarily so they can be
# combined first; apply the strict diameter filter after seam merge.
MIN_SEAM_PART_RADIUS_PX = 2.0

# Save authoritative full-resolution instance ID map:
#     0 = background
#     1 = wire 1
#     2 = wire 2
#     ...
SAVE_INSTANCE_MAP_NPZ = True
INSTANCE_MAP_FILENAME = "wire_instance_map.npz"

# Save a 16-bit PNG ID map when wire count <= 65535.
SAVE_INSTANCE_ID_PNG = True

# Save a colored full-image preview of unique wire IDs.
SAVE_INSTANCE_PREVIEW = True

# Save every final wire contour in GLOBAL stitched-image pixel coordinates.
SAVE_GLOBAL_CONTOURS_JSON = True

# Save tile origin table, useful for debugging local -> global conversion.
SAVE_TILE_MANIFEST = True


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO segmentation on a full stitched image by tiling, "
            "restore every wire to GLOBAL stitched-image coordinates, "
            "merge overlap duplicates, merge the periodic 0/360 seam, "
            "and save a full-resolution instance map for direct matching "
            "with mapped_height_mm.npz."
        )
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf", type=float, default=CONF)
    parser.add_argument("--edge-conf", type=float, default=EDGE_CONF)
    parser.add_argument("--edge-margin-ratio", type=float, default=EDGE_MARGIN_RATIO)
    parser.add_argument("--iou", type=float, default=IOU)
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--merge-distance", type=float, default=MERGE_CENTER_DISTANCE_PX)
    parser.add_argument("--merge-center-ratio", type=float, default=MERGE_CENTER_RADIUS_RATIO)
    parser.add_argument("--merge-overlap-iou", type=float, default=MERGE_OVERLAP_IOU)
    parser.add_argument("--min-diameter", type=float, default=MIN_DIAMETER_PX)
    parser.add_argument("--max-diameter", type=float, default=MAX_DIAMETER_PX)
    parser.add_argument("--min-radius", type=float, default=None)
    parser.add_argument("--max-radius", type=float, default=None)
    parser.add_argument("--radius-scale", type=float, default=RADIUS_SCALE)
    parser.add_argument("--pixel-size-mm", type=float, default=PIXEL_SIZE_MM)
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--seam-band",
        type=float,
        default=PERIODIC_SEAM_BAND_PX,
        help="Search band in pixels near y=0 / y=H for periodic duplicate wires.",
    )
    parser.add_argument(
        "--no-periodic-seam-merge",
        action="store_true",
        help="Disable 0/360 periodic seam merging.",
    )
    parser.add_argument(
        "--draw-contours",
        action="store_true",
        help="Also save a separate debug overlay with the original mask contours.",
    )
    return parser.parse_args()


# =============================================================================
# TILING
# =============================================================================

def make_positions(length, tile_size, step):
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def tile_image(img, tile_size, overlap):
    h, w = img.shape[:2]
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    xs = make_positions(w, tile_size, step)
    ys = make_positions(h, tile_size, step)

    tile_index = 0
    for y in ys:
        for x in xs:
            tile_index += 1
            tile = img[y : y + tile_size, x : x + tile_size]
            yield tile_index, x, y, tile


# =============================================================================
# LOCAL MASK -> GLOBAL STITCHED-IMAGE COORDINATES
# =============================================================================

def mask_to_circle(poly, offset_x, offset_y, radius_scale):
    """
    Ultralytics result.masks.xy is in the coordinate system of the tile that was
    passed to model.predict().

    Convert EVERY polygon point to the GLOBAL stitched-image coordinate system:
        global_x = local_x + tile_x0
        global_y = local_y + tile_y0
    """
    pts = np.asarray(poly, dtype=np.float32)
    if len(pts) < 3:
        return None

    pts_global = pts.copy()
    pts_global[:, 0] += float(offset_x)
    pts_global[:, 1] += float(offset_y)

    contour = np.rint(pts_global).astype(np.int32).reshape(-1, 1, 2)

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])

    area = max(1.0, float(cv2.contourArea(contour)))
    radius = max(
        2.0,
        float(np.sqrt(area / np.pi) * float(radius_scale)),
    )

    return {
        "center_x": cx,
        "center_y": cy,
        "radius_px": radius,
        "area_px": area,
        "contour": contour,
        "contours": [contour],
        "radius_scale": float(radius_scale),
        "seam_merged": False,
        "seam_part_count": 1,
        "source_tiles": [],
    }


# =============================================================================
# CIRCLE DUPLICATE LOGIC
# =============================================================================

def circle_iou(a, b):
    r1 = float(a["radius_px"])
    r2 = float(b["radius_px"])
    dx = float(a["center_x"] - b["center_x"])
    dy = float(a["center_y"] - b["center_y"])
    d = math.hypot(dx, dy)
    area1 = math.pi * r1 * r1
    area2 = math.pi * r2 * r2

    if d >= r1 + r2:
        return 0.0

    if d <= abs(r1 - r2):
        inter = min(area1, area2)
    else:
        cos1 = (d * d + r1 * r1 - r2 * r2) / max(1e-9, 2.0 * d * r1)
        cos2 = (d * d + r2 * r2 - r1 * r1) / max(1e-9, 2.0 * d * r2)
        cos1 = max(-1.0, min(1.0, cos1))
        cos2 = max(-1.0, min(1.0, cos2))

        part1 = r1 * r1 * math.acos(cos1)
        part2 = r2 * r2 * math.acos(cos2)
        part3 = 0.5 * math.sqrt(
            max(
                0.0,
                (-d + r1 + r2)
                * (d + r1 - r2)
                * (d - r1 + r2)
                * (d + r1 + r2),
            )
        )
        inter = part1 + part2 - part3

    union = area1 + area2 - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def circle_iou_periodic_y(a, b, image_height):
    """
    Same circle IoU, but y is periodic:
        y=0 and y=H are neighbors.
    """
    r1 = float(a["radius_px"])
    r2 = float(b["radius_px"])

    dx = float(a["center_x"] - b["center_x"])
    dy_raw = abs(float(a["center_y"] - b["center_y"]))
    dy = min(dy_raw, max(0.0, float(image_height) - dy_raw))
    d = math.hypot(dx, dy)

    area1 = math.pi * r1 * r1
    area2 = math.pi * r2 * r2

    if d >= r1 + r2:
        return 0.0

    if d <= abs(r1 - r2):
        inter = min(area1, area2)
    else:
        cos1 = (d * d + r1 * r1 - r2 * r2) / max(1e-9, 2.0 * d * r1)
        cos2 = (d * d + r2 * r2 - r1 * r1) / max(1e-9, 2.0 * d * r2)
        cos1 = max(-1.0, min(1.0, cos1))
        cos2 = max(-1.0, min(1.0, cos2))

        part1 = r1 * r1 * math.acos(cos1)
        part2 = r2 * r2 * math.acos(cos2)
        part3 = 0.5 * math.sqrt(
            max(
                0.0,
                (-d + r1 + r2)
                * (d + r1 - r2)
                * (d - r1 + r2)
                * (d + r1 + r2),
            )
        )
        inter = part1 + part2 - part3

    union = area1 + area2 - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def is_duplicate_circle(
    circle,
    old,
    merge_distance,
    merge_center_ratio,
    merge_overlap_iou,
):
    dx = float(circle["center_x"] - old["center_x"])
    dy = float(circle["center_y"] - old["center_y"])
    dist2 = dx * dx + dy * dy

    min_radius = min(
        float(circle["radius_px"]),
        float(old["radius_px"]),
    )
    center_limit = max(
        float(merge_distance),
        float(merge_center_ratio) * min_radius,
    )

    if dist2 <= center_limit * center_limit:
        return True

    return circle_iou(circle, old) >= float(merge_overlap_iou)


def is_duplicate_circle_periodic_y(
    circle,
    old,
    image_height,
    merge_distance,
    merge_center_ratio,
    merge_overlap_iou,
):
    dx = float(circle["center_x"] - old["center_x"])

    dy_raw = abs(float(circle["center_y"] - old["center_y"]))
    dy = min(
        dy_raw,
        max(0.0, float(image_height) - dy_raw),
    )

    dist2 = dx * dx + dy * dy
    min_radius = min(
        float(circle["radius_px"]),
        float(old["radius_px"]),
    )
    center_limit = max(
        float(merge_distance),
        float(merge_center_ratio) * min_radius,
    )

    if dist2 <= center_limit * center_limit:
        return True

    return (
        circle_iou_periodic_y(circle, old, image_height)
        >= float(merge_overlap_iou)
    )


def circle_quality_key(item):
    return (
        float(item.get("confidence", 0.0)),
        float(item.get("tile_center_score", 0.0)),
        float(item.get("area_px", 0.0)),
    )


def _source_tile_signature(tile_info):
    return (
        int(tile_info.get("tile_index", -1)),
        int(tile_info.get("tile_x0", -1)),
        int(tile_info.get("tile_y0", -1)),
    )


def _merge_source_tile_lists(dst, src):
    existing = {
        _source_tile_signature(item)
        for item in dst
    }
    for item in src:
        sig = _source_tile_signature(item)
        if sig not in existing:
            dst.append(item)
            existing.add(sig)


def merge_circles(
    circles,
    merge_distance,
    merge_center_ratio,
    merge_overlap_iou,
):
    """
    Merge duplicate detections created by the 50% overlapping tiles.

    IMPORTANT:
    The centers and contours entering this function are already in GLOBAL
    stitched-image coordinates.
    """
    if not circles:
        return []

    ordered = sorted(
        circles,
        key=circle_quality_key,
        reverse=True,
    )

    kept = []
    max_radius = max(float(circle["radius_px"]) for circle in ordered)
    cell_size = max(
        16.0,
        float(merge_distance),
        max_radius * 2.0,
    )
    grid = {}

    def grid_key(circle):
        return (
            int(math.floor(float(circle["center_x"]) / cell_size)),
            int(math.floor(float(circle["center_y"]) / cell_size)),
        )

    for index, circle in enumerate(ordered, 1):
        duplicate_target = None

        gx, gy = grid_key(circle)
        search_cells = int(
            math.ceil(
                (
                    float(circle["radius_px"])
                    + max_radius
                    + float(merge_distance)
                )
                / cell_size
            )
        )

        for yy in range(gy - search_cells, gy + search_cells + 1):
            for xx in range(gx - search_cells, gx + search_cells + 1):
                for old in grid.get((xx, yy), []):
                    if is_duplicate_circle(
                        circle,
                        old,
                        merge_distance,
                        merge_center_ratio,
                        merge_overlap_iou,
                    ):
                        duplicate_target = old
                        break

                if duplicate_target is not None:
                    break

            if duplicate_target is not None:
                break

        if duplicate_target is None:
            kept.append(circle)
            grid.setdefault((gx, gy), []).append(circle)
        else:
            _merge_source_tile_lists(
                duplicate_target.setdefault("source_tiles", []),
                circle.get("source_tiles", []),
            )

        if index % 5000 == 0:
            print(
                f"overlap merge {index}/{len(ordered)}: "
                f"kept={len(kept)}"
            )

    return kept


# =============================================================================
# PERIODIC 0 / 360 DEGREE SEAM MERGE
# =============================================================================

def _circular_weighted_mean_y(values_y, weights, image_height):
    values_y = np.asarray(values_y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    h = float(image_height)
    theta = 2.0 * np.pi * np.mod(values_y, h) / h

    s = float(np.sum(weights * np.sin(theta)))
    c = float(np.sum(weights * np.cos(theta)))

    if abs(s) < 1e-12 and abs(c) < 1e-12:
        return float(np.mod(values_y[0], h))

    angle = math.atan2(s, c)
    if angle < 0.0:
        angle += 2.0 * math.pi

    return float(angle / (2.0 * math.pi) * h)


def _combine_periodic_group(group, image_height):
    """
    Combine top/bottom pieces of the SAME physical wire.

    Unlike normal tile duplicates, these can be complementary clipped halves,
    so their contour areas are added and all contour pieces are kept under one ID.
    """
    if len(group) == 1:
        item = group[0]
        item["center_y"] = float(item["center_y"] % image_height)
        return item

    primary = max(group, key=circle_quality_key)

    areas = np.asarray(
        [max(1.0, float(item["area_px"])) for item in group],
        dtype=np.float64,
    )
    sum_area = float(np.sum(areas))

    center_x = float(
        np.sum(
            areas
            * np.asarray(
                [float(item["center_x"]) for item in group],
                dtype=np.float64,
            )
        )
        / sum_area
    )

    center_y = _circular_weighted_mean_y(
        [float(item["center_y"]) for item in group],
        areas,
        image_height,
    )

    all_contours = []
    source_tiles = []
    for item in group:
        all_contours.extend(
            item.get("contours", [item["contour"]])
        )
        _merge_source_tile_lists(
            source_tiles,
            item.get("source_tiles", []),
        )

    radius_scale = float(primary.get("radius_scale", 1.0))
    combined_radius = max(
        2.0,
        float(np.sqrt(sum_area / np.pi) * radius_scale),
    )

    combined = dict(primary)
    combined["center_x"] = center_x
    combined["center_y"] = center_y
    combined["area_px"] = sum_area
    combined["radius_px"] = combined_radius
    combined["contours"] = all_contours
    combined["contour"] = all_contours[0]
    combined["source_tiles"] = source_tiles
    combined["seam_merged"] = True
    combined["seam_part_count"] = int(
        sum(int(item.get("seam_part_count", 1)) for item in group)
    )
    combined["confidence"] = float(
        max(float(item.get("confidence", 0.0)) for item in group)
    )
    combined["tile_center_score"] = float(
        max(float(item.get("tile_center_score", 0.0)) for item in group)
    )
    return combined


def merge_periodic_y_seam(
    circles,
    image_height,
    seam_band,
    merge_distance,
    merge_center_ratio,
    merge_overlap_iou,
):
    if not circles:
        return []

    h = float(image_height)
    max_radius = max(float(item["radius_px"]) for item in circles)

    effective_band = max(
        float(seam_band),
        2.0 * max_radius + float(merge_distance),
    )

    top_indices = [
        i
        for i, item in enumerate(circles)
        if float(item["center_y"]) <= effective_band
    ]
    bottom_indices = [
        i
        for i, item in enumerate(circles)
        if float(item["center_y"]) >= h - effective_band
    ]

    if not top_indices or not bottom_indices:
        return circles

    parent = list(range(len(circles)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    candidate_pairs = 0
    matched_pairs = 0

    for i in top_indices:
        for j in bottom_indices:
            if i == j:
                continue

            candidate_pairs += 1

            if is_duplicate_circle_periodic_y(
                circles[i],
                circles[j],
                image_height,
                merge_distance,
                merge_center_ratio,
                merge_overlap_iou,
            ):
                union(i, j)
                matched_pairs += 1

    groups = {}
    for i in range(len(circles)):
        groups.setdefault(find(i), []).append(circles[i])

    merged = [
        _combine_periodic_group(group, image_height)
        for group in groups.values()
    ]

    seam_groups = sum(1 for group in groups.values() if len(group) > 1)

    print(
        "periodic seam merge: "
        f"top={len(top_indices)}, bottom={len(bottom_indices)}, "
        f"candidate_pairs={candidate_pairs}, "
        f"matched_pairs={matched_pairs}, "
        f"merged_groups={seam_groups}, "
        f"count {len(circles)} -> {len(merged)}"
    )

    return merged


# =============================================================================
# FILTERS
# =============================================================================

def is_near_periodic_seam(circle, image_height, seam_band):
    y = float(circle["center_y"])
    return y <= float(seam_band) or y >= float(image_height) - float(seam_band)


def prefilter_circles_keep_seam_parts(
    circles,
    min_radius,
    max_radius,
    image_height,
    seam_band,
):
    """
    Keep normal valid circles, plus small clipped pieces near the 0/360 seam.
    Those seam pieces are strictly filtered again after periodic merging.
    """
    result = []

    for circle in circles:
        radius = float(circle["radius_px"])

        normal_ok = float(min_radius) <= radius <= float(max_radius)

        seam_piece_ok = (
            ENABLE_PERIODIC_Y_SEAM_MERGE
            and is_near_periodic_seam(
                circle,
                image_height,
                seam_band,
            )
            and float(MIN_SEAM_PART_RADIUS_PX) <= radius <= float(max_radius)
        )

        if normal_ok or seam_piece_ok:
            result.append(circle)

    return result


def filter_circles_by_radius(circles, min_radius, max_radius):
    return [
        circle
        for circle in circles
        if float(min_radius)
        <= float(circle["radius_px"])
        <= float(max_radius)
    ]


# =============================================================================
# GLOBAL PHYSICAL COORDINATES
# =============================================================================

def attach_global_physical_coordinates(
    circles,
    image_width,
    image_height,
    pixel_size_mm,
):
    """
    Coordinate convention matched to the fusion / mapped_height_mm system:

        camera stitched-image horizontal center = X = 0 mm
        image row 0 = theta = 0 deg
        image row H approaches theta = 360 deg

    `center_x` and `center_y` remain GLOBAL image pixels for compatibility.
    """
    camera_center_x_px = (float(image_width) - 1.0) / 2.0

    for circle in circles:
        gx = float(circle["center_x"])
        gy = float(circle["center_y"]) % float(image_height)

        circle["center_y"] = gy

        circle["center_x_mm"] = (
            gx - camera_center_x_px
        ) * float(pixel_size_mm)

        circle["center_x_from_left_mm"] = gx * float(pixel_size_mm)

        circle["center_y_mm"] = gy * float(pixel_size_mm)

        circle["theta_deg"] = (
            gy / max(float(image_height), 1.0) * 360.0
        )

        circle["diameter_px"] = float(circle["radius_px"]) * 2.0
        circle["diameter_mm"] = (
            float(circle["diameter_px"]) * float(pixel_size_mm)
        )


# =============================================================================
# DRAWING
# =============================================================================

def draw_full_result(img, circles, out_path, line_width, draw_contours=False):
    overlay = img.copy()

    for circle in circles:
        center = (
            int(round(float(circle["center_x"]))),
            int(round(float(circle["center_y"]))),
        )
        radius = max(1, int(round(float(circle["radius_px"]))))

        cv2.circle(
            overlay,
            center,
            radius,
            (0, 0, 255),
            int(line_width),
        )

        if draw_contours:
            for contour in circle.get(
                "contours",
                [circle["contour"]],
            ):
                cv2.polylines(
                    overlay,
                    [contour],
                    True,
                    (0, 255, 255),
                    int(line_width),
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def _wire_color_bgr(wire_id):
    b = (37 * int(wire_id) + 53) % 256
    g = (97 * int(wire_id) + 101) % 256
    r = (17 * int(wire_id) + 173) % 256

    b = max(b, 50)
    g = max(g, 50)
    r = max(r, 50)
    return (b, g, r)


# =============================================================================
# INSTANCE MAP
# =============================================================================

def build_instance_map(img_shape, circles):
    """
    Build an authoritative full-resolution wire ID map.

    The map uses the SAME pixel grid as the original stitched camera image:
        instance_map[y, x]

    Therefore it can be indexed directly together with:
        mapped_height_mm["height_mm"][y, x]
    """
    h, w = img_shape[:2]

    instance_map = np.zeros((h, w), dtype=np.int32)
    score_map = np.full((h, w), -np.inf, dtype=np.float32)

    conflict_pixels = 0

    for wire_id, circle in enumerate(circles, 1):
        temp_mask = np.zeros((h, w), dtype=np.uint8)

        for contour in circle.get(
            "contours",
            [circle["contour"]],
        ):
            cv2.fillPoly(
                temp_mask,
                [contour],
                1,
            )

        region = temp_mask.astype(bool)
        if not np.any(region):
            continue

        score = float(circle.get("confidence", 0.0))
        occupied_other = region & (instance_map != 0)
        conflict_pixels += int(np.count_nonzero(occupied_other))

        replace = region & (score >= score_map)
        instance_map[replace] = int(wire_id)
        score_map[replace] = float(score)

        circle["id"] = int(wire_id)

    for wire_id, circle in enumerate(circles, 1):
        circle["instance_pixel_count"] = int(
            np.count_nonzero(instance_map == int(wire_id))
        )

    return instance_map, conflict_pixels


def save_instance_map_npz(
    path,
    instance_map,
    circles,
    image_path,
    pixel_size_mm,
):
    h, w = instance_map.shape
    camera_center_x_px = (float(w) - 1.0) / 2.0

    np.savez_compressed(
        path,
        instance_map=instance_map.astype(np.int32),
        image_width_px=np.int32(w),
        image_height_px=np.int32(h),
        wire_count=np.int32(len(circles)),
        pixel_size_mm=np.float32(pixel_size_mm),
        camera_center_x_px=np.float32(camera_center_x_px),
        coordinate_system=np.array(
            "GLOBAL_CAMERA_STITCHED_IMAGE: x=axial, y=circumference, "
            "camera_horizontal_center=X0mm, y0=theta0deg"
        ),
        source_camera_image=np.array(str(image_path)),
    )


def save_instance_preview(img, instance_map, out_path):
    h, w = instance_map.shape
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)

    ids = np.unique(instance_map)
    ids = ids[ids > 0]

    for wire_id in ids:
        color_layer[instance_map == int(wire_id)] = _wire_color_bgr(int(wire_id))

    mask = instance_map > 0

    preview = img.copy()
    if np.any(mask):
        blended = cv2.addWeighted(
            img,
            0.50,
            color_layer,
            0.50,
            0.0,
        )
        preview[mask] = blended[mask]

    cv2.imwrite(str(out_path), preview)


# =============================================================================
# OUTPUT TABLES / JSON
# =============================================================================

def _primary_source_tile(circle):
    source_tiles = list(circle.get("source_tiles", []))
    if not source_tiles:
        return {}

    return max(
        source_tiles,
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            float(item.get("tile_center_score", 0.0)),
        ),
    )


def save_csv(path, circles):
    """
    Keep old fields for compatibility and add explicit GLOBAL fields.
    """
    fields = [
        "id",
        "center_x",
        "center_y",
        "diameter_px",
        "center_x_mm",
        "center_y_mm",
        "diameter_mm",
        "global_center_x_px",
        "global_center_y_px",
        "theta_deg",
        "radius_px",
        "area_px",
        "instance_pixel_count",
        "confidence",
        "camera_center_is_x0",
        "center_x_from_left_mm",
        "seam_merged",
        "seam_part_count",
        "source_tile_count",
        "source_tile_indices",
        "primary_tile_index",
        "primary_tile_x0",
        "primary_tile_y0",
        "primary_local_center_x_px",
        "primary_local_center_y_px",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for idx, circle in enumerate(circles, 1):
            primary = _primary_source_tile(circle)
            source_tiles = circle.get("source_tiles", [])

            writer.writerow(
                {
                    "id": idx,
                    "center_x": float(circle["center_x"]),
                    "center_y": float(circle["center_y"]),
                    "diameter_px": float(circle["diameter_px"]),
                    "center_x_mm": float(circle["center_x_mm"]),
                    "center_y_mm": float(circle["center_y_mm"]),
                    "diameter_mm": float(circle["diameter_mm"]),
                    "global_center_x_px": float(circle["center_x"]),
                    "global_center_y_px": float(circle["center_y"]),
                    "theta_deg": float(circle["theta_deg"]),
                    "radius_px": float(circle["radius_px"]),
                    "area_px": float(circle["area_px"]),
                    "instance_pixel_count": int(
                        circle.get("instance_pixel_count", 0)
                    ),
                    "confidence": float(circle.get("confidence", 0.0)),
                    "camera_center_is_x0": 1,
                    "center_x_from_left_mm": float(
                        circle["center_x_from_left_mm"]
                    ),
                    "seam_merged": int(bool(circle.get("seam_merged", False))),
                    "seam_part_count": int(circle.get("seam_part_count", 1)),
                    "source_tile_count": len(source_tiles),
                    "source_tile_indices": ";".join(
                        str(int(item.get("tile_index", -1)))
                        for item in source_tiles
                    ),
                    "primary_tile_index": primary.get("tile_index", ""),
                    "primary_tile_x0": primary.get("tile_x0", ""),
                    "primary_tile_y0": primary.get("tile_y0", ""),
                    "primary_local_center_x_px": primary.get(
                        "local_center_x_px",
                        "",
                    ),
                    "primary_local_center_y_px": primary.get(
                        "local_center_y_px",
                        "",
                    ),
                }
            )


def save_summary_csv(path, circles):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wire_count",
                "periodic_seam_merged_wire_count",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "wire_count": len(circles),
                "periodic_seam_merged_wire_count": sum(
                    1
                    for circle in circles
                    if circle.get("seam_merged", False)
                ),
            }
        )


def save_tile_manifest(path, tile_records):
    fields = [
        "tile_index",
        "tile_x0",
        "tile_y0",
        "tile_width",
        "tile_height",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(tile_records)


def _contour_to_xy_list(contour):
    pts = np.asarray(contour, dtype=np.int32).reshape(-1, 2)
    return [
        [int(x), int(y)]
        for x, y in pts
    ]


def save_global_contours_json(
    path,
    circles,
    image_path,
    image_width,
    image_height,
    pixel_size_mm,
):
    payload = {
        "source_camera_image": str(image_path),
        "image_width_px": int(image_width),
        "image_height_px": int(image_height),
        "pixel_size_mm": float(pixel_size_mm),
        "camera_center_x_px": (float(image_width) - 1.0) / 2.0,
        "coordinate_system": {
            "pixel": (
                "GLOBAL stitched-camera coordinates; "
                "array indexing is [y, x]"
            ),
            "x_mm": (
                "camera horizontal center is X=0 mm; "
                "X_mm=(x_px-camera_center_x_px)*pixel_size_mm"
            ),
            "theta_deg": (
                "theta_deg=(y_px/image_height_px)*360"
            ),
            "periodic_y": True,
        },
        "wires": [],
    }

    for idx, circle in enumerate(circles, 1):
        payload["wires"].append(
            {
                "id": int(idx),
                "global_center_x_px": float(circle["center_x"]),
                "global_center_y_px": float(circle["center_y"]),
                "center_x_mm": float(circle["center_x_mm"]),
                "theta_deg": float(circle["theta_deg"]),
                "diameter_px": float(circle["diameter_px"]),
                "diameter_mm": float(circle["diameter_mm"]),
                "area_px": float(circle["area_px"]),
                "confidence": float(circle.get("confidence", 0.0)),
                "seam_merged": bool(circle.get("seam_merged", False)),
                "contours_global_px": [
                    _contour_to_xy_list(contour)
                    for contour in circle.get(
                        "contours",
                        [circle["contour"]],
                    )
                ],
            }
        )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_metadata_json(
    path,
    args,
    image_path,
    image_width,
    image_height,
    circles,
    conflict_pixels,
):
    payload = {
        "source_camera_image": str(image_path),
        "image_width_px": int(image_width),
        "image_height_px": int(image_height),
        "wire_count": int(len(circles)),
        "instance_map_shape_rows_cols": [
            int(image_height),
            int(image_width),
        ],
        "instance_map_indexing": "instance_map[y, x]",
        "mapped_height_direct_match": (
            "wire_instance_map.npz and mapped_height_mm.npz must be generated "
            "from the same stitched camera image. Then both arrays use the same "
            "[y, x] camera grid and can be indexed directly."
        ),
        "pixel_size_mm": float(args.pixel_size_mm),
        "camera_center_x_px": (float(image_width) - 1.0) / 2.0,
        "camera_center_is_x0_mm": True,
        "theta_definition": "theta_deg=(global_y_px/image_height_px)*360",
        "periodic_y_0_360": True,
        "periodic_seam_merge_enabled": not bool(args.no_periodic_seam_merge),
        "periodic_seam_band_px": float(args.seam_band),
        "instance_map_overlap_conflict_pixels": int(conflict_pixels),
        "tile_size_px": int(args.tile_size),
        "tile_overlap": float(args.overlap),
        "local_to_global_formula": {
            "global_x_px": "local_x_px + tile_x0",
            "global_y_px": "local_y_px + tile_y0",
        },
    }

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    image_height, image_width = img.shape[:2]

    model = YOLO(str(args.model))

    raw_circles = []
    tile_records = []
    tile_count = 0

    edge_margin = image_width * float(args.edge_margin_ratio)
    inference_conf = min(float(args.conf), float(args.edge_conf))

    for tile_index, x, y, tile in tile_image(
        img,
        args.tile_size,
        args.overlap,
    ):
        tile_count += 1

        tile_h, tile_w = tile.shape[:2]
        tile_records.append(
            {
                "tile_index": int(tile_index),
                "tile_x0": int(x),
                "tile_y0": int(y),
                "tile_width": int(tile_w),
                "tile_height": int(tile_h),
            }
        )

        result = model.predict(
            source=tile,
            imgsz=args.imgsz,
            conf=inference_conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
            retina_masks=True,
        )[0]

        if (
            result.masks is None
            or result.masks.xy is None
            or result.boxes is None
        ):
            print(
                f"tile {tile_index}: "
                f"x={x}, y={y}, detections=0, raw={len(raw_circles)}"
            )
            continue

        confidences = result.boxes.conf.cpu().tolist()

        tile_detections = 0

        for poly, confidence in zip(
            result.masks.xy,
            confidences,
        ):
            circle = mask_to_circle(
                poly,
                x,
                y,
                args.radius_scale,
            )
            if circle is None:
                continue

            center_x = float(circle["center_x"])

            is_edge = (
                center_x < edge_margin
                or center_x > image_width - edge_margin
            )
            threshold = (
                float(args.edge_conf)
                if is_edge
                else float(args.conf)
            )
            if float(confidence) < threshold:
                continue

            circle["confidence"] = float(confidence)

            local_cx = float(circle["center_x"]) - float(x)
            local_cy = float(circle["center_y"]) - float(y)

            distance_to_tile_edge = min(
                local_cx,
                local_cy,
                float(tile_w) - local_cx,
                float(tile_h) - local_cy,
            )

            circle["tile_center_score"] = float(
                np.clip(
                    distance_to_tile_edge
                    / max(
                        1.0,
                        min(float(tile_w), float(tile_h)) * 0.5,
                    ),
                    0.0,
                    1.0,
                )
            )

            source_tile_info = {
                "tile_index": int(tile_index),
                "tile_x0": int(x),
                "tile_y0": int(y),
                "tile_width": int(tile_w),
                "tile_height": int(tile_h),
                "local_center_x_px": float(local_cx),
                "local_center_y_px": float(local_cy),
                "confidence": float(confidence),
                "tile_center_score": float(circle["tile_center_score"]),
            }
            circle["source_tiles"] = [source_tile_info]

            raw_circles.append(circle)
            tile_detections += 1

        print(
            f"tile {tile_index}: x={x}, y={y}, "
            f"detections={tile_detections}, raw={len(raw_circles)}"
        )

    min_radius_px = (
        float(args.min_radius)
        if args.min_radius is not None
        else float(args.min_diameter) * 0.5
    )
    max_radius_px = (
        float(args.max_radius)
        if args.max_radius is not None
        else float(args.max_diameter) * 0.5
    )

    effective_seam_band = max(
        float(args.seam_band),
        2.0 * float(max_radius_px) + float(args.merge_distance),
    )

    prefiltered = prefilter_circles_keep_seam_parts(
        raw_circles,
        min_radius_px,
        max_radius_px,
        image_height,
        effective_seam_band,
    )

    circles = merge_circles(
        prefiltered,
        args.merge_distance,
        args.merge_center_ratio,
        args.merge_overlap_iou,
    )

    if (
        ENABLE_PERIODIC_Y_SEAM_MERGE
        and not args.no_periodic_seam_merge
    ):
        circles = merge_periodic_y_seam(
            circles,
            image_height,
            effective_seam_band,
            args.merge_distance,
            args.merge_center_ratio,
            args.merge_overlap_iou,
        )

    circles = filter_circles_by_radius(
        circles,
        min_radius_px,
        max_radius_px,
    )

    circles.sort(
        key=lambda item: (
            float(item["center_y"]) % float(image_height),
            float(item["center_x"]),
        )
    )

    attach_global_physical_coordinates(
        circles,
        image_width,
        image_height,
        args.pixel_size_mm,
    )

    instance_map, conflict_pixels = build_instance_map(
        img.shape,
        circles,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = args.out_dir / "wire_full_circle_overlay.png"
    draw_full_result(
        img,
        circles,
        overlay_path,
        args.line_width,
        False,
    )

    debug_overlay_path = (
        args.out_dir / "wire_full_circle_and_contour_debug.png"
    )
    if args.draw_contours:
        draw_full_result(
            img,
            circles,
            debug_overlay_path,
            args.line_width,
            True,
        )

    positions_csv = args.out_dir / "wire_full_predictions.csv"
    summary_csv = args.out_dir / "wire_summary.csv"

    save_csv(positions_csv, circles)
    save_summary_csv(summary_csv, circles)

    instance_npz_path = args.out_dir / INSTANCE_MAP_FILENAME
    if SAVE_INSTANCE_MAP_NPZ:
        save_instance_map_npz(
            instance_npz_path,
            instance_map,
            circles,
            args.image,
            args.pixel_size_mm,
        )

    instance_id_png_path = (
        args.out_dir / "wire_instance_id_map_uint16.png"
    )
    if SAVE_INSTANCE_ID_PNG:
        if len(circles) <= np.iinfo(np.uint16).max:
            cv2.imwrite(
                str(instance_id_png_path),
                instance_map.astype(np.uint16),
            )
        else:
            print(
                "WARNING: wire count > 65535, "
                "skipping uint16 ID-map PNG. NPZ remains authoritative."
            )

    instance_preview_path = (
        args.out_dir / "wire_instance_map_preview.png"
    )
    if SAVE_INSTANCE_PREVIEW:
        save_instance_preview(
            img,
            instance_map,
            instance_preview_path,
        )

    contours_json_path = (
        args.out_dir / "wire_contours_global.json"
    )
    if SAVE_GLOBAL_CONTOURS_JSON:
        save_global_contours_json(
            contours_json_path,
            circles,
            args.image,
            image_width,
            image_height,
            args.pixel_size_mm,
        )

    tile_manifest_path = (
        args.out_dir / "tiles_manifest.csv"
    )
    if SAVE_TILE_MANIFEST:
        save_tile_manifest(
            tile_manifest_path,
            tile_records,
        )

    metadata_path = (
        args.out_dir / "wire_instance_map_metadata.json"
    )
    save_metadata_json(
        metadata_path,
        args,
        args.image,
        image_width,
        image_height,
        circles,
        conflict_pixels,
    )

    seam_wire_count = sum(
        1
        for circle in circles
        if circle.get("seam_merged", False)
    )

    print("")
    print("Done")
    print(f"Image: {args.image}")
    print(
        f"Full stitched-image size: "
        f"W={image_width}, H={image_height}"
    )
    print(f"Tiles: {tile_count}")
    print(f"Raw instances: {len(raw_circles)}")
    print(
        f"Prefiltered instances "
        f"(including temporary seam parts): {len(prefiltered)}"
    )
    print(f"Final merged instances: {len(circles)}")
    print(f"Wire count: {len(circles)}")
    print(f"Periodic seam-merged wires: {seam_wire_count}")
    print(f"Instance-map overlap conflict pixels: {conflict_pixels}")

    print("")
    print("Coordinate system:")
    print("  center_x / center_y = GLOBAL stitched-image pixels")
    print(
        "  global_x = local_x + tile_x0, "
        "global_y = local_y + tile_y0"
    )
    print(
        f"  camera X=0 physical origin = "
        f"{(image_width - 1) / 2.0:.3f} px"
    )
    print(
        "  X_mm = "
        "(global_x - camera_center_x_px) * pixel_size_mm"
    )
    print(
        "  theta_deg = "
        "global_y / image_height * 360"
    )
    print("  Y is periodic: row 0 <-> row H")

    print("")
    print(
        f"Confidence: center={args.conf:.2f}, "
        f"edge={args.edge_conf:.2f}, "
        f"edge margin={edge_margin:.1f}px"
    )
    print(
        f"Circle filters: "
        f"diameter={min_radius_px * 2.0:.1f}-"
        f"{max_radius_px * 2.0:.1f}px, "
        f"radius={min_radius_px:.1f}-{max_radius_px:.1f}px, "
        f"merge distance={args.merge_distance:.1f}px, "
        f"merge center ratio={args.merge_center_ratio:.2f}, "
        f"merge IoU={args.merge_overlap_iou:.2f}"
    )
    print(
        f"Periodic seam band: "
        f"{effective_seam_band:.1f}px"
    )
    print(
        f"Pixel size: "
        f"{args.pixel_size_mm:.8f} mm/px"
    )

    print("")
    print("Outputs:")
    print(f"  Overlay: {overlay_path}")
    if args.draw_contours:
        print(f"  Debug overlay: {debug_overlay_path}")
    print(f"  Positions CSV: {positions_csv}")
    print(f"  Summary CSV: {summary_csv}")

    if SAVE_INSTANCE_MAP_NPZ:
        print(f"  Instance map NPZ: {instance_npz_path}")
        print(
            f"    shape = {instance_map.shape} "
            "(EXACT full camera stitched-image grid)"
        )
        print("    0=background, 1..N=wire_id")

    if SAVE_INSTANCE_ID_PNG and len(circles) <= np.iinfo(np.uint16).max:
        print(f"  Instance ID PNG: {instance_id_png_path}")

    if SAVE_INSTANCE_PREVIEW:
        print(f"  Instance preview: {instance_preview_path}")

    if SAVE_GLOBAL_CONTOURS_JSON:
        print(f"  Global contours JSON: {contours_json_path}")

    if SAVE_TILE_MANIFEST:
        print(f"  Tile manifest: {tile_manifest_path}")

    print(f"  Metadata JSON: {metadata_path}")

    print("")
    print("Direct matching with mapped_height_mm.npz:")
    print("  wire_mask = (instance_map == wire_id)")
    print("  valid = mapped_height_valid_mask.astype(bool)")
    print("  wire_laser_mask = wire_mask & valid")
    print("  wire_heights_mm = mapped_height_mm[wire_laser_mask]")


if __name__ == "__main__":
    main()
