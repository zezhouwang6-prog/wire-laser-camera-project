from __future__ import annotations

import argparse
import json
import math
import sys


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


configure_console_encoding()
from datetime import datetime
from pathlib import Path
from config import CAMERA_CALIBRATION_DIR
from typing import Any

import cv2
import numpy as np

# 自动标定比例保存变量
AUTO_PX_PER_MM_RESULT = None
AUTO_MM_PER_PX_RESULT = None


# =============================================================================
# Configuration: these values must match the physical calibration board.
# =============================================================================
EXPECTED_CAMERA_MODEL = "acA2440-20gc"
CAMERA_SERIAL_NUMBER = "23680766"

SQUARES_X = 11
SQUARES_Y = 9
SQUARE_LENGTH_MM = 8.0
MARKER_LENGTH_MM = 5.6
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_100

CAMERA_WIDTH = 2448
CAMERA_HEIGHT = 2048
PREVIEW_WIDTH = 1224
PREVIEW_HEIGHT = 1024
INITIAL_EXPOSURE_US = 30000.0

MIN_CAPTURE_CORNERS = 12
MIN_CAPTURE_SHARPNESS = 20.0
MIN_INTRINSIC_IMAGES = 12
MIN_PLANE_CORNERS = 12
CALIBRATION_ROOT = CAMERA_CALIBRATION_DIR

INTRINSIC_IMAGES_DIR = CALIBRATION_ROOT / "intrinsic_images"
PLANE_IMAGES_DIR = CALIBRATION_ROOT / "plane_images"
MOTOR_EXTRINSIC_IMAGES_DIR = CALIBRATION_ROOT / "motor_extrinsic_images"
RESULTS_DIR = CALIBRATION_ROOT / "results"
DETECTION_DIR = RESULTS_DIR / "detections"
UNDISTORTED_DIR = RESULTS_DIR / "undistorted"

INTRINSIC_NPZ = RESULTS_DIR / "intrinsic_calibration.npz"
PLANE_NPZ = RESULTS_DIR / "metric_plane_calibration.npz"
MOTOR_EXTRINSIC_NPZ = RESULTS_DIR / "camera_to_motor_extrinsic.npz"
INTRINSIC_JSON = RESULTS_DIR / "intrinsic_calibration.json"
PLANE_JSON = RESULTS_DIR / "metric_plane_calibration.json"
MOTOR_EXTRINSIC_JSON = RESULTS_DIR / "camera_to_motor_extrinsic.json"
MOTOR_BOARD_POSE_JSON = RESULTS_DIR / "motor_extrinsic_board_pose.json"
CALIBRATION_YAML = RESULTS_DIR / "calibration_aca2440_20gc.yaml"

PREVIEW_WINDOW = "Basler acA2440-20gc Calibration"


def ensure_directories() -> None:
    for directory in (
        INTRINSIC_IMAGES_DIR,
        PLANE_IMAGES_DIR,
        MOTOR_EXTRINSIC_IMAGES_DIR,
        RESULTS_DIR,
        DETECTION_DIR,
        UNDISTORTED_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def board_description() -> dict[str, Any]:
    return {
        "type": "ChArUco",
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_mm": SQUARE_LENGTH_MM,
        "marker_length_mm": MARKER_LENGTH_MM,
        "dictionary": "DICT_4X4_100",
        "pattern_width_mm": SQUARES_X * SQUARE_LENGTH_MM,
        "pattern_height_mm": SQUARES_Y * SQUARE_LENGTH_MM,
    }


def create_board_and_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        float(SQUARE_LENGTH_MM),
        float(MARKER_LENGTH_MM),
        dictionary,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


def detect_charuco(image: np.ndarray, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    count = 0 if charuco_ids is None else int(len(charuco_ids))
    return charuco_corners, charuco_ids, marker_corners, marker_ids, count


def measure_board_sharpness(
    image: np.ndarray,
    charuco_corners,
) -> float:
    if charuco_corners is None or len(charuco_corners) < 4:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)
    x, y, width, height = cv2.boundingRect(points)
    x0 = max(0, x - 4)
    y0 = max(0, y - 4)
    x1 = min(gray.shape[1], x + width + 4)
    y1 = min(gray.shape[0], y + height + 4)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0

    local_points = points - np.array([x0, y0], dtype=np.float32)
    hull = cv2.convexHull(local_points).astype(np.int32)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    laplacian = cv2.Laplacian(crop, cv2.CV_64F)
    values = laplacian[mask > 0]
    return float(values.var()) if values.size else 0.0


def draw_detection(
    image: np.ndarray,
    charuco_corners,
    charuco_ids,
    marker_corners,
    marker_ids,
) -> np.ndarray:
    display = image.copy()
    if marker_ids is not None and len(marker_ids):
        cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
    if charuco_ids is not None and len(charuco_ids):
        cv2.aruco.drawDetectedCornersCharuco(
            display, charuco_corners, charuco_ids, (0, 0, 255)
        )
    return display


def save_json(path: Path, data: dict[str, Any]) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value)!r}")

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=convert),
        encoding="utf-8",
    )


def next_image_path(directory: Path, prefix: str) -> Path:
    existing = sorted(directory.glob(f"{prefix}_*.png"))
    used_numbers = []
    for path in existing:
        try:
            used_numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    next_number = max(used_numbers, default=0) + 1
    return directory / f"{prefix}_{next_number:04d}.png"


def set_enum_if_available(camera, node_name: str, value: str) -> bool:
    try:
        node = getattr(camera, node_name)
        if value in list(node.Symbolics):
            node.SetValue(value)
            return True
    except Exception:
        pass
    return False


def set_bool_if_available(camera, node_name: str, value: bool) -> bool:
    try:
        getattr(camera, node_name).SetValue(bool(value))
        return True
    except Exception:
        return False


def set_integer_node(camera, node_name: str, desired: int) -> int | None:
    try:
        node = getattr(camera, node_name)
        minimum = int(node.GetMin())
        maximum = int(node.GetMax())
        increment = max(1, int(node.GetInc()))
        value = max(minimum, min(maximum, int(desired)))
        value = minimum + ((value - minimum) // increment) * increment
        node.SetValue(value)
        return int(value)
    except Exception:
        return None


def set_float_node(camera, node_names: tuple[str, ...], desired: float) -> float:
    for node_name in node_names:
        try:
            node = getattr(camera, node_name)
            value = max(float(node.GetMin()), min(float(node.GetMax()), float(desired)))
            node.SetValue(value)
            return float(node.GetValue())
        except Exception:
            pass
    raise RuntimeError(f"Camera has none of these nodes: {node_names}")


def get_float_node(camera, node_names: tuple[str, ...], fallback: float) -> float:
    for node_name in node_names:
        try:
            return float(getattr(camera, node_name).GetValue())
        except Exception:
            pass
    return float(fallback)


def open_basler_camera():
    try:
        from pypylon import pylon
    except ImportError as exc:
        raise RuntimeError(
            "没有安装 pypylon。请运行: pip install pypylon"
        ) from exc

    factory = pylon.TlFactory.GetInstance()
    devices = list(factory.EnumerateDevices())
    if not devices:
        raise RuntimeError("没有找到 Basler 相机，请检查电源、网线和 pylon Viewer。")

    print("\n发现的 Basler 相机:")
    for index, device in enumerate(devices):
        print(
            f"  [{index}] {device.GetModelName()}  "
            f"S/N={device.GetSerialNumber()}"
        )

    selected = None
    for device in devices:
        if str(device.GetSerialNumber()) == CAMERA_SERIAL_NUMBER:
            selected = device
            break
    if selected is None:
        if len(devices) == 1:
            selected = devices[0]
            print(
                f"警告: 未找到序列号 {CAMERA_SERIAL_NUMBER}，"
                "将使用唯一检测到的相机。"
            )
        else:
            raise RuntimeError(
                f"没有找到序列号 {CAMERA_SERIAL_NUMBER}。"
                "请修改文件顶部 CAMERA_SERIAL_NUMBER。"
            )

    camera = pylon.InstantCamera(factory.CreateDevice(selected))
    camera.Open()

    model = str(camera.GetDeviceInfo().GetModelName())
    serial = str(camera.GetDeviceInfo().GetSerialNumber())
    if EXPECTED_CAMERA_MODEL.lower() not in model.lower():
        print(f"警告: 当前型号是 {model}，预期型号是 {EXPECTED_CAMERA_MODEL}。")

    if camera.IsGrabbing():
        camera.StopGrabbing()

    set_enum_if_available(camera, "TriggerMode", "Off")
    set_enum_if_available(camera, "AcquisitionMode", "Continuous")
    set_enum_if_available(camera, "ExposureAuto", "Off")
    set_enum_if_available(camera, "GainAuto", "Off")
    set_enum_if_available(camera, "BalanceWhiteAuto", "Off")

    try:
        gain_node = getattr(camera, "Gain")
        gain_node.SetValue(float(gain_node.GetMin()))
    except Exception:
        try:
            gain_node = getattr(camera, "GainRaw")
            gain_node.SetValue(int(gain_node.GetMin()))
        except Exception:
            pass

    set_bool_if_available(camera, "GammaEnable", False)

    set_bool_if_available(camera, "CenterX", False)
    set_bool_if_available(camera, "CenterY", False)
    for node_name in (
        "BinningHorizontal",
        "BinningVertical",
        "DecimationHorizontal",
        "DecimationVertical",
    ):
        set_integer_node(camera, node_name, 1)

    offset_x = set_integer_node(camera, "OffsetX", 0)
    offset_y = set_integer_node(camera, "OffsetY", 0)
    width = set_integer_node(camera, "Width", CAMERA_WIDTH)
    height = set_integer_node(camera, "Height", CAMERA_HEIGHT)
    if width != CAMERA_WIDTH or height != CAMERA_HEIGHT:
        raise RuntimeError(
            f"相机无法设置为完整画幅 {CAMERA_WIDTH}x{CAMERA_HEIGHT}，"
            f"当前为 {width}x{height}"
        )
    exposure = set_float_node(
        camera, ("ExposureTime", "ExposureTimeAbs"), INITIAL_EXPOSURE_US
    )

    metadata = {
        "model": model,
        "serial_number": serial,
        "width": width,
        "height": height,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "exposure_us": exposure,
        "expected_model": EXPECTED_CAMERA_MODEL,
    }
    print(
        f"已连接 {model}, S/N={serial}, "
        f"{width}x{height}, OffsetX={offset_x}, OffsetY={offset_y}, "
        f"exposure={exposure:.0f} us"
    )
    return camera, pylon, metadata


def capture_images(mode: str) -> None:
    ensure_directories()
    board, detector = create_board_and_detector()
    camera, pylon, metadata = open_basler_camera()

    if mode == "intrinsic":
        output_dir = INTRINSIC_IMAGES_DIR
        prefix = "intrinsic"
        stop_after_save = False
    elif mode == "plane":
        output_dir = PLANE_IMAGES_DIR
        prefix = "plane"
        stop_after_save = True
    elif mode == "motor":
        output_dir = MOTOR_EXTRINSIC_IMAGES_DIR
        prefix = "motor_extrinsic"
        stop_after_save = True
    else:
        raise ValueError(f"unknown capture mode: {mode}")

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    print("\n拍照按键:")
    print("  S 或空格: 保存当前照片")
    print("  U / J:    增加 / 减少曝光")
    print("  D:        删除本次拍摄的最后一张")
    print("  Q 或 ESC: 结束拍摄")
    print(
        f"正常保存要求角点不少于 {MIN_CAPTURE_CORNERS}，"
        f"清晰度不低于 {MIN_CAPTURE_SHARPNESS:.0f}；"
        "F 可强制保存。"
    )
    if mode == "intrinsic":
        print("建议拍20至30张，让标定板角点覆盖画面中心、四角和边缘。")
    else:
        print("请把标定板放在丝端面所在高度，保持相机正式工作位置。")

    captured_this_run: list[Path] = []
    captured_records: list[dict[str, Any]] = []
    cv2.namedWindow(
        PREVIEW_WINDOW,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )
    cv2.resizeWindow(PREVIEW_WINDOW, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_Return)
            if not grab.IsValid():
                continue
            try:
                if not grab.GrabSucceeded():
                    continue
                frame = converter.Convert(grab).GetArray().copy()
            finally:
                grab.Release()

            (
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
                count,
            ) = detect_charuco(frame, detector)
            sharpness = measure_board_sharpness(frame, charuco_corners)
            display = draw_detection(
                frame,
                charuco_corners,
                charuco_ids,
                marker_corners,
                marker_ids,
            )

            exposure = get_float_node(
                camera,
                ("ExposureTime", "ExposureTimeAbs"),
                INITIAL_EXPOSURE_US,
            )
            quality_ok = (
                count >= MIN_CAPTURE_CORNERS
                and sharpness >= MIN_CAPTURE_SHARPNESS
            )
            status_color = (0, 255, 0) if quality_ok else (0, 0, 255)
            cv2.putText(
                display,
                (
                    f"corners={count}  sharpness={sharpness:.1f}  "
                    f"saved={len(captured_this_run)}"
                ),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2,
            )
            cv2.putText(
                display,
                f"exposure={exposure:.0f} us | S/Space save | U/J exposure | Q quit",
                (20, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
            )
            cv2.imshow(PREVIEW_WINDOW, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("u"), ord("U")):
                new_value = set_float_node(
                    camera,
                    ("ExposureTime", "ExposureTimeAbs"),
                    exposure * 1.2,
                )
                print(f"曝光增加到 {new_value:.0f} us")
                continue
            if key in (ord("j"), ord("J")):
                new_value = set_float_node(
                    camera,
                    ("ExposureTime", "ExposureTimeAbs"),
                    max(50.0, exposure / 1.2),
                )
                print(f"曝光减少到 {new_value:.0f} us")
                continue
            if key in (ord("d"), ord("D")):
                if captured_this_run:
                    last_path = captured_this_run.pop()
                    captured_records.pop()
                    last_path.unlink(missing_ok=True)
                    print(f"已删除 {last_path.name}")
                continue

            force_save = key in (ord("f"), ord("F"))
            normal_save = key in (ord("s"), ord("S"), 32)
            if not (normal_save or force_save):
                continue
            if not quality_ok and not force_save:
                reasons = []
                if count < MIN_CAPTURE_CORNERS:
                    reasons.append(
                        f"角点 {count} < {MIN_CAPTURE_CORNERS}"
                    )
                if sharpness < MIN_CAPTURE_SHARPNESS:
                    reasons.append(
                        f"清晰度 {sharpness:.1f} < {MIN_CAPTURE_SHARPNESS:.1f}"
                    )
                print(f"当前图像未保存: {', '.join(reasons)}。")
                continue

            path = next_image_path(output_dir, prefix)
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"保存失败: {path}")
            captured_this_run.append(path)
            captured_records.append(
                {
                    "image": str(path),
                    "corners": count,
                    "sharpness": sharpness,
                    "exposure_us": exposure,
                    "forced": force_save,
                }
            )
            print(
                f"已保存 {path}，角点 {count}，"
                f"清晰度 {sharpness:.1f}，曝光 {exposure:.0f} us"
            )
            if stop_after_save:
                break
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    metadata.update(
        {
            "capture_mode": mode,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "images_saved_this_run": [str(path) for path in captured_this_run],
            "image_records": captured_records,
            "board": board_description(),
        }
    )
    save_json(output_dir / "capture_metadata.json", metadata)
    print(f"\n本次共保存 {len(captured_this_run)} 张，目录: {output_dir}")


def collect_calibration_points(image_paths: list[Path]):
    board, detector = create_board_and_detector()
    object_points = []
    image_points = []
    used_paths = []
    rejected = []
    image_size = None

    DETECTION_DIR.mkdir(parents=True, exist_ok=True)
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": str(path), "reason": "cannot read"})
            continue
        current_size = (image.shape[1], image.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected.append(
                {
                    "image": str(path),
                    "reason": f"size {current_size} differs from {image_size}",
                }
            )
            continue

        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
            count,
        ) = detect_charuco(image, detector)
        if count < MIN_CAPTURE_CORNERS:
            rejected.append(
                {
                    "image": str(path),
                    "reason": f"only {count} ChArUco corners",
                }
            )
            continue
        sharpness = measure_board_sharpness(image, charuco_corners)
        if sharpness < MIN_CAPTURE_SHARPNESS:
            rejected.append(
                {
                    "image": str(path),
                    "reason": (
                        f"sharpness {sharpness:.2f} below "
                        f"{MIN_CAPTURE_SHARPNESS:.2f}"
                    ),
                }
            )
            continue

        obj, img = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj is None or img is None or len(obj) < MIN_CAPTURE_CORNERS:
            rejected.append({"image": str(path), "reason": "point matching failed"})
            continue

        object_points.append(np.asarray(obj, dtype=np.float32))
        image_points.append(np.asarray(img, dtype=np.float32))
        used_paths.append(path)

        annotated = draw_detection(
            image,
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        )
        cv2.imwrite(str(DETECTION_DIR / f"{path.stem}_detected.jpg"), annotated)

    return object_points, image_points, used_paths, rejected, image_size


def compute_view_errors(
    object_points,
    image_points,
    rvecs,
    tvecs,
    camera_matrix,
    distortion,
) -> list[float]:
    errors = []
    for obj, img, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            obj, rvec, tvec, camera_matrix, distortion
        )
        difference = img.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))))
    return errors


def calibrate_intrinsics() -> None:
    ensure_directories()
    image_paths = sorted(INTRINSIC_IMAGES_DIR.glob("*.png"))
    if len(image_paths) < MIN_INTRINSIC_IMAGES:
        raise RuntimeError(
            f"内参照片只有 {len(image_paths)} 张，至少需要 "
            f"{MIN_INTRINSIC_IMAGES} 张，建议20至30张。"
        )

    (
        object_points,
        image_points,
        used_paths,
        rejected,
        image_size,
    ) = collect_calibration_points(image_paths)
    if len(used_paths) < MIN_INTRINSIC_IMAGES:
        raise RuntimeError(
            f"只有 {len(used_paths)} 张照片检测合格，"
            f"至少需要 {MIN_INTRINSIC_IMAGES} 张。"
        )

    print(f"\n正在使用 {len(used_paths)} 张图片计算 K 和 D...")
    (
        rms,
        camera_matrix,
        distortion,
        rvecs,
        tvecs,
    ) = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    view_errors = compute_view_errors(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )
    median_error = float(np.median(view_errors))
    rejection_limit = max(0.5, median_error * 2.5)
    keep_indices = [
        index for index, error in enumerate(view_errors) if error <= rejection_limit
    ]

    removed_as_outliers = []
    if (
        len(keep_indices) >= MIN_INTRINSIC_IMAGES
        and len(keep_indices) < len(used_paths)
    ):
        for index, error in enumerate(view_errors):
            if index not in keep_indices:
                removed_as_outliers.append(
                    {
                        "image": str(used_paths[index]),
                        "reason": f"reprojection error {error:.4f} px",
                    }
                )
        filtered_object = [object_points[i] for i in keep_indices]
        filtered_image = [image_points[i] for i in keep_indices]
        filtered_paths = [used_paths[i] for i in keep_indices]

        (
            rms,
            camera_matrix,
            distortion,
            rvecs,
            tvecs,
        ) = cv2.calibrateCamera(
            filtered_object,
            filtered_image,
            image_size,
            None,
            None,
        )
        object_points = filtered_object
        image_points = filtered_image
        used_paths = filtered_paths
        view_errors = compute_view_errors(
            object_points,
            image_points,
            rvecs,
            tvecs,
            camera_matrix,
            distortion,
        )

    new_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        image_size,
        1.0,
        image_size,
    )
    mean_error = float(np.mean(view_errors))
    max_error = float(np.max(view_errors))

    np.savez_compressed(
        INTRINSIC_NPZ,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        new_camera_matrix=new_camera_matrix,
        image_size=np.asarray(image_size, dtype=np.int32),
        valid_roi=np.asarray(valid_roi, dtype=np.int32),
        rms=np.asarray(rms, dtype=np.float64),
        mean_reprojection_error_px=np.asarray(mean_error, dtype=np.float64),
        max_reprojection_error_px=np.asarray(max_error, dtype=np.float64),
        square_length_mm=np.asarray(SQUARE_LENGTH_MM, dtype=np.float64),
        marker_length_mm=np.asarray(MARKER_LENGTH_MM, dtype=np.float64),
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera_model": EXPECTED_CAMERA_MODEL,
        "camera_serial_number": CAMERA_SERIAL_NUMBER,
        "image_size": list(image_size),
        "board": board_description(),
        "camera_matrix": camera_matrix,
        "distortion_coefficients": distortion,
        "new_camera_matrix": new_camera_matrix,
        "valid_roi": list(valid_roi),
        "rms": float(rms),
        "mean_reprojection_error_px": mean_error,
        "max_reprojection_error_px": max_error,
        "images_used": [str(path) for path in used_paths],
        "per_image_error_px": {
            path.name: error for path, error in zip(used_paths, view_errors)
        },
        "images_rejected": rejected + removed_as_outliers,
    }
    save_json(INTRINSIC_JSON, report)

    sample_count = min(5, len(used_paths))
    UNDISTORTED_DIR.mkdir(parents=True, exist_ok=True)
    for path in used_paths[:sample_count]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        undistorted = cv2.undistort(
            image,
            camera_matrix,
            distortion,
            None,
            new_camera_matrix,
        )
        cv2.imwrite(
            str(UNDISTORTED_DIR / f"{path.stem}_undistorted.png"),
            undistorted,
        )

    print("\n内参标定完成")
    print(f"RMS:          {float(rms):.4f} px")
    print(f"平均重投影误差: {mean_error:.4f} px")
    print(f"最大重投影误差: {max_error:.4f} px")
    print("\nK =")
    print(np.array2string(camera_matrix, precision=8, suppress_small=False))
    print("\nD =")
    print(np.array2string(distortion, precision=8, suppress_small=False))
    print(f"\n结果: {INTRINSIC_NPZ}")
    if mean_error > 0.3:
        print("警告: 平均误差偏大，建议补拍覆盖四角和边缘的清晰照片。")


def latest_plane_image() -> Path:
    paths = sorted(
        PLANE_IMAGES_DIR.glob("*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise RuntimeError("没有测量平面照片，请先选择菜单3拍摄。")
    return paths[-1]


def load_intrinsics():
    if not INTRINSIC_NPZ.exists():
        raise RuntimeError("找不到内参结果，请先选择菜单2计算 K 和 D。")
    data = np.load(INTRINSIC_NPZ)
    return (
        data["camera_matrix"].astype(np.float64),
        data["distortion_coefficients"].astype(np.float64),
        data["new_camera_matrix"].astype(np.float64),
        tuple(int(value) for value in data["image_size"].tolist()),
    )


def calibrate_metric_plane() -> None:
    ensure_directories()
    board, detector = create_board_and_detector()
    camera_matrix, distortion, new_camera_matrix, calibration_size = (
        load_intrinsics()
    )
    image_path = latest_plane_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取 {image_path}")
    image_size = (image.shape[1], image.shape[0])
    if image_size != calibration_size:
        raise RuntimeError(
            f"平面照片尺寸 {image_size} 与内参尺寸 {calibration_size} 不一致。"
        )

    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
        count,
    ) = detect_charuco(image, detector)
    if count < MIN_PLANE_CORNERS:
        raise RuntimeError(
            f"平面照片只检测到 {count} 个角点，"
            f"至少需要 {MIN_PLANE_CORNERS} 个。"
        )

    object_points, raw_image_points = board.matchImagePoints(
        charuco_corners, charuco_ids
    )
    object_xy_mm = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)[:, :2]
    raw_image_points = np.asarray(raw_image_points, dtype=np.float64).reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(
        raw_image_points,
        camera_matrix,
        distortion,
        P=new_camera_matrix,
    ).reshape(-1, 2)

    # ==========================================================
    # 自动计算真实像素比例 px/mm
    # 不再使用固定 TARGET_PX_PER_MM
    # ==========================================================
    pixel_distances = []
    mm_distances = []

    for i in range(len(object_xy_mm)):
        for j in range(i + 1, len(object_xy_mm)):
            dm = np.linalg.norm(
                object_xy_mm[i] - object_xy_mm[j]
            )
            dp = np.linalg.norm(
                undistorted_points[i] - undistorted_points[j]
            )

            # 只使用合理范围的点间距离，避免远距离误差放大
            if dm > 0.5 and dm < 20:
                mm_distances.append(dm)
                pixel_distances.append(dp)

    if len(mm_distances) == 0:
        raise RuntimeError("无法计算自动像素比例")

    px_per_mm = float(
        np.median(
            np.asarray(pixel_distances)
            /
            np.asarray(mm_distances)
        )
    )

    mm_per_px = 1.0 / px_per_mm

    global AUTO_PX_PER_MM_RESULT, AUTO_MM_PER_PX_RESULT
    AUTO_PX_PER_MM_RESULT = px_per_mm
    AUTO_MM_PER_PX_RESULT = mm_per_px

    print("\n自动计算真实比例:")
    print(f"px_per_mm = {px_per_mm:.6f}")
    print(f"mm_per_px = {mm_per_px:.8f}")


    h_image_to_plane_mm, inlier_mask = cv2.findHomography(
        undistorted_points,
        object_xy_mm,
        method=cv2.RANSAC,
        ransacReprojThreshold=0.15,
    )
    if h_image_to_plane_mm is None:
        raise RuntimeError("无法计算平面单应性矩阵 H。")

    predicted_mm = cv2.perspectiveTransform(
        undistorted_points.reshape(-1, 1, 2),
        h_image_to_plane_mm,
    ).reshape(-1, 2)
    errors_mm = np.linalg.norm(predicted_mm - object_xy_mm, axis=1)
    inliers = (
        inlier_mask.reshape(-1).astype(bool)
        if inlier_mask is not None
        else np.ones(len(errors_mm), dtype=bool)
    )
    inlier_errors = errors_mm[inliers]
    mean_error_mm = float(np.mean(inlier_errors))
    max_error_mm = float(np.max(inlier_errors))

    undistorted_image = cv2.undistort(
        image,
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
    )
    image_corners = np.float64(
        [
            [0, 0],
            [image_size[0] - 1, 0],
            [image_size[0] - 1, image_size[1] - 1],
            [0, image_size[1] - 1],
        ]
    ).reshape(-1, 1, 2)
    image_corners_mm = cv2.perspectiveTransform(
        image_corners, h_image_to_plane_mm
    ).reshape(-1, 2)

    min_xy_mm = np.floor(np.min(image_corners_mm, axis=0) * 1000.0) / 1000.0
    max_xy_mm = np.ceil(np.max(image_corners_mm, axis=0) * 1000.0) / 1000.0
    field_size_mm = max_xy_mm - min_xy_mm
    output_width = int(math.ceil(field_size_mm[0] * px_per_mm))
    output_height = int(math.ceil(field_size_mm[1] * px_per_mm))
    if (
        output_width <= 0
        or output_height <= 0
        or output_width > 20000
        or output_height > 20000
    ):
        raise RuntimeError(
            f"校正输出尺寸异常: {output_width}x{output_height}。"
            "请检查标定板尺寸配置和角点识别。"
        )

    plane_mm_to_metric_px = np.array(
        [
            [px_per_mm, 0.0, -min_xy_mm[0] * px_per_mm],
            [0.0, px_per_mm, -min_xy_mm[1] * px_per_mm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    h_image_to_metric_px = plane_mm_to_metric_px @ h_image_to_plane_mm
    rectified = cv2.warpPerspective(
        undistorted_image,
        h_image_to_metric_px,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    scale_bar_length_mm = 10.0
    scale_bar_length_px = int(round(scale_bar_length_mm * px_per_mm))
    bar_y = max(30, output_height - 40)
    bar_x = 30
    if bar_x + scale_bar_length_px < output_width:
        cv2.line(
            rectified,
            (bar_x, bar_y),
            (bar_x + scale_bar_length_px, bar_y),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            rectified,
            f"{scale_bar_length_mm:g} mm",
            (bar_x, bar_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    rectified_path = RESULTS_DIR / "metric_rectified_preview.png"
    undistorted_path = RESULTS_DIR / "plane_undistorted.png"
    detected_path = RESULTS_DIR / "plane_detected.png"
    cv2.imwrite(str(rectified_path), rectified)
    cv2.imwrite(str(undistorted_path), undistorted_image)
    cv2.imwrite(
        str(detected_path),
        draw_detection(
            image,
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ),
    )

    np.savez_compressed(
        PLANE_NPZ,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        new_camera_matrix=new_camera_matrix,
        image_size=np.asarray(image_size, dtype=np.int32),
        h_undistorted_image_to_plane_mm=h_image_to_plane_mm,
        h_undistorted_image_to_metric_px=h_image_to_metric_px,
        plane_mm_to_metric_px=plane_mm_to_metric_px,
        px_per_mm=np.asarray(px_per_mm, dtype=np.float64),
        mm_per_px=np.asarray(mm_per_px, dtype=np.float64),
        rectified_origin_mm=min_xy_mm,
        rectified_field_size_mm=field_size_mm,
        rectified_output_size_px=np.asarray(
            [output_width, output_height], dtype=np.int32
        ),
        mean_plane_error_mm=np.asarray(mean_error_mm, dtype=np.float64),
        max_plane_error_mm=np.asarray(max_error_mm, dtype=np.float64),
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera_model": EXPECTED_CAMERA_MODEL,
        "camera_serial_number": CAMERA_SERIAL_NUMBER,
        "source_image": str(image_path),
        "image_size": list(image_size),
        "board": board_description(),
        "detected_corners": count,
        "inlier_corners": int(np.count_nonzero(inliers)),
        "px_per_mm": px_per_mm,
        "mm_per_px": mm_per_px,
        "h_undistorted_image_to_plane_mm": h_image_to_plane_mm,
        "h_undistorted_image_to_metric_px": h_image_to_metric_px,
        "rectified_origin_mm": min_xy_mm,
        "rectified_field_size_mm": field_size_mm,
        "rectified_output_size_px": [output_width, output_height],
        "mean_plane_error_mm": mean_error_mm,
        "max_plane_error_mm": max_error_mm,
        "preview": str(rectified_path),
    }
    save_json(PLANE_JSON, report)
    save_opencv_yaml(
        camera_matrix,
        distortion,
        new_camera_matrix,
        h_image_to_plane_mm,
        h_image_to_metric_px,
        image_size,
        min_xy_mm,
        field_size_mm,
        output_width,
        output_height,
    )

    print("\n测量平面标定完成")
    print(f"检测角点:      {count}")
    print(f"有效角点:      {int(np.count_nonzero(inliers))}")
    print(f"平均平面误差:  {mean_error_mm:.5f} mm")
    print(f"最大平面误差:  {max_error_mm:.5f} mm")
    print(f"输出比例:      {px_per_mm:.4f} px/mm")
    print(f"实际比例:      {mm_per_px:.8f} mm/px")
    print(
        f"校正视野:      {field_size_mm[0]:.3f} x "
        f"{field_size_mm[1]:.3f} mm"
    )
    print("\nH (undistorted image pixel -> plane mm) =")
    print(np.array2string(h_image_to_plane_mm, precision=10))
    print(f"\n结果: {PLANE_NPZ}")
    print(f"预览: {rectified_path}")
    if mean_error_mm > 0.05:
        print("警告: 平面误差偏大，请确认板子尺寸、平整度和测量高度。")



def latest_motor_extrinsic_image() -> Path:
    paths = sorted(
        MOTOR_EXTRINSIC_IMAGES_DIR.glob("*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise RuntimeError(
            "No motor extrinsic image. Run capture-motor-extrinsic first."
        )
    return paths[-1]


def normalize_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError(f"{name} is zero")
    return vector / norm


def write_motor_board_pose_template() -> None:
    ensure_directories()
    if MOTOR_BOARD_POSE_JSON.exists():
        print(f"Board pose template already exists: {MOTOR_BOARD_POSE_JSON}")
        return
    template = {
        "description": (
            "Define where the ChArUco board is in the motor-axis coordinate "
            "system. Units are millimeters. board_origin_motor_mm is the "
            "motor-coordinate position of the board coordinate (0,0,0). "
            "board_x_axis_motor points along board +X. board_y_axis_motor "
            "points along board +Y. They do not need to include scale; the "
            "program normalizes them."
        ),
        "board_origin_motor_mm": [0.0, 75.0, 0.0],
        "board_x_axis_motor": [1.0, 0.0, 0.0],
        "board_y_axis_motor": [0.0, 0.0, 1.0],
        "motor_coordinate_note": {
            "X": "laser width direction, for example left/right across the wheel",
            "Y": "from motor axis center toward the observed wheel surface",
            "Z": "tangential/circumference direction at the front of the wheel"
        },
        "measure_this_before_calibration": [
            "Distance from motor axis origin to the board origin",
            "Which physical board edge is board +X",
            "Which physical board edge is board +Y",
            "Confirm the board is fixed and the camera is in final working position"
        ]
    }
    save_json(MOTOR_BOARD_POSE_JSON, template)
    print(f"Wrote board pose template: {MOTOR_BOARD_POSE_JSON}")
    print("Edit this JSON with your measured board pose, then run calibrate-motor-extrinsic.")


def load_motor_board_pose() -> dict[str, Any]:
    if not MOTOR_BOARD_POSE_JSON.exists():
        write_motor_board_pose_template()
        raise RuntimeError(
            f"Please edit {MOTOR_BOARD_POSE_JSON} with the measured board pose, "
            "then run calibrate-motor-extrinsic again."
        )
    data = json.loads(MOTOR_BOARD_POSE_JSON.read_text(encoding="utf-8"))
    origin = np.asarray(data["board_origin_motor_mm"], dtype=np.float64).reshape(3)
    axis_x = normalize_vector(data["board_x_axis_motor"], "board_x_axis_motor")
    axis_y_raw = normalize_vector(data["board_y_axis_motor"], "board_y_axis_motor")

    # Make a clean orthonormal board frame in motor coordinates.
    axis_y = axis_y_raw - axis_x * float(np.dot(axis_y_raw, axis_x))
    axis_y = normalize_vector(axis_y, "board_y_axis_motor orthogonal part")
    axis_z = np.cross(axis_x, axis_y)
    axis_z = normalize_vector(axis_z, "board normal")

    return {
        "raw": data,
        "origin": origin,
        "axis_x": axis_x,
        "axis_y": axis_y,
        "axis_z": axis_z,
        "R_board_to_motor": np.column_stack([axis_x, axis_y, axis_z]),
    }


def board_object_points_to_motor(object_points_board: np.ndarray, pose: dict[str, Any]) -> np.ndarray:
    board_points = np.asarray(object_points_board, dtype=np.float64).reshape(-1, 3)
    origin = pose["origin"]
    axis_x = pose["axis_x"]
    axis_y = pose["axis_y"]
    axis_z = pose["axis_z"]
    motor_points = (
        origin[None, :]
        + board_points[:, 0:1] * axis_x[None, :]
        + board_points[:, 1:2] * axis_y[None, :]
        + board_points[:, 2:3] * axis_z[None, :]
    )
    return motor_points.astype(np.float64)


def draw_projected_motor_axes(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvec_motor_to_camera: np.ndarray,
    tvec_motor_to_camera: np.ndarray,
    origin_motor: np.ndarray,
    axis_length_mm: float = 20.0,
) -> np.ndarray:
    points_motor = np.asarray(
        [
            origin_motor,
            origin_motor + np.array([axis_length_mm, 0.0, 0.0]),
            origin_motor + np.array([0.0, axis_length_mm, 0.0]),
            origin_motor + np.array([0.0, 0.0, axis_length_mm]),
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        points_motor,
        rvec_motor_to_camera,
        tvec_motor_to_camera,
        camera_matrix,
        distortion,
    )
    pts = projected.reshape(-1, 2).round().astype(int)
    out = image.copy()
    o = tuple(pts[0])
    cv2.circle(out, o, 5, (255, 255, 255), -1)
    cv2.line(out, o, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(out, o, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(out, o, tuple(pts[3]), (255, 0, 0), 3)
    cv2.putText(out, "X", tuple(pts[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, "Y", tuple(pts[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(out, "Z", tuple(pts[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return out


def calibrate_camera_to_motor() -> None:
    ensure_directories()
    board, detector = create_board_and_detector()
    camera_matrix, distortion, new_camera_matrix, calibration_size = load_intrinsics()
    pose = load_motor_board_pose()
    image_path = latest_motor_extrinsic_image()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read {image_path}")
    image_size = (image.shape[1], image.shape[0])
    if image_size != calibration_size:
        raise RuntimeError(
            f"Motor extrinsic image size {image_size} differs from intrinsic size {calibration_size}."
        )

    (
        charuco_corners,
        charuco_ids,
        marker_corners,
        marker_ids,
        count,
    ) = detect_charuco(image, detector)
    if count < MIN_PLANE_CORNERS:
        raise RuntimeError(
            f"Only detected {count} ChArUco corners; need at least {MIN_PLANE_CORNERS}."
        )

    object_points_board, image_points = board.matchImagePoints(
        charuco_corners, charuco_ids
    )
    if object_points_board is None or image_points is None:
        raise RuntimeError("Board point matching failed.")

    object_points_motor = board_object_points_to_motor(object_points_board, pose)
    image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points_motor,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=2.0,
        iterationsCount=200,
        confidence=0.999,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points_motor,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inlier_indices = np.arange(len(image_points), dtype=np.int32)
    else:
        inlier_indices = inliers.reshape(-1).astype(np.int32)

    if not ok:
        raise RuntimeError("solvePnP failed for camera-to-motor extrinsic.")

    if len(inlier_indices) >= 6:
        ok, rvec, tvec = cv2.solvePnP(
            object_points_motor[inlier_indices],
            image_points[inlier_indices],
            camera_matrix,
            distortion,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP refinement failed.")

    projected, _ = cv2.projectPoints(
        object_points_motor,
        rvec,
        tvec,
        camera_matrix,
        distortion,
    )
    error_px = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    mean_error_px = float(np.mean(error_px))
    max_error_px = float(np.max(error_px))

    R_motor_to_camera, _ = cv2.Rodrigues(rvec)
    t_motor_to_camera = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    R_camera_to_motor = R_motor_to_camera.T
    t_camera_to_motor = -R_camera_to_motor @ t_motor_to_camera
    camera_origin_motor_mm = t_camera_to_motor.reshape(3)

    detected_path = RESULTS_DIR / "motor_extrinsic_detected.png"
    axes_path = RESULTS_DIR / "motor_extrinsic_axes_preview.png"
    cv2.imwrite(
        str(detected_path),
        draw_detection(image, charuco_corners, charuco_ids, marker_corners, marker_ids),
    )
    cv2.imwrite(
        str(axes_path),
        draw_projected_motor_axes(
            image,
            camera_matrix,
            distortion,
            rvec,
            tvec,
            pose["origin"],
        ),
    )

    np.savez_compressed(
        MOTOR_EXTRINSIC_NPZ,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        new_camera_matrix=new_camera_matrix,
        image_size=np.asarray(image_size, dtype=np.int32),
        rvec_motor_to_camera=rvec,
        tvec_motor_to_camera=t_motor_to_camera,
        R_motor_to_camera=R_motor_to_camera,
        t_motor_to_camera=t_motor_to_camera,
        R_camera_to_motor=R_camera_to_motor,
        t_camera_to_motor=t_camera_to_motor,
        camera_origin_motor_mm=camera_origin_motor_mm,
        board_origin_motor_mm=pose["origin"],
        board_x_axis_motor=pose["axis_x"],
        board_y_axis_motor=pose["axis_y"],
        board_z_axis_motor=pose["axis_z"],
        mean_reprojection_error_px=np.asarray(mean_error_px, dtype=np.float64),
        max_reprojection_error_px=np.asarray(max_error_px, dtype=np.float64),
        object_points_motor=object_points_motor,
        image_points=image_points,
    )

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_image": str(image_path),
        "image_size": list(image_size),
        "board": board_description(),
        "board_pose_file": str(MOTOR_BOARD_POSE_JSON),
        "board_origin_motor_mm": pose["origin"],
        "board_x_axis_motor": pose["axis_x"],
        "board_y_axis_motor": pose["axis_y"],
        "board_z_axis_motor": pose["axis_z"],
        "detected_corners": count,
        "inlier_corners": int(len(inlier_indices)),
        "mean_reprojection_error_px": mean_error_px,
        "max_reprojection_error_px": max_error_px,
        "rvec_motor_to_camera": rvec,
        "tvec_motor_to_camera": t_motor_to_camera,
        "R_motor_to_camera": R_motor_to_camera,
        "t_motor_to_camera": t_motor_to_camera,
        "R_camera_to_motor": R_camera_to_motor,
        "t_camera_to_motor": t_camera_to_motor,
        "camera_origin_motor_mm": camera_origin_motor_mm,
        "detected_preview": str(detected_path),
        "axes_preview": str(axes_path),
    }
    save_json(MOTOR_EXTRINSIC_JSON, report)

    print("\nCamera-to-motor extrinsic calibration complete")
    print(f"Detected ChArUco corners: {count}")
    print(f"Inlier corners:          {int(len(inlier_indices))}")
    print(f"Mean reprojection error: {mean_error_px:.4f} px")
    print(f"Max reprojection error:  {max_error_px:.4f} px")
    print("\nCamera origin in motor coordinates [mm] =")
    print(np.array2string(camera_origin_motor_mm, precision=6))
    print("\nR_camera_to_motor =")
    print(np.array2string(R_camera_to_motor, precision=8, suppress_small=False))
    print("\nt_camera_to_motor [mm] =")
    print(np.array2string(t_camera_to_motor, precision=8, suppress_small=False))
    print(f"\nResult: {MOTOR_EXTRINSIC_NPZ}")
    print(f"JSON:   {MOTOR_EXTRINSIC_JSON}")
    print(f"Axes preview: {axes_path}")
    if mean_error_px > 1.0:
        print("WARNING: reprojection error is large. Check board pose measurements and corner detection.")

def save_opencv_yaml(
    camera_matrix,
    distortion,
    new_camera_matrix,
    h_image_to_plane_mm,
    h_image_to_metric_px,
    image_size,
    origin_mm,
    field_size_mm,
    output_width,
    output_height,
) -> None:
    storage = cv2.FileStorage(str(CALIBRATION_YAML), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise RuntimeError(f"无法写入 {CALIBRATION_YAML}")
    try:
        storage.write("camera_model", EXPECTED_CAMERA_MODEL)
        storage.write("camera_serial_number", CAMERA_SERIAL_NUMBER)
        storage.write("image_width", int(image_size[0]))
        storage.write("image_height", int(image_size[1]))
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coefficients", distortion)
        storage.write("new_camera_matrix", new_camera_matrix)
        storage.write(
            "h_undistorted_image_to_plane_mm", h_image_to_plane_mm
        )
        storage.write(
            "h_undistorted_image_to_metric_px", h_image_to_metric_px
        )
        storage.write("px_per_mm", float(AUTO_PX_PER_MM_RESULT))
        storage.write("mm_per_px", float(AUTO_MM_PER_PX_RESULT))
        storage.write("rectified_origin_mm", np.asarray(origin_mm))
        storage.write("rectified_field_size_mm", np.asarray(field_size_mm))
        storage.write("rectified_output_width", int(output_width))
        storage.write("rectified_output_height", int(output_height))
    finally:
        storage.release()


def show_status() -> None:
    ensure_directories()
    intrinsic_count = len(list(INTRINSIC_IMAGES_DIR.glob("*.png")))
    plane_count = len(list(PLANE_IMAGES_DIR.glob("*.png")))
    motor_count = len(list(MOTOR_EXTRINSIC_IMAGES_DIR.glob("*.png")))

    print("\n当前状态")
    print("-" * 64)
    print(f"标定总目录:             {CALIBRATION_ROOT}")
    print(f"1 内参照片数量:          {intrinsic_count} 张")
    print(f"2 内参结果 K/D:          {'已有' if INTRINSIC_NPZ.exists() else '没有'}")
    print(f"3 平面照片数量:          {plane_count} 张")
    print(f"3 平面结果 H/px_per_mm:  {'已有' if PLANE_NPZ.exists() else '没有'}")
    print(f"4 电机外参模板:          {'已有' if MOTOR_BOARD_POSE_JSON.exists() else '没有'}")
    print(f"5 电机外参照片数量:      {motor_count} 张")
    print(f"5 相机到电机外参:        {'已有' if MOTOR_EXTRINSIC_JSON.exists() else '没有'}")
    print(f"综合 YAML:              {'已有' if CALIBRATION_YAML.exists() else '没有'}")
    print(f"棋盘格参数:             {SQUARES_X}x{SQUARES_Y}, "
          f"square={SQUARE_LENGTH_MM} mm, marker={MARKER_LENGTH_MM} mm")


def _count_pngs(folder: Path) -> int:
    return len(list(folder.glob("*.png")))


def run_plane_workflow() -> None:
    """Capture one plane image, then calculate the metric plane result."""
    ensure_directories()
    before_count = _count_pngs(PLANE_IMAGES_DIR)
    print("\n第 3 步：平面标定")
    print("请把标定板放到你想当作毫米平面的那个位置。")
    print("程序会先拍 1 张图，然后立刻计算像素和毫米的关系。")
    capture_images("plane")
    after_count = _count_pngs(PLANE_IMAGES_DIR)
    if after_count <= before_count:
        print("\n没有检测到新的平面照片，所以先不计算。")
        print("请重新按 3，看到画面后按 SPACE 或 ENTER 保存。")
        return
    calibrate_metric_plane()


def prepare_motor_extrinsic_pose() -> None:
    """Create the JSON file that tells the program where the board is in motor coordinates."""
    ensure_directories()
    write_motor_board_pose_template()
    print("\n第 4 步完成：已经准备好电机坐标模板。")
    print("你现在要做一件事：打开下面这个文件，把里面的数字改成真实测量值。")
    print(f"文件位置: {MOTOR_BOARD_POSE_JSON}")
    print("\n最重要的是这 4 个东西：")
    print("1. origin_motor_mm：标定板左上角内角点，在电机坐标系里的 X/Y/Z 毫米坐标。")
    print("2. x_axis_motor：标定板横向往右，在电机坐标系里朝哪个方向。")
    print("3. y_axis_motor：标定板纵向往下，在电机坐标系里朝哪个方向。")
    print("4. square_length_mm：棋盘格一个小方格的真实边长。")
    print("\n改完保存，再回到这个菜单按 5。")


def run_motor_extrinsic_workflow() -> None:
    """Capture one motor-extrinsic image, then calculate camera-to-motor transform."""
    ensure_directories()
    if not MOTOR_BOARD_POSE_JSON.exists():
        print("\n还没有电机坐标模板。先自动生成一个模板。")
        prepare_motor_extrinsic_pose()
        print("\n这次先不计算。你把模板数字改好以后，再按 5。")
        return

    print("\n第 5 步：相机外参 -> 电机坐标系")
    print("请确认：标定板没有动，电机/轮/相机也没有动。")
    print("程序会先拍 1 张图，然后用第 4 步的模板计算相机到电机坐标系的关系。")
    before_count = _count_pngs(MOTOR_EXTRINSIC_IMAGES_DIR)
    capture_images("motor")
    after_count = _count_pngs(MOTOR_EXTRINSIC_IMAGES_DIR)
    if after_count <= before_count:
        print("\n没有检测到新的电机外参照片，所以先不计算。")
        print("请重新按 5，看到画面后按 SPACE 或 ENTER 保存。")
        return
    calibrate_camera_to_motor()


def interactive_menu() -> None:
    ensure_directories()
    actions = {
        "1": ("拍摄相机内参照片", lambda: capture_images("intrinsic")),
        "2": ("计算相机内参 K 和畸变 D", calibrate_intrinsics),
        "3": ("平面标定：拍 1 张图并计算像素/毫米", run_plane_workflow),
        "4": ("准备相机外参到电机坐标系的模板", prepare_motor_extrinsic_pose),
        "5": ("拍照并计算相机外参到电机坐标系", run_motor_extrinsic_workflow),
    }

    while True:
        print("\n" + "=" * 70)
        print("标定功能菜单：只按数字就可以")
        print("=" * 70)
        print("建议顺序：1 -> 2 -> 3 -> 4 -> 5")
        print("-" * 70)
        for key, (name, _) in actions.items():
            print(f"  {key}. {name}")
        print("  s. 查看当前状态")
        print("  0. 退出")

        choice = input("\n请输入数字，然后按回车: ").strip().lower()
        if choice == "0":
            print("已退出。")
            return
        if choice == "s":
            show_status()
            continue
        if choice not in actions:
            print("这个数字没有对应功能。请按 1、2、3、4、5、s 或 0。")
            continue

        name, action = actions[choice]
        print("\n" + "-" * 70)
        print(f"开始执行：{choice}. {name}")
        print("-" * 70)
        try:
            action()
        except Exception as exc:
            print(f"\n操作失败: {exc}")
            print("详细错误如下：")
            import traceback
            traceback.print_exc()
        finally:
            input("\n本步结束。按回车回到菜单...")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Basler acA2440-20gc ChArUco calibration tool"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "capture-intrinsic",
            "calibrate-intrinsic",
            "capture-plane",
            "calibrate-plane",
            "init-motor-board-pose",
            "capture-motor-extrinsic",
            "calibrate-motor-extrinsic",
            "status",
        ),
        help="Omit this argument to use the interactive menu.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()
    commands = {
        "capture-intrinsic": lambda: capture_images("intrinsic"),
        "calibrate-intrinsic": calibrate_intrinsics,
        "capture-plane": lambda: capture_images("plane"),
        "calibrate-plane": calibrate_metric_plane,
        "init-motor-board-pose": write_motor_board_pose_template,
        "capture-motor-extrinsic": lambda: capture_images("motor"),
        "calibrate-motor-extrinsic": calibrate_camera_to_motor,
        "status": show_status,
    }
    try:
        if args.command is None:
            interactive_menu()
        else:
            commands[args.command]()
        return 0
    except KeyboardInterrupt:
        print("\n用户取消。")
        return 130
    except Exception as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())