from pathlib import Path

# ============================================================
# Main project paths
# ============================================================
PROJECT_ROOT = Path(r"D:\project")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RUNS_DIR = PROJECT_ROOT / "runs"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"
DOCS_DIR = PROJECT_ROOT / "docs"

# ============================================================
# Change this name for each experiment
# ============================================================
# One experiment folder. Example: test1, test2, K-320X1.0
RUN_NAME = "test1"

# Use the combined workflow output by default:
#   motor_laser_camera_sync.py -> D:\project\runs\testX\laser_then_camera\run1, run2, ...
# Leave COMBINED_RUN_NAME empty to automatically use the latest run folder.
# If you want to process an older run, set it manually, for example: "run1".
USE_COMBINED_ACQUISITION = True
COMBINED_RUN_NAME = "run34" 

# These two are only used when USE_COMBINED_ACQUISITION = False.
# motor_camera.py creates record1, record2, ... under CAMERA_RAW_ROOT.
# motor_laser_sync.py creates test1, test2, ... under LASER_RAW_ROOT.
CAMERA_RECORD_NAME = "record1"
LASER_RECORD_NAME = "test1"

RUN_DIR = RUNS_DIR / RUN_NAME
COMBINED_ACQUISITION_ROOT = RUN_DIR / "laser_then_camera"


def _latest_numbered_dir(root: Path, prefix: str) -> Path:
    """Return latest prefixN folder, or prefix1 if none exists yet."""
    max_index = 0
    if root.exists():
        for item in root.iterdir():
            if not item.is_dir():
                continue
            name = item.name.lower()
            if not name.startswith(prefix.lower()):
                continue
            suffix = item.name[len(prefix):]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return root / f"{prefix}{max_index if max_index > 0 else 1}"


def _combined_run_dir() -> Path:
    if COMBINED_RUN_NAME.strip():
        return COMBINED_ACQUISITION_ROOT / COMBINED_RUN_NAME.strip()
    return _latest_numbered_dir(COMBINED_ACQUISITION_ROOT, "run")


ACTIVE_COMBINED_RUN_DIR = _combined_run_dir()

# ============================================================
# Camera acquisition and processing paths
# ============================================================
CAMERA_RAW_ROOT = RUN_DIR / "camera_raw"
CAMERA_RECORD_DIR = CAMERA_RAW_ROOT / CAMERA_RECORD_NAME

if USE_COMBINED_ACQUISITION:
    CAMERA_IMAGES_DIR = ACTIVE_COMBINED_RUN_DIR / "camera" / "images"
    CAMERA_INFO_CSV = ACTIVE_COMBINED_RUN_DIR / "camera" / "capture_info.csv"
    CAMERA_MOTOR_FEEDBACK_CSV = ACTIVE_COMBINED_RUN_DIR / "camera" / "camera_motor_feedback.csv"
    CAMERA_UNDISTORTED_DIR = ACTIVE_COMBINED_RUN_DIR / "camera" / "undistorted_images"
    CAMERA_STITCHED_DIR = ACTIVE_COMBINED_RUN_DIR / "camera" / "stitched"
else:
    CAMERA_IMAGES_DIR = CAMERA_RECORD_DIR / "images"
    CAMERA_INFO_CSV = CAMERA_RECORD_DIR / "capture_info.csv"
    CAMERA_MOTOR_FEEDBACK_CSV = CAMERA_RECORD_DIR / "camera_motor_feedback.csv"
    CAMERA_UNDISTORTED_DIR = RUN_DIR / "camera_undistorted"
    CAMERA_STITCHED_DIR = RUN_DIR / "camera_stitched"

CAMERA_STITCHED_IMAGE = CAMERA_STITCHED_DIR / "fine_wire_angle_unwrapped_stitched_laser_height.png"

# ============================================================
# Laser acquisition and restoration paths
# ============================================================
LASER_RAW_ROOT = RUN_DIR / "laser_raw"

if USE_COMBINED_ACQUISITION:
    LASER_RECORD_DIR = ACTIVE_COMBINED_RUN_DIR / "laser"
    LASER_RESTORED_DIR = ACTIVE_COMBINED_RUN_DIR / "laser" / "huanyuan_result"
else:
    LASER_RECORD_DIR = LASER_RAW_ROOT / LASER_RECORD_NAME
    LASER_RESTORED_DIR = RUN_DIR / "laser_restored"

LASER_HEIGHT_CSV = LASER_RECORD_DIR / "laser_height_mm.csv"
LASER_SYNC_CSV = LASER_RECORD_DIR / "sync_index.csv"
LASER_MOTOR_FEEDBACK_CSV = LASER_RECORD_DIR / "motor_feedback.csv"
LASER_RESTORED_IMAGE = LASER_RESTORED_DIR / "laser_height_mm_true_scale_detail_height.png"

# ============================================================
# Surface-image fusion inputs
# ============================================================
# Camera surface texture image after angle unwrapping and stitching.
SURFACE_FUSION_CAMERA_IMAGE = CAMERA_STITCHED_IMAGE

# Laser surface-detail visualization at the unique physical scale:
# 1 px = 0.025 mm.
SURFACE_FUSION_LASER_IMAGE = LASER_RESTORED_IMAGE

# The only physical image scale used for surface fusion.
SURFACE_FUSION_MM_PER_PIXEL = 0.05353937

# ============================================================
# Downstream analysis paths
# ============================================================
if USE_COMBINED_ACQUISITION:
    SEGMENTATION_DIR = ACTIVE_COMBINED_RUN_DIR / "segmentation"
    FUSION_DIR = ACTIVE_COMBINED_RUN_DIR / "fusion"
else:
    SEGMENTATION_DIR = RUN_DIR / "segmentation"
    FUSION_DIR = RUN_DIR / "fusion"

# YOLO segmentation model path used by fenge.py.
YOLO_MODEL_PATH = PROJECT_ROOT / "model" / "segment" / "train_mixed_v3" / "weights" / "best.pt"

# ============================================================
# Wire - laser correspondence analysis paths
# ============================================================
WIRE_INSTANCE_MAP_NPZ = SEGMENTATION_DIR / "wire_instance_map.npz"
WIRE_PREDICTIONS_CSV = SEGMENTATION_DIR / "cusi_full_predictions.csv"
MAPPED_HEIGHT_MM_NPZ = FUSION_DIR / "mapped_height_mm.npz"

if USE_COMBINED_ACQUISITION:
    WIRE_LASER_ANALYSIS_DIR = ACTIVE_COMBINED_RUN_DIR / "wire_laser_analysis"
else:
    WIRE_LASER_ANALYSIS_DIR = RUN_DIR / "wire_laser_analysis"

WIRE_LASER_CORRESPONDENCE_CSV = (
    WIRE_LASER_ANALYSIS_DIR / "wire_laser_correspondence.csv"
)
WIRE_LASER_CORRESPONDENCE_NPZ = (
    WIRE_LASER_ANALYSIS_DIR / "wire_laser_correspondence.npz"
)
WIRE_LASER_SUMMARY_JSON = (
    WIRE_LASER_ANALYSIS_DIR / "wire_laser_summary.json"
)
WIRE_LASER_SUMMARY_TXT = (
    WIRE_LASER_ANALYSIS_DIR / "wire_laser_summary.txt"
)
WIRE_LASER_READY_OVERLAY = (
    WIRE_LASER_ANALYSIS_DIR / "wire_laser_ready_overlay.png"
)
WIRE_LASER_COVERAGE_HISTOGRAM = (
    WIRE_LASER_ANALYSIS_DIR / "wire_laser_coverage_histogram.png"
)

# ============================================================
# Calibration paths
# ============================================================
CAMERA_CALIBRATION_DIR = CALIBRATION_DIR / "camera_calibration"
INTRINSIC_IMAGES_DIR = CAMERA_CALIBRATION_DIR / "intrinsic_images"
PLANE_IMAGES_DIR = CAMERA_CALIBRATION_DIR / "plane_images"
MOTOR_EXTRINSIC_IMAGES_DIR = CAMERA_CALIBRATION_DIR / "motor_extrinsic_images"
CALIBRATION_RESULTS_DIR = CAMERA_CALIBRATION_DIR / "results"
INTRINSIC_NPZ = CALIBRATION_RESULTS_DIR / "intrinsic_calibration.npz"
METRIC_PLANE_NPZ = CALIBRATION_RESULTS_DIR / "metric_plane_calibration.npz"

CONE_CALIBRATION_DIR = CALIBRATION_DIR / "cone_angle_calibration"
ANGLE_CALIBRATION_DIR = CALIBRATION_DIR / "angle_calibration"


def ensure_dirs() -> None:
    """Create the common output folders for the current RUN_NAME."""
    for path in (
        RUN_DIR,
        CAMERA_RAW_ROOT,
        CAMERA_UNDISTORTED_DIR,
        CAMERA_STITCHED_DIR,
        LASER_RAW_ROOT,
        LASER_RESTORED_DIR,
        SEGMENTATION_DIR,
        FUSION_DIR,
        WIRE_LASER_ANALYSIS_DIR,
        CAMERA_CALIBRATION_DIR,
        CAMERA_CALIBRATION_DIR / "results",
        CONE_CALIBRATION_DIR,
        ANGLE_CALIBRATION_DIR,
        DOCS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)