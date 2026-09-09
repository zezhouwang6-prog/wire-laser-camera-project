from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import shutil
from pathlib import Path
from config import LASER_RESTORED_IMAGE, CAMERA_STITCHED_IMAGE, FUSION_DIR, CONE_CALIBRATION_DIR

try:
    from config import CALIBRATION_DIR as _CONFIG_CALIBRATION_DIR
except Exception:
    _CONFIG_CALIBRATION_DIR = Path(CONE_CALIBRATION_DIR).parent

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    import cv2
except Exception:
    cv2 = None


# ===================== USER SETTINGS =====================
# Change these paths for different data sets.
# Use the 0.0504 mm/px laser restoration by default. This scale is closer to
# the natural camera stitch scale and is mainly for visual fusion.
LASER_IMAGE_PATH = str(Path(LASER_RESTORED_IMAGE).with_name("laser_height_mm_true_scale_detail_height.png"))
CAMERA_IMAGE_PATH = str(
    Path(CAMERA_STITCHED_IMAGE).with_name("fine_wire_angle_unwrapped_stitched_laser_height.png")
)

# For image alignment, use the already restored laser height PNG by default.
# This is much lighter than reading the full *_height_map.npz matrix.
# Set this to False only when you specifically want numeric height_mm from NPZ
# to drive masks/contours instead of the restored PNG.
USE_LASER_RESTORED_PNG_FOR_ALIGNMENT = True

# Leave empty to auto-find the matching *_height_map.npz beside the laser image.
# It is kept for later numeric analysis, but is not used for alignment while
# USE_LASER_RESTORED_PNG_FOR_ALIGNMENT is True.
LASER_NPZ_PATH = r""


# -------------------------------------------------------------------------
# SAVE REAL NUMERIC LASER HEIGHT IN THE FINAL CAMERA COORDINATE SYSTEM
# -------------------------------------------------------------------------
# This is the key output for the next 3-D textured-mesh step.
#
# Output:
#     <run folder>/mapped_height_mm.npz
#
# The saved `height_mm` array has EXACTLY the same HxW as the final camera
# stitched image. It uses the SAME:
#   - the same complete physical circular affine mapping,
#   - and any optional post-affine residual transform,
# as the final visible fusion.
#
# No 8-bit color/gray conversion is used for this output.
SAVE_MAPPED_HEIGHT_MM_NPZ = True
MAPPED_HEIGHT_MM_FILENAME = "mapped_height_mm.npz"

# If True, fail explicitly when the source numeric *_height_map.npz cannot be
# found. This prevents accidentally making a "3-D height" file from an 8-bit PNG.
REQUIRE_NUMERIC_HEIGHT_NPZ_FOR_MAPPED_OUTPUT = True

# Also save a simple diagnostic preview of the mapped REAL height values.
SAVE_MAPPED_HEIGHT_MM_PREVIEW = True
MAPPED_HEIGHT_PREVIEW_LOW_PERCENTILE = 2.0
MAPPED_HEIGHT_PREVIEW_HIGH_PERCENTILE = 98.0

# Leave empty to save beside this script in "result".
OUTPUT_DIR = str(FUSION_DIR)
OUTPUT_NAME = "fusion_2d.png"

# Output directly into OUTPUT_DIR (= config.FUSION_DIR).
# Do NOT create timestamp/run-name subfolders. Existing files with the same
# names will be overwritten by the next run.
RUN_NAME = r""

# -------------------------------------------------------------------------
# GLOBAL Y PHASE: LOCK TO THE VERIFIED RUN26 PHYSICAL REFERENCE
# -------------------------------------------------------------------------
# The later 46.085355877 deg cone/jog measurement is kept below only as a
# diagnostic value. For fusion, run26 demonstrated that the physically
# correct camera-laser circumferential phase is approximately:
#
#     1040 px / 8609 rows * 360 deg = 43.48937158787315 deg
#
# This phase is treated as a FIXED sensor-to-sensor relation. Therefore if the
# unfolded image height changes, the corresponding pixel offset is recalculated
# from the same angle instead of keeping 1040 px hard-coded.
#
# Example:
#     H = 8609 -> ~1040 px
#     H = 8802 -> ~1063 px
#
# This prevents dense/repetitive wire texture from making a large automatic jump
# to a neighbouring wire row.

# Latest independently measured value; diagnostic only in this version.
MEASURED_ANGLE_OFFSET_DEG_DIAGNOSTIC = 48.544

# The global Y owner for fusion.
USE_RUN26_REFERENCE_Y_PHASE = True
REFERENCE_CAMERA_LASER_PHASE_SOURCE_HEIGHT_PX = 8609
REFERENCE_CAMERA_LASER_PHASE_SOURCE_OFFSET_PX = 1040.0
REFERENCE_CAMERA_LASER_PHASE_DEG = (
    REFERENCE_CAMERA_LASER_PHASE_SOURCE_OFFSET_PX
    / REFERENCE_CAMERA_LASER_PHASE_SOURCE_HEIGHT_PX
    * 360.0
)

# Keep the legacy variable because several diagnostic helpers refer to it.
# In this version it intentionally equals the verified run26 phase.
ANGLE_OFFSET_DEG = REFERENCE_CAMERA_LASER_PHASE_DEG

# Optional manual correction AFTER the fixed reference phase.
# Start at zero. If the whole image is still uniformly off by exactly 1-2 px,
# change only FINAL_FINE_TUNE_PX.
FINAL_FINE_TUNE_DEG = 0.0
FINAL_FINE_TUNE_PX = 0

# Only a SMALL automatic Y residual search is allowed around the run26 phase.
AUTO_FINAL_Y_FINE_TUNE = True
FINAL_Y_FINE_TUNE_SEARCH_PX = 8
FINAL_Y_FINE_TUNE_STEP_PX = 1
FINAL_Y_FINE_TUNE_PREVIEW_WIDTH = 180
FINAL_Y_FINE_TUNE_PREVIEW_HEIGHT = 2400

# IMPORTANT:
# Disable the old large joint X/Y fixed-slide search. On dense repeated brush
# wires it can select a neighbouring wire row. Final X is controlled by the
# physical affine/X=0 calibration, and only small residual X/Y is left to the
# wire-center RANSAC.

# These values remain only for diagnostics/optional later experiments.
FIXED_SLIDE_Y_SEARCH_PX = 8
FIXED_SLIDE_Y_STEP_PX = 1
FIXED_SLIDE_X_SEARCH_PX = 8
FIXED_SLIDE_X_STEP_PX = 1
FIXED_SLIDE_PREVIEW_WIDTH = 180
FIXED_SLIDE_PREVIEW_HEIGHT = 2400
FIXED_SLIDE_TOP_CANDIDATES = 12

# -------------------------------------------------------------------------
# SAFE GLOBAL Y ALIGNMENT GUARDS
# -------------------------------------------------------------------------
# Dense brush-wire textures are highly repetitive.  A whole-image correlation
# search can therefore jump to a neighbouring wire row even when the measured
# sensor angle is already close.  These guards make the measured angle the
# "owner" of the global phase and allow the automatic stages to change it only
# when the score is sufficiently convincing.
#
# The values below are intentionally chosen to separate the previously observed
# good run26 behaviour from the bad run29 behaviour:
#
#   run26 (good):
#       Auto-Y best score ~0.0340, gain ~0.0062
#       Fixed-slide best score ~0.0445
#       Fixed-slide dx=+16 px, dy=-24 px
#
#   run29 (bad):
#       Auto-Y best score ~0.0061, gain ~0.0011
#       Fixed-slide best score ~0.0188
#       Fixed-slide dx=-60 px (exact X search boundary), dy=-52 px
#
# IMPORTANT:
# These guards do NOT change X/Y scale and do NOT alter the affine/physical-X
# calibration. They only decide whether a proposed automatic global translation
# is trustworthy enough to apply.

USE_SAFE_GLOBAL_Y_GUARDS = True

# ----- Guard for AUTO_FINAL_Y_FINE_TUNE -----
AUTO_Y_GUARD_MIN_BEST_SCORE = 0.015
AUTO_Y_GUARD_MIN_SCORE_GAIN = 0.0020

# If the best Auto-Y candidate lands on the +/- search boundary, accept it only
# when it is very strong.  This keeps run26's strong -36 px solution possible,
# while rejecting weak boundary solutions on repetitive texture.
AUTO_Y_GUARD_EDGE_MARGIN_PX = 0.5
AUTO_Y_GUARD_EDGE_MIN_BEST_SCORE = 0.030
AUTO_Y_GUARD_EDGE_MIN_SCORE_GAIN = 0.0040

# Never let this stage jump farther than its configured search range.
AUTO_Y_GUARD_MAX_ABS_DELTA_PX = float(FINAL_Y_FINE_TUNE_SEARCH_PX)

# ----- Guard for FIXED_ANGLE_SLIDE_ALIGNMENT -----
# The physical X calibration should already put X close to the correct place.
# A fixed-slide optimum at the X search boundary is therefore a strong false-
# match signal and the associated Y should NOT be trusted.
FIXED_SLIDE_GUARD_REJECT_X_BOUNDARY = True
FIXED_SLIDE_GUARD_X_BOUNDARY_MARGIN_PX = 0.5 * float(FIXED_SLIDE_X_STEP_PX)

# Additional conservative quality gates.
FIXED_SLIDE_GUARD_MIN_BEST_SCORE = 0.025
FIXED_SLIDE_GUARD_MIN_SCORE_GAIN = 0.0030

# A very large residual X after physical calibration is suspicious even if it
# does not hit the exact search boundary.
FIXED_SLIDE_GUARD_MAX_ABS_DELTA_X_PX = 8.0

# Large Y jumps are allowed only up to this value.  If the measured angle is
# farther away than this, fix ANGLE_OFFSET_DEG rather than letting repeated
# texture choose an arbitrary neighbouring row.
FIXED_SLIDE_GUARD_MAX_ABS_DELTA_Y_PX = 8.0

# Save a compact decision JSON so every run clearly records which global-Y
# proposals were accepted/rejected and why.
SAVE_SAFE_Y_GUARD_JSON = True
SAFE_Y_GUARD_JSON_NAME = "safe_y_alignment_guard.json"

# Vertical behavior along the wheel rotation direction.
# True  = circular roll: top overflow is pasted to the bottom.
# False = no circular roll: overflow is left blank.
CAMERA_VERTICAL_WRAP = True
WRAP_SEAM_FEATHER_PX = 80

# Automatically search near ANGLE_OFFSET_DEG for a better vertical angular offset.
SEARCH_RANGE_DEG = 0.6
SEARCH_STEP_DEG = 0.10
REFINE_RANGE_DEG = 0.20
REFINE_STEP_DEG = 0.02

# Automatic angle search guard.
# If the best result is found at the search boundary, it usually means the score
# is being fooled by repeated wire texture or a wrap seam, not that the angle is
# truly better. In that case keep the measured base angle.
AUTO_ALIGN_REJECT_EDGE_BEST = True
AUTO_ALIGN_EDGE_MARGIN_DEG = 0.10
# A negative or near-zero raw correlation means the automatic match has not found
# a useful common structure between the laser feature image and camera feature image.
AUTO_ALIGN_REJECT_WEAK_SCORE = True
AUTO_ALIGN_MIN_RAW_SCORE = 0.0
# Prefer small corrections unless the match score is clearly better.
AUTO_ALIGN_CENTER_PENALTY_PER_DEG = 0.002

# Try -1 first. If the fused image is shifted in the wrong vertical direction,
# change this to 1 and run again.
ROLL_DIRECTION = -1

# The camera image may be mirrored relative to the laser X direction.

# Also search a small horizontal shift on the preview image.
SEARCH_X_SHIFT_PREVIEW_PX = 32
SEARCH_X_SHIFT_STEP_PREVIEW_PX = 4

# Low-resolution image size for automatic alignment search.
ALIGN_PREVIEW_WIDTH = 256
ALIGN_PREVIEW_HEIGHT = 4096

# Fusion weight. 0.70 means camera color is 70%, laser height color is 30%.
CAMERA_ALPHA = 0.70
LASER_ALPHA = 0.30

# Process image in strips to avoid using too much memory for very tall true-scale maps.
TILE_HEIGHT_PX = 512

# Save an extra image with the laser height image and aligned camera image side by side.

# Save a full-height side-by-side correspondence image:
# left = complete laser height image, right = complete aligned camera stitched image.

# Save another side-by-side image at the camera stitched image scale:
# left = laser height image downsampled to camera size, right = original camera image.

# Save laser-visible camera outputs.

# Save the preferred fusion result:
# full camera image as the base, valid laser height pixels as a transparent overlay.
LASER_ON_CAMERA_ALPHA = 0.55

# Map the laser width only onto the main wire area in the camera image.
# If SELECT_CAMERA_MAIN_X_RANGE is True, a window will open and you can drag
# the left/right boundary with the mouse. Only the X range is used.
CAMERA_MAIN_X0 = 55
CAMERA_MAIN_X1 = 335
# Manual selector uses a fixed laser-band width. Drag the band left/right;
# its width cannot be changed in the selector.
CAMERA_MAIN_FIXED_WIDTH_PX = 317
# The laser head measures 16 mm in X. For the natural stitched camera image,
# use this mm/px scale to place a 16 mm-wide scan band on the camera image.
LASER_EFFECTIVE_WIDTH_MM = 16.0
CAMERA_NATURAL_MM_PER_PX = 0.05473705
CAMERA_SELECTOR_PREVIEW_WIDTH = 900

# -------------------------------------------------------------------------
# PRIMARY FUSION MODEL: FULL PHYSICAL CIRCULAR AFFINE
# -------------------------------------------------------------------------
# True = use the complete physical affine saved by the latest circular
# calibration:
#
#   CAMERA [X_mm, unwrapped_arc_mm]
#       -> full 2x3 affine
#   LASER [X_mm, base_arc_mm]
#
# The final camera->laser sampling map therefore keeps:
#   - affine X scale,
#   - affine X<-Y coupling,
#   - affine Y<-X coupling,
#   - affine Y scale,
#   - affine X/Y translation,
#   - circular +/-360-degree branch handling.
#
# IMPORTANT:
# In this mode the old run26 fixed Y phase, precision-Y curve and the old
# X-scale=1 translation-only placement are NOT used for the final mapping.
USE_FULL_PHYSICAL_CIRCULAR_AFFINE = True

# Empty = automatically use the newest valid affine_calibration.json /
# affine_calibration_physical_circular.json under the calibration folders.
FULL_AFFINE_CALIBRATION_JSON_PATH = r""

# The calibration model is already physical and 2-D.  Keep wire-center RANSAC
# OFF for the first verification so the output shows the calibration itself,
# without a second repeated-texture alignment stage moving it afterwards.
# After the physical affine is verified, this can be enabled for a small
# residual experiment if necessary.

# -------------------------------------------------------------------------
# FINAL X-ONLY GEOMETRIC RESIDUAL AFTER FULL PHYSICAL AFFINE
# -------------------------------------------------------------------------
# The full physical affine is now the global mapping authority.
#
# After that mapping, only one final degree of freedom is allowed:
#
#       laser X translation dx
#
# Y is LOCKED:
#       dy = 0
#
# Scale is LOCKED:
#       sx = sy = 1
#
# This stage does NOT use whole-image / window texture correlation.
# It detects wire-end centers independently in camera and laser data, forms
# tight geometric correspondences around the already-correct full-affine
# result, rejects Y-inconsistent pairs, uses MAD outlier filtering, checks
# several vertical portions of the 360-degree image, and finally applies only
# a small robust X translation.

# Final permitted X correction.  The image is already close after affine.
FULL_AFFINE_X_ONLY_SEARCH_RANGE_PX = 4.0

# Candidate center pairs must be vertically consistent; Y itself is never moved.
FULL_AFFINE_X_ONLY_MAX_PAIR_DY_PX = 2.5

# The initial correspondence radius must be large enough to see a <=4 px X
# residual while still remaining local.
FULL_AFFINE_X_ONLY_MATCH_RADIUS_PX = 5.0

# Robust MAD rejection on pairwise X residuals.
FULL_AFFINE_X_ONLY_MAD_MULTIPLIER = 3.5
FULL_AFFINE_X_ONLY_MIN_MAD_GATE_PX = 0.45

# Require enough pairs at each detector threshold.
FULL_AFFINE_X_ONLY_MIN_MATCHES_PER_THRESHOLD = 80
FULL_AFFINE_X_ONLY_MIN_ROBUST_MATCHES_PER_THRESHOLD = 60

# Split the 360-degree image vertically; the same X correction should be
# visible in multiple portions rather than only one local patch.
FULL_AFFINE_X_ONLY_SEGMENT_COUNT = 5
FULL_AFFINE_X_ONLY_MIN_POINTS_PER_SEGMENT = 12
FULL_AFFINE_X_ONLY_MIN_AGREEING_SEGMENTS = 3
FULL_AFFINE_X_ONLY_MAX_SEGMENT_DEVIATION_PX = 1.10

# Multi-threshold consensus: at least 2 of the 3 detector thresholds must agree.
FULL_AFFINE_X_ONLY_MIN_AGREEING_THRESHOLDS = 2
FULL_AFFINE_X_ONLY_MAX_THRESHOLD_DEVIATION_PX = 0.85

# Avoid a needless resampling if the remaining shift is negligible.
FULL_AFFINE_X_ONLY_MIN_EFFECT_PX = 0.10

SAVE_FULL_AFFINE_X_ONLY_JSON = True
SAVE_FULL_AFFINE_X_ONLY_MATCH_CSV = True

# -------------------------------------------------------------------------
# NEW FINAL X RESIDUAL:
# WIRE-PITCH PHASE SEARCH + SYMMETRIC CHAMFER GEOMETRY
# -------------------------------------------------------------------------
# The previous nearest-neighbour center X-only residual is disabled above.
#
# Reason:
# dense/repetitive brush wires can make a laser wire match the neighbouring
# camera wire, causing an apparently tiny residual (for example +0.15 px)
# even when the actual phase is off by approximately one wire pitch.
#
# New policy:
#   1) keep the complete physical circular affine as the global authority;
#   2) detect wire-end cap geometry independently in both modalities;
#   3) estimate the robust wire pitch from detected centers;
#   4) explicitly test integer wire-phase hypotheses k=-2,-1,0,+1,+2;
#   5) around each k*pitch, search only a small local X interval;
#   6) score GEOMETRIC END-FACE EDGES with symmetric Chamfer distance;
#   7) require vertical-segment and multi-threshold phase consensus;
#   8) apply X translation only.
#
# Y and scale remain locked:
#       dy = 0, sx = 1, sy = 1
#
# No image-window / texture correlation is used.
FULL_AFFINE_USE_WIRE_PITCH_CHAMFER_X_RESIDUAL = True

# Integer wire phase candidates.
FULL_AFFINE_CHAMFER_PHASE_MIN_K = -2
FULL_AFFINE_CHAMFER_PHASE_MAX_K = 2

# Local search around each k * estimated_wire_pitch.
FULL_AFFINE_CHAMFER_LOCAL_RADIUS_PX = 3.0
FULL_AFFINE_CHAMFER_COARSE_STEP_PX = 0.50

# After phase selection, subpixel refinement around the selected X.
FULL_AFFINE_CHAMFER_REFINE_RADIUS_PX = 1.25
FULL_AFFINE_CHAMFER_REFINE_STEP_PX = 0.25
FULL_AFFINE_CHAMFER_FINE_RADIUS_PX = 0.40
FULL_AFFINE_CHAMFER_FINE_STEP_PX = 0.10

# Edge-distance scoring.
FULL_AFFINE_CHAMFER_DISTANCE_CLIP_PX = 4.0
FULL_AFFINE_CHAMFER_CLOSE_DISTANCE_PX = 1.25
FULL_AFFINE_CHAMFER_CLOSE_FRACTION_WEIGHT = 0.35

# Keep the scoring computationally bounded and spatially balanced.
FULL_AFFINE_CHAMFER_SEGMENT_COUNT = 5
FULL_AFFINE_CHAMFER_MAX_EDGE_POINTS_PER_SEGMENT = 4500
FULL_AFFINE_CHAMFER_MIN_EDGE_POINTS_PER_SEGMENT = 180

# At least this many vertical segments must support the same integer wire phase.
FULL_AFFINE_CHAMFER_MIN_PHASE_AGREEING_SEGMENTS = 3

# Multi-threshold consensus. The existing 66/72/78% detector thresholds are used.
FULL_AFFINE_CHAMFER_MIN_AGREEING_THRESHOLDS = 2

# Threshold-level final dx values supporting the same phase should remain close.
FULL_AFFINE_CHAMFER_MAX_THRESHOLD_DX_DEVIATION_PX = 1.50

# Estimated pitch is clipped to a physically plausible pixel range.
FULL_AFFINE_CHAMFER_MIN_PITCH_PX = 4.0
FULL_AFFINE_CHAMFER_MAX_PITCH_PX = 20.0

# If the winning shift sits on the OUTER phase search boundary (k=+/-2 and
# also at the outward local edge), reject rather than silently searching farther.
FULL_AFFINE_CHAMFER_REJECT_OUTER_BOUNDARY = True
FULL_AFFINE_CHAMFER_BOUNDARY_TOL_PX = 0.26

# Do not resample for a negligible final shift.
FULL_AFFINE_CHAMFER_MIN_EFFECT_PX = 0.10

SAVE_FULL_AFFINE_CHAMFER_JSON = True
SAVE_FULL_AFFINE_CHAMFER_SEARCH_CSV = True
SAVE_FULL_AFFINE_CHAMFER_EDGE_DEBUG_PNG = True

# Save the exact 2-D camera->laser pixel correspondence map separately.
SAVE_FULL_AFFINE_CORRESPONDENCE_MAPS = True

# Candidate unwrapped camera branches used around the 0/360-degree seam.
FULL_AFFINE_CAMERA_BRANCHES = (-1, 0, 1)
CAMERA_SELECTOR_PREVIEW_HEIGHT = 900

# In the laser height PNG, invalid/no-data pixels are nearly white.
# Increase this if too much is marked valid; decrease it if valid laser points are missing.
# This is only used when no NPZ height map is found.
LASER_INVALID_WHITE_THRESHOLD = 245

# Slightly expand the valid laser mask so it covers the corresponding camera wire top.
MASK_DILATE_PIXELS = 0

# Extra visualization outputs. These do not replace the original fusion result.
# Contour lines are selected from valid laser height percentiles, so they stay useful
# when the absolute height range changes between experiments.
CONTOUR_PERCENTILES = (45, 60, 75, 88, 96)
CONTOUR_COLOR = (255, 245, 20)      # yellow contour lines
CONTOUR_HIGH_COLOR = (255, 255, 255) # white highest contour lines
BOUNDARY_COLOR = (255, 30, 30)      # red laser-valid boundary
BOUNDARY_DILATE_PIXELS = 1
# Save local side-by-side crops: left=laser height, middle=camera, right=contour overlay.
DETAIL_CROP_HEIGHT_PX = 1200
DETAIL_CROP_POSITIONS = (0.18, 0.50, 0.82)

# Segmented vertical alignment: first keep the global fixed/affine coarse alignment,
# then split the full image vertically, search only a small local Y translation per segment,
# and smooth the offset curve. This is for the case where the top aligns but lower rows
# slowly drift by a few wire rows.
# The previous wide-range segmented correction is kept in the file for reference,
# but the preferred final output now uses the precision refinement pipeline below.
SEGMENT_ALIGN_HEIGHT_PX = 900
SEGMENT_ALIGN_STEP_PX = 450
SEGMENT_SEARCH_RANGE_PX = 48
SEGMENT_SEARCH_STEP_PX = 1
SEGMENT_ALIGN_WIDTH = 192
SEGMENT_SMOOTH_WINDOW = 7
SEGMENT_MAX_JUMP_PX = 16
SEGMENT_MIN_VALID_FRACTION = 0.08

# -------------------------------------------------------------------------
# PRECISION FINAL Y REFINEMENT
# -------------------------------------------------------------------------
# The coarse alignment is already close.  First refine the global Y phase
# AFTER the 4-pixel fixed-slide search, then estimate only +/-6 px local
# residuals with a confidence gate and subpixel refinement.
USE_PRECISION_Y_REFINEMENT = True

PRECISION_GLOBAL_Y_SEARCH_RANGE_PX = 6.0
PRECISION_GLOBAL_Y_STEP_PX = 0.5

PRECISION_SEGMENT_HEIGHT_PX = 600
PRECISION_SEGMENT_STEP_PX = 250
PRECISION_SEGMENT_SEARCH_RANGE_PX = 3
PRECISION_SEGMENT_INTEGER_STEP_PX = 1
PRECISION_SEGMENT_SUBPIXEL_RADIUS_PX = 1.0
PRECISION_SEGMENT_SUBPIXEL_STEP_PX = 0.25
PRECISION_SEGMENT_ALIGN_WIDTH = 192

PRECISION_SEGMENT_MIN_SCORE_GAIN = 0.010
PRECISION_SEGMENT_BOUNDARY_MIN_SCORE_GAIN = 0.025
PRECISION_SEGMENT_REJECT_WEAK_BOUNDARY = True

PRECISION_SEGMENT_SMOOTH_WINDOW = 3
PRECISION_SEGMENT_MAX_JUMP_PX = 2.0

SAVE_PRECISION_Y_CURVE_CSV = True
SAVE_PRECISION_SEGMENT_CSV = True

# -------------------------------------------------------------------------
# LOCK LASER HORIZONTAL SIZE + REUSE THE PREVIOUS X=0 ORIGIN CALIBRATION
# -------------------------------------------------------------------------
# 1) The laser keeps its native X width: NO 299 -> 360 stretch.
# 2) Horizontal placement is NOT image-center placement.
# 3) Reuse the previous physical-origin convention:
#       camera stitched ROI horizontal center = X_camera = 0 mm
#       laser native physical X=0             = X_laser  = 0 mm
#    Therefore the laser X=0 column is placed exactly on the camera X=0 column.
LOCK_LASER_NATIVE_X_SIZE_ON_CAMERA = True
LASER_NATIVE_X_SCALE = 1.0

# Empty = automatically find the newest affine_calibration_physical_mm.json /
# affine_calibration.json made by cone_affine_calibration_roi_center_x0.py.
ORIGIN_CALIBRATION_JSON_PATH = r""

# These two switches are used by the legacy/diagnostic physical-X=0 helper and
# are also written to the run report.  They were referenced in the previous
# version but accidentally not defined, which caused a NameError at the very
# end of an otherwise successful run.
PREFER_EXACT_RAW_LASER_X_AXIS = True

USE_PREVIOUS_X0_ORIGIN_CALIBRATION = True
REQUIRE_PREVIOUS_X0_ORIGIN_CALIBRATION = True

# IMPORTANT:
# Horizontal placement now REUSES the previous clicked affine calibration.
#
# The saved affine calibration contains:
#   camera physical X (ROI center = X_camera 0 mm)
#   laser physical X (the laser 0..16 mm coordinate used during that calibration)
#
# We DO NOT apply the old fitted X scale (~1.045), because that would visibly
# stretch/compress the laser.  Instead, we refit ONLY ONE horizontal translation
# from the SAME calibration inlier points:
#
#       X_laser_mm = X_camera_mm + TX_LOCKED_MM
#
# This keeps X scale exactly 1.0 while preserving the previously calibrated
# horizontal position.
USE_HORIZONTAL_CALIBRATION_SCALE1 = True
REQUIRE_HORIZONTAL_CALIBRATION = True

# Empty = auto-find the newest suitable affine_calibration.json.
# You may set the exact file explicitly if desired.
HORIZONTAL_CALIBRATION_JSON_PATH = r""

# Use the RANSAC inliers saved by the original calibration.
HORIZONTAL_CALIB_USE_SAVED_INLIERS = True

# Robust translation fit. The primary result is the mean of the saved inlier
# offsets (least-squares translation-only refit). Median is also reported.

# Manual tiny X translation after the calibrated scale-1 placement.
# This is pure translation only and never changes laser width.
# IMPORTANT for this version: keep this at 0.0.  The final X/Y residual is
# estimated automatically below, so do not manually use -4/+4 px here.
HORIZONTAL_FINE_TUNE_PX = 0.0

# -------------------------------------------------------------------------
# FINAL RANSAC-GUIDED INLIER RESIDUAL REFINEMENT
# -------------------------------------------------------------------------
# The previous image-window final residual XY method has been removed.
#
# This version does NOT compare whole image windows and does NOT use camera
# texture / laser-height texture correlation for the last few pixels.
#
# Instead:
#   1) global wire-center RANSAC determines the main residual translation;
#   2) optional topology may make only its existing bounded micro correction;
#   3) the accepted transform seeds a TIGHT wire-center re-match;
#   4) remaining pair residuals are filtered by MAD;
#   5) several vertical portions of the 360-degree image must agree;
#   6) only then is one final small translation added.
#
# X/Y scale remains exactly 1.0. No rotation, shear, or manual movement.
USE_RANSAC_INLIER_RESIDUAL_REFINEMENT = True

RANSAC_RESIDUAL_REMATCH_RADIUS_PX = 3.2
RANSAC_RESIDUAL_MIN_MATCHES = 80

RANSAC_RESIDUAL_MAD_MULTIPLIER = 3.5
RANSAC_RESIDUAL_MIN_MAD_GATE_PX = 0.55

RANSAC_RESIDUAL_SEGMENT_COUNT = 5
RANSAC_RESIDUAL_MIN_POINTS_PER_SEGMENT = 18
RANSAC_RESIDUAL_MIN_VALID_SEGMENTS = 3

RANSAC_RESIDUAL_MAX_SEGMENT_DEVIATION_X_PX = 1.35
RANSAC_RESIDUAL_MAX_SEGMENT_DEVIATION_Y_PX = 1.35

RANSAC_RESIDUAL_MAX_FINAL_DX_PX = 2.5
RANSAC_RESIDUAL_MAX_FINAL_DY_PX = 2.5

RANSAC_RESIDUAL_MIN_EFFECT_PX = 0.15
RANSAC_RESIDUAL_MIN_ROBUST_INLIER_RATIO = 0.55

SAVE_RANSAC_RESIDUAL_MATCH_CSV = True
SAVE_RANSAC_RESIDUAL_JSON = True


# -------------------------------------------------------------------------
# FINAL WIRE-CENTER POINT-SET ALIGNMENT + RANSAC
# -------------------------------------------------------------------------
# Preferred final residual alignment for dense brush-wire data.
# Unlike the old whole-image correlation search, this stage works on detected
# wire-end centers and their 2-D geometry.
USE_WIRE_CENTER_RANSAC_ALIGNMENT = True

# Center detection is repeated at several thresholds.  A correction is accepted
# only when these independent point sets agree, which suppresses repeated-wire
# false matches.
WIRE_CENTER_DETECTION_PERCENTILES = (66.0, 72.0, 78.0)
WIRE_CENTER_LOCAL_SIGMA_SMALL = 1.2
WIRE_CENTER_LOCAL_SIGMA_LARGE = 4.8
WIRE_CENTER_NMS_RADIUS_PX = 5
WIRE_CENTER_MIN_DISTANCE_VALUE_PX = 1.6
WIRE_CENTER_MAX_POINTS = 18000
WIRE_CENTER_EDGE_MARGIN_X_PX = 4

# Search is performed on center geometry, not image intensity.
WIRE_CENTER_X_SEARCH_RANGE_PX = 12
WIRE_CENTER_Y_SEARCH_RANGE_PX = 8
WIRE_CENTER_SEARCH_STEP_PX = 1
WIRE_CENTER_POINT_CLOSE_RADIUS_PX = 2.6
WIRE_CENTER_POINT_CLIP_DISTANCE_PX = 7.0
WIRE_CENTER_CENTER_PENALTY_PER_PX = 0.0008

# Candidate correspondence and RANSAC validation.
WIRE_CENTER_MATCH_RADIUS_PX = 5.0
WIRE_CENTER_RANSAC_INLIER_THRESHOLD_PX = 2.6
WIRE_CENTER_RANSAC_MAX_SEEDS = 1800
WIRE_CENTER_MIN_MATCHES = 40
WIRE_CENTER_MIN_INLIERS = 24
WIRE_CENTER_MIN_INLIER_RATIO = 0.12
WIRE_CENTER_MAX_TRANSLATION_DISAGREEMENT_PX = 2.5
WIRE_CENTER_REJECT_SEARCH_BOUNDARY = True

# Optional small diagonal scale correction. Translation is always tested first.
# A scale is used only when it improves the RANSAC inlier fit clearly and remains
# very close to 1.0. Rotation and shear are NEVER introduced here.
WIRE_CENTER_SCALE_MIN = 0.985
WIRE_CENTER_SCALE_MAX = 1.015
WIRE_CENTER_SCALE_MIN_RMSE_IMPROVEMENT_PX = 0.35
WIRE_CENTER_SCALE_MIN_RELATIVE_IMPROVEMENT = 0.18
WIRE_CENTER_SCALE_MIN_X_SPAN_RATIO = 0.35
WIRE_CENTER_SCALE_MIN_Y_SPAN_RATIO = 0.35

SAVE_WIRE_CENTER_MATCH_CSV = True
SAVE_WIRE_CENTER_RANSAC_JSON = True

# -------------------------------------------------------------------------
# HIGH-CONFIDENCE WIRE-END + LOCAL TOPOLOGY MATCHING
# -------------------------------------------------------------------------
# Dense brush wires are quasi-periodic: a geometrically consistent nearest
# point can still be the neighboring physical wire.  This stage therefore
# does NOT accept a center match from position alone.  It first keeps
# high-confidence end-face centers, describes the local neighbor layout around
# every center, requires forward/reverse consistency, and only then sends the
# correspondences to RANSAC.
USE_WIRE_TOPOLOGY_MATCHING = True

# High-confidence end-face filter.  Strength/radius/shape gates are deliberately
# robust-percentile based so a different wire type does not require fixed gray
# thresholds.  If a strict filter leaves too few points, the code automatically
# relaxes the confidence percentile but never falls back to raw nearest matching.
WIRE_TOPOLOGY_CONFIDENCE_PERCENTILE = 48.0
WIRE_TOPOLOGY_RELAXED_CONFIDENCE_PERCENTILE = 30.0
WIRE_TOPOLOGY_MIN_HIGH_CONF_POINTS = 350
WIRE_TOPOLOGY_COMPONENT_MIN_ASPECT = 0.28
WIRE_TOPOLOGY_COMPONENT_MIN_FILL = 0.20
WIRE_TOPOLOGY_RADIUS_LOW_PERCENTILE = 4.0
WIRE_TOPOLOGY_RADIUS_HIGH_PERCENTILE = 96.0
WIRE_TOPOLOGY_AREA_HIGH_PERCENTILE = 96.0

# Local topology descriptor.  Neighbor offsets are normalized by each modality's
# robust nearest-neighbor spacing, but their ANGLES remain in the image coordinate
# system.  Rotation/shear is therefore not silently introduced.
WIRE_TOPOLOGY_NEIGHBOR_RADIUS_FACTOR = 3.6
WIRE_TOPOLOGY_MIN_NEIGHBORS = 3
WIRE_TOPOLOGY_MAX_NEIGHBORS = 14
WIRE_TOPOLOGY_SECTORS = 12
WIRE_TOPOLOGY_RADIAL_SLOTS = 2
WIRE_TOPOLOGY_DESCRIPTOR_CLIP = 3.2
WIRE_TOPOLOGY_MISSING_VALUE = 3.2

# Candidate identity match after a coarse translation hypothesis.
WIRE_TOPOLOGY_MATCH_RADIUS_PX = 7.0
WIRE_TOPOLOGY_POSITION_WEIGHT = 0.28
WIRE_TOPOLOGY_DESCRIPTOR_WEIGHT = 0.72
WIRE_TOPOLOGY_MAX_DESCRIPTOR_DISTANCE = 0.82
WIRE_TOPOLOGY_RATIO_TEST = 0.965
WIRE_TOPOLOGY_MIN_SECOND_BEST_MARGIN = 0.025
WIRE_TOPOLOGY_REQUIRE_MUTUAL = True

# Do not trust only the single peak of the old point-set translation search.
# Evaluate several spatially separated peaks and let topology-verified RANSAC
# choose the physical-wire identity hypothesis.
WIRE_TOPOLOGY_MAX_TRANSLATION_HYPOTHESES = 18
WIRE_TOPOLOGY_HYPOTHESIS_MIN_SEPARATION_PX = 2.5
WIRE_TOPOLOGY_ROWS_CONSIDERED_PER_THRESHOLD = 220
WIRE_TOPOLOGY_MIN_MATCHES = 45
WIRE_TOPOLOGY_MIN_INLIERS = 28
WIRE_TOPOLOGY_MIN_INLIER_RATIO = 0.16
WIRE_TOPOLOGY_MAX_MEDIAN_DESCRIPTOR_DISTANCE = 0.70
WIRE_TOPOLOGY_MIN_X_COVERAGE_RATIO = 0.22
WIRE_TOPOLOGY_MIN_Y_COVERAGE_RATIO = 0.35

SAVE_WIRE_TOPOLOGY_HYPOTHESES_CSV = True

# -------------------------------------------------------------------------
# TOPOLOGY IS MICRO-REFINEMENT ONLY (NEVER THE GLOBAL ALIGNMENT OWNER)
# -------------------------------------------------------------------------
# The ordinary multi-threshold wire-center RANSAC below owns the global X/Y.
# Topology may only make a tiny correction around that already-accepted result.
# If topology fails, disagrees, or wants a larger jump, the global RANSAC result
# is kept unchanged.
WIRE_TOPOLOGY_MICRO_MAX_DX_PX = 2.0
WIRE_TOPOLOGY_MICRO_MAX_DY_PX = 2.0
WIRE_TOPOLOGY_MICRO_HYPOTHESIS_RADIUS_PX = 2
WIRE_TOPOLOGY_MICRO_HYPOTHESIS_STEP_PX = 1

# -------------------------------------------------------------------------
# WIRE-CENTER LOCAL Y REFINEMENT
# -------------------------------------------------------------------------
# After the global wire-center RANSAC translation has aligned X and the global
# Y phase, keep X fixed and estimate ONLY the remaining slowly varying Y error
# from the actual RANSAC wire-center correspondences.  This avoids returning to
# whole-image texture correlation, which is ambiguous for dense repeated wires.

# Sliding periodic windows along the 0..360-degree camera Y axis.
WIRE_CENTER_LOCAL_Y_WINDOW_PX = 900
WIRE_CENTER_LOCAL_Y_STEP_PX = 300
WIRE_CENTER_LOCAL_Y_MIN_POINTS_PER_WINDOW = 18
WIRE_CENTER_LOCAL_Y_MIN_TOTAL_POINTS = 80

# Only small residuals are allowed after the global RANSAC.  Larger values are
# much more likely to be a wrong neighboring-wire correspondence.
WIRE_CENTER_LOCAL_Y_MAX_INPUT_RESIDUAL_PX = 3.25
WIRE_CENTER_LOCAL_Y_MAX_APPLIED_PX = 2.50
WIRE_CENTER_LOCAL_Y_MAD_MULTIPLIER = 3.5
WIRE_CENTER_LOCAL_Y_MIN_MAD_GATE_PX = 0.60

# Smooth the segment medians and limit abrupt changes.
WIRE_CENTER_LOCAL_Y_SMOOTH_WINDOW = 5
WIRE_CENTER_LOCAL_Y_MAX_SEGMENT_JUMP_PX = 0.90
WIRE_CENTER_LOCAL_Y_MIN_VALID_SEGMENT_RATIO = 0.55
WIRE_CENTER_LOCAL_Y_MIN_EFFECT_PX = 0.20


# The laser native width is kept exactly. No 299 -> 360 X resize.
LOCK_LASER_NATIVE_X_SIZE_ON_CAMERA = True
LASER_NATIVE_X_SCALE = 1.0

# Save X calibration diagnostics.

# Fine translation AFTER physical X=0 alignment. This is not a scale.
# Keep 0.0 unless a later measured mechanical offset needs a tiny correction.
LASER_NATIVE_X_OFFSET_PX = 0.0

# Use subpixel X translation so a fractional X=0 location is not rounded away.
ORIGIN_X_USE_SUBPIXEL_TRANSLATION = True

# Save an origin diagnostic image and JSON.

# Save the precision camera-base overlay as the main fusion_2d.png.
SAVE_PRECISION_CAMERA_BASE_AS_MAIN_OUTPUT = True

# End-face guided segmented alignment.
# The original segmented alignment only compared texture/edge images. This extra
# map enhances round/elliptical wire end faces before each segment searches its
# local Y shift, which is more useful for the laser-vs-camera wire-wheel data.
USE_END_FACE_GUIDED_SEGMENT_ALIGNMENT = True
END_FACE_FEATURE_WEIGHT = 0.68
END_FACE_SMALL_BLUR_PX = 1.2
END_FACE_LARGE_BLUR_PX = 6.0
END_FACE_EDGE_WEIGHT = 0.25
END_FACE_THRESHOLD_PERCENTILE = 70.0

# Feature mutual-information + 2D similarity/affine optimization.
# This is a global alignment method. It does not replace the original fixed-angle
# result; it saves extra files whose names start with "feature_mi_affine".
AFFINE_PREVIEW_WIDTH = 180
AFFINE_PREVIEW_HEIGHT = 1800
AFFINE_Y_SEARCH_RANGE_DEG = 1.2
AFFINE_Y_SEARCH_STEP_DEG = 0.15
AFFINE_X_SEARCH_RANGE_PX = 16
AFFINE_X_SEARCH_STEP_PX = 8
AFFINE_REFINE_Y_RANGE_DEG = 0.25
AFFINE_REFINE_Y_STEP_DEG = 0.05
AFFINE_REFINE_X_RANGE_PX = 6
AFFINE_REFINE_X_STEP_PX = 2
AFFINE_X_SCALE_VALUES = (1.00,)
AFFINE_Y_SCALE_VALUES = (0.997, 0.998, 0.999, 1.000, 1.001, 1.002, 1.003)
AFFINE_SHEAR_VALUES = (0.0,)
USE_FEATURE_MAP_FOR_MI = True
FEATURE_MI_WEIGHT = 0.70
FEATURE_EDGE_WEIGHT = 0.25
FEATURE_VALID_TEXTURE_WEIGHT = 0.05
FEATURE_MI_BINS = 32
# Prevent a false match where only one vertical part aligns well.
# The score is reduced when upper/lower image halves give very different NMI scores.
FEATURE_SPLIT_CONSISTENCY_WEIGHT = 0.18
FEATURE_SPLIT_MIN_VALID_PIXELS = 800

# Feature-point RANSAC alignment.
# This is an extra diagnostic/alignment path: it keeps the existing fixed-angle
# and feature-NMI outputs, then saves additional files named ransac_*.
RANSAC_PREVIEW_WIDTH = 420
RANSAC_PREVIEW_HEIGHT = 4200
RANSAC_MAX_FEATURES = 6000
RANSAC_RATIO_TEST = 0.82
RANSAC_REPROJ_THRESHOLD_PX = 8.0
RANSAC_MAX_ITERS = 6000
RANSAC_CONFIDENCE = 0.995
RANSAC_MIN_MATCHES = 16
RANSAC_MIN_INLIERS = 8
RANSAC_MATCH_DRAW_LIMIT = 180
RANSAC_NMI_SHIFT_SEARCH_PX = 12
RANSAC_NMI_SHIFT_STEP_PX = 2
RANSAC_NMI_PREVIEW_WIDTH = 240
RANSAC_NMI_PREVIEW_HEIGHT = 2400

# Extra corrections after the global feature-NMI affine alignment.
# 1) soften the 0/360 degree circular wrap seam;
# 2) search one additional smooth Y correction for the back half.
AFFINE_CLOSE_WRAP_SEAM = True
AFFINE_WRAP_SEAM_FEATHER_PX = 72
AFFINE_WRAP_SEAM_SEARCH_PX = 40
AFFINE_BACK_HALF_START_RATIO = 0.55
AFFINE_BACK_HALF_SEARCH_RANGE_DEG = 3.0
AFFINE_BACK_HALF_SEARCH_STEP_DEG = 0.1

# Three-point cone calibration affine alignment.
# This uses cone_calibration_click.py output. It maps the stitched camera image
# into the laser height image coordinate system using the measured point pairs.
# Leave empty to use the newest cone_calibration.json under:
#   D:\project\calibration\cone_angle_calibration
CONE_CALIBRATION_JSON_PATH = r""
# The wheel is a closed 360-degree surface, so Y can wrap around at the seam.
CONE_AFFINE_WRAP_Y = True
# The cone calibration may come from a reference run, for example run1, while
# the current fusion may use run4. In that case, convert the reference pixel
# affine matrix through normalized image coordinates before applying it.
CONE_AFFINE_ALLOW_SIZE_TRANSFER = True
CONE_AFFINE_REQUIRE_MATCHING_IMAGE_SIZE = True
CONE_AFFINE_SIZE_TOLERANCE_RATIO = 0.02
# =========================================================




def resolve_output_dir(path_text: str) -> Path:
    if str(path_text).strip():
        return Path(path_text).resolve()
    return Path(__file__).resolve().parent / "result"


def safe_folder_name(text: str) -> str:
    bad = '<>:"/\\|?*'
    cleaned = "".join("_" if ch in bad else ch for ch in text.strip())
    cleaned = "_".join(cleaned.split())
    return cleaned or datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_output_dir(base_dir: Path, run_name: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir

def resolve_image_path(path: Path, patterns: tuple[str, ...], label: str) -> Path:
    if path.exists():
        return path

    search_dir = path.parent if path.parent.exists() else path.parent.parent
    if search_dir.exists():
        for pattern in patterns:
            matches = sorted(search_dir.glob(pattern))
            if matches:
                print(f"Warning: {label} path not found, using: {matches[0]}")
                return matches[0]

    raise FileNotFoundError(path)


def resolve_laser_npz_path(laser_path: Path) -> Path | None:
    configured = str(LASER_NPZ_PATH).strip()
    if configured:
        path = Path(configured).resolve()
        if path.exists():
            return path
        print(f"Warning: configured LASER_NPZ_PATH not found: {path}")

    folder = laser_path.parent
    name = laser_path.name
    candidates = []
    for suffix in (
        "_true_scale_0p0504mm_detail_height.png",
        "_true_scale_0p0504mm_absolute_height.png",
        "_true_scale_detail_height.png",
        "_true_scale_absolute_height.png",
        "_detail_height.png",
        "_absolute_height.png",
    ):
        if name.endswith(suffix):
            candidates.append(folder / (name[: -len(suffix)] + "_height_map.npz"))
    candidates.extend(sorted(folder.glob("*_height_map.npz")))

    for path in candidates:
        if path.exists():
            return path
    return None












def laser_valid_mask(laser_strip: Image.Image) -> np.ndarray:
    arr = np.asarray(laser_strip.convert("RGB"), dtype=np.uint8)
    near_white = np.all(arr >= LASER_INVALID_WHITE_THRESHOLD, axis=2)
    mask = ~near_white

    for _ in range(max(0, MASK_DILATE_PIXELS)):
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="edge")
        mask = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return mask








def load_laser_height_value_image(laser_path: Path, laser_size: tuple[int, int]) -> tuple[Image.Image, list[int], Path | None]:
    """Return a normalized 0..255 laser height image and useful contour levels.

    For registration/alignment we prefer the restored laser PNG. It is already an
    unfolded height image, is much smaller than the raw height matrix, and matches
    the visual image-space used by the camera stitch.
    """
    if USE_LASER_RESTORED_PNG_FOR_ALIGNMENT:
        laser_gray = Image.open(laser_path).convert("L")
        if laser_gray.size != laser_size:
            laser_gray = laser_gray.resize(laser_size, Image.Resampling.BILINEAR)
        valid = np.asarray(laser_gray, dtype=np.uint8) < LASER_INVALID_WHITE_THRESHOLD
        levels: list[int] = []
        if np.any(valid):
            values = np.asarray(laser_gray, dtype=np.uint8)[valid]
            for pct in CONTOUR_PERCENTILES:
                level = int(round(float(np.percentile(values, pct))))
                if 4 <= level <= 251 and level not in levels:
                    levels.append(level)
        if not levels:
            levels = [45, 80, 115, 150, 185, 220]
        return laser_gray, levels, None

    npz_path = resolve_laser_npz_path(laser_path)
    if npz_path is None:
        laser_gray = Image.open(laser_path).convert("L")
        if laser_gray.size != laser_size:
            laser_gray = laser_gray.resize(laser_size, Image.Resampling.BILINEAR)
        levels = [45, 80, 115, 150, 185, 220]
        return laser_gray, levels, None

    with np.load(npz_path) as data:
        if "height_mm" not in data:
            raise KeyError(f"{npz_path} does not contain height_mm")
        height_mm = data["height_mm"].astype(np.float32)

    valid = np.isfinite(height_mm) & (height_mm > -90.0)
    if not np.any(valid):
        gray = np.zeros(height_mm.shape, dtype=np.uint8)
        return Image.fromarray(gray, "L").resize(laser_size, Image.Resampling.BILINEAR), [64, 128, 192], npz_path

    lo, hi = np.nanpercentile(height_mm[valid], [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(height_mm[valid]))
        hi = float(np.nanmax(height_mm[valid]))
    if hi <= lo:
        hi = lo + 1.0

    norm = np.zeros(height_mm.shape, dtype=np.float32)
    norm[valid] = np.clip((height_mm[valid] - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0).astype(np.uint8)
    levels = []
    for pct in CONTOUR_PERCENTILES:
        value = float(np.nanpercentile(height_mm[valid], pct))
        level = int(round(np.clip((value - lo) / (hi - lo), 0.0, 1.0) * 255.0))
        if 4 <= level <= 251 and level not in levels:
            levels.append(level)
    if not levels:
        levels = [64, 128, 192]

    image = Image.fromarray(gray, "L")
    if image.size != laser_size:
        image = image.resize(laser_size, Image.Resampling.BILINEAR)
    return image, levels, npz_path











def load_laser_valid_mask_image(laser_path: Path, laser_size: tuple[int, int]) -> tuple[Image.Image, Path | None, float]:
    if USE_LASER_RESTORED_PNG_FOR_ALIGNMENT:
        print(f"Laser valid mask source: restored PNG white-threshold mask ({laser_path})")
        laser = Image.open(laser_path).convert("RGB")
        if laser.size != laser_size:
            laser = laser.resize(laser_size, Image.Resampling.BILINEAR)
        mask = laser_valid_mask(laser)
        valid_fraction = float(np.mean(mask))
        return mask_to_rgb(mask).convert("L"), None, valid_fraction

    npz_path = resolve_laser_npz_path(laser_path)
    if npz_path is None:
        print("Warning: no *_height_map.npz found; using PNG white-threshold mask.")
        laser = Image.open(laser_path).convert("RGB")
        mask = laser_valid_mask(laser)
        valid_fraction = float(np.mean(mask))
        return mask_to_rgb(mask).convert("L"), None, valid_fraction

    print(f"Laser valid mask source: {npz_path}")
    with np.load(npz_path) as data:
        if "height_mm" not in data:
            raise KeyError(f"{npz_path} does not contain height_mm")
        height_mm = data["height_mm"]
        valid = np.isfinite(height_mm) & (height_mm > -90.0)

    valid_fraction_raw = float(np.mean(valid))
    mask_small = Image.fromarray((valid.astype(np.uint8) * 255), "L")
    if mask_small.size != laser_size:
        mask_img = mask_small.resize(laser_size, Image.Resampling.NEAREST)
    else:
        mask_img = mask_small
    return mask_img, npz_path, valid_fraction_raw

def mask_to_rgb(mask: np.ndarray) -> Image.Image:
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask] = [255, 255, 255]
    return Image.fromarray(rgb, "RGB")


























def normalize_01(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    if mask is not None and np.any(mask):
        values = arr[mask]
    else:
        values = arr.reshape(-1)
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def gradient_magnitude(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    gx = np.zeros_like(arr, dtype=np.float32)
    gy = np.zeros_like(arr, dtype=np.float32)
    gx[:, 1:-1] = (arr[:, 2:] - arr[:, :-2]) * 0.5
    gx[:, 0] = arr[:, 1] - arr[:, 0] if arr.shape[1] > 1 else 0.0
    gx[:, -1] = arr[:, -1] - arr[:, -2] if arr.shape[1] > 1 else 0.0
    gy[1:-1, :] = (arr[2:, :] - arr[:-2, :]) * 0.5
    gy[0, :] = arr[1, :] - arr[0, :] if arr.shape[0] > 1 else 0.0
    gy[-1, :] = arr[-1, :] - arr[-2, :] if arr.shape[0] > 1 else 0.0
    return np.sqrt(gx * gx + gy * gy)




def box_blur_01(arr: np.ndarray, radius: int) -> np.ndarray:
    arr = np.clip(arr.astype(np.float32, copy=False), 0.0, 1.0)
    if radius <= 0:
        return arr.copy()
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "L")
    blurred = img.filter(ImageFilter.BoxBlur(radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def local_detail_map(arr: np.ndarray, mask: np.ndarray | None = None, radius: int = 4) -> np.ndarray:
    base = normalize_01(arr, mask)
    if mask is not None and np.any(mask):
        filled = base.copy()
        filled[~mask] = float(np.median(base[mask]))
    else:
        filled = base
    blur = box_blur_01(filled, radius)
    return normalize_01(np.abs(filled - blur), mask)


def laser_feature_scalar(height_scalar: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    base = normalize_01(height_scalar, valid_mask)
    if np.any(valid_mask):
        filled = base.copy()
        filled[~valid_mask] = float(np.median(base[valid_mask]))
    else:
        filled = base
    edge = normalize_01(gradient_magnitude(filled), valid_mask)
    detail = local_detail_map(filled, valid_mask, radius=4)
    feature = 0.20 * base + 0.45 * edge + 0.35 * detail
    if np.any(valid_mask):
        feature = feature.copy()
        feature[~valid_mask] = 0.0
    return normalize_01(feature, valid_mask)


def camera_feature_scalar(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = np.max(arr, axis=2) - np.min(arr, axis=2)
    green_dominance = np.clip(g - np.maximum(r, b), 0.0, 1.0)
    edge = normalize_01(gradient_magnitude(gray))
    detail = local_detail_map(gray, None, radius=4)
    color_texture = normalize_01(0.55 * saturation + 0.45 * green_dominance)
    return normalize_01(0.50 * edge + 0.40 * detail + 0.10 * color_texture)





























































def endface_feature_map(feature: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Enhance round wire end-face candidates for local alignment scoring."""
    if feature.size == 0:
        return feature.astype(np.float32, copy=False)
    mask = np.isfinite(feature)
    if valid_mask is not None:
        mask &= valid_mask.astype(bool, copy=False)
    base = normalize_01(feature, mask if np.any(mask) else None)
    filled = base.astype(np.float32, copy=True)
    if np.any(mask):
        filled[~mask] = float(np.median(filled[mask]))
    else:
        filled[~np.isfinite(filled)] = 0.0

    img = Image.fromarray(np.clip(filled * 255.0, 0, 255).astype(np.uint8), "L")
    small = np.asarray(img.filter(ImageFilter.GaussianBlur(float(END_FACE_SMALL_BLUR_PX))), dtype=np.float32) / 255.0
    large = np.asarray(img.filter(ImageFilter.GaussianBlur(float(END_FACE_LARGE_BLUR_PX))), dtype=np.float32) / 255.0

    # Bright rounded caps become positive in the DoG image. The gradient term keeps
    # cap boundaries useful when the height/color plateau is not very bright.
    dog = np.clip(small - large, 0.0, 1.0)
    edge = normalize_01(gradient_magnitude(filled), mask if np.any(mask) else None)
    blob = normalize_01((1.0 - float(END_FACE_EDGE_WEIGHT)) * dog + float(END_FACE_EDGE_WEIGHT) * edge, mask if np.any(mask) else None)

    if np.any(mask):
        values = blob[mask]
    else:
        values = blob[np.isfinite(blob)]
    if values.size >= 64:
        threshold = float(np.percentile(values, float(END_FACE_THRESHOLD_PERCENTILE)))
        if threshold < 0.98:
            strong = np.clip((blob - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)
            blob = normalize_01(0.35 * blob + 0.65 * strong, mask if np.any(mask) else None)
    if np.any(mask):
        blob = blob.astype(np.float32, copy=True)
        blob[~mask] = 0.0
    return blob.astype(np.float32, copy=False)

























# =============================================================================
# HORIZONTAL CALIBRATION REUSE WITH X SCALE LOCKED TO 1.0
# =============================================================================

def find_horizontal_affine_calibration_json() -> Path | None:
    """Find the newest reusable physical circular affine calibration."""
    explicit = (
        str(FULL_AFFINE_CALIBRATION_JSON_PATH).strip()
        if USE_FULL_PHYSICAL_CIRCULAR_AFFINE
        else str(HORIZONTAL_CALIBRATION_JSON_PATH).strip()
    )
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"HORIZONTAL_CALIBRATION_JSON_PATH not found: {p}"
            )
        return p

    roots: list[Path] = []
    for root in (
        Path(_CONFIG_CALIBRATION_DIR),
        Path(CONE_CALIBRATION_DIR),
        Path(CONE_CALIBRATION_DIR).parent,
    ):
        try:
            root = root.resolve()
        except Exception:
            pass
        if root.exists() and root not in roots:
            roots.append(root)

    candidates: list[Path] = []
    for root in roots:
        try:
            candidates.extend(root.rglob("affine_calibration.json"))
            candidates.extend(root.rglob("affine_calibration_physical_circular.json"))
            candidates.extend(root.rglob("affine_calibration_physical_mm.json"))
        except Exception:
            pass

    valid = []
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            has_points = (
                isinstance(data.get("camera_points_physical"), list)
                and isinstance(data.get("laser_points_physical"), list)
            )
            has_matrix = (
                data.get("camera_physical_to_laser_physical_affine_2x3") is not None
                or data.get("camera_mm_to_laser_mm_affine_2x3") is not None
                or data.get("affine_matrix_2x3") is not None
            )
            center_x0 = bool(data.get("camera_roi_center_is_x0_mm", False))
            if has_points and has_matrix and center_x0:
                valid.append(p)
        except Exception:
            continue

    if not valid:
        return None

    return max(
        valid,
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
    ).resolve()



def load_full_physical_circular_affine_calibration(
    current_laser_size_wh: tuple[int, int],
    current_camera_size_wh: tuple[int, int],
) -> dict:
    """Load the COMPLETE camera-physical -> laser-physical circular affine.

    No affine coefficient is discarded in this model.
    """
    path = find_horizontal_affine_calibration_json()
    if path is None:
        raise FileNotFoundError(
            "Could not find a reusable physical circular affine calibration JSON."
        )

    data = json.loads(path.read_text(encoding="utf-8-sig"))

    matrix_raw = (
        data.get("camera_physical_to_laser_physical_affine_2x3")
        or data.get("camera_mm_to_laser_mm_affine_2x3")
        or data.get("affine_matrix_2x3")
    )
    matrix = np.asarray(matrix_raw, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            f"Invalid 2x3 physical affine matrix in: {path}"
        )

    if not bool(data.get("camera_roi_center_is_x0_mm", False)):
        raise RuntimeError(
            "The selected calibration does not use the required convention: "
            "camera stitched ROI horizontal center = X 0 mm."
        )

    coord = data.get("coordinate_system", {})
    if not isinstance(coord, dict):
        raise RuntimeError(
            f"Missing coordinate_system metadata in: {path}"
        )

    x_meta = coord.get("x", {})
    y_meta = coord.get("circular_y", {})
    if not isinstance(x_meta, dict) or not isinstance(y_meta, dict):
        raise RuntimeError(
            "The calibration JSON must contain coordinate_system.x and "
            "coordinate_system.circular_y."
        )

    def _finite_float(value, name: str) -> float:
        try:
            f = float(value)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid {name} in calibration JSON: {value!r}"
            ) from exc
        if not np.isfinite(f):
            raise RuntimeError(
                f"Non-finite {name} in calibration JSON."
            )
        return f

    camera_mm_per_px = _finite_float(
        x_meta.get("camera_x_mm_per_pixel"),
        "camera_x_mm_per_pixel",
    )
    laser_x_min_mm = _finite_float(
        x_meta.get("laser_x_min_mm"),
        "laser_x_min_mm",
    )
    laser_x_max_mm = _finite_float(
        x_meta.get("laser_x_max_mm"),
        "laser_x_max_mm",
    )
    circumference_mm = _finite_float(
        y_meta.get("circumference_mm"),
        "circumference_mm",
    )
    period_deg = _finite_float(
        y_meta.get("period_deg", 360.0),
        "period_deg",
    )
    wheel_diameter_mm = _finite_float(
        y_meta.get("wheel_diameter_mm"),
        "wheel_diameter_mm",
    )

    if camera_mm_per_px <= 0:
        raise RuntimeError("camera_x_mm_per_pixel must be positive.")
    if laser_x_max_mm <= laser_x_min_mm:
        raise RuntimeError("laser_x_max_mm must be larger than laser_x_min_mm.")
    if circumference_mm <= 0 or period_deg <= 0:
        raise RuntimeError("Invalid circular calibration metadata.")

    laser_w, laser_h = [int(v) for v in current_laser_size_wh]
    camera_w, camera_h = [int(v) for v in current_camera_size_wh]
    if laser_w < 2 or laser_h < 2 or camera_w < 2 or camera_h < 2:
        raise RuntimeError(
            "Current camera/laser images are too small for physical mapping."
        )

    camera_zero_px = (camera_w - 1.0) / 2.0
    laser_mm_per_px_current = (
        (laser_x_max_mm - laser_x_min_mm)
        / float(laser_w - 1)
    )

    created_at = data.get("created_at")
    saved_camera_size = data.get("camera_image_size_px")
    saved_laser_size = data.get("laser_image_size_px")

    print("Full physical circular affine calibration:")
    print(f"  JSON: {path}")
    if created_at:
        print(f"  created_at: {created_at}")
    print("  matrix:")
    print(matrix)
    print(
        f"  camera X scale = {camera_mm_per_px:.9f} mm/px; "
        f"current ROI center = {camera_zero_px:.3f}px -> X=0 mm"
    )
    print(
        f"  laser X physical range = "
        f"{laser_x_min_mm:.6f}..{laser_x_max_mm:.6f} mm; "
        f"current native step = {laser_mm_per_px_current:.9f} mm/px"
    )
    print(
        f"  circular Y = {period_deg:.6f} deg, "
        f"circumference = {circumference_mm:.6f} mm"
    )
    if saved_camera_size is not None or saved_laser_size is not None:
        print(
            f"  saved sizes: camera={saved_camera_size}, laser={saved_laser_size}; "
            f"current sizes: camera={[camera_w, camera_h]}, laser={[laser_w, laser_h]}"
        )

    return {
        "calibration_json": str(path),
        "created_at": created_at,
        "matrix_2x3": matrix,
        "matrix_2x3_list": matrix.tolist(),
        "camera_mm_per_px": float(camera_mm_per_px),
        "camera_zero_px": float(camera_zero_px),
        "laser_x_min_mm": float(laser_x_min_mm),
        "laser_x_max_mm": float(laser_x_max_mm),
        "laser_mm_per_px_current": float(laser_mm_per_px_current),
        "circumference_mm": float(circumference_mm),
        "period_deg": float(period_deg),
        "wheel_diameter_mm": float(wheel_diameter_mm),
        "saved_camera_size_px": saved_camera_size,
        "saved_laser_size_px": saved_laser_size,
        "current_camera_size_px": [camera_w, camera_h],
        "current_laser_size_px": [laser_w, laser_h],
        "fit_method": data.get("fit_method"),
        "rmse_mm": data.get("rmse_mm"),
        "max_error_mm": data.get("max_error_mm"),
        "all_point_rmse_mm": data.get("all_point_rmse_mm"),
        "all_point_max_error_mm": data.get("all_point_max_error_mm"),
        "inlier_count": data.get("inlier_count"),
        "transform_direction": data.get("transform_direction"),
        "coordinate_system": coord,
        "source_json": data,
    }


def build_full_affine_camera_to_laser_maps(
    camera_size_wh: tuple[int, int],
    laser_size_wh: tuple[int, int],
    calibration: dict,
) -> dict:
    """Build exact 2-D inverse-sampling maps on the camera final canvas.

    For each camera output pixel:
      camera pixel -> camera physical [X_mm, unwrapped arc_mm]
                   -> FULL 2x3 physical affine
                   -> laser physical [X_mm, base arc_mm]
                   -> current laser pixel [x_px, y_px]

    Around the circular seam, camera branches k=-1,0,+1 are tested and the
    branch whose predicted laser arc falls in/nearest the base 0..C interval
    is selected.  This is the inverse-sampling counterpart of the calibration
    program's three-branch circular debug warp.
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for full physical affine fusion."
        )

    camera_w, camera_h = [int(v) for v in camera_size_wh]
    laser_w, laser_h = [int(v) for v in laser_size_wh]

    matrix = np.asarray(
        calibration["matrix_2x3"],
        dtype=np.float64,
    )
    camera_mm_per_px = float(calibration["camera_mm_per_px"])
    camera_zero_px = float(calibration["camera_zero_px"])
    laser_x_min = float(calibration["laser_x_min_mm"])
    laser_x_max = float(calibration["laser_x_max_mm"])
    circumference = float(calibration["circumference_mm"])

    # Broadcasted current-camera physical coordinates.
    camera_x_mm = (
        np.arange(camera_w, dtype=np.float64)[None, :]
        - camera_zero_px
    ) * camera_mm_per_px

    camera_arc_base_mm = (
        np.arange(camera_h, dtype=np.float64)[:, None]
        / float(camera_h)
        * circumference
    )

    best_cost = np.full(
        (camera_h, camera_w),
        np.inf,
        dtype=np.float64,
    )
    best_laser_x_mm = np.zeros(
        (camera_h, camera_w),
        dtype=np.float64,
    )
    best_laser_arc_mm = np.zeros(
        (camera_h, camera_w),
        dtype=np.float64,
    )
    best_branch = np.zeros(
        (camera_h, camera_w),
        dtype=np.int8,
    )

    # Full physical affine coefficients.
    a00, a01, tx = [float(v) for v in matrix[0]]
    a10, a11, ty = [float(v) for v in matrix[1]]

    for branch_k in FULL_AFFINE_CAMERA_BRANCHES:
        camera_arc = (
            camera_arc_base_mm
            + float(branch_k) * circumference
        )

        laser_x_mm = (
            a00 * camera_x_mm
            + a01 * camera_arc
            + tx
        )
        laser_arc_mm = (
            a10 * camera_x_mm
            + a11 * camera_arc
            + ty
        )

        # The calibration is camera-unwrapped -> laser BASE branch.
        # Choose the equivalent camera branch whose predicted laser arc lies
        # in, or is closest to, [0, circumference).
        arc_outside_distance = np.where(
            laser_arc_mm < 0.0,
            -laser_arc_mm,
            np.where(
                laser_arc_mm >= circumference,
                laser_arc_mm - circumference,
                0.0,
            ),
        )

        # Tiny deterministic tie-breaker only; it never competes with a real
        # outside-interval distance.
        cost = (
            arc_outside_distance
            + 1e-12 * np.abs(
                laser_arc_mm - 0.5 * circumference
            )
        )

        update = cost < best_cost
        best_cost[update] = cost[update]
        best_laser_x_mm[update] = laser_x_mm[update]
        best_laser_arc_mm[update] = laser_arc_mm[update]
        best_branch[update] = np.int8(branch_k)

    map_x = (
        (best_laser_x_mm - laser_x_min)
        / (laser_x_max - laser_x_min)
        * float(laser_w - 1)
    ).astype(np.float32)

    map_y = np.mod(
        best_laser_arc_mm
        / circumference
        * float(laser_h),
        float(laser_h),
    ).astype(np.float32)

    valid_x = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= float(laser_w - 1))
    )

    # Invalid X must not accidentally sample a border pixel.
    map_x_safe = map_x.copy()
    map_y_safe = map_y.copy()
    map_x_safe[~valid_x] = -1000.0
    map_y_safe[~valid_x] = -1000.0

    unique_k, counts_k = np.unique(
        best_branch,
        return_counts=True,
    )
    branch_counts = {
        str(int(k)): int(v)
        for k, v in zip(unique_k, counts_k)
    }

    print("Full-affine current camera->laser map:")
    print(
        f"  valid X fraction = {float(np.mean(valid_x)):.3%}"
    )
    print(f"  selected circular branches = {branch_counts}")
    print(
        f"  mapped laser X range = "
        f"{float(np.nanmin(map_x)):.3f}..{float(np.nanmax(map_x)):.3f} px"
    )

    return {
        "map_x": map_x_safe,
        "map_y": map_y_safe,
        "map_x_raw": map_x,
        "map_y_raw": map_y,
        "valid_x": valid_x,
        "camera_branch_k": best_branch,
        "branch_counts": branch_counts,
        "max_branch_outside_distance_mm": float(
            np.nanmax(best_cost)
        ),
    }


def _vertical_wrap_extend_image_array(src: np.ndarray) -> np.ndarray:
    """Add one wrapped row above and below, but NEVER wrap X."""
    arr = np.asarray(src)
    return np.concatenate(
        [arr[-1:, ...], arr, arr[:1, ...]],
        axis=0,
    )


def remap_pil_full_affine(
    source: Image.Image,
    maps: dict,
    *,
    fill,
    nearest: bool,
) -> Image.Image:
    """Sample a laser image on the camera canvas with periodic Y and bounded X."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required.")

    mode = "RGB" if source.mode == "RGB" else "L"
    src = np.asarray(
        source.convert(mode),
        dtype=np.uint8,
    )
    src_ext = _vertical_wrap_extend_image_array(src)

    map_x = np.asarray(
        maps["map_x"],
        dtype=np.float32,
    )
    map_y = (
        np.asarray(maps["map_y"], dtype=np.float32)
        + 1.0
    )

    interpolation = (
        cv2.INTER_NEAREST
        if nearest
        else cv2.INTER_LINEAR
    )

    if mode == "RGB":
        if isinstance(fill, str):
            border = (
                (255, 255, 255)
                if fill.lower() == "white"
                else (0, 0, 0)
            )
        else:
            fv = np.asarray(fill).reshape(-1)
            border = tuple(
                int(v)
                for v in (
                    np.repeat(fv, 3)
                    if fv.size == 1
                    else fv[:3]
                )
            )
    else:
        border = int(
            fill
            if not isinstance(fill, tuple)
            else fill[0]
        )

    out = cv2.remap(
        src_ext,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return Image.fromarray(out, mode)


def remap_numeric_full_affine_nanaware(
    values: np.ndarray,
    maps: dict,
) -> np.ndarray:
    """NaN-aware bilinear full-affine sampling with circular Y."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required.")

    src = np.asarray(values, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(
            f"Expected 2-D numeric laser grid, got {src.shape}"
        )

    finite = np.isfinite(src)
    data = np.nan_to_num(
        src,
        nan=0.0,
    ).astype(np.float32)
    weight = finite.astype(np.float32)

    data_ext = _vertical_wrap_extend_image_array(data)
    weight_ext = _vertical_wrap_extend_image_array(weight)

    map_x = np.asarray(
        maps["map_x"],
        dtype=np.float32,
    )
    map_y = (
        np.asarray(maps["map_y"], dtype=np.float32)
        + 1.0
    )

    numerator = cv2.remap(
        data_ext,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    denominator = cv2.remap(
        weight_ext,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    out = np.full(
        map_x.shape,
        np.nan,
        dtype=np.float32,
    )
    good = (
        denominator > 1e-4
    ) & np.asarray(maps["valid_x"], dtype=bool)
    out[good] = numerator[good] / denominator[good]
    return out


def save_full_affine_correspondence_maps(
    maps: dict,
    calibration: dict,
    output_path: Path,
) -> Path | None:
    if not SAVE_FULL_AFFINE_CORRESPONDENCE_MAPS:
        return None

    path = output_path.with_name(
        output_path.stem
        + "_full_affine_correspondence_maps.npz"
    )
    np.savez_compressed(
        path,
        camera_to_laser_x_px=np.asarray(
            maps["map_x_raw"],
            dtype=np.float32,
        ),
        camera_to_laser_y_px=np.asarray(
            maps["map_y_raw"],
            dtype=np.float32,
        ),
        camera_branch_k=np.asarray(
            maps["camera_branch_k"],
            dtype=np.int8,
        ),
        affine_base_valid_x_mask=np.asarray(
            maps["valid_x"],
            dtype=np.uint8,
        ),
        physical_affine_2x3=np.asarray(
            calibration["matrix_2x3"],
            dtype=np.float64,
        ),
        camera_x_mm_per_pixel=np.float64(
            calibration["camera_mm_per_px"]
        ),
        laser_x_min_mm=np.float64(
            calibration["laser_x_min_mm"]
        ),
        laser_x_max_mm=np.float64(
            calibration["laser_x_max_mm"]
        ),
        circumference_mm=np.float64(
            calibration["circumference_mm"]
        ),
        calibration_json=np.array(
            calibration["calibration_json"]
        ),
        note=np.array(
            "Maps describe the base full physical circular affine before "
            "any optional post-affine wire-center residual translation."
        ),
    )
    return path







# =============================================================================
# PHYSICAL X=0 ORIGIN CALIBRATION HELPERS
# =============================================================================























# =============================================================================
# PRECISION REFINEMENT HELPERS
# =============================================================================























# =============================================================================
# REAL NUMERIC HEIGHT -> FINAL CAMERA COORDINATE SYSTEM
# =============================================================================

def resolve_numeric_height_npz_for_mapped_output(
    laser_path: Path,
) -> Path | None:
    """Find the REAL numeric height-map NPZ matching the restored laser PNG."""
    direct = resolve_laser_npz_path(laser_path)
    if direct is not None and direct.exists():
        return direct.resolve()

    # Additional conservative search in nearby folders.
    laser_path = Path(laser_path).resolve()
    stem = laser_path.stem
    base = stem
    for suffix in (
        "_true_scale_0p0504mm_detail_height",
        "_true_scale_0p0504mm_absolute_height",
        "_true_scale_detail_height",
        "_true_scale_absolute_height",
        "_detail_height",
        "_absolute_height",
    ):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    expected_name = f"{base}_height_map.npz"

    roots = [
        laser_path.parent,
        laser_path.parent.parent,
    ]
    candidates: list[Path] = []

    for root in roots:
        if not root.exists():
            continue
        exact = root / expected_name
        if exact.exists():
            return exact.resolve()
        try:
            candidates.extend(root.glob("*_height_map.npz"))
        except Exception:
            pass

    if candidates:
        # Prefer the newest only as a final fallback.
        return max(
            candidates,
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        ).resolve()

    return None


def _nearest_resample_numeric_grid(
    arr: np.ndarray,
    target_size_wh: tuple[int, int],
) -> np.ndarray:
    """Nearest-neighbor array resampling matching the restored true-scale PNG.

    This is used only to move the original numeric matrix onto the already
    established laser image grid. It does NOT stretch the laser to the camera.
    """
    src = np.asarray(arr, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(
            f"Numeric laser height must be 2-D, got shape {src.shape}"
        )

    target_w, target_h = [int(v) for v in target_size_wh]
    src_h, src_w = src.shape

    if (src_w, src_h) == (target_w, target_h):
        return src.copy()

    # PIL-style nearest sample using pixel-center mapping.
    xx = np.floor(
        (np.arange(target_w, dtype=np.float64) + 0.5)
        * float(src_w)
        / max(float(target_w), 1.0)
    ).astype(np.int64)
    yy = np.floor(
        (np.arange(target_h, dtype=np.float64) + 0.5)
        * float(src_h)
        / max(float(target_h), 1.0)
    ).astype(np.int64)

    xx = np.clip(xx, 0, src_w - 1)
    yy = np.clip(yy, 0, src_h - 1)

    return src[np.ix_(yy, xx)].astype(
        np.float32,
        copy=False,
    )


def load_numeric_laser_height_on_visual_grid(
    laser_path: Path,
    laser_size_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None, Path, dict]:
    """Load real height_mm/residual_mm and put them on the laser PNG grid."""
    npz_path = resolve_numeric_height_npz_for_mapped_output(
        laser_path
    )

    if npz_path is None:
        message = (
            "Could not find the source numeric *_height_map.npz needed to "
            "write mapped_height_mm.npz.\n"
            f"Laser image: {laser_path}\n"
            "Set LASER_NPZ_PATH explicitly to the matching height-map NPZ."
        )
        if REQUIRE_NUMERIC_HEIGHT_NPZ_FOR_MAPPED_OUTPUT:
            raise FileNotFoundError(message)
        print("WARNING:", message)
        raise RuntimeError(message)

    with np.load(npz_path, allow_pickle=False) as data:
        if "height_mm" not in data:
            raise KeyError(
                f"{npz_path} does not contain numeric height_mm"
            )

        source_height = np.asarray(
            data["height_mm"],
            dtype=np.float32,
        )

        source_residual = (
            np.asarray(
                data["residual_mm"],
                dtype=np.float32,
            )
            if "residual_mm" in data
            else None
        )

        source_meta = {
            "wheel_diameter_mm": (
                float(data["wheel_diameter_mm"])
                if "wheel_diameter_mm" in data
                else None
            ),
            "rotation_deg": (
                float(data["rotation_deg"])
                if "rotation_deg" in data
                else 360.0
            ),
            "true_scale_mm_per_pixel": (
                float(data["true_scale_mm_per_pixel"])
                if "true_scale_mm_per_pixel" in data
                else None
            ),
            "laser_width_mm": (
                float(data["laser_width_mm"])
                if "laser_width_mm" in data
                else None
            ),
        }

    # Preserve invalid values as NaN.
    source_height = source_height.astype(
        np.float32,
        copy=False,
    )
    source_height[
        ~np.isfinite(source_height)
        | (source_height <= -90.0)
    ] = np.nan

    if source_residual is not None:
        source_residual = source_residual.astype(
            np.float32,
            copy=False,
        )
        source_residual[
            ~np.isfinite(source_residual)
            | (source_residual <= -90.0)
        ] = np.nan

    target_height = _nearest_resample_numeric_grid(
        source_height,
        laser_size_wh,
    )
    target_residual = (
        _nearest_resample_numeric_grid(
            source_residual,
            laser_size_wh,
        )
        if source_residual is not None
        else None
    )

    info = {
        "source_npz": str(npz_path),
        "source_shape_rows_cols": [
            int(source_height.shape[0]),
            int(source_height.shape[1]),
        ],
        "laser_visual_grid_size_wh": [
            int(laser_size_wh[0]),
            int(laser_size_wh[1]),
        ],
        "resampling_to_laser_visual_grid": (
            "nearest_neighbor_matching_true_scale_png_grid"
        ),
        **source_meta,
    }

    return (
        target_height,
        target_residual,
        npz_path,
        info,
    )






def _save_mapped_height_preview_png(
    height_mm: np.ndarray,
    valid_mask: np.ndarray,
    out_path: Path,
) -> None:
    if not SAVE_MAPPED_HEIGHT_MM_PREVIEW:
        return

    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(height_mm)
    )
    preview = np.zeros(
        height_mm.shape,
        dtype=np.uint8,
    )

    if np.any(valid):
        values = height_mm[valid]
        lo, hi = np.percentile(
            values,
            [
                float(MAPPED_HEIGHT_PREVIEW_LOW_PERCENTILE),
                float(MAPPED_HEIGHT_PREVIEW_HIGH_PERCENTILE),
            ],
        )
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(values))
            hi = float(np.nanmax(values))
        if hi <= lo:
            hi = lo + 1.0

        preview[valid] = np.clip(
            (
                height_mm[valid]
                - float(lo)
            )
            / float(hi - lo)
            * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

    # Invalid = white, matching the laser visualization convention.
    preview[~valid] = 255
    Image.fromarray(
        preview,
        "L",
    ).save(out_path)



# =============================================================================
# FINAL 2-D RESIDUAL X/Y ALIGNMENT
# =============================================================================






















# =============================================================================
# WIRE-END CENTER POINT-SET ALIGNMENT + RANSAC
# =============================================================================

def _wire_center_normalize(arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    values = a[mask] if mask is not None and np.any(mask) else a[np.isfinite(a)]
    if values.size < 32:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(values, [2.0, 98.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    out = np.clip((a - float(lo)) / float(hi - lo), 0.0, 1.0)
    if mask is not None:
        out = np.where(mask, out, 0.0)
    return out.astype(np.float32)


def _wire_center_score_map(
    scalar: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    """Bright local-cap score used only to locate candidate wire centers."""
    base = _wire_center_normalize(scalar, valid_mask)
    small = cv2.GaussianBlur(
        base,
        (0, 0),
        float(WIRE_CENTER_LOCAL_SIGMA_SMALL),
    )
    large = cv2.GaussianBlur(
        base,
        (0, 0),
        float(WIRE_CENTER_LOCAL_SIGMA_LARGE),
    )
    dog = np.maximum(small - large, 0.0)
    dog = _wire_center_normalize(dog, valid_mask)

    # The small-blur brightness term keeps flat-topped wire ends detectable;
    # DoG suppresses slowly varying illumination/background.
    score = _wire_center_normalize(
        0.72 * dog + 0.28 * small,
        valid_mask,
    )
    if valid_mask is not None:
        score[~valid_mask] = 0.0
    return score.astype(np.float32)


def _dilate_bool_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    k = int(radius) * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def _detect_wire_end_centers(
    scalar: np.ndarray,
    valid_mask: np.ndarray | None,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wire-center candidates with local shape metadata.

    Columns are:
        x, y, strength, inscribed_radius_px, center_score,
        component_area_px, component_aspect, component_fill

    A thresholded local-cap score is converted to a distance transform.  Local
    maxima of that distance field provide one stable point near each end face,
    including touching/packed wires better than raw connected-component centroids.
    """
    score = _wire_center_score_map(scalar, valid_mask)
    values = score[valid_mask] if valid_mask is not None and np.any(valid_mask) else score.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 64:
        return np.empty((0, 8), dtype=np.float64), score, np.zeros_like(score, dtype=np.uint8)

    threshold = float(np.percentile(values, float(percentile)))
    binary = (score >= threshold).astype(np.uint8)
    if valid_mask is not None:
        binary[~valid_mask] = 0

    kernel3 = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=1)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    r = max(2, int(WIRE_CENTER_NMS_RADIUS_PX))
    nms_kernel = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
    local_max = cv2.dilate(dist, nms_kernel)
    peaks = (
        (dist >= local_max - 1e-6)
        & (dist >= float(WIRE_CENTER_MIN_DISTANCE_VALUE_PX))
    )
    if valid_mask is not None:
        peaks &= valid_mask

    # Shape metadata is measured on the thresholded cap component.  It is not
    # trusted as an exact segmentation; it is only used as a confidence gate.
    comp_n, comp_labels, comp_stats, _comp_centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=8,
    )

    # Collapse plateau maxima to one point before greedy NMS.
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        peaks.astype(np.uint8),
        connectivity=8,
    )
    raw: list[tuple[float, float, float, float, float, float, float, float]] = []
    h, w = score.shape
    x_margin = max(0, int(WIRE_CENTER_EDGE_MARGIN_X_PX))
    for label in range(1, n):
        cx, cy = centroids[label]
        if not np.isfinite(cx) or not np.isfinite(cy):
            continue
        if cx < x_margin or cx > (w - 1 - x_margin):
            continue
        xi = int(np.clip(round(cx), 0, w - 1))
        yi = int(np.clip(round(cy), 0, h - 1))
        radius_px = float(dist[yi, xi])
        center_score = float(score[yi, xi])
        strength = radius_px + 0.35 * center_score
        comp_id = int(comp_labels[yi, xi]) if 0 <= yi < h and 0 <= xi < w else 0
        if 0 < comp_id < comp_n:
            area = float(comp_stats[comp_id, cv2.CC_STAT_AREA])
            bw = float(comp_stats[comp_id, cv2.CC_STAT_WIDTH])
            bh = float(comp_stats[comp_id, cv2.CC_STAT_HEIGHT])
            aspect = float(min(bw, bh) / max(bw, bh, 1.0))
            fill = float(area / max(bw * bh, 1.0))
        else:
            area, aspect, fill = 0.0, 0.0, 0.0
        raw.append((
            float(cx), float(cy), strength, radius_px, center_score,
            area, aspect, fill,
        ))

    if not raw:
        return np.empty((0, 8), dtype=np.float64), score, binary

    raw.sort(key=lambda item: item[2], reverse=True)
    suppress = np.zeros((h, w), dtype=np.uint8)
    points: list[tuple[float, ...]] = []
    max_points = max(100, int(WIRE_CENTER_MAX_POINTS))
    for item in raw:
        cx, cy, strength = item[:3]
        xi = int(np.clip(round(cx), 0, w - 1))
        yi = int(np.clip(round(cy), 0, h - 1))
        if suppress[yi, xi]:
            continue
        points.append(tuple(float(v) for v in item))
        cv2.circle(suppress, (xi, yi), r, 1, -1)
        if len(points) >= max_points:
            break

    return np.asarray(points, dtype=np.float64), score, binary








def _build_spatial_hash(points_xy: np.ndarray, cell_size: float) -> dict:
    cell = max(1.0, float(cell_size))
    table: dict[tuple[int, int], list[int]] = {}
    for i, (x, y) in enumerate(np.asarray(points_xy, dtype=np.float64)):
        key = (int(math.floor(x / cell)), int(math.floor(y / cell)))
        table.setdefault(key, []).append(i)
    return {'cell': cell, 'table': table}




















def _periodic_extended_points(points_xy: np.ndarray, height: int):
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.int32)
    ext = np.vstack([
        np.column_stack([pts[:, 0], pts[:, 1] - height]),
        pts,
        np.column_stack([pts[:, 0], pts[:, 1] + height]),
    ])
    orig = np.tile(np.arange(len(pts), dtype=np.int32), 3)
    return ext, orig


def _query_spatial_candidates(
    table: dict,
    cell: float,
    qx: float,
    qy: float,
    radius: float,
) -> list[int]:
    reach = max(1, int(math.ceil(float(radius) / max(float(cell), 1e-9))))
    bx = int(math.floor(float(qx) / cell))
    by = int(math.floor(float(qy) / cell))
    out: list[int] = []
    for gx in range(bx - reach, bx + reach + 1):
        for gy in range(by - reach, by + reach + 1):
            out.extend(table.get((gx, gy), []))
    return out


def _robust_wire_spacing(points_xy: np.ndarray, height: int) -> float:
    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 8:
        return 8.0
    search_radius = max(18.0, float(WIRE_CENTER_NMS_RADIUS_PX) * 5.0)
    ext, orig = _periodic_extended_points(pts, height)
    spatial = _build_spatial_hash(ext, max(4.0, search_radius / 2.0))
    cell, table = spatial['cell'], spatial['table']
    nearest = []
    # Deterministic sub-sample if there are many centers.
    if len(pts) > 6000:
        indices = np.unique(np.linspace(0, len(pts)-1, 6000, dtype=np.int32))
    else:
        indices = np.arange(len(pts), dtype=np.int32)
    for i in indices:
        x, y = pts[int(i)]
        best = float('inf')
        for ei in _query_spatial_candidates(table, cell, x, y, search_radius):
            oi = int(orig[ei])
            if oi == int(i):
                continue
            dx = float(ext[ei, 0] - x)
            dy = float(ext[ei, 1] - y)
            d = math.hypot(dx, dy)
            if 1e-6 < d < best and d <= search_radius:
                best = d
        if np.isfinite(best):
            nearest.append(best)
    if len(nearest) < 8:
        return 8.0
    spacing = float(np.median(np.asarray(nearest, dtype=np.float64)))
    return float(np.clip(spacing, 3.0, 24.0))























def _cap_boundary_mask(
    binary: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    """Inner boundary of thresholded wire-end caps."""
    b = (np.asarray(binary, dtype=np.uint8) > 0)
    if valid_mask is not None:
        b &= np.asarray(valid_mask, dtype=bool)

    if not np.any(b):
        return np.zeros(b.shape, dtype=bool)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(
        b.astype(np.uint8),
        kernel,
        iterations=1,
    ) > 0
    edge = b & (~eroded)

    if valid_mask is not None:
        edge &= np.asarray(valid_mask, dtype=bool)

    return edge


def _edge_distance_map(edge_mask: np.ndarray) -> np.ndarray:
    """Distance in pixels to the nearest edge pixel."""
    edge = np.asarray(edge_mask, dtype=bool)
    canvas = np.ones(edge.shape, dtype=np.uint8)
    canvas[edge] = 0
    return cv2.distanceTransform(
        canvas,
        cv2.DIST_L2,
        5,
    ).astype(np.float32)


def _edge_points_for_y_segment(
    edge_mask: np.ndarray,
    y0: int,
    y1: int,
    max_points: int,
) -> np.ndarray:
    """Deterministically sample edge points inside one vertical segment."""
    edge = np.asarray(edge_mask, dtype=bool)
    h, w = edge.shape
    lo = max(0, int(y0))
    hi = min(h, int(y1))
    if hi <= lo:
        return np.empty((0, 2), dtype=np.float64)

    yy, xx = np.nonzero(edge[lo:hi])
    if xx.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    yy = yy.astype(np.float64) + float(lo)
    xx = xx.astype(np.float64)

    limit = max(1, int(max_points))
    if xx.size > limit:
        idx = np.unique(
            np.linspace(
                0,
                xx.size - 1,
                limit,
                dtype=np.int64,
            )
        )
        xx = xx[idx]
        yy = yy[idx]

    return np.column_stack([xx, yy]).astype(np.float64)


def _sample_distance_map_x_only(
    distance_map: np.ndarray,
    points_xy: np.ndarray,
    dx_px: float,
) -> np.ndarray:
    """Bilinear X-only sampling of a distance map at shifted point positions."""
    dist = np.asarray(distance_map, dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float64)

    if pts.size == 0:
        return np.empty((0,), dtype=np.float32)

    h, w = dist.shape

    x = pts[:, 0] + float(dx_px)
    y = np.rint(pts[:, 1]).astype(np.int64)

    valid = (
        np.isfinite(x)
        & (x >= 0.0)
        & (x <= float(w - 1))
        & (y >= 0)
        & (y < h)
    )
    if not np.any(valid):
        return np.empty((0,), dtype=np.float32)

    xv = x[valid]
    yv = y[valid]

    x0 = np.floor(xv).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    t = (xv - x0).astype(np.float32)

    a = dist[yv, x0]
    b = dist[yv, x1]
    return (
        a * (1.0 - t)
        + b * t
    ).astype(np.float32)


def _symmetric_chamfer_cost_x_only(
    laser_edge_points: np.ndarray,
    camera_edge_points: np.ndarray,
    camera_distance_map: np.ndarray,
    laser_distance_map: np.ndarray,
    dx_px: float,
) -> tuple[float, dict]:
    """Symmetric edge Chamfer cost for one X translation.

    Laser edges are shifted by +dx when compared with camera edges.
    The reverse term samples the laser distance map at camera_x - dx.
    """
    laser_to_camera = _sample_distance_map_x_only(
        camera_distance_map,
        laser_edge_points,
        float(dx_px),
    )
    camera_to_laser = _sample_distance_map_x_only(
        laser_distance_map,
        camera_edge_points,
        -float(dx_px),
    )

    min_required = max(
        24,
        int(FULL_AFFINE_CHAMFER_MIN_EDGE_POINTS_PER_SEGMENT // 4),
    )
    if (
        laser_to_camera.size < min_required
        or camera_to_laser.size < min_required
    ):
        return float("inf"), {
            "valid": False,
            "laser_to_camera_count": int(laser_to_camera.size),
            "camera_to_laser_count": int(camera_to_laser.size),
        }

    clip = max(
        0.5,
        float(FULL_AFFINE_CHAMFER_DISTANCE_CLIP_PX),
    )
    close_d = max(
        0.1,
        float(FULL_AFFINE_CHAMFER_CLOSE_DISTANCE_PX),
    )

    l_clip = np.minimum(laser_to_camera, clip)
    c_clip = np.minimum(camera_to_laser, clip)

    mean_distance = 0.5 * (
        float(np.mean(l_clip))
        + float(np.mean(c_clip))
    )
    close_fraction = 0.5 * (
        float(np.mean(laser_to_camera <= close_d))
        + float(np.mean(camera_to_laser <= close_d))
    )

    # Lower is better.
    cost = (
        mean_distance
        + float(FULL_AFFINE_CHAMFER_CLOSE_FRACTION_WEIGHT)
        * (1.0 - close_fraction)
    )

    return float(cost), {
        "valid": True,
        "mean_clipped_distance_px": float(mean_distance),
        "close_fraction": float(close_fraction),
        "laser_to_camera_count": int(laser_to_camera.size),
        "camera_to_laser_count": int(camera_to_laser.size),
    }


def _wire_pitch_from_detected_centers(
    laser_pts: np.ndarray,
    camera_pts: np.ndarray,
    height: int,
) -> dict:
    """Robust common wire pitch from both modalities."""
    laser_pitch = _robust_wire_spacing(
        laser_pts[:, :2]
        if len(laser_pts)
        else np.empty((0, 2), dtype=np.float64),
        int(height),
    )
    camera_pitch = _robust_wire_spacing(
        camera_pts[:, :2]
        if len(camera_pts)
        else np.empty((0, 2), dtype=np.float64),
        int(height),
    )

    pitches = np.asarray(
        [laser_pitch, camera_pitch],
        dtype=np.float64,
    )
    pitches = pitches[np.isfinite(pitches)]

    if pitches.size == 0:
        pitch = 8.0
    else:
        pitch = float(np.median(pitches))

    pitch = float(
        np.clip(
            pitch,
            float(FULL_AFFINE_CHAMFER_MIN_PITCH_PX),
            float(FULL_AFFINE_CHAMFER_MAX_PITCH_PX),
        )
    )

    return {
        "laser_pitch_px": float(laser_pitch),
        "camera_pitch_px": float(camera_pitch),
        "common_pitch_px": float(pitch),
    }


def _phase_candidate_grid(
    pitch_px: float,
) -> list[tuple[int, float]]:
    """Coarse X candidates labelled by integer wire phase k."""
    k0 = int(FULL_AFFINE_CHAMFER_PHASE_MIN_K)
    k1 = int(FULL_AFFINE_CHAMFER_PHASE_MAX_K)
    radius = max(
        0.0,
        float(FULL_AFFINE_CHAMFER_LOCAL_RADIUS_PX),
    )
    step = max(
        0.05,
        float(FULL_AFFINE_CHAMFER_COARSE_STEP_PX),
    )

    offsets = np.arange(
        -radius,
        radius + 0.5 * step,
        step,
        dtype=np.float64,
    )

    candidates: list[tuple[int, float]] = []
    for k in range(k0, k1 + 1):
        base = float(k) * float(pitch_px)
        for local in offsets:
            candidates.append(
                (int(k), float(base + local))
            )
    return candidates


def _search_chamfer_segment_phase(
    laser_points: np.ndarray,
    camera_points: np.ndarray,
    camera_dist: np.ndarray,
    laser_dist: np.ndarray,
    pitch_px: float,
    threshold_percentile: float,
    segment_index: int,
    search_rows: list[dict],
) -> dict:
    """Select integer wire phase and coarse dx for one vertical segment."""
    best = None

    for k, dx in _phase_candidate_grid(pitch_px):
        cost, detail = _symmetric_chamfer_cost_x_only(
            laser_points,
            camera_points,
            camera_dist,
            laser_dist,
            float(dx),
        )

        row = {
            "threshold_percentile": float(threshold_percentile),
            "segment_index": int(segment_index),
            "stage": "phase_coarse",
            "phase_k": int(k),
            "dx_px": float(dx),
            "cost": (
                None if not np.isfinite(cost) else float(cost)
            ),
            "mean_clipped_distance_px": detail.get(
                "mean_clipped_distance_px"
            ),
            "close_fraction": detail.get("close_fraction"),
            "laser_edge_point_count": int(len(laser_points)),
            "camera_edge_point_count": int(len(camera_points)),
            "valid": bool(detail.get("valid", False)),
        }
        search_rows.append(row)

        if (
            not bool(detail.get("valid", False))
            or not np.isfinite(cost)
        ):
            continue

        candidate = {
            "phase_k": int(k),
            "dx_px": float(dx),
            "cost": float(cost),
            **detail,
        }
        if (
            best is None
            or candidate["cost"] < best["cost"]
        ):
            best = candidate

    if best is None:
        return {
            "ok": False,
            "reason": "no_valid_chamfer_candidate",
            "segment_index": int(segment_index),
        }

    return {
        "ok": True,
        "reason": "ok",
        "segment_index": int(segment_index),
        **best,
    }


def _aggregate_chamfer_cost_for_segments(
    segment_data: list[dict],
    camera_dist: np.ndarray,
    laser_dist: np.ndarray,
    dx_px: float,
) -> tuple[float, list[float]]:
    costs: list[float] = []

    for seg in segment_data:
        cost, detail = _symmetric_chamfer_cost_x_only(
            seg["laser_points"],
            seg["camera_points"],
            camera_dist,
            laser_dist,
            float(dx_px),
        )
        if (
            bool(detail.get("valid", False))
            and np.isfinite(cost)
        ):
            costs.append(float(cost))

    if not costs:
        return float("inf"), []

    # Median makes one unusual vertical region unable to dominate the result.
    return float(np.median(np.asarray(costs))), costs


def _refine_chamfer_dx(
    segment_data: list[dict],
    camera_dist: np.ndarray,
    laser_dist: np.ndarray,
    seed_dx: float,
    threshold_percentile: float,
    phase_k: int,
    search_rows: list[dict],
) -> dict:
    """Two-stage subpixel refinement around the selected wire phase."""
    current = float(seed_dx)
    current_cost = float("inf")
    current_segment_costs: list[float] = []

    stages = [
        (
            "phase_refine",
            float(FULL_AFFINE_CHAMFER_REFINE_RADIUS_PX),
            float(FULL_AFFINE_CHAMFER_REFINE_STEP_PX),
        ),
        (
            "phase_fine",
            float(FULL_AFFINE_CHAMFER_FINE_RADIUS_PX),
            float(FULL_AFFINE_CHAMFER_FINE_STEP_PX),
        ),
    ]

    for stage_name, radius, step in stages:
        radius = max(0.0, radius)
        step = max(0.02, step)

        values = np.arange(
            current - radius,
            current + radius + 0.5 * step,
            step,
            dtype=np.float64,
        )

        best_stage = None
        for dx in values:
            cost, segment_costs = (
                _aggregate_chamfer_cost_for_segments(
                    segment_data,
                    camera_dist,
                    laser_dist,
                    float(dx),
                )
            )

            search_rows.append({
                "threshold_percentile": float(
                    threshold_percentile
                ),
                "segment_index": -1,
                "stage": stage_name,
                "phase_k": int(phase_k),
                "dx_px": float(dx),
                "cost": (
                    None
                    if not np.isfinite(cost)
                    else float(cost)
                ),
                "mean_clipped_distance_px": None,
                "close_fraction": None,
                "laser_edge_point_count": int(
                    sum(
                        len(seg["laser_points"])
                        for seg in segment_data
                    )
                ),
                "camera_edge_point_count": int(
                    sum(
                        len(seg["camera_points"])
                        for seg in segment_data
                    )
                ),
                "valid": bool(np.isfinite(cost)),
            })

            if not np.isfinite(cost):
                continue

            candidate = (
                float(cost),
                float(dx),
                segment_costs,
            )
            if (
                best_stage is None
                or candidate[0] < best_stage[0]
            ):
                best_stage = candidate

        if best_stage is None:
            return {
                "ok": False,
                "reason": f"{stage_name}_failed",
            }

        current_cost, current, current_segment_costs = best_stage

    return {
        "ok": True,
        "reason": "ok",
        "dx_px": float(current),
        "cost": float(current_cost),
        "segment_costs": [
            float(v) for v in current_segment_costs
        ],
    }


def _save_chamfer_edge_debug(
    camera: Image.Image,
    laser_edge: np.ndarray,
    camera_edge: np.ndarray,
    dx_px: float,
    path: Path,
) -> None:
    """Save a compact visual diagnostic of camera/laser cap edges."""
    if not SAVE_FULL_AFFINE_CHAMFER_EDGE_DEBUG_PNG:
        return

    cam = np.asarray(
        camera.convert("RGB"),
        dtype=np.uint8,
    ).copy()
    h, w = laser_edge.shape

    # Shift laser edge using the same X-only transform convention.
    xx = np.arange(w, dtype=np.float32)
    map_x_row = (
        xx - float(dx_px)
    ).astype(np.float32)
    map_x = np.broadcast_to(
        map_x_row[None, :],
        (h, w),
    ).copy()
    map_y = np.broadcast_to(
        np.arange(h, dtype=np.float32)[:, None],
        (h, w),
    ).copy()

    shifted_laser_edge = cv2.remap(
        laser_edge.astype(np.uint8) * 255,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0

    # Camera edges = red, shifted laser edges = cyan, overlap = white.
    out = cam.copy()
    ce = np.asarray(camera_edge, dtype=bool)
    le = shifted_laser_edge

    out[ce] = np.array([255, 40, 40], dtype=np.uint8)
    out[le] = np.array([30, 255, 255], dtype=np.uint8)
    overlap = ce & le
    out[overlap] = np.array([255, 255, 255], dtype=np.uint8)

    Image.fromarray(out, "RGB").save(path)


def optimize_full_affine_wire_pitch_chamfer_x(
    camera: Image.Image,
    mapped_height: Image.Image,
    mapped_mask: Image.Image,
    out_dir: Path,
    stem: str,
) -> dict:
    """Resolve an integer wire-pitch X phase, then refine X by edge Chamfer.

    This is specifically designed for the case where the full physical affine
    is visually close but is horizontally displaced by approximately one
    repeated wire spacing.
    """
    base_result = {
        "enabled": bool(
            FULL_AFFINE_USE_WIRE_PITCH_CHAMFER_X_RESIDUAL
        ),
        "applied": False,
        "reason": "disabled",
        "method": (
            "full_affine_wire_pitch_phase_plus_"
            "symmetric_chamfer_x_only"
        ),
        "uses_image_window_correlation": False,
        "uses_nearest_neighbour_center_residual": False,
        "y_locked": True,
        "scale_locked": True,
        "applied_dx_px": 0.0,
        "applied_dy_px": 0.0,
        "applied_x_scale": 1.0,
        "applied_y_scale": 1.0,
    }

    if not FULL_AFFINE_USE_WIRE_PITCH_CHAMFER_X_RESIDUAL:
        return base_result

    if cv2 is None:
        base_result["reason"] = "opencv_cv2_unavailable"
        return base_result

    if (
        camera.size != mapped_height.size
        or camera.size != mapped_mask.size
    ):
        raise RuntimeError(
            "Wire-pitch Chamfer X residual requires equal image sizes."
        )

    width, height = camera.size

    laser_mask = (
        np.asarray(
            mapped_mask.convert("L"),
            dtype=np.uint8,
        )
        > 127
    )

    camera_analysis_mask = _dilate_bool_mask(
        laser_mask,
        int(
            math.ceil(
                2.0
                * float(FULL_AFFINE_CHAMFER_MAX_PITCH_PX)
                + float(FULL_AFFINE_CHAMFER_LOCAL_RADIUS_PX)
                + 3.0
            )
        ),
    )

    laser_scalar = (
        np.asarray(
            mapped_height.convert("L"),
            dtype=np.float32,
        )
        / 255.0
    )

    cam_rgb = (
        np.asarray(
            camera.convert("RGB"),
            dtype=np.float32,
        )
        / 255.0
    )
    camera_scalar = (
        0.299 * cam_rgb[:, :, 0]
        + 0.587 * cam_rgb[:, :, 1]
        + 0.114 * cam_rgb[:, :, 2]
    ).astype(np.float32)

    threshold_results: list[dict] = []
    search_rows: list[dict] = []
    debug_candidates: list[tuple[float, np.ndarray, np.ndarray]] = []

    segment_count = max(
        1,
        int(FULL_AFFINE_CHAMFER_SEGMENT_COUNT),
    )

    for percentile in WIRE_CENTER_DETECTION_PERCENTILES:
        laser_pts, _laser_score, laser_binary = (
            _detect_wire_end_centers(
                laser_scalar,
                laser_mask,
                float(percentile),
            )
        )
        camera_pts, _camera_score, camera_binary = (
            _detect_wire_end_centers(
                camera_scalar,
                camera_analysis_mask,
                float(percentile),
            )
        )

        pitch_info = _wire_pitch_from_detected_centers(
            laser_pts,
            camera_pts,
            height,
        )
        pitch = float(
            pitch_info["common_pitch_px"]
        )

        laser_edge = _cap_boundary_mask(
            laser_binary,
            laser_mask,
        )
        camera_edge = _cap_boundary_mask(
            camera_binary,
            camera_analysis_mask,
        )

        laser_dist = _edge_distance_map(
            laser_edge
        )
        camera_dist = _edge_distance_map(
            camera_edge
        )

        segment_results: list[dict] = []
        segment_payload: list[dict] = []

        for si in range(segment_count):
            y0 = int(
                round(
                    float(si)
                    / segment_count
                    * height
                )
            )
            y1 = int(
                round(
                    float(si + 1)
                    / segment_count
                    * height
                )
            )

            laser_edge_pts = _edge_points_for_y_segment(
                laser_edge,
                y0,
                y1,
                int(
                    FULL_AFFINE_CHAMFER_MAX_EDGE_POINTS_PER_SEGMENT
                ),
            )
            camera_edge_pts = _edge_points_for_y_segment(
                camera_edge,
                y0,
                y1,
                int(
                    FULL_AFFINE_CHAMFER_MAX_EDGE_POINTS_PER_SEGMENT
                ),
            )

            if (
                len(laser_edge_pts)
                < int(
                    FULL_AFFINE_CHAMFER_MIN_EDGE_POINTS_PER_SEGMENT
                )
                or len(camera_edge_pts)
                < int(
                    FULL_AFFINE_CHAMFER_MIN_EDGE_POINTS_PER_SEGMENT
                )
            ):
                segment_results.append({
                    "ok": False,
                    "reason": "too_few_edge_points",
                    "segment_index": int(si),
                    "laser_edge_point_count": int(
                        len(laser_edge_pts)
                    ),
                    "camera_edge_point_count": int(
                        len(camera_edge_pts)
                    ),
                })
                continue

            result = _search_chamfer_segment_phase(
                laser_edge_pts,
                camera_edge_pts,
                camera_dist,
                laser_dist,
                pitch,
                float(percentile),
                si,
                search_rows,
            )
            result["y0"] = int(y0)
            result["y1"] = int(y1)
            result["laser_edge_point_count"] = int(
                len(laser_edge_pts)
            )
            result["camera_edge_point_count"] = int(
                len(camera_edge_pts)
            )
            segment_results.append(result)

            if bool(result.get("ok", False)):
                segment_payload.append({
                    "segment_index": int(si),
                    "phase_k": int(
                        result["phase_k"]
                    ),
                    "coarse_dx_px": float(
                        result["dx_px"]
                    ),
                    "laser_points": laser_edge_pts,
                    "camera_points": camera_edge_pts,
                })

        valid_segments = [
            row
            for row in segment_results
            if bool(row.get("ok", False))
        ]

        threshold_result = {
            "percentile": float(percentile),
            "laser_center_count": int(len(laser_pts)),
            "camera_center_count": int(len(camera_pts)),
            **pitch_info,
            "valid_segment_count": int(
                len(valid_segments)
            ),
            "segment_count": int(segment_count),
            "segments": segment_results,
            "ok": False,
            "reason": "",
        }

        if len(valid_segments) < int(
            FULL_AFFINE_CHAMFER_MIN_PHASE_AGREEING_SEGMENTS
        ):
            threshold_result["reason"] = (
                "too_few_valid_segments"
            )
            threshold_results.append(
                threshold_result
            )
            continue

        # Integer phase vote.
        phase_counts: dict[int, int] = {}
        for row in valid_segments:
            k = int(row["phase_k"])
            phase_counts[k] = (
                phase_counts.get(k, 0) + 1
            )

        winning_phase = sorted(
            phase_counts.items(),
            key=lambda item: (
                -item[1],
                abs(item[0]),
                item[0],
            ),
        )[0][0]

        agreeing_segments = [
            row
            for row in segment_payload
            if int(row["phase_k"]) == int(
                winning_phase
            )
        ]

        if len(agreeing_segments) < int(
            FULL_AFFINE_CHAMFER_MIN_PHASE_AGREEING_SEGMENTS
        ):
            threshold_result.update({
                "reason": (
                    "insufficient_phase_segment_consensus:"
                    f"{len(agreeing_segments)}"
                ),
                "phase_vote_counts": {
                    str(k): int(v)
                    for k, v in phase_counts.items()
                },
                "winning_phase_k": int(
                    winning_phase
                ),
                "agreeing_segment_count": int(
                    len(agreeing_segments)
                ),
            })
            threshold_results.append(
                threshold_result
            )
            continue

        seed_dx = float(
            np.median(
                np.asarray(
                    [
                        float(row["coarse_dx_px"])
                        for row in agreeing_segments
                    ],
                    dtype=np.float64,
                )
            )
        )

        refined = _refine_chamfer_dx(
            agreeing_segments,
            camera_dist,
            laser_dist,
            seed_dx,
            float(percentile),
            int(winning_phase),
            search_rows,
        )

        if not bool(refined.get("ok", False)):
            threshold_result.update({
                "reason": refined.get(
                    "reason",
                    "refinement_failed",
                ),
                "phase_vote_counts": {
                    str(k): int(v)
                    for k, v in phase_counts.items()
                },
                "winning_phase_k": int(
                    winning_phase
                ),
                "agreeing_segment_count": int(
                    len(agreeing_segments)
                ),
                "seed_dx_px": float(seed_dx),
            })
            threshold_results.append(
                threshold_result
            )
            continue

        final_dx = float(
            refined["dx_px"]
        )

        # Outer-boundary safety.
        outer_hit = False
        if FULL_AFFINE_CHAMFER_REJECT_OUTER_BOUNDARY:
            local_r = float(
                FULL_AFFINE_CHAMFER_LOCAL_RADIUS_PX
            )
            tol = float(
                FULL_AFFINE_CHAMFER_BOUNDARY_TOL_PX
            )
            kmin = int(
                FULL_AFFINE_CHAMFER_PHASE_MIN_K
            )
            kmax = int(
                FULL_AFFINE_CHAMFER_PHASE_MAX_K
            )

            if int(winning_phase) == kmin:
                outward = (
                    float(kmin) * pitch
                    - local_r
                )
                if final_dx <= outward + tol:
                    outer_hit = True

            if int(winning_phase) == kmax:
                outward = (
                    float(kmax) * pitch
                    + local_r
                )
                if final_dx >= outward - tol:
                    outer_hit = True

        if outer_hit:
            threshold_result.update({
                "reason": "winning_phase_hits_outer_search_boundary",
                "phase_vote_counts": {
                    str(k): int(v)
                    for k, v in phase_counts.items()
                },
                "winning_phase_k": int(
                    winning_phase
                ),
                "agreeing_segment_count": int(
                    len(agreeing_segments)
                ),
                "seed_dx_px": float(seed_dx),
                "final_dx_px": float(final_dx),
                "final_cost": float(
                    refined["cost"]
                ),
            })
            threshold_results.append(
                threshold_result
            )
            continue

        threshold_result.update({
            "ok": True,
            "reason": "ok",
            "phase_vote_counts": {
                str(k): int(v)
                for k, v in phase_counts.items()
            },
            "winning_phase_k": int(
                winning_phase
            ),
            "agreeing_segment_count": int(
                len(agreeing_segments)
            ),
            "seed_dx_px": float(seed_dx),
            "final_dx_px": float(final_dx),
            "final_cost": float(
                refined["cost"]
            ),
            "refined_segment_costs": (
                refined.get(
                    "segment_costs",
                    [],
                )
            ),
        })
        threshold_results.append(
            threshold_result
        )

        debug_candidates.append(
            (
                float(percentile),
                laser_edge,
                camera_edge,
            )
        )

    valid_thresholds = [
        row
        for row in threshold_results
        if bool(row.get("ok", False))
    ]

    if len(valid_thresholds) < int(
        FULL_AFFINE_CHAMFER_MIN_AGREEING_THRESHOLDS
    ):
        base_result.update({
            "reason": (
                "too_few_valid_thresholds:"
                f"{len(valid_thresholds)}"
            ),
            "thresholds": threshold_results,
        })
        result = base_result
    else:
        # Multi-threshold INTEGER PHASE consensus first.
        phase_counts: dict[int, int] = {}
        for row in valid_thresholds:
            k = int(row["winning_phase_k"])
            phase_counts[k] = (
                phase_counts.get(k, 0) + 1
            )

        winning_phase = sorted(
            phase_counts.items(),
            key=lambda item: (
                -item[1],
                abs(item[0]),
                item[0],
            ),
        )[0][0]

        phase_thresholds = [
            row
            for row in valid_thresholds
            if int(row["winning_phase_k"])
            == int(winning_phase)
        ]

        if len(phase_thresholds) < int(
            FULL_AFFINE_CHAMFER_MIN_AGREEING_THRESHOLDS
        ):
            base_result.update({
                "reason": (
                    "insufficient_multithreshold_phase_consensus:"
                    f"{len(phase_thresholds)}"
                ),
                "threshold_phase_vote_counts": {
                    str(k): int(v)
                    for k, v in phase_counts.items()
                },
                "thresholds": threshold_results,
            })
            result = base_result
        else:
            dx_values = np.asarray(
                [
                    float(row["final_dx_px"])
                    for row in phase_thresholds
                ],
                dtype=np.float64,
            )
            dx_median = float(
                np.median(dx_values)
            )

            dx_agree = (
                np.abs(
                    dx_values - dx_median
                )
                <= float(
                    FULL_AFFINE_CHAMFER_MAX_THRESHOLD_DX_DEVIATION_PX
                )
            )

            agreeing_thresholds = [
                row
                for row, keep in zip(
                    phase_thresholds,
                    dx_agree,
                )
                if bool(keep)
            ]

            if len(agreeing_thresholds) < int(
                FULL_AFFINE_CHAMFER_MIN_AGREEING_THRESHOLDS
            ):
                base_result.update({
                    "reason": (
                        "threshold_dx_values_too_scattered:"
                        f"{len(agreeing_thresholds)}"
                    ),
                    "threshold_phase_vote_counts": {
                        str(k): int(v)
                        for k, v in phase_counts.items()
                    },
                    "winning_phase_k": int(
                        winning_phase
                    ),
                    "thresholds": threshold_results,
                })
                result = base_result
            else:
                final_dx = float(
                    np.median(
                        np.asarray(
                            [
                                float(row["final_dx_px"])
                                for row in agreeing_thresholds
                            ],
                            dtype=np.float64,
                        )
                    )
                )

                applied = bool(
                    abs(final_dx)
                    >= float(
                        FULL_AFFINE_CHAMFER_MIN_EFFECT_PX
                    )
                )
                applied_dx = (
                    float(final_dx)
                    if applied
                    else 0.0
                )

                result = {
                    "enabled": True,
                    "applied": bool(applied),
                    "reason": (
                        "ok_wire_pitch_phase_chamfer_x"
                        if applied
                        else "x_residual_below_min_effect"
                    ),
                    "method": (
                        "full_affine_wire_pitch_phase_plus_"
                        "symmetric_chamfer_x_only"
                    ),
                    "uses_image_window_correlation": False,
                    "uses_nearest_neighbour_center_residual": False,
                    "y_locked": True,
                    "scale_locked": True,
                    "phase_min_k": int(
                        FULL_AFFINE_CHAMFER_PHASE_MIN_K
                    ),
                    "phase_max_k": int(
                        FULL_AFFINE_CHAMFER_PHASE_MAX_K
                    ),
                    "winning_phase_k": int(
                        winning_phase
                    ),
                    "threshold_phase_vote_counts": {
                        str(k): int(v)
                        for k, v in phase_counts.items()
                    },
                    "valid_threshold_count": int(
                        len(valid_thresholds)
                    ),
                    "agreeing_threshold_count": int(
                        len(agreeing_thresholds)
                    ),
                    "raw_consensus_dx_px": float(
                        final_dx
                    ),
                    "applied_dx_px": float(
                        applied_dx
                    ),
                    "applied_dy_px": 0.0,
                    "applied_x_scale": 1.0,
                    "applied_y_scale": 1.0,
                    "thresholds": threshold_results,
                }

    # --------------------------------------------------------------
    # Diagnostics.
    # --------------------------------------------------------------
    csv_path = (
        out_dir
        / f"{stem}_full_affine_wire_pitch_chamfer_search.csv"
    )
    if (
        SAVE_FULL_AFFINE_CHAMFER_SEARCH_CSV
        and search_rows
    ):
        keys: list[str] = []
        seen: set[str] = set()
        for row in search_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=keys,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(search_rows)

        result["search_csv"] = str(
            csv_path
        )
    else:
        result["search_csv"] = None

    json_path = (
        out_dir
        / f"{stem}_full_affine_wire_pitch_chamfer.json"
    )
    if SAVE_FULL_AFFINE_CHAMFER_JSON:
        json_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result["result_json"] = str(
            json_path
        )

    # Edge debug uses the threshold nearest the middle detector percentile.
    edge_debug_path = (
        out_dir
        / f"{stem}_full_affine_wire_pitch_chamfer_edges.png"
    )
    if (
        SAVE_FULL_AFFINE_CHAMFER_EDGE_DEBUG_PNG
        and debug_candidates
    ):
        target_pct = float(
            np.median(
                np.asarray(
                    WIRE_CENTER_DETECTION_PERCENTILES,
                    dtype=np.float64,
                )
            )
        )
        debug_candidates.sort(
            key=lambda item: abs(
                item[0] - target_pct
            )
        )
        _pct, laser_edge, camera_edge = (
            debug_candidates[0]
        )
        _save_chamfer_edge_debug(
            camera,
            laser_edge,
            camera_edge,
            float(
                result.get(
                    "applied_dx_px",
                    0.0,
                )
            ),
            edge_debug_path,
        )
        if edge_debug_path.exists():
            result["edge_debug_png"] = str(
                edge_debug_path
            )

    print("Full-affine wire-pitch phase + Chamfer X residual:")
    for row in threshold_results:
        if bool(row.get("ok", False)):
            print(
                f"  pct={row['percentile']:.1f}: "
                f"pitch={row['common_pitch_px']:.3f}px, "
                f"k={row['winning_phase_k']:+d}, "
                f"dx={row['final_dx_px']:.3f}px, "
                f"segments={row['agreeing_segment_count']}/"
                f"{row['segment_count']}"
            )
        else:
            print(
                f"  pct={row['percentile']:.1f}: "
                f"rejected ({row.get('reason')})"
            )

    print(
        f"  winning phase k = "
        f"{result.get('winning_phase_k', 'N/A')}"
    )
    print(
        f"  final X correction = "
        f"{result.get('applied_dx_px', 0.0):.3f}px"
    )
    print("  Y correction = 0.000px (LOCKED)")
    print(
        f"  applied = "
        f"{result.get('applied', False)}; "
        f"reason={result.get('reason')}"
    )

    return result





























def _warp_image_scale_translation(
    image: Image.Image,
    dx_px: float,
    dy_px: float,
    x_scale: float,
    y_scale: float,
    *,
    fill,
    nearest: bool = False,
) -> Image.Image:
    """Diagonal scale about camera center + translation; Y is circular, X is not."""
    mode = 'RGB' if image.mode == 'RGB' else 'L'
    src = np.asarray(image.convert(mode), dtype=np.uint8)
    h, w = src.shape[:2]
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5
    sx = max(float(x_scale), 1e-9)
    sy = max(float(y_scale), 1e-9)
    xx = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)
    map_x_row = ((xx - cx) / sx + cx - float(dx_px)).astype(np.float32)
    map_y_col = np.mod((yy - cy) / sy + cy - float(dy_px), h).astype(np.float32)
    map_x = np.broadcast_to(map_x_row[None, :], (h, w)).copy()
    map_y = np.broadcast_to(map_y_col[:, None], (h, w)).copy()
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if mode == 'RGB':
        if isinstance(fill, str):
            border = (255,255,255) if fill.lower() == 'white' else (0,0,0)
        else:
            fv = np.asarray(fill).reshape(-1)
            border = tuple(int(v) for v in (np.repeat(fv,3) if fv.size == 1 else fv[:3]))
    else:
        border = int(fill if not isinstance(fill, tuple) else fill[0])
    out = cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return Image.fromarray(out, mode)


def _warp_numeric_grid_scale_translation_nanaware(
    values: np.ndarray,
    dx_px: float,
    dy_px: float,
    x_scale: float,
    y_scale: float,
) -> np.ndarray:
    src = np.asarray(values, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError(f'Expected 2-D numeric grid, got {src.shape}')
    h, w = src.shape
    cx = (w - 1.0) * 0.5
    cy = (h - 1.0) * 0.5
    xx = np.arange(w, dtype=np.float32)
    yy = np.arange(h, dtype=np.float32)
    map_x_row = ((xx - cx) / max(float(x_scale),1e-9) + cx - float(dx_px)).astype(np.float32)
    map_y_col = np.mod((yy - cy) / max(float(y_scale),1e-9) + cy - float(dy_px), h).astype(np.float32)
    map_x = np.broadcast_to(map_x_row[None, :], (h, w)).copy()
    map_y = np.broadcast_to(map_y_col[:, None], (h, w)).copy()
    valid = np.isfinite(src).astype(np.float32)
    data = np.nan_to_num(src, nan=0.0).astype(np.float32)
    numerator = cv2.remap(data, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    denominator = cv2.remap(valid, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = np.full((h,w), np.nan, dtype=np.float32)
    good = denominator > 1e-4
    out[good] = numerator[good] / denominator[good]
    return out












def save_mapped_height_mm_npz_full_affine(
    laser_path: Path,
    laser_size_wh: tuple[int, int],
    camera: Image.Image,
    calibration: dict,
    maps: dict,
    mapped_visual_mask: np.ndarray,
    output_path: Path,
    final_residual_dx_px: float = 0.0,
    final_residual_dy_px: float = 0.0,
    final_residual_x_scale: float = 1.0,
    final_residual_y_scale: float = 1.0,
) -> dict:
    """Save REAL laser height in the final camera grid using full physical affine."""
    (
        laser_height_mm,
        laser_residual_mm,
        source_npz_path,
        source_info,
    ) = load_numeric_laser_height_on_visual_grid(
        laser_path,
        laser_size_wh,
    )

    camera_w, camera_h = camera.size

    mapped_height = remap_numeric_full_affine_nanaware(
        laser_height_mm,
        maps,
    )

    mapped_residual = None
    if laser_residual_mm is not None:
        mapped_residual = remap_numeric_full_affine_nanaware(
            laser_residual_mm,
            maps,
        )

    residual_dx = float(final_residual_dx_px)
    residual_dy = float(final_residual_dy_px)
    residual_sx = float(final_residual_x_scale)
    residual_sy = float(final_residual_y_scale)

    transform_needed = (
        abs(residual_dx) > 1e-12
        or abs(residual_dy) > 1e-12
        or abs(residual_sx - 1.0) > 1e-12
        or abs(residual_sy - 1.0) > 1e-12
    )
    if transform_needed:
        mapped_height = _warp_numeric_grid_scale_translation_nanaware(
            mapped_height,
            residual_dx,
            residual_dy,
            residual_sx,
            residual_sy,
        )
        if mapped_residual is not None:
            mapped_residual = _warp_numeric_grid_scale_translation_nanaware(
                mapped_residual,
                residual_dx,
                residual_dy,
                residual_sx,
                residual_sy,
            )

    visual_mask = np.asarray(
        mapped_visual_mask,
        dtype=bool,
    )
    if visual_mask.shape != mapped_height.shape:
        raise RuntimeError(
            "Mapped visual mask and numeric height grid have different shapes: "
            f"{visual_mask.shape} vs {mapped_height.shape}"
        )

    valid_mask = (
        visual_mask
        & np.isfinite(mapped_height)
    )
    mapped_height[~valid_mask] = np.nan
    if mapped_residual is not None:
        mapped_residual[~valid_mask] = np.nan

    camera_zero_px = float(
        calibration["camera_zero_px"]
    )
    camera_mm_per_px = float(
        calibration["camera_mm_per_px"]
    )
    camera_x_axis_mm = (
        (
            np.arange(
                camera_w,
                dtype=np.float64,
            )
            - camera_zero_px
        )
        * camera_mm_per_px
    ).astype(np.float32)

    angle_axis_deg = (
        np.arange(
            camera_h,
            dtype=np.float64,
        )
        / float(camera_h)
        * float(calibration["period_deg"])
    ).astype(np.float32)

    # Backward-compatible 1-D diagnostics only.  The exact mapping is 2-D and
    # is saved separately in *_full_affine_correspondence_maps.npz.
    center_col = int(round(camera_zero_px))
    center_row = int(camera_h // 2)
    row_map_centerline = np.asarray(
        maps["map_y_raw"][:, center_col],
        dtype=np.float32,
    )
    col_map_centerline = np.asarray(
        maps["map_x_raw"][center_row, :],
        dtype=np.float32,
    )

    out_npz = (
        output_path.parent
        / str(MAPPED_HEIGHT_MM_FILENAME)
    )

    payload = {
        "height_mm": mapped_height.astype(np.float32),
        "valid_mask": valid_mask.astype(np.uint8),
        "camera_x_axis_mm": camera_x_axis_mm,
        "angle_axis_deg": angle_axis_deg,

        # Kept for compatibility, but these are only centerline diagnostics
        # because the full affine mapping is genuinely 2-D.
        "camera_row_to_laser_row_px": row_map_centerline,
        "camera_col_to_laser_col_px": col_map_centerline,

        "camera_width_px": np.int32(camera_w),
        "camera_height_px": np.int32(camera_h),
        "laser_visual_width_px": np.int32(laser_size_wh[0]),
        "laser_visual_height_px": np.int32(laser_size_wh[1]),
        "camera_x_mm_per_pixel": np.float32(camera_mm_per_px),
        "camera_zero_x_px": np.float32(camera_zero_px),

        "physical_affine_2x3": np.asarray(
            calibration["matrix_2x3"],
            dtype=np.float64,
        ),
        "laser_x_min_mm": np.float64(
            calibration["laser_x_min_mm"]
        ),
        "laser_x_max_mm": np.float64(
            calibration["laser_x_max_mm"]
        ),
        "circumference_mm": np.float64(
            calibration["circumference_mm"]
        ),
        "wheel_diameter_mm": np.float64(
            calibration["wheel_diameter_mm"]
        ),

        "final_residual_dx_px": np.float32(residual_dx),
        "final_residual_dy_px": np.float32(residual_dy),
        "final_residual_x_scale": np.float32(residual_sx),
        "final_residual_y_scale": np.float32(residual_sy),

        "source_numeric_height_npz": np.array(
            str(source_npz_path)
        ),
        "calibration_json": np.array(
            calibration["calibration_json"]
        ),
        "mapping_method": np.array(
            "full_physical_circular_affine_2d"
        ),
        "exact_correspondence_map_note": np.array(
            "Exact 2-D camera->laser pixel maps are saved separately in "
            "fusion_2d_full_affine_correspondence_maps.npz."
        ),
    }

    if mapped_residual is not None:
        payload["residual_mm"] = mapped_residual.astype(
            np.float32
        )

    np.savez_compressed(
        out_npz,
        **payload,
    )

    preview_path = (
        output_path.parent
        / "mapped_height_mm_preview.png"
    )
    _save_mapped_height_preview_png(
        mapped_height,
        valid_mask,
        preview_path,
    )

    valid_values = mapped_height[valid_mask]
    info = {
        "npz": str(out_npz),
        "preview_png": (
            str(preview_path)
            if SAVE_MAPPED_HEIGHT_MM_PREVIEW
            else None
        ),
        "source_numeric_npz": str(source_npz_path),
        "camera_shape_rows_cols": [
            int(camera_h),
            int(camera_w),
        ],
        "laser_visual_size_wh": [
            int(laser_size_wh[0]),
            int(laser_size_wh[1]),
        ],
        "valid_points": int(np.count_nonzero(valid_mask)),
        "valid_fraction": float(np.mean(valid_mask)),
        "height_min_mm": (
            float(np.min(valid_values))
            if valid_values.size
            else None
        ),
        "height_median_mm": (
            float(np.median(valid_values))
            if valid_values.size
            else None
        ),
        "height_max_mm": (
            float(np.max(valid_values))
            if valid_values.size
            else None
        ),
        "mapping_method": "full_physical_circular_affine_2d",
        "calibration_json": calibration["calibration_json"],
        "physical_affine_2x3": calibration["matrix_2x3_list"],
        "camera_mm_per_px": float(camera_mm_per_px),
        "laser_x_min_mm": float(calibration["laser_x_min_mm"]),
        "laser_x_max_mm": float(calibration["laser_x_max_mm"]),
        "laser_native_mm_per_px_current": float(
            calibration["laser_mm_per_px_current"]
        ),
        "circumference_mm": float(
            calibration["circumference_mm"]
        ),
        "final_residual_dx_px": float(residual_dx),
        "final_residual_dy_px": float(residual_dy),
        "final_residual_x_scale": float(residual_sx),
        "final_residual_y_scale": float(residual_sy),
        "source_info": source_info,
    }
    return info









def parse_args():
    parser = argparse.ArgumentParser(description="Fuse a laser height unfold image and a camera unfold image.")
    parser.add_argument("--laser", default=LASER_IMAGE_PATH, help="Laser height image path.")
    parser.add_argument("--camera", default=CAMERA_IMAGE_PATH, help="Camera unfolded image path.")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="Output folder.")
    parser.add_argument("--name", default=OUTPUT_NAME, help="Output image file name.")
    parser.add_argument("--run-name", default=RUN_NAME, help="Ignored; output goes directly to --out-dir.")
    return parser.parse_args()


def main():
    args = parse_args()
    base_out_dir = resolve_output_dir(args.out_dir)
    out_dir = make_run_output_dir(base_out_dir, args.run_name)
    print(f"Run output folder: {out_dir}")
    output_path = out_dir / args.name
    fuse_2d(Path(args.laser).resolve(), Path(args.camera).resolve(), output_path)
    print("Done.")

import sys

# =============================================================================
# FINAL Y-ONLY REFINEMENT SETTINGS
# =============================================================================
PATCH_MARKER = "FUSION_FINAL_KEEP_ORIGINAL_X_FEATURE_AUTO_Y"

USE_FINAL_FEATURE_AUTO_Y = True
FEATURE_Y_SEARCH_RANGE_PX = 12.0
FEATURE_Y_COARSE_STEP_PX = 0.10
FEATURE_Y_FINE_RADIUS_PX = 0.30
FEATURE_Y_FINE_STEP_PX = 0.01
FEATURE_Y_MIN_EFFECT_PX = 0.01
FEATURE_Y_ANALYSIS_WIDTH_PX = 192
FEATURE_Y_WEIGHT_ENDFACE_2D = 0.36
FEATURE_Y_WEIGHT_EDGE_2D = 0.28
FEATURE_Y_WEIGHT_STRONG_DICE = 0.14
FEATURE_Y_WEIGHT_ROW_PROFILE = 0.22
FEATURE_Y_SEGMENT_COUNT = 5
FEATURE_Y_SEGMENT_MEDIAN_WEIGHT = 0.18
FEATURE_Y_SEGMENT_DISPERSION_WEIGHT = 0.08
FEATURE_Y_STRONG_PERCENTILE = 72.0
FEATURE_Y_MIN_VALID_PIXELS = 5000
FEATURE_Y_BOUNDARY_MARGIN_PX = 0.015
FEATURE_Y_BOUNDARY_MIN_SCORE_GAIN = 0.0020
FEATURE_Y_MIN_SCORE_GAIN_NONBOUNDARY = 0.0
SAVE_FEATURE_Y_JSON = True
SAVE_FEATURE_Y_SEARCH_CSV = True

USE_FINAL_LOCAL_FEATURE_Y = True
LOCAL_FEATURE_Y_WINDOW_HEIGHT_PX = 650
LOCAL_FEATURE_Y_STEP_PX = 300
LOCAL_FEATURE_Y_SEARCH_RANGE_PX = 2.00
LOCAL_FEATURE_Y_COARSE_STEP_PX = 0.05
LOCAL_FEATURE_Y_FINE_RADIUS_PX = 0.12
LOCAL_FEATURE_Y_FINE_STEP_PX = 0.01
LOCAL_FEATURE_Y_WEIGHT_ENDFACE_2D = 0.34
LOCAL_FEATURE_Y_WEIGHT_EDGE_2D = 0.28
LOCAL_FEATURE_Y_WEIGHT_STRONG_DICE = 0.14
LOCAL_FEATURE_Y_WEIGHT_ROW_PROFILE = 0.24
LOCAL_FEATURE_Y_MIN_SCORE_GAIN = 0.00020
LOCAL_FEATURE_Y_BOUNDARY_MIN_SCORE_GAIN = 0.00150
LOCAL_FEATURE_Y_BOUNDARY_MARGIN_PX = 0.015
LOCAL_FEATURE_Y_MIN_VALID_PIXELS = 1800
LOCAL_FEATURE_Y_MIN_VALID_WINDOWS = 6
LOCAL_FEATURE_Y_MAX_ABS_PX = 2.00
LOCAL_FEATURE_Y_MAX_ADJACENT_JUMP_PX = 0.35
LOCAL_FEATURE_Y_SMOOTH_WINDOW = 5
LOCAL_FEATURE_Y_SMOOTH_PASSES = 2
LOCAL_FEATURE_Y_MIN_EFFECT_PX = 0.015
SAVE_LOCAL_FEATURE_Y_JSON = True
SAVE_LOCAL_FEATURE_Y_WINDOWS_CSV = True
SAVE_LOCAL_FEATURE_Y_CURVE_CSV = True

# -------------------------------------------------------------------------
# KALMAN LOCAL-Y SENSOR-FUSION EXPERIMENT
# -------------------------------------------------------------------------
# IMPORTANT:
#   - The complete physical affine remains the global mapping authority.
#   - The original wire-pitch/Chamfer X stage remains unchanged.
#   - The original local-Y post-processing remains available as BASELINE.
#   - Kalman uses the SAME accepted local-Y observations (raw_dy_px), but
#     weights each observation by its local matching confidence.
#   - The main legacy outputs (fusion_2d.png and mapped_height_mm.npz) remain
#     BASELINE for backward compatibility.  Kalman outputs are saved separately.
USE_KALMAN_LOCAL_Y = True

# Constant-velocity state: [local_dy_px, local_dy_change_per_window_px].
KALMAN_LOCAL_Y_MODEL = "constant_velocity"
KALMAN_LOCAL_Y_PERIODIC_CYCLES = 2

# Dynamic measurement uncertainty R_k.  High score-gain -> smaller sigma;
# weak/boundary matches -> larger sigma.
KALMAN_LOCAL_Y_MIN_MEAS_SIGMA_PX = 0.06
KALMAN_LOCAL_Y_MAX_MEAS_SIGMA_PX = 0.45
KALMAN_LOCAL_Y_BOUNDARY_SIGMA_MULTIPLIER = 1.75

# Optional independently measured sensor localization uncertainties.
# Set these when such values are available.  Zero means that no additional
# sensor-specific term is added beyond the confidence-derived local R_k.
KALMAN_CAMERA_LOCALIZATION_SIGMA_PX = 0.0
KALMAN_LLTS_LOCALIZATION_SIGMA_PX = 0.0

# Process uncertainty Q is estimated robustly from changes between accepted
# neighbouring raw local-Y observations and clipped to this range.
KALMAN_LOCAL_Y_PROCESS_SIGMA_FALLBACK_PX = 0.08
KALMAN_LOCAL_Y_PROCESS_SIGMA_MIN_PX = 0.02
KALMAN_LOCAL_Y_PROCESS_SIGMA_MAX_PX = 0.20

# The physical-affine registration RMSE (converted mm -> circular-Y px) is used
# as the initial position covariance when present in the calibration JSON.
KALMAN_LOCAL_Y_USE_AFFINE_RMSE_FOR_INITIAL_P = True
KALMAN_LOCAL_Y_INITIAL_SIGMA_FALLBACK_PX = 0.75
KALMAN_LOCAL_Y_INITIAL_SIGMA_MIN_PX = 0.30
KALMAN_LOCAL_Y_INITIAL_SIGMA_MAX_PX = 4.00
KALMAN_LOCAL_Y_INITIAL_VELOCITY_SIGMA_PX = 0.25

# Keep Kalman correction within exactly the same safety envelope as baseline.
KALMAN_LOCAL_Y_MAX_ABS_PX = LOCAL_FEATURE_Y_MAX_ABS_PX
KALMAN_LOCAL_Y_MIN_EFFECT_PX = LOCAL_FEATURE_Y_MIN_EFFECT_PX

SAVE_KALMAN_LOCAL_Y_WINDOWS_CSV = True
SAVE_KALMAN_LOCAL_Y_CURVE_CSV = True
SAVE_KALMAN_LOCAL_Y_JSON = True
SAVE_LOCAL_Y_BASELINE_KALMAN_COMPARISON_CSV = True

# Legacy constant-velocity Kalman is preserved only as an experimental comparator.
KALMAN_FUSION_SUFFIX = "_kalman_cv"
BASELINE_FUSION_SUFFIX = "_baseline"
MAPPED_HEIGHT_MM_KALMAN_FILENAME = "mapped_height_mm_kalman_cv.npz"
MAPPED_HEIGHT_MM_BASELINE_FILENAME = "mapped_height_mm_baseline.npz"

# -------------------------------------------------------------------------
# RECOMMENDED 1-D ADAPTIVE KALMAN LOCAL-Y
# -------------------------------------------------------------------------
# State: x_k = local_dy_k only.  Random-walk model:
#       dy_k = dy_(k-1) + w_k
# This removes the velocity-state inertia/overshoot observed with the previous
# constant-velocity model.
USE_KALMAN_1D_LOCAL_Y = True
KALMAN_1D_LOCAL_Y_MODEL = "random_walk_1d"
KALMAN_1D_PERIODIC_CYCLES = 3

# Confidence-dependent measurement uncertainty.
KALMAN_1D_MIN_MEAS_SIGMA_PX = 0.08
KALMAN_1D_MAX_MEAS_SIGMA_PX = 0.50

# A result at +/- LOCAL_FEATURE_Y_SEARCH_RANGE_PX is censored: the true optimum
# may lie outside the search interval.  Therefore even a high feature score must
# not be treated as a high-precision measurement.
KALMAN_1D_BOUNDARY_SIGMA_MULTIPLIER = 2.50
KALMAN_1D_BOUNDARY_MIN_SIGMA_PX = 0.30

# Random-walk process uncertainty, estimated from adjacent reliable NON-boundary
# observations whenever possible and clipped to a conservative range.
KALMAN_1D_PROCESS_SIGMA_FALLBACK_PX = 0.07
KALMAN_1D_PROCESS_SIGMA_MIN_PX = 0.025
KALMAN_1D_PROCESS_SIGMA_MAX_PX = 0.16

# Initial covariance still uses the physical-affine RMSE only as an initial
# uncertainty scale.  Repeated circular passes rapidly reduce cold-start effects.
KALMAN_1D_USE_AFFINE_RMSE_FOR_INITIAL_P = True
KALMAN_1D_INITIAL_SIGMA_FALLBACK_PX = 0.75
KALMAN_1D_INITIAL_SIGMA_MIN_PX = 0.30
KALMAN_1D_INITIAL_SIGMA_MAX_PX = 4.00

KALMAN_1D_MAX_ABS_PX = LOCAL_FEATURE_Y_MAX_ABS_PX
KALMAN_1D_MIN_EFFECT_PX = LOCAL_FEATURE_Y_MIN_EFFECT_PX

SAVE_KALMAN_1D_WINDOWS_CSV = True
SAVE_KALMAN_1D_CURVE_CSV = True
SAVE_KALMAN_1D_JSON = True
SAVE_LOCAL_Y_THREE_WAY_COMPARISON_CSV = True

KALMAN_1D_FUSION_SUFFIX = "_kalman_1d"
MAPPED_HEIGHT_MM_KALMAN_1D_FILENAME = "mapped_height_mm_kalman_1d.npz"

# -------------------------------------------------------------------------
# RECOMMENDED FIXED-INTERVAL RTS SMOOTHER AFTER THE 1-D ADAPTIVE KALMAN
# -------------------------------------------------------------------------
# The forward 1-D Kalman can lag in low-confidence regions because it only sees
# past measurements.  The Rauch-Tung-Striebel (RTS) smoother uses the already
# collected full 360-degree sequence and propagates information backward.
# For the scalar random-walk state F=1:
#       C_k = P_k|k / P_(k+1|k)
#       x_k|N = x_k|k + C_k * (x_(k+1)|N - x_(k+1|k))
# No new image matching, X motion, scale, rotation or shear is introduced.
USE_KALMAN_RTS_LOCAL_Y = True
KALMAN_RTS_MODEL = "rts_fixed_interval_scalar_random_walk"
KALMAN_RTS_MAX_ABS_PX = LOCAL_FEATURE_Y_MAX_ABS_PX
KALMAN_RTS_MIN_EFFECT_PX = LOCAL_FEATURE_Y_MIN_EFFECT_PX
SAVE_KALMAN_RTS_WINDOWS_CSV = True
SAVE_KALMAN_RTS_CURVE_CSV = True
SAVE_KALMAN_RTS_JSON = True
SAVE_LOCAL_Y_BASELINE_1D_RTS_COMPARISON_CSV = True
KALMAN_RTS_FUSION_SUFFIX = "_kalman_rts"
MAPPED_HEIGHT_MM_KALMAN_RTS_FILENAME = "mapped_height_mm_kalman_rts.npz"

# Which local-Y branch owns the historical main outputs fusion_2d.png and
# mapped_height_mm.npz.  Every named variant is still saved separately.
# Allowed: "baseline", "kalman_cv", "kalman_1d", "kalman_rts".
MAIN_LOCAL_Y_VARIANT = "kalman_rts"


def _float_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")

    count = int(
        math.floor(
            (stop - start) / step + 0.5
        )
    ) + 1

    values = (
        float(start)
        + np.arange(
            max(count, 0),
            dtype=np.float64,
        )
        * float(step)
    )

    values = values[
        (values >= start - 1e-10)
        & (values <= stop + 1e-10)
    ]

    return np.round(values, 10)


def _resize_x_only(
    base,
    arr: np.ndarray,
    target_width: int,
    *,
    nearest: bool,
) -> np.ndarray:
    src = np.asarray(arr)
    h, w = src.shape[:2]

    tw = max(
        8,
        min(int(target_width), int(w)),
    )

    if tw == w:
        return src.copy()

    interpolation = (
        base.cv2.INTER_NEAREST
        if nearest
        else base.cv2.INTER_AREA
    )

    resized = base.cv2.resize(
        src,
        (tw, h),
        interpolation=interpolation,
    )

    return resized


def _shift_periodic_y_float(
    arr: np.ndarray,
    dy_px: float,
) -> np.ndarray:
    """
    Shift content vertically by dy using the SAME sign convention as:

        _warp_image_scale_translation(..., dy_px=dy)

    Original code samples:
        source_y = output_y - dy

    Therefore positive dy moves content downward.
    """
    src = np.asarray(
        arr,
        dtype=np.float32,
    )

    h = int(src.shape[0])

    out_y = np.arange(
        h,
        dtype=np.float64,
    )

    source_y = np.mod(
        out_y - float(dy_px),
        float(h),
    )

    y0f = np.floor(source_y)
    y0 = y0f.astype(np.int64)
    y1 = (y0 + 1) % h

    weight = (
        source_y - y0f
    ).astype(np.float32)

    if src.ndim == 1:
        return (
            src[y0] * (1.0 - weight)
            + src[y1] * weight
        ).astype(np.float32)

    return (
        src[y0, ...] * (1.0 - weight[:, None])
        + src[y1, ...] * weight[:, None]
    ).astype(np.float32)


def _shift_periodic_y_mask(
    mask: np.ndarray,
    dy_px: float,
) -> np.ndarray:
    shifted = _shift_periodic_y_float(
        np.asarray(mask, dtype=np.float32),
        float(dy_px),
    )

    return shifted >= 0.50


def _masked_ncc(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
) -> float:
    aa = np.asarray(
        a,
        dtype=np.float32,
    )
    bb = np.asarray(
        b,
        dtype=np.float32,
    )
    mm = np.asarray(
        mask,
        dtype=bool,
    )

    valid = (
        mm
        & np.isfinite(aa)
        & np.isfinite(bb)
    )

    count = int(
        np.count_nonzero(valid)
    )

    if count < 128:
        return -1.0

    av = aa[valid].astype(
        np.float64,
        copy=False,
    )
    bv = bb[valid].astype(
        np.float64,
        copy=False,
    )

    av = av - float(np.mean(av))
    bv = bv - float(np.mean(bv))

    sa = float(
        np.sqrt(
            np.mean(av * av)
        )
    )
    sb = float(
        np.sqrt(
            np.mean(bv * bv)
        )
    )

    if sa < 1e-8 or sb < 1e-8:
        return -1.0

    return float(
        np.mean(
            (av / sa)
            * (bv / sb)
        )
    )


def _dice_score(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
) -> float:
    aa = (
        np.asarray(a, dtype=bool)
        & np.asarray(mask, dtype=bool)
    )
    bb = (
        np.asarray(b, dtype=bool)
        & np.asarray(mask, dtype=bool)
    )

    na = int(
        np.count_nonzero(aa)
    )
    nb = int(
        np.count_nonzero(bb)
    )

    if na + nb <= 0:
        return 0.0

    inter = int(
        np.count_nonzero(
            aa & bb
        )
    )

    return float(
        2.0 * inter
        / float(na + nb)
    )


def _normalize_1d(
    values: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(
        values,
        dtype=np.float32,
    ).reshape(-1)

    finite = np.isfinite(arr)

    if int(np.count_nonzero(finite)) < 16:
        return np.zeros_like(
            arr,
            dtype=np.float32,
        )

    out = arr.copy()

    median = float(
        np.median(out[finite])
    )
    out[~finite] = median

    # Remove slow illumination / density drift along 360 degrees.
    # Sigma ~12 rows is much larger than the 0~2 px residual we are estimating.
    try:
        import cv2
        low = cv2.GaussianBlur(
            out[:, None],
            (0, 0),
            sigmaX=0.0,
            sigmaY=12.0,
            borderType=cv2.BORDER_WRAP,
        )[:, 0]
    except Exception:
        radius = 25
        kernel = np.ones(
            2 * radius + 1,
            dtype=np.float32,
        )
        kernel /= float(
            np.sum(kernel)
        )
        ext = np.concatenate(
            [
                out[-radius:],
                out,
                out[:radius],
            ]
        )
        low = np.convolve(
            ext,
            kernel,
            mode="valid",
        )

    high = (
        out - low
    ).astype(np.float32)

    high -= float(
        np.mean(high)
    )

    std = float(
        np.std(high)
    )

    if std > 1e-8:
        high /= std
    else:
        high[:] = 0.0

    return high.astype(
        np.float32,
        copy=False,
    )


def _row_profile(
    feature: np.ndarray,
    column_support: np.ndarray,
) -> np.ndarray:
    feat = np.asarray(
        feature,
        dtype=np.float32,
    )

    cols = np.asarray(
        column_support,
        dtype=bool,
    )

    if (
        feat.ndim != 2
        or cols.ndim != 1
        or len(cols) != feat.shape[1]
        or int(np.count_nonzero(cols)) < 4
    ):
        return np.zeros(
            feat.shape[0],
            dtype=np.float32,
        )

    values = np.mean(
        feat[:, cols],
        axis=1,
    )

    return _normalize_1d(values)


def _row_ncc(
    laser_row: np.ndarray,
    camera_row: np.ndarray,
    dy_px: float,
    y0: int | None = None,
    y1: int | None = None,
) -> float:
    shifted = _shift_periodic_y_float(
        laser_row,
        float(dy_px),
    )

    camera = np.asarray(
        camera_row,
        dtype=np.float32,
    )

    if y0 is not None or y1 is not None:
        lo = (
            0
            if y0 is None
            else max(0, int(y0))
        )
        hi = (
            len(camera)
            if y1 is None
            else min(
                len(camera),
                int(y1),
            )
        )
        shifted = shifted[lo:hi]
        camera = camera[lo:hi]

    if len(camera) < 32:
        return -1.0

    shifted = (
        shifted
        - float(np.mean(shifted))
    )
    camera = (
        camera
        - float(np.mean(camera))
    )

    sa = float(
        np.std(shifted)
    )
    sb = float(
        np.std(camera)
    )

    if sa < 1e-8 or sb < 1e-8:
        return -1.0

    return float(
        np.mean(
            shifted / sa
            * camera / sb
        )
    )


def _build_feature_y_data(
    base,
    camera: Image.Image,
    mapped_height: Image.Image,
    mapped_mask: Image.Image,
) -> dict:
    """
    Build modality-normalized end-face feature maps.

    No RGB-to-laser direct correlation is used.
    Each modality first goes through the original program's own feature
    extraction, then the comparable end-face representation is aligned.
    """
    if base.cv2 is None:
        raise RuntimeError(
            "OpenCV is required for feature Y alignment."
        )

    mask_full = (
        np.asarray(
            mapped_mask.convert("L"),
            dtype=np.uint8,
        )
        > 127
    )

    if (
        int(np.count_nonzero(mask_full))
        < int(FEATURE_Y_MIN_VALID_PIXELS)
    ):
        raise RuntimeError(
            "Too few valid mapped laser pixels for final feature Y alignment."
        )

    laser_scalar_full = (
        np.asarray(
            mapped_height.convert("L"),
            dtype=np.float32,
        )
        / 255.0
    )

    # Original fusion program's modality-specific feature builders.
    laser_feature_full = (
        base.laser_feature_scalar(
            laser_scalar_full,
            mask_full,
        )
    )

    camera_feature_full = (
        base.camera_feature_scalar(
            camera
        )
    )

    # End-face enhancement is the common representation.
    laser_end_full = (
        base.endface_feature_map(
            laser_feature_full,
            mask_full,
        )
    )

    camera_end_full = (
        base.endface_feature_map(
            camera_feature_full,
            None,
        )
    )

    laser_edge_full = (
        base.normalize_01(
            base.gradient_magnitude(
                laser_end_full
            ),
            mask_full,
        )
    )

    camera_edge_full = (
        base.normalize_01(
            base.gradient_magnitude(
                camera_end_full
            ),
            None,
        )
    )

    analysis_width = min(
        int(FEATURE_Y_ANALYSIS_WIDTH_PX),
        camera.width,
    )

    laser_end = _resize_x_only(
        base,
        laser_end_full,
        analysis_width,
        nearest=False,
    ).astype(np.float32)

    camera_end = _resize_x_only(
        base,
        camera_end_full,
        analysis_width,
        nearest=False,
    ).astype(np.float32)

    laser_edge = _resize_x_only(
        base,
        laser_edge_full,
        analysis_width,
        nearest=False,
    ).astype(np.float32)

    camera_edge = _resize_x_only(
        base,
        camera_edge_full,
        analysis_width,
        nearest=False,
    ).astype(np.float32)

    mask = _resize_x_only(
        base,
        mask_full.astype(np.uint8),
        analysis_width,
        nearest=True,
    ) > 0

    # Thresholds are calculated from each modality independently.
    laser_values = (
        laser_end[mask]
        if np.any(mask)
        else laser_end.reshape(-1)
    )

    # Camera threshold is estimated in a slightly expanded version of the
    # physical laser band, but the final scoring still uses the shifted laser mask.
    camera_support = base._dilate_bool_mask(
        mask,
        2,
    )

    camera_values = (
        camera_end[camera_support]
        if np.any(camera_support)
        else camera_end.reshape(-1)
    )

    laser_strong_threshold = float(
        np.percentile(
            laser_values,
            float(FEATURE_Y_STRONG_PERCENTILE),
        )
    )

    camera_strong_threshold = float(
        np.percentile(
            camera_values,
            float(FEATURE_Y_STRONG_PERCENTILE),
        )
    )

    laser_strong = (
        laser_end
        >= laser_strong_threshold
    )

    camera_strong = (
        camera_end
        >= camera_strong_threshold
    )

    # X is already fixed. Use columns that repeatedly contain mapped laser
    # information to build an independent Y row signature.
    column_coverage = np.mean(
        mask.astype(np.float32),
        axis=0,
    )

    column_support = (
        column_coverage
        >= max(
            0.01,
            float(
                np.percentile(
                    column_coverage,
                    30.0,
                )
            )
            * 0.35,
        )
    )

    if int(np.count_nonzero(column_support)) < 8:
        column_support[:] = True

    laser_row_end = _row_profile(
        laser_end,
        column_support,
    )
    camera_row_end = _row_profile(
        camera_end,
        column_support,
    )

    laser_row_edge = _row_profile(
        laser_edge,
        column_support,
    )
    camera_row_edge = _row_profile(
        camera_edge,
        column_support,
    )

    return {
        "laser_end": laser_end,
        "camera_end": camera_end,
        "laser_edge": laser_edge,
        "camera_edge": camera_edge,
        "laser_strong": laser_strong,
        "camera_strong": camera_strong,
        "mask": mask,
        "laser_row_end": laser_row_end,
        "camera_row_end": camera_row_end,
        "laser_row_edge": laser_row_edge,
        "camera_row_edge": camera_row_edge,
        "analysis_width": int(analysis_width),
        "height": int(camera.height),
        "laser_strong_threshold": float(
            laser_strong_threshold
        ),
        "camera_strong_threshold": float(
            camera_strong_threshold
        ),
    }


def _score_feature_y(
    data: dict,
    dy_px: float,
) -> dict:
    laser_end_shifted = (
        _shift_periodic_y_float(
            data["laser_end"],
            float(dy_px),
        )
    )

    laser_edge_shifted = (
        _shift_periodic_y_float(
            data["laser_edge"],
            float(dy_px),
        )
    )

    laser_strong_shifted = (
        _shift_periodic_y_float(
            data["laser_strong"].astype(
                np.float32
            ),
            float(dy_px),
        )
        >= 0.50
    )

    mask_shifted = _shift_periodic_y_mask(
        data["mask"],
        float(dy_px),
    )

    valid_count = int(
        np.count_nonzero(mask_shifted)
    )

    if valid_count < int(
        FEATURE_Y_MIN_VALID_PIXELS
    ):
        return {
            "valid": False,
            "score": -1e9,
            "valid_pixels": int(valid_count),
        }

    endface_ncc = _masked_ncc(
        laser_end_shifted,
        data["camera_end"],
        mask_shifted,
    )

    edge_ncc = _masked_ncc(
        laser_edge_shifted,
        data["camera_edge"],
        mask_shifted,
    )

    strong_dice = _dice_score(
        laser_strong_shifted,
        data["camera_strong"],
        mask_shifted,
    )

    row_end = _row_ncc(
        data["laser_row_end"],
        data["camera_row_end"],
        float(dy_px),
    )

    row_edge = _row_ncc(
        data["laser_row_edge"],
        data["camera_row_edge"],
        float(dy_px),
    )

    row_profile_score = (
        0.55 * float(row_end)
        + 0.45 * float(row_edge)
    )

    main_score = (
        float(FEATURE_Y_WEIGHT_ENDFACE_2D)
        * float(endface_ncc)
        + float(FEATURE_Y_WEIGHT_EDGE_2D)
        * float(edge_ncc)
        + float(FEATURE_Y_WEIGHT_STRONG_DICE)
        * float(strong_dice)
        + float(FEATURE_Y_WEIGHT_ROW_PROFILE)
        * float(row_profile_score)
    )

    # Soft vertical consistency using row signals only. This is cheap and,
    # unlike old logic, NEVER rejects the global result by itself.
    segment_scores = []

    segment_count = max(
        1,
        int(FEATURE_Y_SEGMENT_COUNT),
    )

    h = int(data["height"])

    for seg in range(segment_count):
        y0 = int(
            round(
                seg / float(segment_count)
                * h
            )
        )
        y1 = int(
            round(
                (seg + 1)
                / float(segment_count)
                * h
            )
        )

        se = _row_ncc(
            data["laser_row_end"],
            data["camera_row_end"],
            float(dy_px),
            y0,
            y1,
        )

        sg = _row_ncc(
            data["laser_row_edge"],
            data["camera_row_edge"],
            float(dy_px),
            y0,
            y1,
        )

        segment_scores.append(
            0.55 * float(se)
            + 0.45 * float(sg)
        )

    seg_arr = np.asarray(
        segment_scores,
        dtype=np.float64,
    )

    seg_median = float(
        np.median(seg_arr)
    )

    seg_dispersion = float(
        np.median(
            np.abs(
                seg_arr - seg_median
            )
        )
    )

    final_score = (
        float(main_score)
        + float(
            FEATURE_Y_SEGMENT_MEDIAN_WEIGHT
        )
        * float(seg_median)
        - float(
            FEATURE_Y_SEGMENT_DISPERSION_WEIGHT
        )
        * float(seg_dispersion)
    )

    return {
        "valid": True,
        "score": float(final_score),
        "main_score": float(main_score),
        "endface_ncc": float(endface_ncc),
        "edge_ncc": float(edge_ncc),
        "strong_dice": float(strong_dice),
        "row_end_ncc": float(row_end),
        "row_edge_ncc": float(row_edge),
        "row_profile_score": float(
            row_profile_score
        ),
        "segment_median_score": float(
            seg_median
        ),
        "segment_dispersion": float(
            seg_dispersion
        ),
        "valid_pixels": int(valid_count),
    }


def _parabolic_peak_refine(
    rows: list[dict],
    best_dy: float,
) -> float:
    """
    Three-point quadratic peak refinement.
    This can return a value between the 0.01 px fine-grid locations.
    """
    fine = [
        row
        for row in rows
        if (
            row.get("stage") == "fine"
            and row.get("valid", False)
            and np.isfinite(
                float(
                    row.get(
                        "score",
                        -1e9,
                    )
                )
            )
        )
    ]

    if len(fine) < 3:
        return float(best_dy)

    fine.sort(
        key=lambda row: float(
            row["dy_px"]
        )
    )

    index = min(
        range(len(fine)),
        key=lambda i: abs(
            float(fine[i]["dy_px"])
            - float(best_dy)
        ),
    )

    if index <= 0 or index >= len(fine) - 1:
        return float(best_dy)

    left = fine[index - 1]
    mid = fine[index]
    right = fine[index + 1]

    x1 = float(left["dy_px"])
    x2 = float(mid["dy_px"])
    x3 = float(right["dy_px"])

    y1 = float(left["score"])
    y2 = float(mid["score"])
    y3 = float(right["score"])

    step1 = x2 - x1
    step2 = x3 - x2

    if (
        abs(step1 - step2) > 1e-8
        or abs(step1) < 1e-12
    ):
        return float(best_dy)

    denominator = (
        y1
        - 2.0 * y2
        + y3
    )

    if abs(denominator) < 1e-12:
        return float(best_dy)

    delta = (
        0.5
        * (y1 - y3)
        / denominator
        * step1
    )

    delta = float(
        np.clip(
            delta,
            -abs(step1),
            +abs(step1),
        )
    )

    candidate = x2 + delta

    return float(
        np.clip(
            candidate,
            -float(FEATURE_Y_SEARCH_RANGE_PX),
            +float(FEATURE_Y_SEARCH_RANGE_PX),
        )
    )


def optimize_final_feature_auto_y(
    base,
    camera: Image.Image,
    mapped_height: Image.Image,
    mapped_mask: Image.Image,
    out_dir: Path,
    stem: str,
) -> dict:
    result = {
        "enabled": bool(
            USE_FINAL_FEATURE_AUTO_Y
        ),
        "applied": False,
        "reason": "disabled",
        "method": (
            "final_y_only_multimodal_endface_feature_"
            "correlation_fullheight_subpixel"
        ),
        "manual_y_used": False,
        "x_translation_locked_px": 0.0,
        "x_scale_locked": 1.0,
        "y_scale_locked": 1.0,
        "rotation_locked_deg": 0.0,
        "shear_locked": 0.0,
        "search_range_px": float(
            FEATURE_Y_SEARCH_RANGE_PX
        ),
        "coarse_step_px": float(
            FEATURE_Y_COARSE_STEP_PX
        ),
        "fine_step_px": float(
            FEATURE_Y_FINE_STEP_PX
        ),
        "applied_dx_px": 0.0,
        "applied_dy_px": 0.0,
    }

    if not USE_FINAL_FEATURE_AUTO_Y:
        return result

    if base.cv2 is None:
        result["reason"] = (
            "opencv_cv2_unavailable"
        )
        return result

    if (
        camera.size != mapped_height.size
        or camera.size != mapped_mask.size
    ):
        raise RuntimeError(
            "Feature Y-only alignment requires equal image sizes."
        )

    data = _build_feature_y_data(
        base,
        camera,
        mapped_height,
        mapped_mask,
    )

    result["analysis_width_px"] = int(
        data["analysis_width"]
    )
    result["analysis_height_px"] = int(
        data["height"]
    )
    result["laser_strong_threshold"] = float(
        data["laser_strong_threshold"]
    )
    result["camera_strong_threshold"] = float(
        data["camera_strong_threshold"]
    )

    rows: list[dict] = []

    coarse = _float_grid(
        -float(FEATURE_Y_SEARCH_RANGE_PX),
        +float(FEATURE_Y_SEARCH_RANGE_PX),
        float(FEATURE_Y_COARSE_STEP_PX),
    )

    best = None

    for dy in coarse:
        score = _score_feature_y(
            data,
            float(dy),
        )

        row = {
            "stage": "coarse",
            "dy_px": float(dy),
            **score,
        }

        rows.append(row)

        if (
            score.get("valid", False)
            and (
                best is None
                or float(score["score"])
                > float(best["score"])
            )
        ):
            best = row

    if best is None:
        result["reason"] = (
            "no_valid_feature_y_candidate"
        )
        return result

    fine_lo = max(
        -float(FEATURE_Y_SEARCH_RANGE_PX),
        float(best["dy_px"])
        - float(FEATURE_Y_FINE_RADIUS_PX),
    )

    fine_hi = min(
        +float(FEATURE_Y_SEARCH_RANGE_PX),
        float(best["dy_px"])
        + float(FEATURE_Y_FINE_RADIUS_PX),
    )

    fine = _float_grid(
        fine_lo,
        fine_hi,
        float(FEATURE_Y_FINE_STEP_PX),
    )

    for dy in fine:
        score = _score_feature_y(
            data,
            float(dy),
        )

        row = {
            "stage": "fine",
            "dy_px": float(dy),
            **score,
        }

        rows.append(row)

        if (
            score.get("valid", False)
            and float(score["score"])
            > float(best["score"])
        ):
            best = row

    grid_best_dy = float(
        best["dy_px"]
    )

    parabola_dy = _parabolic_peak_refine(
        rows,
        grid_best_dy,
    )

    parabola_score = _score_feature_y(
        data,
        float(parabola_dy),
    )

    if (
        parabola_score.get("valid", False)
        and float(parabola_score["score"])
        >= float(best["score"])
    ):
        best_dy = float(
            parabola_dy
        )
        best_score_dict = (
            parabola_score
        )
        result[
            "parabolic_subpixel_refinement_used"
        ] = True
    else:
        best_dy = float(
            grid_best_dy
        )
        best_score_dict = {
            key: value
            for key, value in best.items()
            if key
            not in (
                "stage",
                "dy_px",
            )
        }
        result[
            "parabolic_subpixel_refinement_used"
        ] = False

    zero_score_dict = _score_feature_y(
        data,
        0.0,
    )

    zero_score = float(
        zero_score_dict.get(
            "score",
            -1e9,
        )
    )

    best_score = float(
        best_score_dict.get(
            "score",
            -1e9,
        )
    )

    score_gain = float(
        best_score - zero_score
    )

    boundary_hit = bool(
        abs(best_dy)
        >= (
            float(FEATURE_Y_SEARCH_RANGE_PX)
            - float(FEATURE_Y_BOUNDARY_MARGIN_PX)
        )
    )

    result.update({
        "grid_best_dy_px": float(
            grid_best_dy
        ),
        "candidate_dy_px": float(
            best_dy
        ),
        "best_score": float(
            best_score
        ),
        "zero_score": float(
            zero_score
        ),
        "score_gain_vs_zero": float(
            score_gain
        ),
        "search_boundary_hit": bool(
            boundary_hit
        ),
        "best_components": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_),
                )
                else (
                    int(value)
                    if isinstance(
                        value,
                        (int, np.integer),
                    )
                    else (
                        float(value)
                        if isinstance(
                            value,
                            (
                                float,
                                np.floating,
                            ),
                        )
                        else value
                    )
                )
            )
            for key, value
            in best_score_dict.items()
        },
        "zero_components": {
            key: (
                bool(value)
                if isinstance(
                    value,
                    (bool, np.bool_),
                )
                else (
                    int(value)
                    if isinstance(
                        value,
                        (int, np.integer),
                    )
                    else (
                        float(value)
                        if isinstance(
                            value,
                            (
                                float,
                                np.floating,
                            ),
                        )
                        else value
                    )
                )
            )
            for key, value
            in zero_score_dict.items()
        },
    })

    if (
        boundary_hit
        and score_gain
        < float(
            FEATURE_Y_BOUNDARY_MIN_SCORE_GAIN
        )
    ):
        result["reason"] = (
            "weak_peak_at_y_search_boundary"
        )

    elif (
        not boundary_hit
        and score_gain
        < float(
            FEATURE_Y_MIN_SCORE_GAIN_NONBOUNDARY
        )
    ):
        result["reason"] = (
            "feature_score_not_better_than_zero"
        )

    elif abs(best_dy) < float(
        FEATURE_Y_MIN_EFFECT_PX
    ):
        result["reason"] = (
            "feature_y_residual_below_min_effect"
        )

    else:
        result["applied"] = True
        result["applied_dy_px"] = float(
            best_dy
        )
        result["reason"] = (
            "accepted_feature_auto_y"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_FEATURE_Y_SEARCH_CSV:
        csv_path = (
            out_dir
            / f"{stem}_feature_auto_y_search.csv"
        )

        fieldnames = [
            "stage",
            "dy_px",
            "score",
            "main_score",
            "endface_ncc",
            "edge_ncc",
            "strong_dice",
            "row_end_ncc",
            "row_edge_ncc",
            "row_profile_score",
            "segment_median_score",
            "segment_dispersion",
            "valid_pixels",
            "valid",
        ]

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()

            for row in rows:
                writer.writerow(row)

            writer.writerow({
                "stage": "parabolic_candidate",
                "dy_px": float(
                    parabola_dy
                ),
                **parabola_score,
            })

            writer.writerow({
                "stage": "zero_reference",
                "dy_px": 0.0,
                **zero_score_dict,
            })

        result["search_csv"] = str(
            csv_path
        )

    if SAVE_FEATURE_Y_JSON:
        json_path = (
            out_dir
            / f"{stem}_feature_auto_y.json"
        )

        json_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result["result_json"] = str(
            json_path
        )

    print("")
    print(
        "Final feature-correlation AUTO Y-only:"
    )
    print(
        f"  grid best dy = "
        f"{result.get('grid_best_dy_px', 0.0):+.3f}px"
    )
    print(
        f"  subpixel candidate dy = "
        f"{result.get('candidate_dy_px', 0.0):+.4f}px"
    )
    print(
        f"  zero score = "
        f"{result.get('zero_score', 0.0):.6f}"
    )
    print(
        f"  best score = "
        f"{result.get('best_score', 0.0):.6f}"
    )
    print(
        f"  score gain = "
        f"{result.get('score_gain_vs_zero', 0.0):+.6f}"
    )
    print(
        f"  FINAL applied dy = "
        f"{result.get('applied_dy_px', 0.0):+.4f}px"
    )
    print(
        f"  applied={result.get('applied', False)}, "
        f"reason={result.get('reason')}"
    )

    return result




# =============================================================================
# Y WIRE-PHASE ESTIMATION AND SEARCH
# =============================================================================


def _periodic_output_rows(
    center_y: float,
    window_height: int,
    height: int,
) -> np.ndarray:
    """Integer camera rows for one periodic local window."""
    wh = max(32, int(window_height))
    h = max(1, int(height))
    start = int(round(float(center_y) - 0.5 * wh))
    return np.mod(
        start + np.arange(wh, dtype=np.int64),
        h,
    )


def _sample_shifted_periodic_window_2d(
    source: np.ndarray,
    output_rows: np.ndarray,
    dy_px: float,
) -> np.ndarray:
    """
    Sample a source feature map after a Y-only content shift.

    Sign convention is identical to the global feature-Y stage:
        source_y = output_y - dy
    so positive dy moves laser content downward.
    """
    src = np.asarray(source, dtype=np.float32)
    h = int(src.shape[0])

    yy = np.mod(
        np.asarray(output_rows, dtype=np.float64) - float(dy_px),
        float(h),
    )
    y0f = np.floor(yy)
    y0 = y0f.astype(np.int64)
    y1 = (y0 + 1) % h
    w = (yy - y0f).astype(np.float32)

    if src.ndim == 1:
        return (
            src[y0] * (1.0 - w)
            + src[y1] * w
        ).astype(np.float32)

    return (
        src[y0, ...] * (1.0 - w[:, None])
        + src[y1, ...] * w[:, None]
    ).astype(np.float32)


def _sample_shifted_periodic_window_mask(
    source: np.ndarray,
    output_rows: np.ndarray,
    dy_px: float,
) -> np.ndarray:
    sampled = _sample_shifted_periodic_window_2d(
        np.asarray(source, dtype=np.float32),
        output_rows,
        float(dy_px),
    )
    return sampled >= 0.50


def _local_row_ncc_window(
    laser_row: np.ndarray,
    camera_row: np.ndarray,
    output_rows: np.ndarray,
    dy_px: float,
) -> float:
    shifted = _sample_shifted_periodic_window_2d(
        np.asarray(laser_row, dtype=np.float32),
        output_rows,
        float(dy_px),
    ).reshape(-1)

    camera = np.asarray(
        camera_row,
        dtype=np.float32,
    )[np.asarray(output_rows, dtype=np.int64)].reshape(-1)

    if len(camera) < 32:
        return -1.0

    shifted = shifted - float(np.mean(shifted))
    camera = camera - float(np.mean(camera))

    sa = float(np.std(shifted))
    sb = float(np.std(camera))

    if sa < 1e-8 or sb < 1e-8:
        return -1.0

    return float(
        np.mean(
            (shifted / sa)
            * (camera / sb)
        )
    )


def _score_local_feature_y_window(
    data: dict,
    output_rows: np.ndarray,
    dy_px: float,
) -> dict:
    rows = np.asarray(output_rows, dtype=np.int64)

    laser_end = _sample_shifted_periodic_window_2d(
        data["laser_end"],
        rows,
        float(dy_px),
    )
    laser_edge = _sample_shifted_periodic_window_2d(
        data["laser_edge"],
        rows,
        float(dy_px),
    )
    laser_strong = (
        _sample_shifted_periodic_window_2d(
            data["laser_strong"].astype(np.float32),
            rows,
            float(dy_px),
        )
        >= 0.50
    )
    shifted_mask = _sample_shifted_periodic_window_mask(
        data["mask"],
        rows,
        float(dy_px),
    )

    camera_end = data["camera_end"][rows]
    camera_edge = data["camera_edge"][rows]
    camera_strong = data["camera_strong"][rows]

    valid_pixels = int(np.count_nonzero(shifted_mask))
    if valid_pixels < int(LOCAL_FEATURE_Y_MIN_VALID_PIXELS):
        return {
            "valid": False,
            "score": -1e9,
            "valid_pixels": int(valid_pixels),
        }

    endface_ncc = _masked_ncc(
        laser_end,
        camera_end,
        shifted_mask,
    )
    edge_ncc = _masked_ncc(
        laser_edge,
        camera_edge,
        shifted_mask,
    )
    strong_dice = _dice_score(
        laser_strong,
        camera_strong,
        shifted_mask,
    )

    row_end = _local_row_ncc_window(
        data["laser_row_end"],
        data["camera_row_end"],
        rows,
        float(dy_px),
    )
    row_edge = _local_row_ncc_window(
        data["laser_row_edge"],
        data["camera_row_edge"],
        rows,
        float(dy_px),
    )
    row_profile = (
        0.55 * float(row_end)
        + 0.45 * float(row_edge)
    )

    score = (
        float(LOCAL_FEATURE_Y_WEIGHT_ENDFACE_2D)
        * float(endface_ncc)
        + float(LOCAL_FEATURE_Y_WEIGHT_EDGE_2D)
        * float(edge_ncc)
        + float(LOCAL_FEATURE_Y_WEIGHT_STRONG_DICE)
        * float(strong_dice)
        + float(LOCAL_FEATURE_Y_WEIGHT_ROW_PROFILE)
        * float(row_profile)
    )

    return {
        "valid": True,
        "score": float(score),
        "endface_ncc": float(endface_ncc),
        "edge_ncc": float(edge_ncc),
        "strong_dice": float(strong_dice),
        "row_end_ncc": float(row_end),
        "row_edge_ncc": float(row_edge),
        "row_profile_score": float(row_profile),
        "valid_pixels": int(valid_pixels),
    }


def _refine_local_peak_parabola(
    candidate_rows: list[dict],
    best_dy: float,
) -> float:
    valid = [
        row
        for row in candidate_rows
        if (
            row.get("valid", False)
            and np.isfinite(float(row.get("score", -1e9)))
        )
    ]
    if len(valid) < 3:
        return float(best_dy)

    valid.sort(key=lambda row: float(row["dy_px"]))
    index = min(
        range(len(valid)),
        key=lambda i: abs(float(valid[i]["dy_px"]) - float(best_dy)),
    )

    if index <= 0 or index >= len(valid) - 1:
        return float(best_dy)

    left = valid[index - 1]
    mid = valid[index]
    right = valid[index + 1]

    x1, x2, x3 = (
        float(left["dy_px"]),
        float(mid["dy_px"]),
        float(right["dy_px"]),
    )
    y1, y2, y3 = (
        float(left["score"]),
        float(mid["score"]),
        float(right["score"]),
    )

    step1 = x2 - x1
    step2 = x3 - x2

    if abs(step1 - step2) > 1e-8 or abs(step1) < 1e-12:
        return float(best_dy)

    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-12:
        return float(best_dy)

    delta = 0.5 * (y1 - y3) / denom * step1
    delta = float(
        np.clip(
            delta,
            -abs(step1),
            +abs(step1),
        )
    )

    return float(
        np.clip(
            x2 + delta,
            -float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
            +float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
        )
    )


def _periodic_interpolate_nan_windows(
    centers: np.ndarray,
    values: np.ndarray,
    height: int,
) -> np.ndarray:
    """Fill weak local windows by periodic interpolation from reliable ones."""
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    h = float(height)

    valid = np.isfinite(values)

    if int(np.count_nonzero(valid)) == 0:
        return np.zeros(len(values), dtype=np.float64)

    if int(np.count_nonzero(valid)) == 1:
        return np.full(
            len(values),
            float(values[valid][0]),
            dtype=np.float64,
        )

    xv = centers[valid]
    vv = values[valid]
    order = np.argsort(xv)
    xv = xv[order]
    vv = vv[order]

    xext = np.concatenate([
        xv - h,
        xv,
        xv + h,
    ])
    vext = np.concatenate([
        vv,
        vv,
        vv,
    ])

    return np.interp(
        centers,
        xext,
        vext,
    ).astype(np.float64)


def _circular_median_filter(values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return arr.copy()

    w = max(1, int(width))
    if w % 2 == 0:
        w += 1
    radius = w // 2

    out = np.empty_like(arr)
    for i in range(n):
        idx = [
            (i + j) % n
            for j in range(-radius, radius + 1)
        ]
        out[i] = float(np.median(arr[idx]))

    return out


def _enforce_periodic_jump_limit(
    values: np.ndarray,
    max_jump: float,
    passes: int = 4,
) -> np.ndarray:
    """
    Conservative circular jump limiter.

    It never changes a point by a whole wire pitch: every neighbouring window
    is constrained to the configured subpixel-scale jump.
    """
    arr = np.asarray(values, dtype=np.float64).copy()
    n = len(arr)
    if n < 2:
        return arr

    jump = max(1e-6, float(max_jump))

    for _ in range(max(1, int(passes))):
        # Forward.
        for i in range(1, n):
            arr[i] = np.clip(
                arr[i],
                arr[i - 1] - jump,
                arr[i - 1] + jump,
            )

        # Circular last -> first.
        arr[0] = np.clip(
            arr[0],
            arr[-1] - jump,
            arr[-1] + jump,
        )

        # Backward.
        for i in range(n - 2, -1, -1):
            arr[i] = np.clip(
                arr[i],
                arr[i + 1] - jump,
                arr[i + 1] + jump,
            )

        # Circular first -> last.
        arr[-1] = np.clip(
            arr[-1],
            arr[0] - jump,
            arr[0] + jump,
        )

    return arr


def _circular_smooth_window_values(
    values: np.ndarray,
    width: int,
    passes: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    n = len(arr)
    if n < 3:
        return arr

    w = max(1, int(width))
    if w % 2 == 0:
        w += 1
    radius = w // 2

    # Triangular weights: for width=5 -> 1,2,3,2,1.
    weights = np.asarray(
        [
            radius + 1 - abs(j)
            for j in range(-radius, radius + 1)
        ],
        dtype=np.float64,
    )
    weights /= float(np.sum(weights))

    for _ in range(max(0, int(passes))):
        out = np.empty_like(arr)
        for i in range(n):
            vals = np.asarray(
                [
                    arr[(i + j) % n]
                    for j in range(-radius, radius + 1)
                ],
                dtype=np.float64,
            )
            out[i] = float(np.sum(vals * weights))
        arr = out

    return arr


def _build_periodic_local_curve(
    centers: np.ndarray,
    values: np.ndarray,
    height: int,
) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if len(centers) == 0:
        return np.zeros(int(height), dtype=np.float32)

    if len(centers) == 1:
        return np.full(
            int(height),
            float(values[0]),
            dtype=np.float32,
        )

    order = np.argsort(centers)
    centers = centers[order]
    values = values[order]

    h = float(height)

    xext = np.concatenate([
        centers - h,
        centers,
        centers + h,
    ])
    vext = np.concatenate([
        values,
        values,
        values,
    ])

    yy = np.arange(
        int(height),
        dtype=np.float64,
    )

    curve = np.interp(
        yy,
        xext,
        vext,
    )

    return np.clip(
        curve,
        -float(LOCAL_FEATURE_Y_MAX_ABS_PX),
        +float(LOCAL_FEATURE_Y_MAX_ABS_PX),
    ).astype(np.float32)


def _robust_mad_sigma(values: np.ndarray, fallback: float) -> float:
    """Robust sigma estimate 1.4826*MAD with a deterministic fallback."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float(fallback)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = float(np.std(arr)) if arr.size >= 2 else float(fallback)
    if not np.isfinite(sigma) or sigma <= 1e-9:
        sigma = float(fallback)
    return float(sigma)


def _kalman_filter_local_y(
    centers: np.ndarray,
    raw_values: np.ndarray,
    window_rows: list[dict],
    registration_rmse_px: float | None,
) -> tuple[np.ndarray, dict]:
    """Uncertainty-aware periodic Kalman filtering of local camera-LLTS Y residuals.

    State:
        x_k = [dy_k, v_k]^T
    where dy_k is the local camera/LLTS registration residual in pixels and v_k
    is its slow change per local window.

    Measurement:
        z_k = raw_dy_k

    R_k is confidence dependent.  Missing/rejected windows perform prediction
    only.  The sequence is traversed for two circular cycles and the second
    cycle is retained so the 0/360-degree boundary does not behave like a cold
    start.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    n = int(len(raw))

    report = {
        "enabled": bool(USE_KALMAN_LOCAL_Y),
        "available": False,
        "reason": "disabled",
        "model": str(KALMAN_LOCAL_Y_MODEL),
        "state": ["local_dy_px", "local_dy_change_per_window_px"],
        "measurement": "accepted_local_feature_raw_dy_px",
        "periodic_cycles": int(KALMAN_LOCAL_Y_PERIODIC_CYCLES),
        "registration_rmse_px_for_initial_p": (
            None if registration_rmse_px is None else float(registration_rmse_px)
        ),
        "window_rows": [],
    }

    if not USE_KALMAN_LOCAL_Y:
        return np.zeros(n, dtype=np.float64), report

    if n == 0:
        report["reason"] = "no_local_windows"
        return np.zeros(0, dtype=np.float64), report

    accepted = np.isfinite(raw)
    accepted_count = int(np.count_nonzero(accepted))
    report["accepted_measurement_count"] = int(accepted_count)
    report["window_count"] = int(n)

    if accepted_count < 2:
        report["reason"] = "too_few_accepted_measurements"
        fallback = _periodic_interpolate_nan_windows(
            centers,
            raw,
            max(1, int(round(float(centers[-1] + (centers[1]-centers[0] if n > 1 else 1.0)))))
            if n else 1,
        ) if accepted_count else np.zeros(n, dtype=np.float64)
        return np.clip(
            fallback,
            -float(KALMAN_LOCAL_Y_MAX_ABS_PX),
            +float(KALMAN_LOCAL_Y_MAX_ABS_PX),
        ), report

    # --------------------------------------------------------------
    # Measurement uncertainty R_k from local feature-match confidence.
    # --------------------------------------------------------------
    gains = np.full(n, np.nan, dtype=np.float64)
    boundary = np.zeros(n, dtype=bool)
    for i, row in enumerate(window_rows[:n]):
        try:
            g = row.get("score_gain_vs_zero")
            if g is not None and np.isfinite(float(g)):
                gains[i] = float(g)
        except Exception:
            pass
        boundary[i] = bool(row.get("boundary_hit", False))

    reliable_gains = gains[accepted & np.isfinite(gains)]
    if reliable_gains.size:
        gain_lo, gain_hi = np.percentile(reliable_gains, [20.0, 80.0])
        gain_lo = float(gain_lo)
        gain_hi = float(gain_hi)
    else:
        gain_lo, gain_hi = 0.0, 1.0

    sensor_floor = math.hypot(
        float(KALMAN_CAMERA_LOCALIZATION_SIGMA_PX),
        float(KALMAN_LLTS_LOCALIZATION_SIGMA_PX),
    )

    measurement_sigma = np.full(n, np.nan, dtype=np.float64)
    quality = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not accepted[i]:
            continue
        if np.isfinite(gains[i]) and gain_hi > gain_lo + 1e-12:
            q = float(np.clip((gains[i] - gain_lo) / (gain_hi - gain_lo), 0.0, 1.0))
        else:
            q = 0.5
        quality[i] = q
        sigma_conf = (
            float(KALMAN_LOCAL_Y_MAX_MEAS_SIGMA_PX)
            - q * (
                float(KALMAN_LOCAL_Y_MAX_MEAS_SIGMA_PX)
                - float(KALMAN_LOCAL_Y_MIN_MEAS_SIGMA_PX)
            )
        )
        if boundary[i]:
            sigma_conf *= float(KALMAN_LOCAL_Y_BOUNDARY_SIGMA_MULTIPLIER)
        sigma = math.sqrt(sigma_conf * sigma_conf + sensor_floor * sensor_floor)
        measurement_sigma[i] = float(max(sigma, 1e-6))

    # --------------------------------------------------------------
    # Process uncertainty Q from accepted neighbouring residual changes.
    # Only physically adjacent accepted windows contribute.
    # --------------------------------------------------------------
    neighbour_diffs = []
    for i in range(n):
        j = (i + 1) % n
        if accepted[i] and accepted[j]:
            neighbour_diffs.append(float(raw[j] - raw[i]))

    process_sigma = _robust_mad_sigma(
        np.asarray(neighbour_diffs, dtype=np.float64),
        float(KALMAN_LOCAL_Y_PROCESS_SIGMA_FALLBACK_PX),
    )
    process_sigma = float(np.clip(
        process_sigma,
        float(KALMAN_LOCAL_Y_PROCESS_SIGMA_MIN_PX),
        float(KALMAN_LOCAL_Y_PROCESS_SIGMA_MAX_PX),
    ))

    # Constant-velocity model with unit step between local windows.
    F = np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    H = np.asarray([[1.0, 0.0]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)

    q2 = process_sigma * process_sigma
    Q = q2 * np.asarray(
        [[0.25, 0.50], [0.50, 1.00]],
        dtype=np.float64,
    )

    # Start at the strongest accepted local observation, which avoids beginning
    # at a weak/occluded 0-degree seam location.
    accepted_idx = np.flatnonzero(accepted)
    start_index = int(accepted_idx[0])
    if reliable_gains.size:
        valid_gain_idx = [
            int(i) for i in accepted_idx
            if np.isfinite(gains[int(i)])
        ]
        if valid_gain_idx:
            start_index = max(valid_gain_idx, key=lambda i: float(gains[i]))

    if (
        bool(KALMAN_LOCAL_Y_USE_AFFINE_RMSE_FOR_INITIAL_P)
        and registration_rmse_px is not None
        and np.isfinite(float(registration_rmse_px))
        and float(registration_rmse_px) > 0.0
    ):
        initial_sigma = float(registration_rmse_px)
        initial_sigma_source = "physical_affine_rmse_mm_converted_to_circular_y_px"
    else:
        initial_sigma = float(KALMAN_LOCAL_Y_INITIAL_SIGMA_FALLBACK_PX)
        initial_sigma_source = "configured_fallback"

    initial_sigma = float(np.clip(
        initial_sigma,
        float(KALMAN_LOCAL_Y_INITIAL_SIGMA_MIN_PX),
        float(KALMAN_LOCAL_Y_INITIAL_SIGMA_MAX_PX),
    ))

    x = np.asarray([float(raw[start_index]), 0.0], dtype=np.float64)
    P = np.diag([
        initial_sigma * initial_sigma,
        float(KALMAN_LOCAL_Y_INITIAL_VELOCITY_SIGMA_PX) ** 2,
    ]).astype(np.float64)

    base_order = np.concatenate([
        np.arange(start_index, n, dtype=np.int32),
        np.arange(0, start_index, dtype=np.int32),
    ])
    cycles = max(2, int(KALMAN_LOCAL_Y_PERIODIC_CYCLES))
    order = np.tile(base_order, cycles)

    step_records: list[dict] = []
    for step_no, idx_raw in enumerate(order):
        idx = int(idx_raw)

        # Prediction.
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        used_measurement = bool(accepted[idx])
        innovation = None
        gain_position = 0.0
        gain_velocity = 0.0
        r_value = None

        if used_measurement:
            z = float(raw[idx])
            sigma = float(measurement_sigma[idx])
            R = sigma * sigma
            S = float((H @ P_pred @ H.T)[0, 0] + R)
            if np.isfinite(S) and S > 1e-12:
                K = (P_pred @ H.T) / S
                innovation = float(z - (H @ x_pred)[0])
                x = x_pred + K[:, 0] * innovation
                # Joseph form is stable and keeps P positive semidefinite.
                KH = K @ H
                P = (
                    (I - KH) @ P_pred @ (I - KH).T
                    + (K * R) @ K.T
                )
                gain_position = float(K[0, 0])
                gain_velocity = float(K[1, 0])
                r_value = float(R)
            else:
                x = x_pred
                P = P_pred
                used_measurement = False
        else:
            x = x_pred
            P = P_pred

        x[0] = float(np.clip(
            x[0],
            -float(KALMAN_LOCAL_Y_MAX_ABS_PX),
            +float(KALMAN_LOCAL_Y_MAX_ABS_PX),
        ))

        step_records.append({
            "step_no": int(step_no),
            "cycle_index": int(step_no // n),
            "window_index": int(idx),
            "center_y_px": float(centers[idx]),
            "used_measurement": bool(used_measurement),
            "raw_dy_px": None if not accepted[idx] else float(raw[idx]),
            "score_gain_vs_zero": None if not np.isfinite(gains[idx]) else float(gains[idx]),
            "measurement_quality_0to1": float(quality[idx]) if accepted[idx] else None,
            "measurement_sigma_px": None if not np.isfinite(measurement_sigma[idx]) else float(measurement_sigma[idx]),
            "measurement_variance_r_px2": r_value,
            "predicted_dy_px": float(x_pred[0]),
            "innovation_px": innovation,
            "kalman_gain_position": float(gain_position),
            "kalman_gain_velocity": float(gain_velocity),
            "filtered_dy_px": float(x[0]),
            "filtered_velocity_px_per_window": float(x[1]),
            "posterior_position_sigma_px": float(math.sqrt(max(P[0, 0], 0.0))),
        })

    # Use the last circular cycle after the first cycle has warmed the state
    # through the 0/360-degree boundary.
    selected_steps = step_records[-n:]
    filtered = np.zeros(n, dtype=np.float64)
    row_by_index: dict[int, dict] = {}
    for row in selected_steps:
        idx = int(row["window_index"])
        filtered[idx] = float(row["filtered_dy_px"])
        row_by_index[idx] = row

    filtered = np.clip(
        filtered,
        -float(KALMAN_LOCAL_Y_MAX_ABS_PX),
        +float(KALMAN_LOCAL_Y_MAX_ABS_PX),
    )

    report_rows = []
    for i in range(n):
        row = dict(row_by_index.get(i, {}))
        row.setdefault("window_index", int(i))
        row.setdefault("center_y_px", float(centers[i]))
        row["accepted_by_feature_gate"] = bool(accepted[i])
        row["boundary_hit"] = bool(boundary[i])
        report_rows.append(row)

    report.update({
        "available": True,
        "reason": "ok",
        "start_window_index": int(start_index),
        "initial_position_sigma_px": float(initial_sigma),
        "initial_position_sigma_source": str(initial_sigma_source),
        "process_sigma_px_per_window": float(process_sigma),
        "process_covariance_Q": Q.tolist(),
        "measurement_gain_percentile_20": float(gain_lo),
        "measurement_gain_percentile_80": float(gain_hi),
        "sensor_sigma_floor_px": float(sensor_floor),
        "measurement_sigma_min_used_px": (
            float(np.nanmin(measurement_sigma)) if np.any(np.isfinite(measurement_sigma)) else None
        ),
        "measurement_sigma_max_used_px": (
            float(np.nanmax(measurement_sigma)) if np.any(np.isfinite(measurement_sigma)) else None
        ),
        "filtered_window_min_px": float(np.min(filtered)),
        "filtered_window_max_px": float(np.max(filtered)),
        "filtered_window_mean_px": float(np.mean(filtered)),
        "window_rows": report_rows,
    })

    return filtered.astype(np.float64), report



def _kalman_filter_local_y_1d(
    centers: np.ndarray,
    raw_values: np.ndarray,
    window_rows: list[dict],
    registration_rmse_px: float | None,
) -> tuple[np.ndarray, dict]:
    """Recommended uncertainty-aware 1-D random-walk Kalman local-Y filter.

    State:
        x_k = dy_k

    Process model:
        dy_k = dy_(k-1) + w_k

    Measurement:
        z_k = accepted local-feature raw_dy_px

    The measurement variance R_k is adapted from local feature confidence.
    Search-boundary observations are explicitly down-weighted because the true
    optimum can lie outside the finite local search interval. Missing/rejected
    windows perform prediction only. Several circular passes are used and only
    the last pass is retained, avoiding a cold-start discontinuity at 0/360 deg.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    n = int(len(raw))

    report = {
        "enabled": bool(USE_KALMAN_1D_LOCAL_Y),
        "available": False,
        "reason": "disabled",
        "model": str(KALMAN_1D_LOCAL_Y_MODEL),
        "state": ["local_dy_px"],
        "measurement": "accepted_local_feature_raw_dy_px",
        "periodic_cycles": int(KALMAN_1D_PERIODIC_CYCLES),
        "registration_rmse_px_for_initial_p": (
            None if registration_rmse_px is None else float(registration_rmse_px)
        ),
        "window_rows": [],
    }

    if not USE_KALMAN_1D_LOCAL_Y:
        return np.zeros(n, dtype=np.float64), report
    if n == 0:
        report["reason"] = "no_local_windows"
        return np.zeros(0, dtype=np.float64), report

    accepted = np.isfinite(raw)
    accepted_count = int(np.count_nonzero(accepted))
    report["accepted_measurement_count"] = accepted_count
    report["window_count"] = n

    if accepted_count < 2:
        report["reason"] = "too_few_accepted_measurements"
        if accepted_count == 0:
            return np.zeros(n, dtype=np.float64), report
        only = float(raw[np.flatnonzero(accepted)[0]])
        return np.full(
            n,
            np.clip(only, -float(KALMAN_1D_MAX_ABS_PX), +float(KALMAN_1D_MAX_ABS_PX)),
            dtype=np.float64,
        ), report

    # --------------------------------------------------------------
    # R_k: feature-confidence-dependent measurement uncertainty.
    # --------------------------------------------------------------
    gains = np.full(n, np.nan, dtype=np.float64)
    boundary = np.zeros(n, dtype=bool)
    feature_accepted = np.zeros(n, dtype=bool)
    for i, row in enumerate(window_rows[:n]):
        feature_accepted[i] = bool(row.get("accepted", False))
        boundary[i] = bool(row.get("boundary_hit", False))
        try:
            g = row.get("score_gain_vs_zero")
            if g is not None and np.isfinite(float(g)):
                gains[i] = float(g)
        except Exception:
            pass

    reliable_gains = gains[accepted & np.isfinite(gains)]
    if reliable_gains.size:
        gain_lo, gain_hi = np.percentile(reliable_gains, [20.0, 80.0])
        gain_lo, gain_hi = float(gain_lo), float(gain_hi)
    else:
        gain_lo, gain_hi = 0.0, 1.0

    sensor_floor = math.hypot(
        float(KALMAN_CAMERA_LOCALIZATION_SIGMA_PX),
        float(KALMAN_LLTS_LOCALIZATION_SIGMA_PX),
    )

    measurement_sigma = np.full(n, np.nan, dtype=np.float64)
    quality = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not accepted[i]:
            continue
        if np.isfinite(gains[i]) and gain_hi > gain_lo + 1e-12:
            q = float(np.clip((gains[i] - gain_lo) / (gain_hi - gain_lo), 0.0, 1.0))
        else:
            q = 0.5
        quality[i] = q

        sigma_conf = (
            float(KALMAN_1D_MAX_MEAS_SIGMA_PX)
            - q * (
                float(KALMAN_1D_MAX_MEAS_SIGMA_PX)
                - float(KALMAN_1D_MIN_MEAS_SIGMA_PX)
            )
        )

        # A boundary optimum is censored.  It is never allowed to receive the
        # extremely small sigma of an unconstrained high-confidence optimum.
        if boundary[i]:
            sigma_conf = max(
                sigma_conf * float(KALMAN_1D_BOUNDARY_SIGMA_MULTIPLIER),
                float(KALMAN_1D_BOUNDARY_MIN_SIGMA_PX),
            )

        sigma = math.sqrt(sigma_conf * sigma_conf + sensor_floor * sensor_floor)
        measurement_sigma[i] = float(max(sigma, 1e-6))

    # --------------------------------------------------------------
    # Q: estimate random-walk change from adjacent reliable measurements.
    # Prefer NON-boundary pairs, because boundary values are clipped/censored.
    # --------------------------------------------------------------
    diffs_non_boundary: list[float] = []
    diffs_all: list[float] = []
    for i in range(n):
        j = (i + 1) % n
        if accepted[i] and accepted[j]:
            d = float(raw[j] - raw[i])
            diffs_all.append(d)
            if not boundary[i] and not boundary[j]:
                diffs_non_boundary.append(d)

    q_source = "adjacent_non_boundary_raw_dy"
    q_values = np.asarray(diffs_non_boundary, dtype=np.float64)
    if q_values.size < 4:
        q_source = "adjacent_all_raw_dy_fallback"
        q_values = np.asarray(diffs_all, dtype=np.float64)

    process_sigma = _robust_mad_sigma(
        q_values,
        float(KALMAN_1D_PROCESS_SIGMA_FALLBACK_PX),
    )
    process_sigma = float(np.clip(
        process_sigma,
        float(KALMAN_1D_PROCESS_SIGMA_MIN_PX),
        float(KALMAN_1D_PROCESS_SIGMA_MAX_PX),
    ))
    Q = float(process_sigma * process_sigma)

    # Start at the strongest accepted NON-boundary measurement if possible.
    accepted_idx = [int(i) for i in np.flatnonzero(accepted)]
    non_boundary_idx = [i for i in accepted_idx if not boundary[i]]
    candidate_start = non_boundary_idx if non_boundary_idx else accepted_idx
    finite_gain_start = [i for i in candidate_start if np.isfinite(gains[i])]
    if finite_gain_start:
        start_index = max(finite_gain_start, key=lambda i: float(gains[i]))
    else:
        start_index = int(candidate_start[0])

    if (
        bool(KALMAN_1D_USE_AFFINE_RMSE_FOR_INITIAL_P)
        and registration_rmse_px is not None
        and np.isfinite(float(registration_rmse_px))
        and float(registration_rmse_px) > 0.0
    ):
        initial_sigma = float(registration_rmse_px)
        initial_sigma_source = "physical_affine_rmse_mm_converted_to_circular_y_px"
    else:
        initial_sigma = float(KALMAN_1D_INITIAL_SIGMA_FALLBACK_PX)
        initial_sigma_source = "configured_fallback"

    initial_sigma = float(np.clip(
        initial_sigma,
        float(KALMAN_1D_INITIAL_SIGMA_MIN_PX),
        float(KALMAN_1D_INITIAL_SIGMA_MAX_PX),
    ))

    x = float(raw[start_index])
    P = float(initial_sigma * initial_sigma)

    base_order = np.concatenate([
        np.arange(start_index, n, dtype=np.int32),
        np.arange(0, start_index, dtype=np.int32),
    ])
    cycles = max(2, int(KALMAN_1D_PERIODIC_CYCLES))
    order = np.tile(base_order, cycles)

    step_records: list[dict] = []
    for step_no, idx_raw in enumerate(order):
        idx = int(idx_raw)

        # Random-walk prediction: no velocity/trend extrapolation.
        x_pred = float(x)
        P_pred = float(P + Q)

        used_measurement = bool(accepted[idx])
        innovation = None
        K = 0.0
        R = None

        if used_measurement:
            z = float(raw[idx])
            sigma = float(measurement_sigma[idx])
            R = float(sigma * sigma)
            S = float(P_pred + R)
            if np.isfinite(S) and S > 1e-12:
                K = float(P_pred / S)
                innovation = float(z - x_pred)
                x = float(x_pred + K * innovation)
                # Scalar Joseph-equivalent update.
                P = float((1.0 - K) * P_pred)
            else:
                x = x_pred
                P = P_pred
                used_measurement = False
        else:
            x = x_pred
            P = P_pred

        x = float(np.clip(
            x,
            -float(KALMAN_1D_MAX_ABS_PX),
            +float(KALMAN_1D_MAX_ABS_PX),
        ))

        step_records.append({
            "step_no": int(step_no),
            "cycle_index": int(step_no // n),
            "window_index": int(idx),
            "center_y_px": float(centers[idx]),
            "used_measurement": bool(used_measurement),
            "accepted_by_feature_gate": bool(feature_accepted[idx]),
            "boundary_hit": bool(boundary[idx]),
            "raw_dy_px": None if not accepted[idx] else float(raw[idx]),
            "score_gain_vs_zero": None if not np.isfinite(gains[idx]) else float(gains[idx]),
            "measurement_quality_0to1": float(quality[idx]) if accepted[idx] else None,
            "measurement_sigma_px": None if not np.isfinite(measurement_sigma[idx]) else float(measurement_sigma[idx]),
            "measurement_variance_r_px2": R,
            "predicted_dy_px": float(x_pred),
            "predicted_position_variance_px2": float(P_pred),
            "predicted_position_sigma_px": float(math.sqrt(max(P_pred, 0.0))),
            "innovation_px": innovation,
            "kalman_gain_position": float(K),
            "filtered_dy_px": float(x),
            "posterior_position_variance_px2": float(P),
            "posterior_position_sigma_px": float(math.sqrt(max(P, 0.0))),
        })

    # Keep only the final circular pass after warm-up through the seam.
    selected_steps = step_records[-n:]
    filtered = np.zeros(n, dtype=np.float64)
    row_by_index: dict[int, dict] = {}
    for row in selected_steps:
        wi = int(row["window_index"])
        filtered[wi] = float(row["filtered_dy_px"])
        row_by_index[wi] = dict(row)

    filtered = np.clip(
        filtered,
        -float(KALMAN_1D_MAX_ABS_PX),
        +float(KALMAN_1D_MAX_ABS_PX),
    )

    report_rows: list[dict] = []
    for i in range(n):
        row = dict(row_by_index.get(i, {}))
        row.setdefault("window_index", int(i))
        row.setdefault("center_y_px", float(centers[i]))
        row.setdefault("accepted_by_feature_gate", bool(feature_accepted[i]))
        row.setdefault("boundary_hit", bool(boundary[i]))
        report_rows.append(row)

    report.update({
        "available": True,
        "reason": "ok",
        "start_window_index": int(start_index),
        "initial_position_sigma_px": float(initial_sigma),
        "initial_position_sigma_source": str(initial_sigma_source),
        "process_sigma_px_per_window": float(process_sigma),
        "process_variance_Q_px2": float(Q),
        "process_sigma_source": str(q_source),
        "measurement_gain_percentile_20": float(gain_lo),
        "measurement_gain_percentile_80": float(gain_hi),
        "sensor_sigma_floor_px": float(sensor_floor),
        "boundary_sigma_multiplier": float(KALMAN_1D_BOUNDARY_SIGMA_MULTIPLIER),
        "boundary_min_sigma_px": float(KALMAN_1D_BOUNDARY_MIN_SIGMA_PX),
        "boundary_measurement_count": int(np.count_nonzero(boundary & accepted)),
        "measurement_sigma_min_used_px": (
            float(np.nanmin(measurement_sigma)) if np.any(np.isfinite(measurement_sigma)) else None
        ),
        "measurement_sigma_max_used_px": (
            float(np.nanmax(measurement_sigma)) if np.any(np.isfinite(measurement_sigma)) else None
        ),
        "filtered_window_min_px": float(np.min(filtered)),
        "filtered_window_max_px": float(np.max(filtered)),
        "filtered_window_mean_px": float(np.mean(filtered)),
        "filtered_window_rms_px": float(np.sqrt(np.mean(filtered * filtered))),
        "window_rows": report_rows,
        # Two final circular passes are retained only for fixed-interval RTS.
        # The first of these two passes is used as the reported smoothed cycle;
        # the second supplies future information across the 0/360-degree seam.
        "rts_forward_tail_two_cycles": [
            dict(row) for row in step_records[-min(len(step_records), 2 * n):]
        ],
    })

    return filtered.astype(np.float64), report


def _rts_smooth_local_y_1d(
    centers: np.ndarray,
    forward_report: dict,
) -> tuple[np.ndarray, dict]:
    """Fixed-interval scalar RTS smoother for the 1-D random-walk Kalman result.

    The final two circular forward passes are smoothed backward.  We retain the
    first of those two passes, because every retained point then has one full
    future revolution available, including across the 0/360-degree seam.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    n = int(len(centers))
    result = {
        "enabled": bool(USE_KALMAN_RTS_LOCAL_Y),
        "available": False,
        "reason": "disabled",
        "model": str(KALMAN_RTS_MODEL),
        "state": ["local_dy_px"],
        "source": "final_two_circular_passes_of_1d_adaptive_kalman",
        "window_rows": [],
    }

    if not USE_KALMAN_RTS_LOCAL_Y:
        return np.zeros(n, dtype=np.float64), result
    if n == 0:
        result["reason"] = "no_local_windows"
        return np.zeros(0, dtype=np.float64), result
    if not bool(forward_report.get("available", False)):
        result["reason"] = "forward_kalman_unavailable"
        return np.zeros(n, dtype=np.float64), result

    tail = [dict(r) for r in forward_report.get("rts_forward_tail_two_cycles", [])]
    if len(tail) < 2 * n:
        result["reason"] = f"insufficient_forward_tail:{len(tail)}<{2*n}"
        return np.zeros(n, dtype=np.float64), result

    # Keep exactly the last two complete cycles.
    tail = tail[-2 * n:]
    m = len(tail)
    x_f = np.asarray([float(r["filtered_dy_px"]) for r in tail], dtype=np.float64)
    P_f = np.asarray([float(r["posterior_position_variance_px2"]) for r in tail], dtype=np.float64)
    x_pred = np.asarray([float(r["predicted_dy_px"]) for r in tail], dtype=np.float64)
    P_pred = np.asarray([float(r["predicted_position_variance_px2"]) for r in tail], dtype=np.float64)

    x_s = x_f.copy()
    P_s = P_f.copy()
    gains = np.zeros(m - 1, dtype=np.float64)

    # Standard scalar RTS backward recursion for F=1.
    for t in range(m - 2, -1, -1):
        denom = float(P_pred[t + 1])
        if not np.isfinite(denom) or denom <= 1e-12:
            C = 0.0
        else:
            C = float(P_f[t] / denom)
            C = float(np.clip(C, 0.0, 1.0))
        gains[t] = C
        x_s[t] = float(
            x_f[t]
            + C * (x_s[t + 1] - x_pred[t + 1])
        )
        P_s[t] = float(
            P_f[t]
            + C * C * (P_s[t + 1] - P_pred[t + 1])
        )
        P_s[t] = max(P_s[t], 0.0)

    x_s = np.clip(
        x_s,
        -float(KALMAN_RTS_MAX_ABS_PX),
        +float(KALMAN_RTS_MAX_ABS_PX),
    )

    # Retain the first of the final two cycles. It has a complete future cycle
    # available, which removes the ordinary backward-smoother seam endpoint.
    retained = tail[:n]
    retained_x = x_s[:n]
    retained_P = P_s[:n]
    retained_C = gains[:n] if len(gains) >= n else np.pad(gains, (0, n-len(gains)))

    smoothed = np.zeros(n, dtype=np.float64)
    rows_by_index: dict[int, dict] = {}
    for j, row in enumerate(retained):
        wi = int(row["window_index"])
        smoothed[wi] = float(retained_x[j])
        rr = dict(row)
        rr.update({
            "rts_smoother_gain": float(retained_C[j]) if j < len(retained_C) else None,
            "rts_smoothed_dy_px": float(retained_x[j]),
            "rts_smoothed_position_variance_px2": float(retained_P[j]),
            "rts_smoothed_position_sigma_px": float(math.sqrt(max(float(retained_P[j]), 0.0))),
            "rts_minus_forward_px": float(retained_x[j] - float(row["filtered_dy_px"])),
        })
        rows_by_index[wi] = rr

    report_rows: list[dict] = []
    for i in range(n):
        row = dict(rows_by_index.get(i, {}))
        row.setdefault("window_index", int(i))
        row.setdefault("center_y_px", float(centers[i]))
        report_rows.append(row)

    smoothed = np.clip(
        smoothed,
        -float(KALMAN_RTS_MAX_ABS_PX),
        +float(KALMAN_RTS_MAX_ABS_PX),
    )

    forward_by_index = {
        int(r.get("window_index", -1)): float(r.get("filtered_dy_px", 0.0))
        for r in forward_report.get("window_rows", [])
    }
    forward_natural = np.asarray(
        [forward_by_index.get(i, smoothed[i]) for i in range(n)],
        dtype=np.float64,
    )

    result.update({
        "available": True,
        "reason": "ok",
        "retained_cycle": "first_of_final_two_forward_cycles",
        "uses_future_full_revolution": True,
        "smoothed_window_min_px": float(np.min(smoothed)),
        "smoothed_window_max_px": float(np.max(smoothed)),
        "smoothed_window_mean_px": float(np.mean(smoothed)),
        "smoothed_window_rms_px": float(np.sqrt(np.mean(smoothed * smoothed))),
        "mean_abs_rts_minus_forward_px": float(np.mean(np.abs(smoothed - forward_natural))),
        "window_rows": report_rows,
    })
    return smoothed.astype(np.float64), result


def _save_kalman_1d_diagnostics(
    out_dir: Path,
    stem: str,
    centers: np.ndarray,
    baseline_window_values: np.ndarray,
    cv_window_values: np.ndarray,
    one_d_window_values: np.ndarray,
    baseline_curve: np.ndarray,
    cv_curve: np.ndarray,
    one_d_curve: np.ndarray,
    report: dict,
) -> dict:
    """Save paper-ready diagnostics for the recommended 1-D Kalman branch."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    rows = [dict(r) for r in report.get("window_rows", [])]
    by_index = {int(r.get("window_index", -1)): r for r in rows}
    combined_rows: list[dict] = []
    for i, center in enumerate(np.asarray(centers, dtype=np.float64)):
        row = dict(by_index.get(i, {}))
        row["window_index"] = int(i)
        row["center_y_px"] = float(center)
        row["baseline_smoothed_dy_px"] = float(baseline_window_values[i])
        row["kalman_cv_dy_px"] = float(cv_window_values[i])
        row["kalman_1d_dy_px"] = float(one_d_window_values[i])
        combined_rows.append(row)

    if SAVE_KALMAN_1D_WINDOWS_CSV and combined_rows:
        path = out_dir / f"{stem}_kalman_1d_local_y_windows.csv"
        keys: list[str] = []
        seen: set[str] = set()
        for row in combined_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(combined_rows)
        paths["kalman_1d_window_csv"] = str(path)

    if SAVE_KALMAN_1D_CURVE_CSV:
        path = out_dir / f"{stem}_kalman_1d_local_y_curve.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["camera_y_px", "kalman_1d_local_dy_px"])
            for y, dy in enumerate(one_d_curve):
                writer.writerow([int(y), f"{float(dy):.9f}"])
        paths["kalman_1d_curve_csv"] = str(path)

    if SAVE_LOCAL_Y_THREE_WAY_COMPARISON_CSV:
        path = out_dir / f"{stem}_local_y_baseline_vs_kalman_cv_vs_kalman_1d.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "camera_y_px",
                "baseline_local_dy_px",
                "kalman_cv_local_dy_px",
                "kalman_1d_local_dy_px",
                "kalman_cv_minus_baseline_px",
                "kalman_1d_minus_baseline_px",
                "kalman_1d_minus_kalman_cv_px",
            ])
            for y, (b, cv, kd) in enumerate(zip(baseline_curve, cv_curve, one_d_curve)):
                writer.writerow([
                    int(y),
                    f"{float(b):.9f}",
                    f"{float(cv):.9f}",
                    f"{float(kd):.9f}",
                    f"{float(cv-b):.9f}",
                    f"{float(kd-b):.9f}",
                    f"{float(kd-cv):.9f}",
                ])
        paths["three_way_curve_csv"] = str(path)

    if SAVE_KALMAN_1D_JSON:
        path = out_dir / f"{stem}_kalman_1d_local_y.json"
        payload = dict(report)
        payload["diagnostic_paths"] = dict(paths)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["kalman_1d_json"] = str(path)

    return paths


def _save_kalman_rts_diagnostics(
    out_dir: Path,
    stem: str,
    centers: np.ndarray,
    raw_values: np.ndarray,
    baseline_window_values: np.ndarray,
    one_d_window_values: np.ndarray,
    rts_window_values: np.ndarray,
    baseline_curve: np.ndarray,
    one_d_curve: np.ndarray,
    rts_curve: np.ndarray,
    rts_report: dict,
) -> dict:
    """Save window- and row-level diagnostics for Baseline vs 1D vs RTS."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    centers = np.asarray(centers, dtype=np.float64)
    raw = np.asarray(raw_values, dtype=np.float64)
    b = np.asarray(baseline_window_values, dtype=np.float64)
    k = np.asarray(one_d_window_values, dtype=np.float64)
    s = np.asarray(rts_window_values, dtype=np.float64)

    by_index = {
        int(r.get("window_index", -1)): dict(r)
        for r in rts_report.get("window_rows", [])
    }
    rows: list[dict] = []
    for i, center in enumerate(centers):
        row = dict(by_index.get(i, {}))
        row.update({
            "window_index": int(i),
            "center_y_px": float(center),
            "raw_dy_px": None if i >= len(raw) or not np.isfinite(raw[i]) else float(raw[i]),
            "baseline_smoothed_dy_px": float(b[i]),
            "kalman_1d_forward_dy_px": float(k[i]),
            "kalman_rts_dy_px": float(s[i]),
            "rts_minus_forward_px": float(s[i] - k[i]),
            "rts_minus_baseline_px": float(s[i] - b[i]),
        })
        rows.append(row)

    if SAVE_KALMAN_RTS_WINDOWS_CSV and rows:
        path = out_dir / f"{stem}_kalman_rts_local_y_windows.csv"
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        paths["kalman_rts_window_csv"] = str(path)

    if SAVE_KALMAN_RTS_CURVE_CSV:
        path = out_dir / f"{stem}_kalman_rts_local_y_curve.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["camera_y_px", "kalman_rts_local_dy_px"])
            for y, dy in enumerate(rts_curve):
                writer.writerow([int(y), f"{float(dy):.9f}"])
        paths["kalman_rts_curve_csv"] = str(path)

    if SAVE_LOCAL_Y_BASELINE_1D_RTS_COMPARISON_CSV:
        path = out_dir / f"{stem}_local_y_baseline_vs_kalman_1d_vs_rts.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "camera_y_px",
                "baseline_local_dy_px",
                "kalman_1d_forward_local_dy_px",
                "kalman_rts_local_dy_px",
                "kalman_1d_minus_baseline_px",
                "kalman_rts_minus_baseline_px",
                "kalman_rts_minus_kalman_1d_px",
            ])
            for y, (bb, kk, ss) in enumerate(zip(baseline_curve, one_d_curve, rts_curve)):
                writer.writerow([
                    int(y),
                    f"{float(bb):.9f}",
                    f"{float(kk):.9f}",
                    f"{float(ss):.9f}",
                    f"{float(kk-bb):.9f}",
                    f"{float(ss-bb):.9f}",
                    f"{float(ss-kk):.9f}",
                ])
        paths["baseline_1d_rts_curve_csv"] = str(path)

    if SAVE_KALMAN_RTS_JSON:
        path = out_dir / f"{stem}_kalman_rts_local_y.json"
        payload = dict(rts_report)
        payload["diagnostic_paths"] = dict(paths)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["kalman_rts_json"] = str(path)

    return paths


def optimize_final_local_feature_y(
    base,
    camera: Image.Image,
    mapped_height: Image.Image,
    mapped_mask: Image.Image,
    out_dir: Path,
    stem: str,
    registration_rmse_px: float | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Estimate a continuous local residual curve after global feature-Y.

    The SAME local feature measurements are post-processed in two parallel ways:

      1) BASELINE: periodic interpolation + median + jump limiter + smoother.
      2) KALMAN: confidence-weighted periodic Kalman state estimation.

    Both curves are returned so one run can save a strict without/with-Kalman
    comparison without changing the physical affine or the X registration.
    """
    height = int(camera.height)
    zero_curve = np.zeros(height, dtype=np.float32)

    result = {
        "enabled": bool(USE_FINAL_LOCAL_FEATURE_Y),
        "applied": False,
        "reason": "disabled",
        "method": (
            "local_endface_feature_correlation_"
            "periodic_continuous_y_curve"
        ),
        "manual_y_used": False,
        "x_translation_locked_px": 0.0,
        "x_scale_locked": 1.0,
        "y_scale_locked": 1.0,
        "rotation_locked_deg": 0.0,
        "shear_locked": 0.0,
        "window_height_px": int(LOCAL_FEATURE_Y_WINDOW_HEIGHT_PX),
        "step_px": int(LOCAL_FEATURE_Y_STEP_PX),
        "search_range_px": float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
        "coarse_step_px": float(LOCAL_FEATURE_Y_COARSE_STEP_PX),
        "fine_step_px": float(LOCAL_FEATURE_Y_FINE_STEP_PX),
        "window_results": [],
    }

    if not USE_FINAL_LOCAL_FEATURE_Y:
        return result, zero_curve, zero_curve

    data = _build_feature_y_data(
        base,
        camera,
        mapped_height,
        mapped_mask,
    )

    step = max(1, int(LOCAL_FEATURE_Y_STEP_PX))
    centers = np.arange(
        0.5 * step,
        float(height),
        float(step),
        dtype=np.float64,
    )
    if len(centers) == 0:
        centers = np.asarray(
            [0.5 * height],
            dtype=np.float64,
        )

    raw_values = np.full(
        len(centers),
        np.nan,
        dtype=np.float64,
    )

    window_rows: list[dict] = []

    coarse_grid = _float_grid(
        -float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
        +float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
        float(LOCAL_FEATURE_Y_COARSE_STEP_PX),
    )

    for wi, center in enumerate(centers):
        output_rows = _periodic_output_rows(
            float(center),
            int(LOCAL_FEATURE_Y_WINDOW_HEIGHT_PX),
            height,
        )

        candidate_rows: list[dict] = []
        best = None

        for dy in coarse_grid:
            score = _score_local_feature_y_window(
                data,
                output_rows,
                float(dy),
            )
            row = {
                "stage": "coarse",
                "dy_px": float(dy),
                **score,
            }
            candidate_rows.append(row)

            if (
                score.get("valid", False)
                and (
                    best is None
                    or float(score["score"]) > float(best["score"])
                )
            ):
                best = row

        window_report = {
            "window_index": int(wi),
            "center_y_px": float(center),
            "accepted": False,
            "reason": "",
            "raw_dy_px": None,
            "score_gain_vs_zero": None,
            "best_score": None,
            "zero_score": None,
        }

        if best is None:
            window_report["reason"] = "no_valid_candidate"
            window_rows.append(window_report)
            continue

        fine_grid = _float_grid(
            max(
                -float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
                float(best["dy_px"])
                - float(LOCAL_FEATURE_Y_FINE_RADIUS_PX),
            ),
            min(
                +float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX),
                float(best["dy_px"])
                + float(LOCAL_FEATURE_Y_FINE_RADIUS_PX),
            ),
            float(LOCAL_FEATURE_Y_FINE_STEP_PX),
        )

        for dy in fine_grid:
            score = _score_local_feature_y_window(
                data,
                output_rows,
                float(dy),
            )
            row = {
                "stage": "fine",
                "dy_px": float(dy),
                **score,
            }
            candidate_rows.append(row)

            if (
                score.get("valid", False)
                and float(score["score"]) > float(best["score"])
            ):
                best = row

        grid_best = float(best["dy_px"])
        subpixel = _refine_local_peak_parabola(
            [
                row
                for row in candidate_rows
                if row.get("stage") == "fine"
            ],
            grid_best,
        )

        subpixel_score = _score_local_feature_y_window(
            data,
            output_rows,
            float(subpixel),
        )

        if (
            subpixel_score.get("valid", False)
            and float(subpixel_score["score"]) >= float(best["score"])
        ):
            best_dy = float(subpixel)
            best_score = float(subpixel_score["score"])
        else:
            best_dy = float(grid_best)
            best_score = float(best["score"])

        zero = _score_local_feature_y_window(
            data,
            output_rows,
            0.0,
        )
        zero_score = float(
            zero.get("score", -1e9)
        )

        gain = float(
            best_score - zero_score
        )

        boundary_hit = bool(
            abs(best_dy)
            >= (
                float(LOCAL_FEATURE_Y_SEARCH_RANGE_PX)
                - float(LOCAL_FEATURE_Y_BOUNDARY_MARGIN_PX)
            )
        )

        window_report.update({
            "grid_best_dy_px": float(grid_best),
            "raw_dy_px": float(best_dy),
            "best_score": float(best_score),
            "zero_score": float(zero_score),
            "score_gain_vs_zero": float(gain),
            "boundary_hit": bool(boundary_hit),
        })

        if (
            boundary_hit
            and gain < float(LOCAL_FEATURE_Y_BOUNDARY_MIN_SCORE_GAIN)
        ):
            window_report["reason"] = "weak_peak_at_search_boundary"
        elif (
            abs(best_dy) > 0.03
            and gain < float(LOCAL_FEATURE_Y_MIN_SCORE_GAIN)
        ):
            window_report["reason"] = "weak_local_peak"
        else:
            raw_values[wi] = float(best_dy)
            window_report["accepted"] = True
            window_report["reason"] = "accepted_local_feature_y"

        window_rows.append(window_report)

    accepted_count = int(
        np.count_nonzero(
            np.isfinite(raw_values)
        )
    )

    result["window_results"] = window_rows
    result["window_count"] = int(len(window_rows))
    result["accepted_window_count"] = int(accepted_count)

    baseline_window_values = np.zeros(len(centers), dtype=np.float64)
    baseline_curve = zero_curve.copy()
    kalman_window_values = np.zeros(len(centers), dtype=np.float64)
    kalman_curve = zero_curve.copy()
    kalman_report = {
        "enabled": bool(USE_KALMAN_LOCAL_Y),
        "available": False,
        "reason": "not_run",
        "window_rows": [],
    }

    if accepted_count < int(LOCAL_FEATURE_Y_MIN_VALID_WINDOWS):
        result["reason"] = "too_few_reliable_local_windows"
        result["kalman_applied"] = False
        result["kalman_reason"] = "too_few_reliable_local_windows"
    else:
        # ----------------------------------------------------------
        # A) ORIGINAL BASELINE (preserved exactly).
        # ----------------------------------------------------------
        filled = _periodic_interpolate_nan_windows(
            centers,
            raw_values,
            height,
        )

        robust = _circular_median_filter(
            filled,
            3,
        )

        robust = _enforce_periodic_jump_limit(
            robust,
            float(LOCAL_FEATURE_Y_MAX_ADJACENT_JUMP_PX),
            passes=4,
        )

        robust = _circular_smooth_window_values(
            robust,
            int(LOCAL_FEATURE_Y_SMOOTH_WINDOW),
            int(LOCAL_FEATURE_Y_SMOOTH_PASSES),
        )

        robust = np.clip(
            robust,
            -float(LOCAL_FEATURE_Y_MAX_ABS_PX),
            +float(LOCAL_FEATURE_Y_MAX_ABS_PX),
        )
        baseline_window_values = robust.astype(np.float64, copy=True)

        baseline_curve = _build_periodic_local_curve(
            centers,
            baseline_window_values,
            height,
        )

        baseline_max_abs = float(np.max(np.abs(baseline_curve)))
        baseline_rms = float(
            np.sqrt(np.mean(baseline_curve.astype(np.float64) ** 2))
        )

        result.update({
            "smoothed_window_values_px": [
                float(v) for v in baseline_window_values.tolist()
            ],
            "curve_min_px": float(np.min(baseline_curve)),
            "curve_max_px": float(np.max(baseline_curve)),
            "curve_mean_px": float(np.mean(baseline_curve)),
            "curve_median_px": float(np.median(baseline_curve)),
            "curve_rms_px": float(baseline_rms),
            "curve_max_abs_px": float(baseline_max_abs),
            "baseline_postprocessing": (
                "periodic_interpolation+median_filter+jump_limiter+triangular_smoothing"
            ),
        })

        if baseline_max_abs < float(LOCAL_FEATURE_Y_MIN_EFFECT_PX):
            result["reason"] = "local_curve_below_min_effect"
            baseline_curve = zero_curve.copy()
            baseline_window_values[:] = 0.0
        else:
            result["applied"] = True
            result["reason"] = "accepted_continuous_local_y_curve"

        # ----------------------------------------------------------
        # B) KALMAN on the SAME raw local-Y observations.
        # ----------------------------------------------------------
        kalman_window_values, kalman_report = _kalman_filter_local_y(
            centers,
            raw_values,
            window_rows,
            registration_rmse_px,
        )

        if bool(kalman_report.get("available", False)):
            kalman_curve = _build_periodic_local_curve(
                centers,
                kalman_window_values,
                height,
            )
            kalman_max_abs = float(np.max(np.abs(kalman_curve)))
            kalman_rms = float(
                np.sqrt(np.mean(kalman_curve.astype(np.float64) ** 2))
            )
            if kalman_max_abs < float(KALMAN_LOCAL_Y_MIN_EFFECT_PX):
                kalman_curve = zero_curve.copy()
                kalman_window_values[:] = 0.0
                kalman_applied = False
                kalman_reason = "kalman_curve_below_min_effect"
            else:
                kalman_applied = True
                kalman_reason = "accepted_uncertainty_weighted_kalman_local_y"

            result.update({
                "kalman_applied": bool(kalman_applied),
                "kalman_reason": str(kalman_reason),
                "kalman_curve_min_px": float(np.min(kalman_curve)),
                "kalman_curve_max_px": float(np.max(kalman_curve)),
                "kalman_curve_mean_px": float(np.mean(kalman_curve)),
                "kalman_curve_median_px": float(np.median(kalman_curve)),
                "kalman_curve_rms_px": float(kalman_rms),
                "kalman_curve_max_abs_px": float(kalman_max_abs),
                "kalman_window_values_px": [
                    float(v) for v in kalman_window_values.tolist()
                ],
            })
        else:
            result["kalman_applied"] = False
            result["kalman_reason"] = str(kalman_report.get("reason", "unavailable"))

    result["kalman"] = kalman_report

    # Window-level baseline values are attached to the Kalman diagnostic rows.
    if bool(kalman_report.get("available", False)):
        by_index = {
            int(row.get("window_index", -1)): row
            for row in kalman_report.get("window_rows", [])
        }
        for i in range(len(centers)):
            row = by_index.get(i)
            if row is not None:
                row["baseline_smoothed_dy_px"] = float(baseline_window_values[i])

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_LOCAL_FEATURE_Y_WINDOWS_CSV:
        csv_path = out_dir / f"{stem}_local_feature_y_windows.csv"

        fieldnames = [
            "window_index",
            "center_y_px",
            "accepted",
            "reason",
            "grid_best_dy_px",
            "raw_dy_px",
            "best_score",
            "zero_score",
            "score_gain_vs_zero",
            "boundary_hit",
        ]

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(window_rows)

        result["window_csv"] = str(csv_path)

    if SAVE_LOCAL_FEATURE_Y_CURVE_CSV:
        csv_path = out_dir / f"{stem}_local_feature_y_curve.csv"

        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.writer(f)
            writer.writerow([
                "camera_y_px",
                "local_dy_px",
            ])
            for yy, dy in enumerate(baseline_curve):
                writer.writerow([
                    int(yy),
                    f"{float(dy):.9f}",
                ])

        result["curve_csv"] = str(csv_path)

    if SAVE_KALMAN_LOCAL_Y_WINDOWS_CSV and bool(kalman_report.get("available", False)):
        kalman_window_csv = out_dir / f"{stem}_kalman_local_y_windows.csv"
        rows = list(kalman_report.get("window_rows", []))
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        with kalman_window_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        result["kalman_window_csv"] = str(kalman_window_csv)

    if SAVE_KALMAN_LOCAL_Y_CURVE_CSV:
        kalman_curve_csv = out_dir / f"{stem}_kalman_local_y_curve.csv"
        with kalman_curve_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["camera_y_px", "kalman_local_dy_px"])
            for yy, dy in enumerate(kalman_curve):
                writer.writerow([int(yy), f"{float(dy):.9f}"])
        result["kalman_curve_csv"] = str(kalman_curve_csv)

    if SAVE_LOCAL_Y_BASELINE_KALMAN_COMPARISON_CSV:
        comparison_csv = out_dir / f"{stem}_local_y_baseline_vs_kalman.csv"
        with comparison_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow([
                "camera_y_px",
                "baseline_local_dy_px",
                "kalman_local_dy_px",
                "difference_kalman_minus_baseline_px",
            ])
            for yy, (bdy, kdy) in enumerate(zip(baseline_curve, kalman_curve)):
                writer.writerow([
                    int(yy),
                    f"{float(bdy):.9f}",
                    f"{float(kdy):.9f}",
                    f"{float(kdy-bdy):.9f}",
                ])
        result["baseline_vs_kalman_curve_csv"] = str(comparison_csv)

    if SAVE_KALMAN_LOCAL_Y_JSON:
        kalman_json = out_dir / f"{stem}_kalman_local_y.json"
        kalman_json.write_text(
            json.dumps(kalman_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["kalman_json"] = str(kalman_json)

    if SAVE_LOCAL_FEATURE_Y_JSON:
        json_path = out_dir / f"{stem}_local_feature_y.json"

        json_path.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        result["result_json"] = str(json_path)

    print("")
    print("Final LOCAL feature-Y curve:")
    print(
        f"  reliable windows = "
        f"{result.get('accepted_window_count', 0)}/"
        f"{result.get('window_count', 0)}"
    )

    if bool(result.get("applied", False)):
        print(
            f"  local dy(y) range = "
            f"{result.get('curve_min_px', 0.0):+.4f} .. "
            f"{result.get('curve_max_px', 0.0):+.4f}px"
        )
        print(
            f"  local dy(y) rms = "
            f"{result.get('curve_rms_px', 0.0):.4f}px"
        )

    print(
        f"  BASELINE applied={result.get('applied', False)}, "
        f"reason={result.get('reason')}"
    )
    print(
        f"  KALMAN applied={result.get('kalman_applied', False)}, "
        f"reason={result.get('kalman_reason')}"
    )
    if bool(kalman_report.get("available", False)):
        print(
            f"  Kalman Q process sigma = "
            f"{float(kalman_report.get('process_sigma_px_per_window', 0.0)):.4f}px/window; "
            f"initial sigma = "
            f"{float(kalman_report.get('initial_position_sigma_px', 0.0)):.4f}px"
        )

    return result, baseline_curve, kalman_curve


def _sample_periodic_curve(
    curve_px: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    curve = np.asarray(
        curve_px,
        dtype=np.float64,
    ).reshape(-1)

    h = int(len(curve))
    if h <= 0:
        return np.zeros_like(
            np.asarray(y, dtype=np.float64)
        )

    yy = np.mod(
        np.asarray(y, dtype=np.float64),
        float(h),
    )

    y0f = np.floor(yy)
    y0 = y0f.astype(np.int64)
    y1 = (y0 + 1) % h
    w = yy - y0f

    return (
        curve[y0] * (1.0 - w)
        + curve[y1] * w
    )


def _inverse_source_rows_for_local_curve(
    curve_px: np.ndarray,
) -> np.ndarray:
    """
    Invert:
        y_output = y_source + dy(y_source)

    Residuals are small and smooth, so fixed-point iteration converges quickly.
    """
    curve = np.asarray(
        curve_px,
        dtype=np.float64,
    ).reshape(-1)

    h = int(len(curve))
    output_y = np.arange(
        h,
        dtype=np.float64,
    )

    source_y = output_y.copy()

    for _ in range(8):
        source_y = np.mod(
            output_y
            - _sample_periodic_curve(
                curve,
                source_y,
            ),
            float(h),
        )

    return source_y.astype(np.float32)


def _warp_image_with_local_y_curve(
    base,
    image: Image.Image,
    curve_px: np.ndarray,
    *,
    nearest: bool,
) -> Image.Image:
    """
    Apply the row-dependent local Y correction on the camera final canvas.

    X coordinates are identity. Y is periodic.
    """
    mode = (
        "RGB"
        if image.mode == "RGB"
        else "L"
    )

    src = np.asarray(
        image.convert(mode),
        dtype=np.uint8,
    )

    h, w = src.shape[:2]

    curve = np.asarray(
        curve_px,
        dtype=np.float64,
    ).reshape(-1)

    if len(curve) != h:
        raise RuntimeError(
            f"Local Y curve length {len(curve)} != image height {h}"
        )

    if not np.any(
        np.abs(curve) > 1e-12
    ):
        return image.copy()

    source_y = _inverse_source_rows_for_local_curve(
        curve
    )

    map_y = np.broadcast_to(
        source_y[:, None],
        (h, w),
    ).astype(
        np.float32,
        copy=True,
    )

    map_x = np.broadcast_to(
        np.arange(
            w,
            dtype=np.float32,
        )[None, :],
        (h, w),
    ).copy()

    interpolation = (
        base.cv2.INTER_NEAREST
        if nearest
        else base.cv2.INTER_LINEAR
    )

    out = base.cv2.remap(
        src,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=base.cv2.BORDER_WRAP,
    )

    return Image.fromarray(
        out,
        mode,
    )


def _warp_numeric_with_local_y_curve(
    base,
    values: np.ndarray,
    curve_px: np.ndarray,
) -> np.ndarray:
    """
    NaN-aware numeric version of the exact same local Y warp.
    """
    src = np.asarray(
        values,
        dtype=np.float32,
    )

    if src.ndim != 2:
        raise ValueError(
            f"Expected 2-D numeric array, got {src.shape}"
        )

    h, w = src.shape

    curve = np.asarray(
        curve_px,
        dtype=np.float64,
    ).reshape(-1)

    if len(curve) != h:
        raise RuntimeError(
            f"Local Y curve length {len(curve)} != numeric height {h}"
        )

    if not np.any(
        np.abs(curve) > 1e-12
    ):
        return src.copy()

    source_y = _inverse_source_rows_for_local_curve(
        curve
    )

    map_y = np.broadcast_to(
        source_y[:, None],
        (h, w),
    ).astype(
        np.float32,
        copy=True,
    )

    map_x = np.broadcast_to(
        np.arange(
            w,
            dtype=np.float32,
        )[None, :],
        (h, w),
    ).copy()

    finite = np.isfinite(
        src
    ).astype(np.float32)

    data = np.nan_to_num(
        src,
        nan=0.0,
    ).astype(np.float32)

    numerator = base.cv2.remap(
        data,
        map_x,
        map_y,
        interpolation=base.cv2.INTER_LINEAR,
        borderMode=base.cv2.BORDER_WRAP,
    )

    denominator = base.cv2.remap(
        finite,
        map_x,
        map_y,
        interpolation=base.cv2.INTER_LINEAR,
        borderMode=base.cv2.BORDER_WRAP,
    )

    out = np.full(
        (h, w),
        np.nan,
        dtype=np.float32,
    )

    valid = denominator > 1e-4

    out[valid] = (
        numerator[valid]
        / denominator[valid]
    )

    return out


def _patch_mapped_height_npz_with_local_curve(
    base,
    npz_path: Path,
    curve_px: np.ndarray,
    final_visual_mask: np.ndarray,
    global_auto_y_px: float,
    *,
    preview_filename: str = "mapped_height_mm_preview.png",
    mapping_variant: str = "baseline",
) -> dict:
    """
    The base full-affine numeric saver understands scalar dx/dy only.
    It first saves the physical affine + original X + global Y result.
    We then apply the SAME local dy(y) curve to the real numeric height.
    """
    path = Path(npz_path)

    with np.load(
        path,
        allow_pickle=True,
    ) as data:
        payload = {
            key: data[key]
            for key in data.files
        }

    if "height_mm" not in payload:
        raise RuntimeError(
            f"{path} does not contain height_mm"
        )

    height_mm = np.asarray(
        payload["height_mm"],
        dtype=np.float32,
    )

    curve = np.asarray(
        curve_px,
        dtype=np.float32,
    ).reshape(-1)

    if len(curve) != height_mm.shape[0]:
        raise RuntimeError(
            "Local Y curve / mapped height row mismatch: "
            f"{len(curve)} vs {height_mm.shape[0]}"
        )

    if np.any(
        np.abs(curve) > 1e-12
    ):
        height_mm = _warp_numeric_with_local_y_curve(
            base,
            height_mm,
            curve,
        )

        if "residual_mm" in payload:
            payload["residual_mm"] = (
                _warp_numeric_with_local_y_curve(
                    base,
                    np.asarray(
                        payload["residual_mm"],
                        dtype=np.float32,
                    ),
                    curve,
                ).astype(np.float32)
            )

    visual_mask = np.asarray(
        final_visual_mask,
        dtype=bool,
    )

    if visual_mask.shape != height_mm.shape:
        raise RuntimeError(
            "Final visual mask / numeric height shape mismatch: "
            f"{visual_mask.shape} vs {height_mm.shape}"
        )

    final_valid = (
        visual_mask
        & np.isfinite(height_mm)
    )

    height_mm[
        ~final_valid
    ] = np.nan

    payload["height_mm"] = (
        height_mm.astype(np.float32)
    )
    payload["valid_mask"] = (
        final_valid.astype(np.uint8)
    )

    if "residual_mm" in payload:
        residual = np.asarray(
            payload["residual_mm"],
            dtype=np.float32,
        )
        residual[
            ~final_valid
        ] = np.nan
        payload[
            "residual_mm"
        ] = residual

    # Explicit traceability.
    payload[
        "feature_auto_global_y_dy_px"
    ] = np.float32(
        global_auto_y_px
    )

    payload[
        "feature_auto_local_y_curve_px"
    ] = curve.astype(
        np.float32
    )

    payload[
        "feature_auto_local_y_applied"
    ] = np.uint8(
        1
        if np.any(
            np.abs(curve) > 1e-12
        )
        else 0
    )

    payload[
        "manual_y_trim_used"
    ] = np.uint8(0)

    payload[
        "mapping_method"
    ] = np.array(
        "full_physical_circular_affine_plus_original_x_"
        "plus_global_feature_y_plus_continuous_local_feature_y_"
        + str(mapping_variant)
    )
    payload["local_y_filter_variant"] = np.array(str(mapping_variant))

    np.savez_compressed(
        path,
        **payload,
    )

    preview_path = (
        path.parent
        / str(preview_filename)
    )

    if hasattr(
        base,
        "_save_mapped_height_preview_png",
    ):
        base._save_mapped_height_preview_png(
            height_mm,
            final_valid,
            preview_path,
        )

    values = height_mm[
        final_valid
    ]

    return {
        "npz": str(path),
        "preview_png": (
            str(preview_path)
            if preview_path.exists()
            else None
        ),
        "valid_points": int(
            np.count_nonzero(final_valid)
        ),
        "valid_fraction": float(
            np.mean(final_valid)
        ),
        "height_min_mm": (
            float(np.min(values))
            if values.size
            else None
        ),
        "height_median_mm": (
            float(np.median(values))
            if values.size
            else None
        ),
        "height_max_mm": (
            float(np.max(values))
            if values.size
            else None
        ),
        "feature_auto_global_y_dy_px": float(
            global_auto_y_px
        ),
        "feature_auto_local_y_applied": bool(
            np.any(
                np.abs(curve) > 1e-12
            )
        ),
        "feature_auto_local_y_curve_min_px": float(
            np.min(curve)
        ),
        "feature_auto_local_y_curve_max_px": float(
            np.max(curve)
        ),
        "feature_auto_local_y_curve_mean_px": float(
            np.mean(curve)
        ),
        "manual_y_trim_used": False,
        "local_y_filter_variant": str(mapping_variant),
    }

def _compose_camera_laser_overlay(
    camera: Image.Image,
    mapped_laser: Image.Image,
    mapped_mask: Image.Image,
    alpha: float,
) -> tuple[Image.Image, np.ndarray]:
    """Compose one camera-base laser overlay and return its boolean final mask."""
    final_mask = (
        np.asarray(mapped_mask.convert("L"), dtype=np.uint8) > 127
    )
    cam_arr = np.asarray(camera.convert("RGB"), dtype=np.float32)
    las_arr = np.asarray(mapped_laser.convert("RGB"), dtype=np.float32)
    overlay_arr = cam_arr.copy()
    overlay_arr[final_mask] = (
        cam_arr[final_mask] * (1.0 - float(alpha))
        + las_arr[final_mask] * float(alpha)
    )
    overlay = Image.fromarray(
        np.clip(overlay_arr, 0, 255).astype(np.uint8),
        "RGB",
    )
    return overlay, final_mask


def save_full_physical_affine_final_outputs(
    laser_path: Path,
    laser: Image.Image,
    laser_height_value_img: Image.Image,
    laser_valid_mask_img: Image.Image,
    camera: Image.Image,
    output_path: Path,
) -> dict:
    base = sys.modules[__name__]

    # ---------------------------------------------------------------------
    # 1. ORIGINAL COMPLETE PHYSICAL AFFINE
    # ---------------------------------------------------------------------
    calibration = (
        base.load_full_physical_circular_affine_calibration(
            laser.size,
            camera.size,
        )
    )

    maps = (
        base.build_full_affine_camera_to_laser_maps(
            camera.size,
            laser.size,
            calibration,
        )
    )

    mapped_laser = (
        base.remap_pil_full_affine(
            laser,
            maps,
            fill="black",
            nearest=False,
        )
    )

    mapped_height = (
        base.remap_pil_full_affine(
            laser_height_value_img,
            maps,
            fill=0,
            nearest=False,
        )
    )

    mapped_mask = (
        base.remap_pil_full_affine(
            laser_valid_mask_img,
            maps,
            fill=0,
            nearest=True,
        )
    )

    mask_arr = np.array(
        mapped_mask.convert("L"),
        dtype=np.uint8,
        copy=True,
    )

    mask_arr[
        ~np.asarray(
            maps["valid_x"],
            dtype=bool,
        )
    ] = 0

    mapped_mask = Image.fromarray(
        mask_arr,
        "L",
    )

    # ---------------------------------------------------------------------
    # 2. ORIGINAL X RESIDUAL - EXACT ORIGINAL FUNCTION
    # ---------------------------------------------------------------------
    x_alignment = {
        "enabled": True,
        "applied": False,
        "reason": (
            "all_post_affine_residual_stages_disabled"
        ),
        "method": "none",
        "applied_dx_px": 0.0,
        "applied_dy_px": 0.0,
        "applied_x_scale": 1.0,
        "applied_y_scale": 1.0,
    }

    if getattr(
        base,
        "FULL_AFFINE_USE_WIRE_PITCH_CHAMFER_X_RESIDUAL",
        False,
    ):
        x_alignment = (
            base.optimize_full_affine_wire_pitch_chamfer_x(
                camera,
                mapped_height,
                mapped_mask,
                output_path.parent,
                output_path.stem,
            )
        )

    original_dx = float(
        x_alignment.get(
            "applied_dx_px",
            0.0,
        )
    )

    original_dy = float(
        x_alignment.get(
            "applied_dy_px",
            0.0,
        )
    )

    original_sx = float(
        x_alignment.get(
            "applied_x_scale",
            1.0,
        )
    )

    original_sy = float(
        x_alignment.get(
            "applied_y_scale",
            1.0,
        )
    )

    original_transform_needed = bool(
        x_alignment.get(
            "applied",
            False,
        )
    ) and (
        abs(original_dx) > 1e-12
        or abs(original_dy) > 1e-12
        or abs(original_sx - 1.0) > 1e-12
        or abs(original_sy - 1.0) > 1e-12
    )

    if original_transform_needed:
        mapped_laser = (
            base._warp_image_scale_translation(
                mapped_laser,
                original_dx,
                original_dy,
                original_sx,
                original_sy,
                fill="black",
                nearest=False,
            )
        )

        mapped_height = (
            base._warp_image_scale_translation(
                mapped_height,
                original_dx,
                original_dy,
                original_sx,
                original_sy,
                fill=0,
                nearest=False,
            )
        )

        mapped_mask = (
            base._warp_image_scale_translation(
                mapped_mask,
                original_dx,
                original_dy,
                original_sx,
                original_sy,
                fill=0,
                nearest=True,
            )
        )

    # ---------------------------------------------------------------------
    # 3A. WHOLE-WIRE Y PHASE REMOVED
    # ---------------------------------------------------------------------
    y_phase = {
        "enabled": False,
        "applied": False,
        "reason": "whole_wire_y_phase_disabled_in_final",
        "method": "removed_false_switch",
        "applied_dy_px": 0.0,
        "winning_phase_k": 0,
        "estimated_y_pitch_px": 0.0,
    }
    phase_dy = 0.0

    # ---------------------------------------------------------------------
    # 3B. EXISTING GLOBAL FEATURE CORRELATION AUTO Y-ONLY
    #     This now refines only the residual AFTER the wire phase is fixed.
    # ---------------------------------------------------------------------
    auto_y = (
        optimize_final_feature_auto_y(
            base,
            camera,
            mapped_height,
            mapped_mask,
            output_path.parent,
            output_path.stem,
        )
    )

    auto_dy = float(
        auto_y.get(
            "applied_dy_px",
            0.0,
        )
    )

    if (
        bool(
            auto_y.get(
                "applied",
                False,
            )
        )
        and abs(auto_dy) > 1e-12
    ):
        # Strict Y-only:
        # dx=0, sx=1, sy=1.
        mapped_laser = (
            base._warp_image_scale_translation(
                mapped_laser,
                0.0,
                auto_dy,
                1.0,
                1.0,
                fill="black",
                nearest=False,
            )
        )

        mapped_height = (
            base._warp_image_scale_translation(
                mapped_height,
                0.0,
                auto_dy,
                1.0,
                1.0,
                fill=0,
                nearest=False,
            )
        )

        mapped_mask = (
            base._warp_image_scale_translation(
                mapped_mask,
                0.0,
                auto_dy,
                1.0,
                1.0,
                fill=0,
                nearest=True,
            )
        )

    # Mask corresponding exactly to physical affine + ORIGINAL X +
    # scalar global feature-Y. The base numeric saver uses this BEFORE
    # the row-dependent local curve is patched into height_mm.
    pre_local_mask = (
        np.asarray(
            mapped_mask.convert("L"),
            dtype=np.uint8,
        )
        > 127
    )

    # ---------------------------------------------------------------------
    # 3C. LOCAL Y RESIDUAL: BASELINE + KALMAN IN PARALLEL
    # ---------------------------------------------------------------------
    # Convert the physical-affine RMSE (mm) to an approximate circular-Y pixel
    # uncertainty for the Kalman initial covariance P0.  The RMSE is a 2-D
    # registration statistic, so this is used only as an initialization scale,
    # not as the local measurement R_k itself.
    registration_rmse_px = None
    affine_rmse_mm = calibration.get("rmse_mm")
    if affine_rmse_mm is None:
        affine_rmse_mm = calibration.get("all_point_rmse_mm")
    try:
        if (
            affine_rmse_mm is not None
            and np.isfinite(float(affine_rmse_mm))
            and float(affine_rmse_mm) > 0.0
        ):
            circular_y_mm_per_px = (
                float(calibration["circumference_mm"])
                / float(max(camera.height, 1))
            )
            if circular_y_mm_per_px > 0.0:
                registration_rmse_px = (
                    float(affine_rmse_mm)
                    / circular_y_mm_per_px
                )
    except Exception:
        registration_rmse_px = None

    # Preserve the exact pre-local state.  Baseline and Kalman must start from
    # identical physical-affine + X + global-Y data for a fair comparison.
    pre_local_laser_img = mapped_laser.copy()
    pre_local_height_img = mapped_height.copy()
    pre_local_mask_img = mapped_mask.copy()

    local_y, baseline_curve, kalman_cv_curve = (
        optimize_final_local_feature_y(
            base,
            camera,
            pre_local_height_img,
            pre_local_mask_img,
            output_path.parent,
            output_path.stem,
            registration_rmse_px=registration_rmse_px,
        )
    )

    # Reconstruct the exact local observations used by the baseline/CV branch.
    local_rows = list(local_y.get("window_results", []))
    centers_1d = np.asarray(
        [float(r.get("center_y_px", 0.0)) for r in local_rows],
        dtype=np.float64,
    )
    raw_1d = np.asarray([
        (
            float(r["raw_dy_px"])
            if bool(r.get("accepted", False))
            and r.get("raw_dy_px") is not None
            else np.nan
        )
        for r in local_rows
    ], dtype=np.float64)

    # The old optimize function stores its baseline window values explicitly.
    baseline_window_values = np.asarray(
        local_y.get("smoothed_window_values_px", np.zeros(len(centers_1d))),
        dtype=np.float64,
    )
    if baseline_window_values.size != centers_1d.size:
        baseline_window_values = np.zeros(len(centers_1d), dtype=np.float64)

    cv_report = dict(local_y.get("kalman", {}))
    cv_window_values = np.zeros(len(centers_1d), dtype=np.float64)
    if bool(cv_report.get("available", False)):
        cv_by_index = {
            int(r.get("window_index", -1)): r
            for r in cv_report.get("window_rows", [])
        }
        for i in range(len(centers_1d)):
            row = cv_by_index.get(i)
            if row is not None and row.get("filtered_dy_px") is not None:
                cv_window_values[i] = float(row["filtered_dy_px"])

    # Recommended 1-D random-walk Kalman on the SAME measurements.
    kalman_1d_window_values, kalman_1d_report = _kalman_filter_local_y_1d(
        centers_1d,
        raw_1d,
        local_rows,
        registration_rmse_px,
    )

    if bool(kalman_1d_report.get("available", False)) and len(centers_1d):
        kalman_1d_curve = _build_periodic_local_curve(
            centers_1d,
            kalman_1d_window_values,
            int(camera.height),
        )
        if float(np.max(np.abs(kalman_1d_curve))) < float(KALMAN_1D_MIN_EFFECT_PX):
            kalman_1d_curve = np.zeros(int(camera.height), dtype=np.float32)
            kalman_1d_window_values[:] = 0.0
            kalman_1d_available = False
            kalman_1d_reason = "kalman_1d_curve_below_min_effect"
        else:
            kalman_1d_available = True
            kalman_1d_reason = "accepted_adaptive_random_walk_kalman_local_y"
    else:
        kalman_1d_curve = np.zeros(int(camera.height), dtype=np.float32)
        kalman_1d_available = False
        kalman_1d_reason = str(kalman_1d_report.get("reason", "unavailable"))

    # Fixed-interval RTS backward smoother on the SAME 1-D Kalman sequence.
    kalman_rts_window_values, kalman_rts_report = _rts_smooth_local_y_1d(
        centers_1d,
        kalman_1d_report,
    )
    if bool(kalman_rts_report.get("available", False)) and len(centers_1d):
        kalman_rts_curve = _build_periodic_local_curve(
            centers_1d,
            kalman_rts_window_values,
            int(camera.height),
        )
        if float(np.max(np.abs(kalman_rts_curve))) < float(KALMAN_RTS_MIN_EFFECT_PX):
            kalman_rts_curve = np.zeros(int(camera.height), dtype=np.float32)
            kalman_rts_window_values[:] = 0.0
            kalman_rts_available = False
            kalman_rts_reason = "kalman_rts_curve_below_min_effect"
        else:
            kalman_rts_available = True
            kalman_rts_reason = "accepted_1d_kalman_plus_rts_fixed_interval_local_y"
    else:
        kalman_rts_curve = np.zeros(int(camera.height), dtype=np.float32)
        kalman_rts_available = False
        kalman_rts_reason = str(kalman_rts_report.get("reason", "unavailable"))

    kalman_cv_available = bool(
        USE_KALMAN_LOCAL_Y
        and local_y.get("kalman", {}).get("available", False)
    )

    # Save the 1-D diagnostics before image warping.
    kalman_1d_diag_paths = _save_kalman_1d_diagnostics(
        output_path.parent,
        output_path.stem,
        centers_1d,
        baseline_window_values,
        cv_window_values,
        kalman_1d_window_values,
        baseline_curve,
        kalman_cv_curve,
        kalman_1d_curve,
        kalman_1d_report,
    )
    kalman_rts_diag_paths = _save_kalman_rts_diagnostics(
        output_path.parent,
        output_path.stem,
        centers_1d,
        raw_1d,
        baseline_window_values,
        kalman_1d_window_values,
        kalman_rts_window_values,
        baseline_curve,
        kalman_1d_curve,
        kalman_rts_curve,
        kalman_rts_report,
    )

    # BASELINE branch.
    baseline_mapped_laser = _warp_image_with_local_y_curve(
        base, pre_local_laser_img, baseline_curve, nearest=False
    )
    baseline_mapped_height = _warp_image_with_local_y_curve(
        base, pre_local_height_img, baseline_curve, nearest=False
    )
    baseline_mapped_mask = _warp_image_with_local_y_curve(
        base, pre_local_mask_img, baseline_curve, nearest=True
    )

    # Legacy constant-velocity Kalman branch.
    if kalman_cv_available:
        kalman_cv_mapped_laser = _warp_image_with_local_y_curve(
            base, pre_local_laser_img, kalman_cv_curve, nearest=False
        )
        kalman_cv_mapped_height = _warp_image_with_local_y_curve(
            base, pre_local_height_img, kalman_cv_curve, nearest=False
        )
        kalman_cv_mapped_mask = _warp_image_with_local_y_curve(
            base, pre_local_mask_img, kalman_cv_curve, nearest=True
        )
    else:
        kalman_cv_mapped_laser = pre_local_laser_img.copy()
        kalman_cv_mapped_height = pre_local_height_img.copy()
        kalman_cv_mapped_mask = pre_local_mask_img.copy()

    # Recommended 1-D adaptive Kalman branch.
    if kalman_1d_available:
        kalman_1d_mapped_laser = _warp_image_with_local_y_curve(
            base, pre_local_laser_img, kalman_1d_curve, nearest=False
        )
        kalman_1d_mapped_height = _warp_image_with_local_y_curve(
            base, pre_local_height_img, kalman_1d_curve, nearest=False
        )
        kalman_1d_mapped_mask = _warp_image_with_local_y_curve(
            base, pre_local_mask_img, kalman_1d_curve, nearest=True
        )
    else:
        kalman_1d_mapped_laser = pre_local_laser_img.copy()
        kalman_1d_mapped_height = pre_local_height_img.copy()
        kalman_1d_mapped_mask = pre_local_mask_img.copy()

    # Recommended 1-D Adaptive Kalman + RTS fixed-interval smoother branch.
    if kalman_rts_available:
        kalman_rts_mapped_laser = _warp_image_with_local_y_curve(
            base, pre_local_laser_img, kalman_rts_curve, nearest=False
        )
        kalman_rts_mapped_height = _warp_image_with_local_y_curve(
            base, pre_local_height_img, kalman_rts_curve, nearest=False
        )
        kalman_rts_mapped_mask = _warp_image_with_local_y_curve(
            base, pre_local_mask_img, kalman_rts_curve, nearest=True
        )
    else:
        kalman_rts_mapped_laser = pre_local_laser_img.copy()
        kalman_rts_mapped_height = pre_local_height_img.copy()
        kalman_rts_mapped_mask = pre_local_mask_img.copy()

    # Select the owner of historical main output names.
    requested_main_variant = str(MAIN_LOCAL_Y_VARIANT).strip().lower()
    if requested_main_variant == "kalman_rts" and kalman_rts_available:
        main_variant = "kalman_rts"
        mapped_laser = kalman_rts_mapped_laser
        mapped_height = kalman_rts_mapped_height
        mapped_mask = kalman_rts_mapped_mask
        main_local_curve = kalman_rts_curve
    elif requested_main_variant == "kalman_rts" and kalman_1d_available:
        # Safe fallback: if RTS cannot be formed, retain the validated forward 1-D
        # Kalman result rather than unexpectedly falling all the way to baseline.
        main_variant = "kalman_1d"
        mapped_laser = kalman_1d_mapped_laser
        mapped_height = kalman_1d_mapped_height
        mapped_mask = kalman_1d_mapped_mask
        main_local_curve = kalman_1d_curve
    elif requested_main_variant == "kalman_1d" and kalman_1d_available:
        main_variant = "kalman_1d"
        mapped_laser = kalman_1d_mapped_laser
        mapped_height = kalman_1d_mapped_height
        mapped_mask = kalman_1d_mapped_mask
        main_local_curve = kalman_1d_curve
    elif requested_main_variant == "kalman_cv" and kalman_cv_available:
        main_variant = "kalman_cv"
        mapped_laser = kalman_cv_mapped_laser
        mapped_height = kalman_cv_mapped_height
        mapped_mask = kalman_cv_mapped_mask
        main_local_curve = kalman_cv_curve
    else:
        main_variant = "baseline"
        mapped_laser = baseline_mapped_laser
        mapped_height = baseline_mapped_height
        mapped_mask = baseline_mapped_mask
        main_local_curve = baseline_curve
    final_dx = float(
        original_dx
    )

    final_dy = float(
        original_dy
        + phase_dy
        + auto_dy
    )

    final_sx = float(
        original_sx
    )

    final_sy = float(
        original_sy
    )

    # ---------------------------------------------------------------------
    # 4. FINAL FUSION IMAGES: BASELINE + CV KALMAN + 1-D KALMAN + RTS
    # ---------------------------------------------------------------------
    alpha = float(getattr(base, "LASER_ON_CAMERA_ALPHA", 0.55))

    baseline_overlay, baseline_final_mask = _compose_camera_laser_overlay(
        camera, baseline_mapped_laser, baseline_mapped_mask, alpha
    )
    baseline_output_path = output_path.with_name(
        output_path.stem + str(BASELINE_FUSION_SUFFIX) + output_path.suffix
    )
    baseline_overlay.save(baseline_output_path)

    cv_output_path = output_path.with_name(
        output_path.stem + str(KALMAN_FUSION_SUFFIX) + output_path.suffix
    )
    cv_final_mask = np.asarray(kalman_cv_mapped_mask.convert("L"), dtype=np.uint8) > 127
    if kalman_cv_available:
        cv_overlay, cv_final_mask = _compose_camera_laser_overlay(
            camera, kalman_cv_mapped_laser, kalman_cv_mapped_mask, alpha
        )
        cv_overlay.save(cv_output_path)

    one_d_output_path = output_path.with_name(
        output_path.stem + str(KALMAN_1D_FUSION_SUFFIX) + output_path.suffix
    )
    one_d_final_mask = np.asarray(kalman_1d_mapped_mask.convert("L"), dtype=np.uint8) > 127
    if kalman_1d_available:
        one_d_overlay, one_d_final_mask = _compose_camera_laser_overlay(
            camera, kalman_1d_mapped_laser, kalman_1d_mapped_mask, alpha
        )
        one_d_overlay.save(one_d_output_path)

    rts_output_path = output_path.with_name(
        output_path.stem + str(KALMAN_RTS_FUSION_SUFFIX) + output_path.suffix
    )
    rts_final_mask = np.asarray(kalman_rts_mapped_mask.convert("L"), dtype=np.uint8) > 127
    if kalman_rts_available:
        rts_overlay, rts_final_mask = _compose_camera_laser_overlay(
            camera, kalman_rts_mapped_laser, kalman_rts_mapped_mask, alpha
        )
        rts_overlay.save(rts_output_path)

    # Historical fusion_2d.png follows MAIN_LOCAL_Y_VARIANT.
    if main_variant == "kalman_rts" and kalman_rts_available:
        rts_overlay.save(output_path)
        final_mask = rts_final_mask
    elif main_variant == "kalman_1d" and kalman_1d_available:
        one_d_overlay.save(output_path)
        final_mask = one_d_final_mask
    elif main_variant == "kalman_cv" and kalman_cv_available:
        cv_overlay.save(output_path)
        final_mask = cv_final_mask
    else:
        baseline_overlay.save(output_path)
        final_mask = baseline_final_mask

    paths: dict[str, str] = {
        "main_fusion": str(output_path),
        "main_fusion_variant": str(main_variant),
        "baseline_fusion": str(baseline_output_path),
    }
    if kalman_cv_available:
        paths["kalman_cv_fusion"] = str(cv_output_path)
    if kalman_1d_available:
        paths["kalman_1d_fusion"] = str(one_d_output_path)
    if kalman_rts_available:
        paths["kalman_rts_fusion"] = str(rts_output_path)
    for k, v in kalman_1d_diag_paths.items():
        paths[k] = str(v)
    for k, v in kalman_rts_diag_paths.items():
        paths[k] = str(v)

    correspondence_path = (
        base.save_full_affine_correspondence_maps(
            maps,
            calibration,
            output_path,
        )
    )

    if correspondence_path is not None:
        paths[
            "full_affine_correspondence_maps"
        ] = str(
            correspondence_path
        )

    # Preserve original X diagnostics.
    if x_alignment.get(
        "result_json"
    ):
        paths[
            "original_x_alignment_json"
        ] = str(
            Path(
                x_alignment[
                    "result_json"
                ]
            )
        )

    if x_alignment.get(
        "search_csv"
    ):
        paths[
            "original_x_alignment_search_csv"
        ] = str(
            Path(
                x_alignment[
                    "search_csv"
                ]
            )
        )

    if x_alignment.get(
        "edge_debug_png"
    ):
        paths[
            "original_x_alignment_edge_debug"
        ] = str(
            Path(
                x_alignment[
                    "edge_debug_png"
                ]
            )
        )

    if x_alignment.get(
        "match_csv"
    ):
        paths[
            "original_x_alignment_match_csv"
        ] = str(
            Path(
                x_alignment[
                    "match_csv"
                ]
            )
        )

    if y_phase.get(
        "result_json"
    ):
        paths[
            "auto_y_wire_phase_json"
        ] = str(
            Path(
                y_phase[
                    "result_json"
                ]
            )
        )

    if y_phase.get(
        "search_csv"
    ):
        paths[
            "auto_y_wire_phase_search_csv"
        ] = str(
            Path(
                y_phase[
                    "search_csv"
                ]
            )
        )

    if auto_y.get(
        "result_json"
    ):
        paths[
            "feature_auto_y_json"
        ] = str(
            Path(
                auto_y[
                    "result_json"
                ]
            )
        )

    if auto_y.get(
        "search_csv"
    ):
        paths[
            "feature_auto_y_search_csv"
        ] = str(
            Path(
                auto_y[
                    "search_csv"
                ]
            )
        )

    if local_y.get(
        "result_json"
    ):
        paths[
            "local_feature_y_json"
        ] = str(
            Path(
                local_y[
                    "result_json"
                ]
            )
        )

    if local_y.get(
        "window_csv"
    ):
        paths[
            "local_feature_y_windows_csv"
        ] = str(
            Path(
                local_y[
                    "window_csv"
                ]
            )
        )

    if local_y.get(
        "curve_csv"
    ):
        paths[
            "local_feature_y_curve_csv"
        ] = str(
            Path(
                local_y[
                    "curve_csv"
                ]
            )
        )

    for key, path_key in (
        ("kalman_window_csv", "kalman_cv_local_y_windows_csv"),
        ("kalman_curve_csv", "kalman_cv_local_y_curve_csv"),
        ("baseline_vs_kalman_curve_csv", "local_y_baseline_vs_kalman_cv_csv"),
        ("kalman_json", "kalman_cv_local_y_json"),
    ):
        if local_y.get(key):
            paths[path_key] = str(Path(local_y[key]))

    # ---------------------------------------------------------------------
    # 5. REAL NUMERIC HEIGHT: ALL THREE VARIANTS FROM THE SAME BASE MAP
    # ---------------------------------------------------------------------
    mapped_height_mm_info = None
    mapped_height_mm_baseline_info = None
    mapped_height_mm_cv_info = None
    mapped_height_mm_1d_info = None
    mapped_height_mm_rts_info = None

    if getattr(base, "SAVE_MAPPED_HEIGHT_MM_NPZ", True):
        # Create the common physical-affine + X + global-Y numeric base once.
        mapped_height_mm_info = base.save_mapped_height_mm_npz_full_affine(
            laser_path=laser_path,
            laser_size_wh=laser.size,
            camera=camera,
            calibration=calibration,
            maps=maps,
            mapped_visual_mask=pre_local_mask,
            output_path=output_path,
            final_residual_dx_px=final_dx,
            final_residual_dy_px=final_dy,
            final_residual_x_scale=final_sx,
            final_residual_y_scale=final_sy,
        )

        main_npz_path = Path(mapped_height_mm_info["npz"])
        base_npz_path = main_npz_path.with_name("mapped_height_mm_pre_local_base.npz")
        shutil.copy2(main_npz_path, base_npz_path)
        paths["mapped_height_mm_pre_local_base_npz"] = str(base_npz_path)

        # Explicit BASELINE copy.
        baseline_npz_path = main_npz_path.parent / str(MAPPED_HEIGHT_MM_BASELINE_FILENAME)
        shutil.copy2(base_npz_path, baseline_npz_path)
        mapped_height_mm_baseline_info = _patch_mapped_height_npz_with_local_curve(
            base,
            baseline_npz_path,
            baseline_curve,
            baseline_final_mask,
            (phase_dy + auto_dy),
            preview_filename="mapped_height_mm_baseline_preview.png",
            mapping_variant="baseline",
        )
        paths["mapped_height_mm_baseline_npz"] = str(baseline_npz_path)
        if mapped_height_mm_baseline_info.get("preview_png"):
            paths["mapped_height_mm_baseline_preview"] = str(mapped_height_mm_baseline_info["preview_png"])

        # Legacy constant-velocity Kalman copy.
        cv_npz_path = main_npz_path.parent / str(MAPPED_HEIGHT_MM_KALMAN_FILENAME)
        if kalman_cv_available:
            shutil.copy2(base_npz_path, cv_npz_path)
            mapped_height_mm_cv_info = _patch_mapped_height_npz_with_local_curve(
                base,
                cv_npz_path,
                kalman_cv_curve,
                cv_final_mask,
                (phase_dy + auto_dy),
                preview_filename="mapped_height_mm_kalman_cv_preview.png",
                mapping_variant="kalman_constant_velocity_local_y",
            )
            paths["mapped_height_mm_kalman_cv_npz"] = str(cv_npz_path)
            if mapped_height_mm_cv_info.get("preview_png"):
                paths["mapped_height_mm_kalman_cv_preview"] = str(mapped_height_mm_cv_info["preview_png"])

        # Recommended 1-D adaptive Kalman copy.
        one_d_npz_path = main_npz_path.parent / str(MAPPED_HEIGHT_MM_KALMAN_1D_FILENAME)
        if kalman_1d_available:
            shutil.copy2(base_npz_path, one_d_npz_path)
            mapped_height_mm_1d_info = _patch_mapped_height_npz_with_local_curve(
                base,
                one_d_npz_path,
                kalman_1d_curve,
                one_d_final_mask,
                (phase_dy + auto_dy),
                preview_filename="mapped_height_mm_kalman_1d_preview.png",
                mapping_variant="kalman_1d_adaptive_random_walk_local_y",
            )
            paths["mapped_height_mm_kalman_1d_npz"] = str(one_d_npz_path)
            if mapped_height_mm_1d_info.get("preview_png"):
                paths["mapped_height_mm_kalman_1d_preview"] = str(mapped_height_mm_1d_info["preview_png"])

        # Recommended 1-D Kalman + RTS fixed-interval smoother copy.
        rts_npz_path = main_npz_path.parent / str(MAPPED_HEIGHT_MM_KALMAN_RTS_FILENAME)
        if kalman_rts_available:
            shutil.copy2(base_npz_path, rts_npz_path)
            mapped_height_mm_rts_info = _patch_mapped_height_npz_with_local_curve(
                base,
                rts_npz_path,
                kalman_rts_curve,
                rts_final_mask,
                (phase_dy + auto_dy),
                preview_filename="mapped_height_mm_kalman_rts_preview.png",
                mapping_variant="kalman_1d_plus_rts_fixed_interval_local_y",
            )
            paths["mapped_height_mm_kalman_rts_npz"] = str(rts_npz_path)
            if mapped_height_mm_rts_info.get("preview_png"):
                paths["mapped_height_mm_kalman_rts_preview"] = str(mapped_height_mm_rts_info["preview_png"])

        # Historical mapped_height_mm.npz follows the same selected variant as
        # fusion_2d.png.  Copy the already-patched selected file to the main name.
        if main_variant == "kalman_rts" and kalman_rts_available:
            shutil.copy2(rts_npz_path, main_npz_path)
            selected_numeric_info = mapped_height_mm_rts_info
        elif main_variant == "kalman_1d" and kalman_1d_available:
            shutil.copy2(one_d_npz_path, main_npz_path)
            selected_numeric_info = mapped_height_mm_1d_info
        elif main_variant == "kalman_cv" and kalman_cv_available:
            shutil.copy2(cv_npz_path, main_npz_path)
            selected_numeric_info = mapped_height_mm_cv_info
        else:
            shutil.copy2(baseline_npz_path, main_npz_path)
            selected_numeric_info = mapped_height_mm_baseline_info

        # Main preview is also synchronized with the selected numeric branch.
        main_preview = main_npz_path.parent / "mapped_height_mm_preview.png"
        selected_preview = None if selected_numeric_info is None else selected_numeric_info.get("preview_png")
        if selected_preview and Path(selected_preview).exists():
            shutil.copy2(Path(selected_preview), main_preview)

        mapped_height_mm_info.update({
            "npz": str(main_npz_path),
            "preview_png": str(main_preview) if main_preview.exists() else None,
            "mapping_variant": str(main_variant),
        })
        paths["mapped_height_mm_npz"] = str(main_npz_path)
        if main_preview.exists():
            paths["mapped_height_mm_preview"] = str(main_preview)

    # ---------------------------------------------------------------------
    # 6. METADATA
    # ---------------------------------------------------------------------
    metadata_path = (
        output_path.with_name(
            output_path.stem
            + "_full_affine_metadata.json"
        )
    )

    metadata = {
        "method": (
            "full_physical_circular_affine_"
            "plus_original_x_"
            "plus_global_feature_auto_y_"
            "plus_parallel_baseline_cv_kalman_1d_adaptive_kalman_and_rts_local_y"
        ),
        "main_output_variant": str(main_variant),
        "kalman_cv_experiment_enabled": bool(USE_KALMAN_LOCAL_Y),
        "kalman_1d_experiment_enabled": bool(USE_KALMAN_1D_LOCAL_Y),
        "kalman_rts_experiment_enabled": bool(USE_KALMAN_RTS_LOCAL_Y),
        "kalman_cv_output_available": bool(kalman_cv_available),
        "kalman_1d_output_available": bool(kalman_1d_available),
        "kalman_rts_output_available": bool(kalman_rts_available),
        "registration_rmse_mm_used_for_kalman_initialization": (
            None if affine_rmse_mm is None else float(affine_rmse_mm)
        ),
        "registration_rmse_px_used_for_kalman_initialization": (
            None if registration_rmse_px is None else float(registration_rmse_px)
        ),
        "patch_marker": PATCH_MARKER,
        "original_x_algorithm_preserved": True,
        "feature_auto_y_only": True,
        "manual_y_used": False,
        "camera_is_final_canvas": True,
        "camera_resize": False,
        "camera_warp": False,
        "laser_source_horizontal_resize": False,
        "original_x_alignment": (
            x_alignment
        ),
        "automatic_y_wire_phase_alignment": (
            y_phase
        ),
        "feature_auto_y_alignment": (
            auto_y
        ),
        "local_feature_y_alignment": (
            local_y
        ),
        "local_feature_y_curve_summary": {
            "applied": bool(
                local_y.get(
                    "applied",
                    False,
                )
            ),
            "min_px": float(
                np.min(baseline_curve)
            ),
            "max_px": float(
                np.max(baseline_curve)
            ),
            "mean_px": float(
                np.mean(baseline_curve)
            ),
            "rms_px": float(
                np.sqrt(
                    np.mean(
                        baseline_curve.astype(
                            np.float64
                        ) ** 2
                    )
                )
            ),
        },
        "kalman_cv_local_y_curve_summary": {
            "available": bool(kalman_cv_available),
            "applied": bool(local_y.get("kalman_applied", False)),
            "reason": local_y.get("kalman_reason"),
            "min_px": float(np.min(kalman_cv_curve)),
            "max_px": float(np.max(kalman_cv_curve)),
            "mean_px": float(np.mean(kalman_cv_curve)),
            "rms_px": float(np.sqrt(np.mean(kalman_cv_curve.astype(np.float64) ** 2))),
        },
        "kalman_1d_local_y_curve_summary": {
            "available": bool(kalman_1d_available),
            "reason": str(kalman_1d_reason),
            "min_px": float(np.min(kalman_1d_curve)),
            "max_px": float(np.max(kalman_1d_curve)),
            "mean_px": float(np.mean(kalman_1d_curve)),
            "rms_px": float(np.sqrt(np.mean(kalman_1d_curve.astype(np.float64) ** 2))),
            "report": kalman_1d_report,
        },
        "kalman_rts_local_y_curve_summary": {
            "available": bool(kalman_rts_available),
            "reason": str(kalman_rts_reason),
            "min_px": float(np.min(kalman_rts_curve)),
            "max_px": float(np.max(kalman_rts_curve)),
            "mean_px": float(np.mean(kalman_rts_curve)),
            "rms_px": float(np.sqrt(np.mean(kalman_rts_curve.astype(np.float64) ** 2))),
            "report": kalman_rts_report,
        },
        "final_residual": {
            "dx_px": float(
                final_dx
            ),
            "dy_px": float(
                final_dy
            ),
            "x_scale": float(
                final_sx
            ),
            "y_scale": float(
                final_sy
            ),
            "original_x_stage_dx_px": float(
                original_dx
            ),
            "original_stage_dy_px": float(
                original_dy
            ),
            "automatic_y_wire_phase_dy_px": float(
                phase_dy
            ),
            "new_feature_auto_y_dy_px": float(
                auto_dy
            ),
        },
        "new_y_stage_locked": {
            "dx_px": 0.0,
            "x_scale": 1.0,
            "y_scale": 1.0,
            "rotation_deg": 0.0,
            "shear": 0.0,
        },
        "physical_affine_2x3": calibration[
            "matrix_2x3_list"
        ],
        "calibration": {
            key: value
            for key, value
            in calibration.items()
            if key
            not in (
                "matrix_2x3",
                "source_json",
            )
        },
        "mapped_numeric_height_mm": (
            mapped_height_mm_info
        ),
        "mapped_numeric_height_mm_baseline": mapped_height_mm_baseline_info,
        "mapped_numeric_height_mm_kalman_cv": mapped_height_mm_cv_info,
        "mapped_numeric_height_mm_kalman_1d": mapped_height_mm_1d_info,
        "mapped_numeric_height_mm_kalman_rts": mapped_height_mm_rts_info,
        "outputs": paths,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    paths[
        "metadata"
    ] = str(
        metadata_path
    )

    print("")
    print("=" * 82)
    print(
        "FULL AFFINE + ORIGINAL X + FEATURE-CORRELATION AUTO Y-ONLY"
    )
    print("=" * 82)
    print(
        f"Original X method: "
        f"{x_alignment.get('method')}"
    )
    print(
        f"Original X correction: "
        f"{original_dx:+.3f}px"
    )
    print(
        f"Automatic Y wire phase: "
        f"k={int(y_phase.get('winning_phase_k', 0)):+d}, "
        f"pitch={float(y_phase.get('estimated_y_pitch_px', 0.0)):.4f}px, "
        f"dy={phase_dy:+.4f}px"
    )
    print(
        f"Feature AUTO global Y residual: "
        f"{auto_dy:+.4f}px"
    )
    print(
        f"Local feature-Y curve: "
        f"applied={local_y.get('applied', False)}, "
        f"range={float(np.min(baseline_curve)):+.4f}.."
        f"{float(np.max(baseline_curve)):+.4f}px"
    )
    print(
        f"Kalman-CV local-Y curve: "
        f"available={kalman_cv_available}, "
        f"applied={local_y.get('kalman_applied', False)}, "
        f"range={float(np.min(kalman_cv_curve)):+.4f}.."
        f"{float(np.max(kalman_cv_curve)):+.4f}px"
    )
    print(
        f"Kalman-1D local-Y curve: "
        f"available={kalman_1d_available}, "
        f"reason={kalman_1d_reason}, "
        f"range={float(np.min(kalman_1d_curve)):+.4f}.."
        f"{float(np.max(kalman_1d_curve)):+.4f}px"
    )
    print(
        f"Kalman-RTS local-Y curve: "
        f"available={kalman_rts_available}, "
        f"reason={kalman_rts_reason}, "
        f"range={float(np.min(kalman_rts_curve)):+.4f}.."
        f"{float(np.max(kalman_rts_curve)):+.4f}px"
    )
    print(f"MAIN local-Y variant: {main_variant}")
    print(
        "Global/local Y stages changed X: False"
    )
    print(
        f"FINAL residual used by image AND numeric height: "
        f"dx={final_dx:+.3f}px, "
        f"dy={final_dy:+.4f}px"
    )

    if mapped_height_mm_info is not None:
        print(
            f"mapped_height_mm.npz: "
            f"{mapped_height_mm_info['npz']}"
        )
    if mapped_height_mm_cv_info is not None:
        print(f"mapped_height_mm_kalman_cv.npz: {paths.get('mapped_height_mm_kalman_cv_npz')}")
    if mapped_height_mm_1d_info is not None:
        print(f"mapped_height_mm_kalman_1d.npz: {paths.get('mapped_height_mm_kalman_1d_npz')}")
    if mapped_height_mm_rts_info is not None:
        print(f"mapped_height_mm_kalman_rts.npz: {paths.get('mapped_height_mm_kalman_rts_npz')}")

    print("=" * 82)

    return paths


def fuse_2d(laser_path: Path, camera_path: Path, output_path: Path):
    print("=" * 78)
    print("FUSION FINAL: FULL PHYSICAL CIRCULAR AFFINE + ORIGINAL X + AUTO Y")
    print("Primary mapping: camera pixel -> physical X/arc -> complete camera-to-laser affine -> current laser pixel.")
    print("Post-affine X: original wire-pitch + symmetric Chamfer X residual.")
    print("Post-affine Y: global feature Y-only residual + baseline/1D-Kalman/RTS local curves.")
    print("Whole-wire Y phase correction: removed/disabled.")
    print("=" * 78)

    laser_path = resolve_image_path(
        laser_path,
        (
            '*_true_scale_0p0504mm_detail_height.png',
            '*_true_scale_0p0504mm_absolute_height.png',
            '*_true_scale_detail_height.png',
            '*_true_scale_absolute_height.png',
            '*detail_height.png',
        ),
        'laser',
    )
    camera_path = resolve_image_path(
        camera_path,
        ('*manual_full_roi_stitched.png', '*stitched*.png'),
        'camera',
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    laser = Image.open(laser_path).convert('RGB')
    camera = Image.open(camera_path).convert('RGB')
    width, height = laser.size
    laser_valid_mask_img, laser_mask_npz_path, laser_valid_fraction_raw = load_laser_valid_mask_image(
        laser_path, (width, height)
    )
    laser_height_value_img, _laser_contour_levels, laser_height_npz_path = load_laser_height_value_image(
        laser_path, (width, height)
    )

    print(f"Laser image:  {laser_path}")
    print(f"Camera image: {camera_path}")
    print(f"Output:       {output_path}")
    print(f"Laser size:   {laser.size[0]} x {laser.size[1]} px")
    print(f"Camera size:  {camera.size[0]} x {camera.size[1]} px")

    full_affine_paths = save_full_physical_affine_final_outputs(
        laser_path,
        laser,
        laser_height_value_img,
        laser_valid_mask_img,
        camera,
        output_path,
    )

    info_path = output_path.with_suffix(".txt")
    info = {
        "laser_image": str(laser_path),
        "laser_mask_npz": str(laser_mask_npz_path) if laser_mask_npz_path else None,
        "laser_height_npz": str(laser_height_npz_path) if laser_height_npz_path else None,
        "laser_valid_fraction_raw": float(laser_valid_fraction_raw),
        "camera_image": str(camera_path),
        "output_image": str(output_path),
        "fusion_mode": "full_physical_circular_affine_plus_original_x_plus_auto_y",
        "whole_wire_y_phase_removed": True,
        "feature_auto_y_enabled": bool(USE_FINAL_FEATURE_AUTO_Y),
        "local_feature_y_enabled": bool(USE_FINAL_LOCAL_FEATURE_Y),
        "outputs": full_affine_paths,
    }
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Info: {info_path}")



if __name__ == "__main__":
    print("")
    print("#" * 82)
    print("FUSION FINAL: FULL AFFINE + ORIGINAL X + GLOBAL Y + 1D KALMAN + RTS")
    print("#" * 82)
    print(f"Global Y residual: +/-{FEATURE_Y_SEARCH_RANGE_PX:g}px, fine={FEATURE_Y_FINE_STEP_PX:g}px")
    print(f"Local Y: window={LOCAL_FEATURE_Y_WINDOW_HEIGHT_PX}px, step={LOCAL_FEATURE_Y_STEP_PX}px, range=+/-{LOCAL_FEATURE_Y_SEARCH_RANGE_PX:g}px")
    print("Whole-wire Y phase correction: removed/disabled")
    print("Manual Y: none")
    print("#" * 82)
    print("")
    main()
