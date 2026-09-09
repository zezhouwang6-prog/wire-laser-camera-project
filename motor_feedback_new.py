r"""Standalone shared-zero motor + KEYENCE LJ-X8060 + Basler camera workflow.

This file is the standalone form of the original motor_feedback.py.  It keeps the
same laser-first / camera-second acquisition logic, but the code previously loaded
from motor_laser_sync.py and motor_camera.py has been integrated directly here.

It therefore does NOT require these two files:
    D:\project\scripts\motor_laser_sync.py
    D:\project\scripts\motor_camera.py

Hardware/SDK dependencies are still required:
    - control_motor_gsusb.py from the DaMiao motor control package
    - KEYENCE LJXAwrap.py / LJ-X8000A Python SDK
    - python-can
    - numpy
    - OpenCV (cv2)
    - pypylon

Outputs default to D:\project\MotorLaserCamera\run1, run2, ... unless
COMBINED_ACQUISITION_ROOT is available from config.py.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pypylon import pylon

# config.py is optional in this standalone version.  If present, keep using the
# project's centralized output path; otherwise fall back to the historical path.
try:
    from config import COMBINED_ACQUISITION_ROOT
except Exception:
    COMBINED_ACQUISITION_ROOT = Path(r"D:\project\MotorLaserCamera")


# =========================
# USER SETTINGS
# =========================

OUTPUT_ROOT = COMBINED_ACQUISITION_ROOT

# External hardware SDK locations.  These are SDK/library dependencies, not
# motor_laser_sync.py or motor_camera.py.
MOTOR_CONTROL_CODE_DIR = Path(r"D:\Projeck laser\DM-J4310-2EC-master_code")
LASER_CODE_DIR = Path(r"D:\Projeck laser\Zezhou\LJ-X8000A_PyLib_Pruefstand\PYTHON")

CAN_CHANNEL = 0
CAN_BITRATE = 1_000_000
MOTOR_CAN_ID = 0x01
MASTER_ID = 0x00

DIRECTION = -1

# Laser continuous scan: 6 deg/s gives 60 s for 360 deg.
LASER_SPEED_DEG_PER_SEC = 6.0
LASER_ROTATION_DEG = 360.0
LASER_SAMPLE_HZ = 500.0
LASER_TARGET_PROFILES = 0

# MIT laser motor control. These are the parameters that tested smoothest.
MIT_KP = 0.03
MIT_KD = 3.5
MIT_TAU = 0.19
MIT_COMMAND_HZ = 200.0
MIT_STOP_EARLY_DEG = 0.0
MIT_HOLD_AFTER_MOTION_S = 0.5

# After laser scan, rotate this amount from the laser position to camera-facing position.
# Positive value means magnitude. The actual motor target is DIRECTION * CAMERA_OFFSET_DEG.
CAMERA_OFFSET_DEG = 0.0

# Camera stop-and-shoot scan.
CAMERA_SPEED_DEG_PER_SEC = 1.0
CAMERA_ROTATION_DEG = 360.0
CAMERA_ANGLE_STEP_DEG = 1.0
CAMERA_SETTLE_TIME_S = 1.0

# Laser controller.
LASER_MODEL = "LJ-X8060"
LASER_DEVICE_ID = 0
LASER_IP = "192.168.0.1"
LASER_PORT = 24691
LASER_HIGH_SPEED_PORT = 24692
LASER_CHUNK_PROFILES = 1000

# Laser height CSV export settings retained from motor_laser_sync.py.
HEIGHT_INVALID_MM = -99.9996
HEIGHT_CSV_NAME = "laser_height_mm.csv"
HEIGHT_CSV_FMT = "%.4f"

# Basler camera settings retained from motor_camera.py.
EXPECTED_CAMERA_MODEL = "acA2440-20gc"
CAMERA_SERIAL_NUMBER = "23680766"
CAMERA_WIDTH = 2448
CAMERA_HEIGHT = 2048
EXPOSURE_TIME_US = 30_000.0
CAMERA_TIMEOUT_MS = 5_000
PREVIEW_WINDOW_NAME = "Basler Live Preview + Fixed ROI"
PREVIEW_SCREEN_SCALE = 0.85
PREVIEW_MIN_WIDTH = 800
PREVIEW_MIN_HEIGHT = 600

# Fixed ROI used by the current camera stitching workflow.
# These coordinates are only used for LIVE PREVIEW display here.
# Full 2448 x 2048 images are still saved during acquisition.
SHOW_FIXED_ROI_PREVIEW = True
FIXED_ROI_X_PX = 1038
FIXED_ROI_Y_PX = 1035
FIXED_ROI_WIDTH_PX = 339
FIXED_ROI_HEIGHT_PX = 85

# A second window shows only the fixed ROI enlarged in real time, so the
# 85-pixel-high strip remains easy to inspect on screen.
SHOW_FIXED_ROI_ZOOM_WINDOW = True
FIXED_ROI_ZOOM_SCALE = 4.0
FIXED_ROI_WINDOW_NAME = "LIVE Fixed ROI 339x85"
PREVIEW_BUILD_TAG = "FIXED_ROI_PREVIEW_V3_2026-08-25"

# CAN/safety timings retained from the original source modules.
SEND_TIMEOUT_S = 0.5
CAN_SEND_RETRIES = 5
CAN_RETRY_DELAY_S = 0.05
RAW_COMMAND_REPEAT_COUNT = 3
RAW_COMMAND_REPEAT_INTERVAL_S = 0.05
CAMERA_DISABLE_SEND_TIMEOUT_S = 0.02
CAN_DRAIN_TIMEOUT_S = 0.08
CAN_DRAIN_MAX_MESSAGES = 512
SAFETY_DISABLE_REPEAT_COUNT = 8
POSITION_LIMIT_MARGIN_RAD = 0.1
MOTION_TIMEOUT_MARGIN_S = 8.0
CAN_DRAIN_INTERVAL_S = 0.05

STOP_REQUESTED = False
STOP_EVENT = threading.Event()


# =========================
# External SDK imports
# =========================

if not (MOTOR_CONTROL_CODE_DIR / "control_motor_gsusb.py").exists():
    raise FileNotFoundError(
        f"Cannot find control_motor_gsusb.py in: {MOTOR_CONTROL_CODE_DIR}"
    )
if not (LASER_CODE_DIR / "LJXAwrap.py").exists():
    raise FileNotFoundError(
        f"Cannot find LJXAwrap.py in: {LASER_CODE_DIR}"
    )

sys.path.insert(0, str(MOTOR_CONTROL_CODE_DIR))
sys.path.insert(0, str(LASER_CODE_DIR))

import can  # noqa: E402
import LJXAwrap  # noqa: E402
from control_motor_gsusb import (  # noqa: E402
    Control_Type,
    DM_Motor_Type,
    GsUsb,
    GsUsbDmControl,
    Motor,
    float_to_uint8s,
)



# =========================
# Integrated laser/motor helpers
# =========================
def parse_ip(value: str) -> tuple[int, int, int, int]:
    parts = value.strip().split(".")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("IP must look like 192.168.0.1")
    numbers = tuple(int(part) for part in parts)
    if any(part < 0 or part > 255 for part in numbers):
        raise argparse.ArgumentTypeError("IP octets must be 0..255")
    return numbers

def scan_adapters() -> int:
    devices = GsUsb.scan()
    if not devices:
        print("No gs_usb/candleLight USB-CAN adapter found.")
        return 1
    print("Detected gs_usb/candleLight adapters:")
    for index, device in enumerate(devices):
        print(f"  channel {index}: {device}")
    return 0

def send_can_with_retry(bus: can.BusABC, msg: can.Message, description: str) -> None:
    last_exc: Exception | None = None
    for attempt in range(1, CAN_SEND_RETRIES + 1):
        try:
            bus.send(msg, timeout=SEND_TIMEOUT_S)
            return
        except can.CanError as exc:
            last_exc = exc
            if attempt < CAN_SEND_RETRIES:
                time.sleep(CAN_RETRY_DELAY_S)
    raise RuntimeError(f"{description} failed after {CAN_SEND_RETRIES} retries") from last_exc

def send_position_speed_no_feedback(
    bus: can.BusABC,
    motor: Motor,
    position_rad: float,
    velocity_rad_s: float,
) -> None:
    msg = can.Message(
        arbitration_id=0x100 + motor.SlaveID,
        data=float_to_uint8s(float(position_rad)) + float_to_uint8s(float(velocity_rad_s)),
        is_extended_id=False,
    )
    send_can_with_retry(bus, msg, "position/speed command")

def send_position_target(
    bus: can.BusABC,
    motor: Motor,
    position_rad: float,
    velocity_rad_s: float,
) -> None:
    for _ in range(3):
        send_position_speed_no_feedback(bus, motor, position_rad, velocity_rad_s)
        time.sleep(0.02)

def send_control_command_no_feedback(bus: can.BusABC, motor: Motor, command: int) -> None:
    data = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, command])
    for _ in range(RAW_COMMAND_REPEAT_COUNT):
        msg = can.Message(arbitration_id=motor.SlaveID, data=data, is_extended_id=False)
        send_can_with_retry(bus, msg, f"raw motor command 0x{command:02X}")
        time.sleep(RAW_COMMAND_REPEAT_INTERVAL_S)

def switch_to_position_velocity_no_feedback(bus: can.BusABC, motor: Motor) -> None:
    can_id_l = motor.SlaveID & 0xFF
    can_id_h = (motor.SlaveID >> 8) & 0xFF
    mode_bytes = int(Control_Type.POS_VEL).to_bytes(4, "little", signed=False)
    data = bytes([can_id_l, can_id_h, 0x55, 10] + list(mode_bytes))
    for _ in range(RAW_COMMAND_REPEAT_COUNT):
        msg = can.Message(arbitration_id=0x7FF, data=data, is_extended_id=False)
        send_can_with_retry(bus, msg, "switch motor to position-velocity mode")
        time.sleep(RAW_COMMAND_REPEAT_INTERVAL_S)

def send_disable_no_feedback(bus: can.BusABC, motor: Motor) -> None:
    msg = can.Message(
        arbitration_id=motor.SlaveID,
        data=bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]),
        is_extended_id=False,
    )
    send_can_with_retry(bus, msg, "motor disable command")

def open_motor_bus(channel: int, bitrate: int, can_id: int, master_id: int):
    bus = can.Bus(interface="gs_usb", channel=channel, bitrate=bitrate)
    motor = Motor(DM_Motor_Type.DM4310, can_id, master_id)
    ctrl = GsUsbDmControl(bus)
    ctrl.add_motor(motor)
    return bus, ctrl, motor

@dataclass
class LaserChunk:
    chunk_index: int
    callback_wall_time_iso: str
    callback_time_s: float
    height: np.ndarray
    luminance: np.ndarray | None

class LaserHighSpeedRecorder:
    def __init__(
        self,
        *,
        record_dir: Path,
        device_id: int,
        ip: str,
        port: int,
        high_speed_port: int,
        chunk_profiles: int,
        sample_hz: float,
        target_profiles: int,
    ) -> None:
        self.record_dir = record_dir
        self.chunk_dir = record_dir / "laser_chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.device_id = device_id
        self.ip = ip
        self.port = port
        self.high_speed_port = high_speed_port
        self.chunk_profiles = chunk_profiles
        self.sample_hz = sample_hz
        self.target_profiles = target_profiles

        self.ethernet = LJXAwrap.LJX8IF_ETHERNET_CONFIG()
        ip_parts = parse_ip(ip)
        for i, value in enumerate(ip_parts):
            self.ethernet.abyIpAddress[i] = value
        self.ethernet.wPortNo = port

        self.callback_ref = None
        self.profinfo = LJXAwrap.LJX8IF_PROFILE_INFO()
        self.z_unit_um = None
        self.luminance_enabled = 0
        self.x_points = 0
        self.line_count = 0
        self.chunk_count = 0
        self.start_measure_result: int | None = None
        self.queue: queue.Queue[LaserChunk | None] = queue.Queue(maxsize=20)
        self.writer_thread: threading.Thread | None = None
        self.index_handle = None
        self.index_writer = None
        self.start_perf = 0.0
        self.running = False
        self.write_errors: list[str] = []
        self.callback_count = 0
        self.ignored_callback_count = 0
        self.last_notify = None
        self.last_profnum = None

    def _check(self, name: str, result: int) -> None:
        if result != 0:
            raise RuntimeError(f"{name} failed: 0x{result:08X}")

    def open(self) -> None:
        self._check("LJX8IF_EthernetOpen", LJXAwrap.LJX8IF_EthernetOpen(self.device_id, self.ethernet))

    def prepare(self, start_perf: float) -> None:
        self.start_perf = start_perf
        self.index_handle, self.index_writer = open_csv(
            self.record_dir / "laser_index.csv",
            [
                "chunk_index",
                "callback_wall_time_iso",
                "callback_time_s",
                "line_start",
                "line_count",
                "estimated_first_line_time_s",
                "estimated_last_line_time_s",
                "height_file",
                "luminance_file",
            ],
        )

        self.callback_ref = LJXAwrap.LJX8IF_CALLBACK_SIMPLE_ARRAY(self._callback)
        self._check(
            "LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray",
            LJXAwrap.LJX8IF_InitializeHighSpeedDataCommunicationSimpleArray(
                self.device_id,
                self.ethernet,
                self.high_speed_port,
                self.callback_ref,
                self.chunk_profiles,
                0,
            ),
        )

        req = LJXAwrap.LJX8IF_HIGH_SPEED_PRE_START_REQ()
        req.bySendPosition = 2
        self._check(
            "LJX8IF_PreStartHighSpeedDataCommunication",
            LJXAwrap.LJX8IF_PreStartHighSpeedDataCommunication(self.device_id, req, self.profinfo),
        )

        self.x_points = int(self.profinfo.wProfileDataCount)
        self.luminance_enabled = int(self.profinfo.byLuminanceOutput)

        z_unit = ctypes.c_ushort()
        self._check("LJX8IF_GetZUnitSimpleArray", LJXAwrap.LJX8IF_GetZUnitSimpleArray(self.device_id, z_unit))
        self.z_unit_um = z_unit.value / 100.0

        x_axis_mm = np.array(
            [
                (self.profinfo.lXStart + self.profinfo.lXPitch * i) / 100000.0
                for i in range(self.x_points)
            ],
            dtype=np.float64,
        )
        np.savetxt(self.record_dir / "laser_x_axis_mm.csv", x_axis_mm, delimiter=",", header="x_mm", comments="")

        self.writer_thread = threading.Thread(target=self._writer_loop, name="laser-chunk-writer", daemon=True)
        self.writer_thread.start()

    def start(self) -> None:
        self._check("LJX8IF_StartHighSpeedDataCommunication", LJXAwrap.LJX8IF_StartHighSpeedDataCommunication(self.device_id))
        self.start_measure_result = LJXAwrap.LJX8IF_StartMeasure(self.device_id)
        if self.start_measure_result != 0:
            print(
                "Warning: LJX8IF_StartMeasure returned "
                f"0x{self.start_measure_result:08X}; continuing because high-speed data may already be active."
            )
        self.running = True

    def stop(self) -> None:
        if self.running:
            LJXAwrap.LJX8IF_StopMeasure(self.device_id)
            LJXAwrap.LJX8IF_StopHighSpeedDataCommunication(self.device_id)
            self.running = False
        LJXAwrap.LJX8IF_FinalizeHighSpeedDataCommunication(self.device_id)
        try:
            self.queue.put(None, timeout=1.0)
        except queue.Full:
            self.write_errors.append("laser writer queue was full during stop; writer thread was not joined cleanly")
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=5.0)
            if self.writer_thread.is_alive():
                self.write_errors.append("laser writer thread did not stop within 5 seconds")
        if self.index_handle is not None:
            self.index_handle.flush()
            self.index_handle.close()
        LJXAwrap.LJX8IF_CommunicationClose(self.device_id)

    def _callback(
        self,
        _p_header,
        p_height,
        p_lumi,
        luminance_enable,
        xpointnum,
        profnum,
        notify,
        _user,
    ):
        self.callback_count += 1
        self.last_notify = int(notify)
        self.last_profnum = int(profnum)
        if notify not in (0, 0x10000) or profnum == 0:
            self.ignored_callback_count += 1
            return
        callback_time_s = time.perf_counter() - self.start_perf
        height = np.ctypeslib.as_array(p_height, shape=(int(xpointnum) * int(profnum),)).copy()
        height = height.reshape((int(profnum), int(xpointnum)))
        luminance = None
        if luminance_enable == 1 and p_lumi:
            luminance = np.ctypeslib.as_array(p_lumi, shape=(int(xpointnum) * int(profnum),)).copy()
            luminance = luminance.reshape((int(profnum), int(xpointnum)))

        self.chunk_count += 1
        chunk = LaserChunk(
            chunk_index=self.chunk_count,
            callback_wall_time_iso=now_iso(),
            callback_time_s=callback_time_s,
            height=height,
            luminance=luminance,
        )
        try:
            self.queue.put_nowait(chunk)
        except queue.Full:
            self.write_errors.append(f"laser queue full at chunk {self.chunk_count}")
        return

    def captured_profiles(self) -> int:
        queued_profiles = 0
        with self.queue.mutex:
            for item in list(self.queue.queue):
                if item is not None:
                    queued_profiles += int(item.height.shape[0])
        return self.line_count + queued_profiles

    def _writer_loop(self) -> None:
        while True:
            item = self.queue.get()
            if item is None:
                self.queue.task_done()
                break
            try:
                line_start = self.line_count
                original_line_count = int(item.height.shape[0])
                remaining = self.target_profiles - self.line_count if self.target_profiles > 0 else original_line_count
                if remaining <= 0:
                    continue
                line_count = min(original_line_count, remaining)
                height = item.height[:line_count]
                luminance = item.luminance[:line_count] if item.luminance is not None else None
                if line_count < original_line_count:
                    self.write_errors.append(
                        f"trimmed final laser chunk {item.chunk_index}: kept {line_count}/{original_line_count} profiles"
                    )
                height_name = f"height_chunk_{item.chunk_index:06d}.npy"
                lumi_name = ""
                np.save(self.chunk_dir / height_name, height)
                if luminance is not None:
                    lumi_name = f"luminance_chunk_{item.chunk_index:06d}.npy"
                    np.save(self.chunk_dir / lumi_name, luminance)

                estimated_last = item.callback_time_s
                if line_count < original_line_count:
                    estimated_last = item.callback_time_s - (original_line_count - line_count) / self.sample_hz
                estimated_first = estimated_last - max(0, line_count - 1) / self.sample_hz
                self.index_writer.writerow([
                    item.chunk_index,
                    item.callback_wall_time_iso,
                    f"{item.callback_time_s:.6f}",
                    line_start,
                    line_count,
                    f"{estimated_first:.6f}",
                    f"{estimated_last:.6f}",
                    f"laser_chunks/{height_name}",
                    f"laser_chunks/{lumi_name}" if lumi_name else "",
                ])
                if self.index_handle is not None:
                    self.index_handle.flush()
                self.line_count += line_count
            except Exception as exc:  # noqa: BLE001
                self.write_errors.append(f"failed to write laser chunk {item.chunk_index}: {exc}")
            finally:
                self.queue.task_done()

    def metadata(self) -> dict:
        return {
            "laser_model": LASER_MODEL,
            "device_id": self.device_id,
            "ip": self.ip,
            "port": self.port,
            "high_speed_port": self.high_speed_port,
            "sample_hz_for_time_estimation": self.sample_hz,
            "chunk_profiles": self.chunk_profiles,
            "target_profiles": self.target_profiles,
            "x_points": self.x_points,
            "luminance_enabled": self.luminance_enabled,
            "x_start_0p01um": int(self.profinfo.lXStart),
            "x_pitch_0p01um": int(self.profinfo.lXPitch),
            "z_unit_um": self.z_unit_um,
            "height_conversion": "height_raw == 0 is invalid; height_mm = (height_raw - 32768) * z_unit_um / 1000",
            "start_measure_result": self.start_measure_result,
            "line_count": self.line_count,
            "chunk_count": self.chunk_count,
            "callback_count": self.callback_count,
            "ignored_callback_count": self.ignored_callback_count,
            "last_notify": self.last_notify,
            "last_profnum": self.last_profnum,
            "write_errors": self.write_errors,
        }

def load_motor_samples(path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    times: list[float] = []
    pos_deg: list[float] = []
    vel_deg_s: list[float] = []
    torque_nm: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["time_s"]))
            pos_deg.append(float(row["position_deg"]))
            vel_deg_s.append(float(row["velocity_deg_s"]))
            torque_nm.append(float(row["torque_nm"]))
    return times, pos_deg, vel_deg_s, torque_nm

def interp_at(times: list[float], values: list[float], t: float) -> float | None:
    if not times:
        return None
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    lo = 0
    hi = len(times) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if times[mid] <= t:
            lo = mid
        else:
            hi = mid
    span = times[hi] - times[lo]
    if span <= 0:
        return values[lo]
    alpha = (t - times[lo]) / span
    return values[lo] + alpha * (values[hi] - values[lo])

def theory_angle_deg_at(
    *,
    laser_time_s: float,
    motor_start_time_s: float,
    speed_deg_s: float,
    rotation_deg: float,
    direction: int,
) -> float:
    elapsed_s = laser_time_s - motor_start_time_s
    if elapsed_s <= 0:
        return 0.0
    angle = elapsed_s * speed_deg_s
    if angle > rotation_deg:
        angle = rotation_deg
    return angle * direction

def write_sync_index(
    record_dir: Path,
    sample_hz: float,
    *,
    motor_start_time_s: float,
    speed_deg_s: float,
    rotation_deg: float,
    direction: int,
) -> None:
    motor_csv = record_dir / "motor_feedback.csv"
    laser_csv = record_dir / "laser_index.csv"
    sync_csv = record_dir / "sync_index.csv"
    times, pos_deg, vel_deg_s, torque_nm = load_motor_samples(motor_csv)

    with open(laser_csv, newline="", encoding="utf-8-sig") as laser_handle, open(
        sync_csv, "w", newline="", encoding="utf-8-sig"
    ) as sync_handle:
        reader = csv.DictReader(laser_handle)
        writer = csv.writer(sync_handle)
        writer.writerow([
            "laser_line",
            "laser_time_s",
            "theory_position_deg",
            "feedback_position_deg",
            "feedback_velocity_deg_s",
            "feedback_torque_nm",
            "fusion_position_deg",
            "fusion_angle_source",
            "laser_chunk_index",
            "line_in_chunk",
            "height_file",
            "luminance_file",
        ])
        for chunk in reader:
            line_start = int(chunk["line_start"])
            line_count = int(chunk["line_count"])
            last_time = float(chunk["estimated_last_line_time_s"])
            chunk_index = int(chunk["chunk_index"])
            for line_in_chunk in range(line_count):
                laser_line = line_start + line_in_chunk
                laser_time = last_time - (line_count - 1 - line_in_chunk) / sample_hz
                theory_p = theory_angle_deg_at(
                    laser_time_s=laser_time,
                    motor_start_time_s=motor_start_time_s,
                    speed_deg_s=speed_deg_s,
                    rotation_deg=rotation_deg,
                    direction=direction,
                )
                p = interp_at(times, pos_deg, laser_time)
                v = interp_at(times, vel_deg_s, laser_time)
                tau = interp_at(times, torque_nm, laser_time)
                writer.writerow([
                    laser_line,
                    f"{laser_time:.6f}",
                    f"{theory_p:.9f}",
                    "" if p is None else f"{p:.9f}",
                    "" if v is None else f"{v:.9f}",
                    "" if tau is None else f"{tau:.9f}",
                    f"{theory_p:.9f}",
                    "theory",
                    chunk_index,
                    line_in_chunk,
                    chunk["height_file"],
                    chunk["luminance_file"],
                ])

def write_laser_summary(record_dir: Path, laser: LaserHighSpeedRecorder, expected_profiles: int) -> Path:
    summary_path = record_dir / "laser_summary.txt"
    lines = [
        f"laser_line_count={laser.line_count}",
        f"laser_chunk_count={laser.chunk_count}",
        f"laser_callback_count={laser.callback_count}",
        f"laser_ignored_callback_count={laser.ignored_callback_count}",
        f"laser_last_notify={laser.last_notify}",
        f"laser_last_profnum={laser.last_profnum}",
        f"laser_expected_profiles={expected_profiles}",
        f"laser_start_measure_result={laser.start_measure_result}",
        f"laser_write_errors={laser.write_errors}",
    ]
    if laser.line_count == 0:
        lines.extend(
            [
                "",
                "NO LASER DATA WAS SAVED.",
                "Most likely reasons:",
                "1. LJ-X controller is not outputting high-speed profiles.",
                "2. LJ-X Navigator program trigger mode is external trigger, but no trigger is being supplied.",
                "3. Sampling cycle/frequency in LJ-X Navigator does not match this script.",
                "4. Laser is not in measurement/running state, or StartMeasure failed.",
                "5. IP/port/high-speed port is wrong, or Ethernet high-speed communication is blocked.",
                "6. Object is out of measuring range, or the controller program does not output height data.",
                "",
                "Check in LJ-X Navigator:",
                "- Current program number",
                "- Trigger mode: use internal/free-run for this software-only test",
                "- Sampling cycle: 1 kHz if you run with --laser-sample-hz 1000",
                "- High-speed communication / profile output enabled",
                "- Confirm the live height image is updating before running Python",
            ]
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path

def export_height_mm_csv(
    record_dir: Path,
    *,
    z_unit_um: float,
    invalid_mm: float = HEIGHT_INVALID_MM,
    csv_name: str = HEIGHT_CSV_NAME,
    fmt: str = HEIGHT_CSV_FMT,
) -> Path:
    """Export laser height chunks as one Navigator-style height matrix CSV.

    Rows are laser profiles, columns are X points. Raw height value 0 is invalid
    and is written as -99.9996 mm, matching the invalid marker used by LJ-X CSV
    exports in this project.
    """
    laser_index_path = record_dir / "laser_index.csv"
    output_path = record_dir / csv_name

    if not laser_index_path.exists():
        output_path.write_text("", encoding="utf-8")
        return output_path

    with open(laser_index_path, newline="", encoding="utf-8-sig") as index_handle, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as output_handle:
        for row in csv.DictReader(index_handle):
            height_file = row.get("height_file", "")
            if not height_file:
                continue
            height_path = record_dir / height_file
            if not height_path.exists():
                continue

            raw = np.load(height_path)
            raw_i32 = raw.astype(np.int32, copy=False)
            height_mm = (raw_i32.astype(np.float32) - 32768.0) * (float(z_unit_um) / 1000.0)
            height_mm[raw == 0] = float(invalid_mm)
            np.savetxt(output_handle, height_mm, delimiter=",", fmt=fmt)

    return output_path

def _camera_send_disable_no_feedback(bus: can.BusABC, motor: Motor) -> None:
    """Camera-style emergency disable send retained from motor_camera.py."""
    msg = can.Message(
        arbitration_id=motor.SlaveID,
        data=bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]),
        is_extended_id=False,
    )
    bus.send(msg, timeout=CAMERA_DISABLE_SEND_TIMEOUT_S)


# =========================
# Integrated Basler camera helpers
# =========================

def set_integer_node(camera, node_name: str, desired: int):
    try:
        node = getattr(camera, node_name)
        minimum = int(node.GetMin())
        maximum = int(node.GetMax())
        increment = max(1, int(node.GetInc()))
        value = max(minimum, min(maximum, int(desired)))
        value = minimum + ((value - minimum) // increment) * increment
        node.SetValue(value)
        return int(node.GetValue())
    except Exception:
        return None

def set_bool_node(camera, node_name: str, value: bool) -> bool:
    try:
        getattr(camera, node_name).SetValue(bool(value))
        return True
    except Exception:
        return False

def set_exposure(camera, exposure_us: float) -> float:
    for node_name in ("ExposureTime", "ExposureTimeAbs"):
        try:
            node = getattr(camera, node_name)
            value = max(node.GetMin(), min(node.GetMax(), float(exposure_us)))
            node.SetValue(value)
            return float(node.GetValue())
        except Exception:
            pass
    raise RuntimeError("The camera does not expose a writable exposure node.")

def get_exposure(camera) -> float:
    for node_name in ("ExposureTime", "ExposureTimeAbs"):
        try:
            return float(getattr(camera, node_name).GetValue())
        except Exception:
            pass
    return float(EXPOSURE_TIME_US)

def get_preview_window_size() -> tuple[int, int]:
    try:
        screen_width = int(ctypes.windll.user32.GetSystemMetrics(0))
        screen_height = int(ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        screen_width, screen_height = 1707, 1067

    max_width = int(screen_width * PREVIEW_SCREEN_SCALE)
    max_height = int(screen_height * PREVIEW_SCREEN_SCALE)
    image_aspect = CAMERA_WIDTH / CAMERA_HEIGHT

    width = max_width
    height = int(width / image_aspect)
    if height > max_height:
        height = max_height
        width = int(height * image_aspect)

    width = max(PREVIEW_MIN_WIDTH, width)
    height = max(PREVIEW_MIN_HEIGHT, height)
    return width, height

def draw_preview_guides(frame) -> None:
    height, width = frame.shape[:2]
    cx = width // 2
    cy = height // 2
    color = (0, 255, 0)
    cv2.line(frame, (cx, 0), (cx, height - 1), color, 2, cv2.LINE_AA)
    cv2.line(frame, (0, cy), (width - 1, cy), color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 24, color, 2, cv2.LINE_AA)


def validate_fixed_roi(frame) -> tuple[int, int, int, int]:
    if frame is None or frame.size == 0:
        raise RuntimeError("Cannot preview ROI on an empty camera frame.")

    height, width = frame.shape[:2]
    x = int(FIXED_ROI_X_PX)
    y = int(FIXED_ROI_Y_PX)
    w = int(FIXED_ROI_WIDTH_PX)
    h = int(FIXED_ROI_HEIGHT_PX)

    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise RuntimeError(
            "Fixed ROI is outside the live camera frame: "
            f"frame={width}x{height}, ROI=(x={x}, y={y}, w={w}, h={h})"
        )
    return x, y, w, h


def draw_fixed_roi_on_preview(frame) -> None:
    if not SHOW_FIXED_ROI_PREVIEW:
        return

    x, y, w, h = validate_fixed_roi(frame)
    x1 = x + w - 1
    y1 = y + h - 1

    # Thick black halo + yellow line so the ROI stays visible on green wire.
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 0), 7, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 255), 3, cv2.LINE_AA)

    cx = x + w // 2
    cy = y + h // 2
    cv2.drawMarker(
        frame,
        (cx, cy),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=2,
        line_type=cv2.LINE_AA,
    )

    label = f"FIXED ROI {w}x{h}  x={x}, y={y}"
    label_y = max(125, y - 12)
    cv2.putText(
        frame,
        label,
        (20, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label,
        (20, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_fixed_roi_zoom(frame):
    x, y, w, h = validate_fixed_roi(frame)
    roi = frame[y:y + h, x:x + w].copy()

    scale = max(1.0, float(FIXED_ROI_ZOOM_SCALE))
    zoom = cv2.resize(
        roi,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_LINEAR,
    )

    zoom = cv2.copyMakeBorder(
        zoom,
        58,
        12,
        12,
        12,
        cv2.BORDER_CONSTANT,
        value=(22, 22, 22),
    )
    cv2.putText(
        zoom,
        f"LIVE FIXED ROI  {w}x{h}  x={x}, y={y}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        zoom,
        f"{PREVIEW_BUILD_TAG} | S=start | Q/Esc=cancel",
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return zoom

def configure_full_sensor(camera) -> None:
    if camera.IsGrabbing():
        camera.StopGrabbing()

    set_bool_node(camera, "CenterX", False)
    set_bool_node(camera, "CenterY", False)
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
            f"Cannot set full image to {CAMERA_WIDTH}x{CAMERA_HEIGHT}; "
            f"current image is {width}x{height}."
        )

    print(f"Camera image: {width}x{height}, OffsetX={offset_x}, OffsetY={offset_y}")

def setup_camera():
    factory = pylon.TlFactory.GetInstance()
    devices = list(factory.EnumerateDevices())
    if not devices:
        raise RuntimeError("No Basler camera was found.")

    print("Detected cameras:")
    for index, device in enumerate(devices):
        print(
            f"  [{index}] {device.GetModelName()} "
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
                f"Warning: camera S/N {CAMERA_SERIAL_NUMBER} was not found; "
                "using the only detected camera."
            )
        else:
            raise RuntimeError(
                f"Camera S/N {CAMERA_SERIAL_NUMBER} was not found. "
                "Update CAMERA_SERIAL_NUMBER."
            )

    camera = pylon.InstantCamera(factory.CreateDevice(selected))
    camera.Open()

    model = str(camera.GetDeviceInfo().GetModelName())
    serial_number = str(camera.GetDeviceInfo().GetSerialNumber())
    print(f"Opened camera: {model}, S/N={serial_number}")
    if EXPECTED_CAMERA_MODEL.lower() not in model.lower():
        print(f"Warning: camera model is {model}; expected {EXPECTED_CAMERA_MODEL}.")

    configure_full_sensor(camera)

    for node_name in ("ExposureAuto", "GainAuto"):
        try:
            getattr(camera, node_name).SetValue("Off")
        except Exception:
            pass

    actual_exposure = set_exposure(camera, EXPOSURE_TIME_US)
    print(f"Exposure: {actual_exposure:.0f} us")
    return camera

def configure_camera_for_preview(camera) -> None:
    if camera.IsGrabbing():
        camera.StopGrabbing()
    try:
        camera.TriggerMode.SetValue("Off")
    except Exception:
        pass
    try:
        camera.AcquisitionMode.SetValue("Continuous")
    except Exception:
        pass

def configure_camera_for_capture(camera) -> None:
    if camera.IsGrabbing():
        camera.StopGrabbing()
    try:
        camera.TriggerSelector.SetValue("FrameStart")
    except Exception:
        pass
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Software")
    try:
        camera.AcquisitionMode.SetValue("Continuous")
    except Exception:
        pass

def live_preview(camera, skip_preview: bool) -> bool:
    if skip_preview:
        return True

    configure_camera_for_preview(camera)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    preview_width, preview_height = get_preview_window_size()
    cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, preview_width, preview_height)

    if SHOW_FIXED_ROI_PREVIEW and SHOW_FIXED_ROI_ZOOM_WINDOW:
        cv2.namedWindow(FIXED_ROI_WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    exposure_us = get_exposure(camera)

    print("\nPreview controls:")
    print(f"  build: {PREVIEW_BUILD_TAG}")
    print(f"  preview window: {preview_width} x {preview_height}")
    print(f"  saved image: {CAMERA_WIDTH} x {CAMERA_HEIGHT} (full image)")
    if SHOW_FIXED_ROI_PREVIEW:
        print(
            "  fixed ROI ACTIVE: "
            f"x={FIXED_ROI_X_PX}, y={FIXED_ROI_Y_PX}, "
            f"w={FIXED_ROI_WIDTH_PX}, h={FIXED_ROI_HEIGHT_PX}"
        )
        if SHOW_FIXED_ROI_ZOOM_WINDOW:
            print("  second live window: enlarged fixed ROI")
    print("  S: start capture")
    print("  Q or Esc: cancel")
    print("  U: increase exposure")
    print("  J: decrease exposure\n")

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
            if not grab.IsValid():
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return False
                continue

            try:
                if grab.GrabSucceeded():
                    raw_frame = converter.Convert(grab).GetArray()

                    # Keep the raw frame untouched for the ROI crop. Draw all
                    # guides only on a copy used for display.
                    display_frame = raw_frame.copy()
                    draw_preview_guides(display_frame)

                    if SHOW_FIXED_ROI_PREVIEW:
                        draw_fixed_roi_on_preview(display_frame)

                    cv2.putText(
                        display_frame,
                        f"{PREVIEW_BUILD_TAG} | S:start | Q/Esc:cancel",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display_frame,
                        f"Exposure: {exposure_us:.0f} us | U:+ J:-",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    if SHOW_FIXED_ROI_PREVIEW:
                        cv2.putText(
                            display_frame,
                            "YELLOW BOX = fixed ROI used for positioning",
                            (20, 103),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.imshow(PREVIEW_WINDOW_NAME, display_frame)

                    if SHOW_FIXED_ROI_PREVIEW and SHOW_FIXED_ROI_ZOOM_WINDOW:
                        roi_zoom = make_fixed_roi_zoom(raw_frame)
                        cv2.imshow(FIXED_ROI_WINDOW_NAME, roi_zoom)
            finally:
                grab.Release()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                return True
            if key in (ord("q"), 27):
                return False
            if key == ord("u"):
                exposure_us = set_exposure(camera, exposure_us * 1.2)
                print(f"Exposure increased to {exposure_us:.0f} us")
            elif key == ord("j"):
                exposure_us = set_exposure(camera, max(50.0, exposure_us / 1.2))
                print(f"Exposure decreased to {exposure_us:.0f} us")
    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def software_trigger_and_save(camera, filename: Path) -> float:
    if not camera.WaitForFrameTriggerReady(2000, pylon.TimeoutHandling_ThrowException):
        raise RuntimeError("Camera is not ready for software trigger.")

    trigger_time = time.perf_counter()
    camera.ExecuteSoftwareTrigger()
    grab = camera.RetrieveResult(
        CAMERA_TIMEOUT_MS,
        pylon.TimeoutHandling_ThrowException,
    )

    try:
        if not grab.GrabSucceeded():
            raise RuntimeError(
                f"Camera grab failed: {grab.ErrorCode}, {grab.ErrorDescription}"
            )
        image = pylon.PylonImage()
        image.AttachGrabResultBuffer(grab)
        try:
            image.Save(pylon.ImageFileFormat_Png, str(filename))
        finally:
            image.Release()
    finally:
        grab.Release()

    return trigger_time

def drain_can_receive_queue(bus: can.BusABC) -> int:
    """Discard queued CAN replies without allowing an unbounded feedback wait."""
    deadline = time.monotonic() + CAN_DRAIN_TIMEOUT_S
    drained = 0
    while drained < CAN_DRAIN_MAX_MESSAGES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            message = bus.recv(timeout=min(0.002, remaining))
        except Exception:
            break
        if message is None:
            break
        drained += 1
    return drained

def motor_snapshot_csv_values(snapshot: dict[str, float | str | bool]) -> list[str]:
    return [
        "1" if snapshot["feedback_read_ok"] else "0",
        str(snapshot["feedback_wall_time_iso"]),
        f"{float(snapshot['feedback_elapsed_s']):.6f}",
        f"{float(snapshot['feedback_position_rad']):.9f}",
        f"{float(snapshot['feedback_position_deg']):.9f}",
        f"{float(snapshot['feedback_position_deg_mod360']):.9f}",
        f"{float(snapshot['feedback_velocity_rad_s']):.9f}",
        f"{float(snapshot['feedback_velocity_deg_s']):.9f}",
        f"{float(snapshot['feedback_torque_nm']):.9f}",
    ]

def send_disable_best_effort(bus: can.BusABC, motor: Motor) -> None:
    drain_can_receive_queue(bus)
    repeat_count = max(RAW_COMMAND_REPEAT_COUNT, SAFETY_DISABLE_REPEAT_COUNT)
    for _ in range(repeat_count):
        try:
            _camera_send_disable_no_feedback(bus, motor)
        except Exception as exc:
            print(f"Warning: failed to send disable command: {exc}")
            break
        time.sleep(RAW_COMMAND_REPEAT_INTERVAL_S)
        drain_can_receive_queue(bus)



# =========================
# Original motor_feedback workflow
# =========================

def now_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def deg_to_rad(value_deg: float) -> float:
    return math.radians(value_deg)


def rad_to_deg(value_rad: float) -> float:
    return math.degrees(value_rad)


def parse_int(value: str) -> int:
    value = value.strip().lower()
    if value.startswith("0x"):
        return int(value, 16)
    return int(value)


def handle_ctrl_c(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    STOP_EVENT.set()
    raise KeyboardInterrupt


def next_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for item in root.iterdir():
        if item.is_dir() and item.name.lower().startswith("run"):
            suffix = item.name[3:]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    out = root / f"run{max_index + 1}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def open_csv(path: Path, header: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(handle)
    writer.writerow(header)
    return handle, writer


def write_event(writer, start_perf: float, event: str, detail: str = "") -> None:
    writer.writerow([now_iso(), f"{time.perf_counter() - start_perf:.6f}", event, detail])


def drain_bus(bus, max_messages: int = 512) -> int:
    count = 0
    for _ in range(max_messages):
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break
        count += 1
    return count


def wait_with_drain(bus, duration_s: float) -> None:
    end = time.perf_counter() + max(0.0, duration_s)
    next_drain = time.perf_counter() + CAN_DRAIN_INTERVAL_S
    while True:
        if STOP_REQUESTED:
            raise KeyboardInterrupt
        now = time.perf_counter()
        if now >= end:
            break
        if now >= next_drain:
            drain_bus(bus)
            next_drain = now + CAN_DRAIN_INTERVAL_S
        time.sleep(min(0.02, end - now))
    drain_bus(bus)


def check_position_limit(*angles_deg: float) -> None:
    limit_rad = 12.5 - POSITION_LIMIT_MARGIN_RAD
    for angle_deg in angles_deg:
        angle_rad = deg_to_rad(angle_deg)
        if abs(angle_rad) > limit_rad:
            raise RuntimeError(
                f"target {angle_deg:.3f} deg exceeds safe motor limit +/-{rad_to_deg(limit_rad):.1f} deg"
            )


def disable_best_effort(bus, motor) -> None:
    try:
        send_disable_best_effort(bus, motor)
    except Exception:
        try:
            for _ in range(5):
                send_disable_no_feedback(bus, motor)
                time.sleep(0.03)
        except Exception as exc:
            print(f"Warning: failed to disable motor: {exc}")


def switch_to_mit_no_feedback(bus, motor) -> None:
    can_id_l = motor.SlaveID & 0xFF
    can_id_h = (motor.SlaveID >> 8) & 0xFF
    mode_bytes = int(Control_Type.MIT).to_bytes(4, "little", signed=False)
    data = bytes([can_id_l, can_id_h, 0x55, 10] + list(mode_bytes))
    for _ in range(3):
        msg = can.Message(arbitration_id=0x7FF, data=data, is_extended_id=False)
        send_can_with_retry(bus, msg, "switch motor to MIT mode")
        time.sleep(0.02)


def set_shared_zero_and_enable(bus, motor, label: str, *, mode: str) -> None:
    print(f"Setting motor zero: {label}")
    disable_best_effort(bus, motor)
    time.sleep(0.2)
    drain_bus(bus)

    send_control_command_no_feedback(bus, motor, 0xFE)
    time.sleep(0.3)
    drain_bus(bus)

    if mode == "mit":
        switch_to_mit_no_feedback(bus, motor)
    elif mode == "posvel":
        switch_to_position_velocity_no_feedback(bus, motor)
    else:
        raise ValueError(f"unknown motor mode: {mode}")

    time.sleep(0.2)
    drain_bus(bus)

    send_control_command_no_feedback(bus, motor, 0xFC)
    time.sleep(0.5)
    drain_bus(bus)


def move_to_target(bus, motor, target_deg: float, speed_deg_s: float, *, settle_s: float = 0.0) -> float:
    check_position_limit(target_deg)
    target_rad = deg_to_rad(target_deg)
    speed_rad_s = abs(deg_to_rad(speed_deg_s))
    print(f"Motor target: {target_deg:.3f} deg at {speed_deg_s:.3f} deg/s")
    send_position_target(bus, motor, target_rad, speed_rad_s)
    duration_s = abs(target_deg) / speed_deg_s + settle_s
    wait_with_drain(bus, duration_s)
    return target_rad


def send_camera_position_target_once(bus, motor, target_rad: float, speed_rad_s: float) -> None:
    """Send one POS_VEL target frame for camera stop-and-shoot.

    The old repeated-send helper can overload some candleLight adapters during 361
    small steps. For camera capture, one accepted target frame is enough because
    the driver executes the point-to-point move internally.
    """
    msg = can.Message(
        arbitration_id=0x100 + motor.SlaveID,
        data=float_to_uint8s(float(target_rad)) + float_to_uint8s(float(speed_rad_s)),
        is_extended_id=False,
    )
    send_can_with_retry(bus, msg, "camera position-speed command")


def empty_motor_snapshot(start_perf: float) -> dict[str, float | str | bool]:
    return {
        "feedback_read_ok": False,
        "feedback_wall_time_iso": now_iso(),
        "feedback_elapsed_s": time.perf_counter() - start_perf,
        "feedback_position_rad": 0.0,
        "feedback_position_deg": 0.0,
        "feedback_position_deg_mod360": 0.0,
        "feedback_velocity_rad_s": 0.0,
        "feedback_velocity_deg_s": 0.0,
        "feedback_torque_nm": 0.0,
    }


def validate_camera_motion(rotation_deg: float, step_deg: float) -> int:
    if rotation_deg <= 0:
        raise ValueError("camera rotation must be > 0")
    if step_deg <= 0:
        raise ValueError("camera angle step must be > 0")
    count_float = rotation_deg / step_deg
    count = round(count_float)
    if not math.isclose(count_float, count, abs_tol=1e-9):
        raise ValueError("camera rotation must be divisible by angle step")
    return count + 1



# =========================
# MIT laser motion helpers
# =========================

P_MIN = -12.5
P_MAX = 12.5
V_MIN = -30.0
V_MAX = 30.0
T_MIN = -10.0
T_MAX = 10.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def float_to_uint(value: float, value_min: float, value_max: float, bits: int) -> int:
    value = clamp(value, value_min, value_max)
    span = value_max - value_min
    return int((value - value_min) * ((1 << bits) - 1) / span)


def uint_to_float(value: int, value_min: float, value_max: float, bits: int) -> float:
    span = value_max - value_min
    return float(value) * span / float((1 << bits) - 1) + value_min


def pack_mit_command(position_rad: float, velocity_rad_s: float, kp: float, kd: float, tau_nm: float) -> bytes:
    p_uint = float_to_uint(position_rad, P_MIN, P_MAX, 16)
    v_uint = float_to_uint(velocity_rad_s, V_MIN, V_MAX, 12)
    kp_uint = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_uint = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t_uint = float_to_uint(tau_nm, T_MIN, T_MAX, 12)
    return bytes([
        (p_uint >> 8) & 0xFF,
        p_uint & 0xFF,
        (v_uint >> 4) & 0xFF,
        ((v_uint & 0x0F) << 4) | ((kp_uint >> 8) & 0x0F),
        kp_uint & 0xFF,
        (kd_uint >> 4) & 0xFF,
        ((kd_uint & 0x0F) << 4) | ((t_uint >> 8) & 0x0F),
        t_uint & 0xFF,
    ])


def decode_mit_feedback(msg, motor_id: int):
    data = bytes(msg.data)
    if len(data) < 8:
        return None
    frame_motor_id = data[0] & 0x0F
    if frame_motor_id != (motor_id & 0x0F):
        return None
    pos_raw = (data[1] << 8) | data[2]
    vel_raw = (data[3] << 4) | (data[4] >> 4)
    tau_raw = ((data[4] & 0x0F) << 8) | data[5]
    pos_rad = uint_to_float(pos_raw, P_MIN, P_MAX, 16)
    vel_rad_s = uint_to_float(vel_raw, V_MIN, V_MAX, 12)
    tau_nm = uint_to_float(tau_raw, T_MIN, T_MAX, 12)
    return pos_rad, vel_rad_s, tau_nm


def read_latest_mit_feedback(bus, motor_id: int, timeout_s: float = 0.0):
    deadline = time.monotonic() + max(0.0, timeout_s)
    latest = None
    while True:
        remaining = deadline - time.monotonic()
        timeout = 0.0 if timeout_s <= 0 else max(0.0, remaining)
        msg = bus.recv(timeout=timeout)
        if msg is None:
            break
        decoded = decode_mit_feedback(msg, motor_id)
        if decoded is not None:
            latest = decoded
        if timeout_s <= 0 or remaining <= 0:
            break
    return latest


def send_mit_can(bus, motor, data: bytes, description: str) -> None:
    msg = can.Message(arbitration_id=motor.SlaveID, data=data, is_extended_id=False)
    send_can_with_retry(bus, msg, description)


def hold_mit_position_best_effort(bus, motor, position_rad: float, kp: float, kd: float, command_hz: float, hold_s: float) -> None:
    data = pack_mit_command(position_rad, 0.0, kp, kd, 0.0)
    period_s = 1.0 / max(1.0, command_hz)
    deadline = time.perf_counter() + max(0.0, hold_s)
    while time.perf_counter() < deadline:
        try:
            send_mit_can(bus, motor, data, "MIT final hold command")
        except Exception:
            pass
        time.sleep(period_s)


def run_mit_laser_motion(args, bus, motor, start_perf: float, motor_writer) -> tuple[float, int, float]:
    total_rad = deg_to_rad(args.laser_rotation) * args.direction
    duration_s = args.laser_rotation / args.laser_speed_deg_s
    velocity_rad_s = total_rad / duration_s
    period_s = 1.0 / args.mit_hz
    tau_cmd = abs(args.mit_tau) * args.direction
    stop_early_rad = deg_to_rad(max(0.0, args.mit_stop_early_deg))

    print(
        "MIT laser motion: "
        f"{args.laser_rotation:.1f} deg at {args.laser_speed_deg_s:.3f} deg/s, "
        f"KP={args.mit_kp}, KD={args.mit_kd}, TAU={tau_cmd}, Hz={args.mit_hz}."
    )

    start_feedback = None
    probe = pack_mit_command(0.0, 0.0, args.mit_kp, args.mit_kd, 0.0)
    for _ in range(10):
        send_mit_can(bus, motor, probe, "MIT feedback probe")
        decoded = read_latest_mit_feedback(bus, motor.SlaveID, timeout_s=0.02)
        if decoded is not None:
            start_feedback = decoded[0]
            break
        time.sleep(0.02)

    if start_feedback is None:
        print("Warning: no MIT feedback. Laser motor phase will stop by time only.")
        command_origin = 0.0
    else:
        command_origin = start_feedback

    motor_start_time_s = time.perf_counter() - start_perf
    loop_start = time.perf_counter()
    next_tick = loop_start
    sent = 0
    reached_target = False
    last_delta_rad = 0.0
    max_duration_s = duration_s + 3.0

    while True:
        if STOP_REQUESTED:
            raise KeyboardInterrupt
        elapsed = time.perf_counter() - loop_start
        if elapsed >= max_duration_s:
            break

        progress_ratio = min(1.0, elapsed / duration_s)
        theory_deg = args.direction * min(args.laser_rotation, elapsed * args.laser_speed_deg_s)
        target_position = command_origin + total_rad * progress_ratio
        move_data = pack_mit_command(target_position, velocity_rad_s, args.mit_kp, args.mit_kd, tau_cmd)
        send_mit_can(bus, motor, move_data, "MIT laser trajectory command")
        sent += 1

        decoded = read_latest_mit_feedback(bus, motor.SlaveID, timeout_s=0.001)
        if decoded is not None:
            pos_rad, vel_rad_s, torque_nm = decoded
            rel_pos_rad = pos_rad - command_origin
            last_delta_rad = rel_pos_rad * args.direction
            t_now = time.perf_counter() - start_perf
            motor_writer.writerow([
                now_iso(),
                f"{t_now:.6f}",
                f"{theory_deg:.9f}",
                f"{rel_pos_rad:.9f}",
                f"{rad_to_deg(rel_pos_rad):.9f}",
                f"{vel_rad_s:.9f}",
                f"{rad_to_deg(vel_rad_s):.9f}",
                f"{torque_nm:.9f}",
                f"{target_position - command_origin:.9f}",
                f"{rad_to_deg(target_position - command_origin):.9f}",
            ])
            if last_delta_rad >= abs(total_rad) - stop_early_rad:
                reached_target = True
                break
        elif start_feedback is None and elapsed >= duration_s:
            break

        next_tick += period_s
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()

    final_position = command_origin + total_rad
    hold_mit_position_best_effort(bus, motor, final_position, args.mit_kp, args.mit_kd, args.mit_hz, args.mit_hold_after_motion)
    if reached_target:
        print(f"MIT laser motion stopped by feedback at {math.degrees(last_delta_rad):.2f} deg. Sent {sent} frames.")
    else:
        print(f"MIT laser motion stopped by time limit. Sent {sent} frames.")
    return motor_start_time_s, sent, math.degrees(last_delta_rad)


def run_laser_phase(args, run_dir: Path, bus, ctrl, motor, start_perf: float, event_writer) -> Path:
    laser_dir = run_dir / "laser"
    laser_dir.mkdir(parents=True, exist_ok=True)

    expected_duration_s = args.laser_rotation / args.laser_speed_deg_s
    expected_profiles = int(round(expected_duration_s * args.laser_sample_hz))
    target_profiles = expected_profiles if args.laser_target_profiles == 0 else args.laser_target_profiles

    laser = LaserHighSpeedRecorder(
        record_dir=laser_dir,
        device_id=args.laser_device_id,
        ip=args.laser_ip,
        port=args.laser_port,
        high_speed_port=args.laser_high_speed_port,
        chunk_profiles=args.laser_chunk_profiles,
        sample_hz=args.laser_sample_hz,
        target_profiles=target_profiles,
    )

    motor_handle, motor_writer = open_csv(
        laser_dir / "motor_feedback.csv",
        [
            "wall_time_iso",
            "time_s",
            "theory_position_deg",
            "position_rad",
            "position_deg",
            "velocity_rad_s",
            "velocity_deg_s",
            "torque_nm",
            "target_position_rad",
            "target_position_deg",
        ],
    )

    laser_started = False
    motor_start_time_s = 0.0
    mit_sent_frames = 0
    mit_final_angle_deg = 0.0

    try:
        print("Opening and preparing laser...")
        laser.open()
        write_event(event_writer, start_perf, "laser_ethernet_open", args.laser_ip)
        laser.prepare(start_perf)
        write_event(event_writer, start_perf, "laser_prepared", f"x_points={laser.x_points}")

        print("Starting laser acquisition in background...")
        laser.start()
        laser_started = True
        write_event(event_writer, start_perf, "laser_measure_started")

        motor_start_time_s, mit_sent_frames, mit_final_angle_deg = run_mit_laser_motion(
            args, bus, motor, start_perf, motor_writer
        )
        write_event(
            event_writer,
            start_perf,
            "laser_mit_motion_finished",
            f"motor_start_time_s={motor_start_time_s:.6f}, sent_frames={mit_sent_frames}, final_angle_deg={mit_final_angle_deg:.6f}",
        )

        disable_best_effort(bus, motor)
        write_event(event_writer, start_perf, "motor_disabled_after_laser")
        print("Laser phase motion finished; motor disabled.")

    finally:
        try:
            if laser_started:
                laser.stop()
                write_event(event_writer, start_perf, "laser_stopped")
        except Exception as exc:
            write_event(event_writer, start_perf, "laser_stop_error", str(exc))
        motor_handle.flush()
        motor_handle.close()

    write_sync_index(
        laser_dir,
        args.laser_sample_hz,
        motor_start_time_s=motor_start_time_s,
        speed_deg_s=args.laser_speed_deg_s,
        rotation_deg=args.laser_rotation,
        direction=args.direction,
    )
    summary_path = write_laser_summary(laser_dir, laser, expected_profiles)
    height_csv_path = None
    if args.export_height_csv:
        if laser.z_unit_um is None:
            print("Warning: laser z_unit_um unavailable; height CSV not exported.")
        else:
            print("Exporting laser height CSV...")
            height_csv_path = export_height_mm_csv(laser_dir, z_unit_um=float(laser.z_unit_um))

    (laser_dir / "laser_phase_metadata.json").write_text(
        json.dumps(
            {
                "expected_profiles": expected_profiles,
                "target_profiles": target_profiles,
                "actual_profiles": laser.line_count,
                "laser_speed_deg_s": args.laser_speed_deg_s,
                "laser_rotation_deg": args.laser_rotation,
                "direction": args.direction,
                "motor_control_mode": "MIT",
                "mit_kp": args.mit_kp,
                "mit_kd": args.mit_kd,
                "mit_tau": args.mit_tau,
                "mit_hz": args.mit_hz,
                "mit_sent_frames": mit_sent_frames,
                "mit_final_angle_deg": mit_final_angle_deg,
                "summary_path": str(summary_path),
                "height_csv_path": None if height_csv_path is None else str(height_csv_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return laser_dir


def run_camera_phase(args, run_dir: Path, bus, ctrl, motor, camera, start_perf: float, event_writer) -> Path:
    """Camera stop-and-shoot with real feedback and NO background RX thread.

    Only this camera function is changed.

    Safety policy:
    - No thread is created.
    - No active feedback-query command is sent.
    - Every camera target is sent exactly once.
    - bus.recv(timeout=0.0) is used only inside this camera phase.
    - Every receive pass is bounded by max_messages, so it cannot block.
    - Missing feedback never stops motor movement or camera capture.
    """
    camera_dir = run_dir / "camera"
    images_dir = camera_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    capture_csv_path = camera_dir / "capture_info.csv"
    feedback_csv_path = camera_dir / "camera_motor_feedback.csv"

    total_images = validate_camera_motion(args.camera_rotation, args.camera_angle_step)
    camera_offset_target_deg = args.direction * args.camera_offset_deg
    final_target_deg = args.direction * (args.camera_offset_deg + args.camera_rotation)
    check_position_limit(camera_offset_target_deg, final_target_deg)

    configure_camera_for_capture(camera)
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    capture_handle, capture_writer = open_csv(
        capture_csv_path,
        [
            "image_index",
            "camera_sweep_angle_deg",
            "shared_zero_target_angle_deg",
            "camera_offset_deg",
            "move_start_elapsed_s",
            "capture_elapsed_s",
            "settle_time_s",
            "trigger_wall_time_iso",
            "filename",
            "feedback_read_ok",
            "feedback_wall_time_iso",
            "feedback_elapsed_s",
            "feedback_position_rad",
            "feedback_position_deg",
            "feedback_position_deg_mod360",
            "feedback_velocity_rad_s",
            "feedback_velocity_deg_s",
            "feedback_torque_nm",
            "feedback_target_error_deg",
            "feedback_time_from_trigger_ms",
            "feedback_frames_received_for_image",
            "feedback_frames_decoded_for_image",
        ],
    )

    feedback_handle, feedback_writer = open_csv(
        feedback_csv_path,
        [
            "wall_time_iso",
            "time_s",
            "image_index",
            "phase",
            "target_position_deg",
            "arbitration_id",
            "raw_data_hex",
            "decoded_ok",
            "position_rad",
            "position_deg",
            "position_deg_mod360",
            "velocity_rad_s",
            "velocity_deg_s",
            "torque_nm",
            "target_error_deg",
        ],
    )

    def empty_snapshot() -> dict[str, float | str | bool]:
        return empty_motor_snapshot(start_perf)

    def collect_feedback_nonblocking(
        *,
        image_index: int,
        phase: str,
        target_deg: float,
        max_messages: int,
    ):
        """Read already-buffered CAN frames without waiting.

        Returns:
            latest decoded motor snapshot or None,
            number of all CAN frames read,
            number of successfully decoded motor frames,
            buffered CSV rows.
        """
        latest_snapshot = None
        frames_received = 0
        frames_decoded = 0
        rows = []

        for _ in range(max(1, int(max_messages))):
            msg = bus.recv(timeout=0.0)
            if msg is None:
                break

            frames_received += 1
            sample_perf = time.perf_counter()
            wall_time = now_iso()
            arbitration_id = int(getattr(msg, "arbitration_id", 0))
            raw_data_hex = bytes(msg.data).hex(" ")

            decoded = decode_mit_feedback(msg, motor.SlaveID)

            if decoded is None:
                rows.append(
                    [
                        wall_time,
                        f"{sample_perf - start_perf:.6f}",
                        image_index,
                        phase,
                        f"{target_deg:.9f}",
                        arbitration_id,
                        raw_data_hex,
                        False,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue

            frames_decoded += 1
            position_rad, velocity_rad_s, torque_nm = decoded
            position_deg = rad_to_deg(position_rad)
            velocity_deg_s = rad_to_deg(velocity_rad_s)
            target_error_deg = position_deg - target_deg

            latest_snapshot = {
                "feedback_read_ok": True,
                "feedback_wall_time_iso": wall_time,
                "feedback_elapsed_s": sample_perf - start_perf,
                "feedback_position_rad": float(position_rad),
                "feedback_position_deg": float(position_deg),
                "feedback_position_deg_mod360": float(position_deg % 360.0),
                "feedback_velocity_rad_s": float(velocity_rad_s),
                "feedback_velocity_deg_s": float(velocity_deg_s),
                "feedback_torque_nm": float(torque_nm),
                "_sample_perf": float(sample_perf),
                "_target_error_deg": float(target_error_deg),
            }

            rows.append(
                [
                    wall_time,
                    f"{sample_perf - start_perf:.6f}",
                    image_index,
                    phase,
                    f"{target_deg:.9f}",
                    arbitration_id,
                    raw_data_hex,
                    True,
                    f"{position_rad:.9f}",
                    f"{position_deg:.9f}",
                    f"{position_deg % 360.0:.9f}",
                    f"{velocity_rad_s:.9f}",
                    f"{velocity_deg_s:.9f}",
                    f"{torque_nm:.9f}",
                    f"{target_error_deg:.9f}",
                ]
            )

        return latest_snapshot, frames_received, frames_decoded, rows

    def wait_and_collect_nonblocking(
        *,
        duration_s: float,
        image_index: int,
        phase: str,
        target_deg: float,
    ):
        """Wait for motor travel while draining RX only with timeout=0.0."""
        deadline = time.perf_counter() + max(0.0, float(duration_s))
        latest_snapshot = None
        total_received = 0
        total_decoded = 0
        all_rows = []

        while True:
            if STOP_REQUESTED:
                raise KeyboardInterrupt

            snapshot, received, decoded_count, rows = collect_feedback_nonblocking(
                image_index=image_index,
                phase=phase,
                target_deg=target_deg,
                max_messages=32,
            )
            total_received += received
            total_decoded += decoded_count
            all_rows.extend(rows)

            if snapshot is not None:
                latest_snapshot = snapshot

            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                break

            time.sleep(min(0.02, remaining))

        # One final bounded drain after the movement/settle time.
        snapshot, received, decoded_count, rows = collect_feedback_nonblocking(
            image_index=image_index,
            phase=f"{phase}_final",
            target_deg=target_deg,
            max_messages=256,
        )
        total_received += received
        total_decoded += decoded_count
        all_rows.extend(rows)

        if snapshot is not None:
            latest_snapshot = snapshot

        return latest_snapshot, total_received, total_decoded, all_rows

    def write_feedback_rows(rows) -> None:
        for row in rows:
            feedback_writer.writerow(row)

    def write_capture(
        *,
        image_index: int,
        local_angle_deg: float,
        target_deg: float,
        move_start_elapsed_s: float,
        trigger_perf: float,
        trigger_wall_time: str,
        filename: Path,
        snapshot,
        frames_received: int,
        frames_decoded: int,
    ) -> None:
        if snapshot is None:
            snapshot = empty_snapshot()
            target_error_text = ""
            trigger_error_text = ""
        else:
            target_error_text = f"{float(snapshot['_target_error_deg']):.9f}"
            trigger_error_text = (
                f"{(float(snapshot['_sample_perf']) - trigger_perf) * 1000.0:.3f}"
            )

        capture_writer.writerow(
            [
                image_index,
                f"{args.direction * local_angle_deg:.9f}",
                f"{target_deg:.9f}",
                f"{args.camera_offset_deg:.9f}",
                f"{move_start_elapsed_s:.6f}",
                f"{trigger_perf - start_perf:.6f}",
                f"{args.camera_settle_time:.3f}",
                trigger_wall_time,
                filename.name,
            ]
            + motor_snapshot_csv_values(snapshot)
            + [
                target_error_text,
                trigger_error_text,
                frames_received,
                frames_decoded,
            ]
        )

        capture_handle.flush()
        feedback_handle.flush()

        if bool(snapshot.get("feedback_read_ok", False)):
            feedback_text = f"{float(snapshot['feedback_position_deg']):.4f} deg"
        else:
            feedback_text = "missing"

        print(
            f"[{image_index + 1}/{total_images}] saved {filename.name}; "
            f"real_feedback={feedback_text}, "
            f"rx_frames={frames_received}, decoded={frames_decoded}"
        )

    try:
        # Remove only stale frames left before the camera phase begins.
        # This happens after the original laser phase has already completed.
        stale_snapshot, stale_received, stale_decoded, stale_rows = (
            collect_feedback_nonblocking(
                image_index=-1,
                phase="camera_prestart_stale",
                target_deg=0.0,
                max_messages=512,
            )
        )
        write_feedback_rows(stale_rows)
        print(
            "Camera nonblocking feedback mode started; "
            f"cleared/logged {stale_received} stale frames."
        )

        step_speed_rad_s = abs(deg_to_rad(args.camera_speed_deg_s))

        # Initial camera-facing movement: one command only.
        print("Moving cone from laser position to camera-facing position...")
        move_start = time.perf_counter() - start_perf

        send_camera_position_target_once(
            bus,
            motor,
            deg_to_rad(camera_offset_target_deg),
            step_speed_rad_s,
        )

        offset_snapshot, offset_received, offset_decoded, offset_rows = (
            wait_and_collect_nonblocking(
                duration_s=(
                    abs(camera_offset_target_deg) / args.camera_speed_deg_s
                    + args.camera_settle_time
                ),
                image_index=0,
                phase="camera_offset_move",
                target_deg=camera_offset_target_deg,
            )
        )
        write_feedback_rows(offset_rows)

        print("Capturing camera initial image at camera-facing cone position...")
        filename = images_dir / f"img_0000_angle_{camera_offset_target_deg:08.2f}.png"
        trigger_wall_time = now_iso()
        trigger_time = software_trigger_and_save(camera, filename)

        # Collect only frames already available immediately after the trigger.
        post_snapshot, post_received, post_decoded, post_rows = (
            collect_feedback_nonblocking(
                image_index=0,
                phase="camera_initial_post_trigger",
                target_deg=camera_offset_target_deg,
                max_messages=64,
            )
        )
        write_feedback_rows(post_rows)

        if post_snapshot is not None:
            offset_snapshot = post_snapshot

        write_capture(
            image_index=0,
            local_angle_deg=0.0,
            target_deg=camera_offset_target_deg,
            move_start_elapsed_s=move_start,
            trigger_perf=trigger_time,
            trigger_wall_time=trigger_wall_time,
            filename=filename,
            snapshot=offset_snapshot,
            frames_received=offset_received + post_received,
            frames_decoded=offset_decoded + post_decoded,
        )

        write_event(
            event_writer,
            start_perf,
            "camera_offset_reached",
            (
                f"target_deg={camera_offset_target_deg:.9f}, "
                f"feedback_ok={bool(offset_snapshot and offset_snapshot.get('feedback_read_ok', False))}"
            ),
        )

        # Remaining camera steps.
        for index in range(1, total_images):
            if STOP_REQUESTED:
                raise KeyboardInterrupt

            local_angle = index * args.camera_angle_step
            shared_target_deg = args.direction * (
                args.camera_offset_deg + local_angle
            )
            target_rad = deg_to_rad(shared_target_deg)
            move_start = time.perf_counter() - start_perf

            print(
                f"[{index + 1}/{total_images}] moving to shared target "
                f"{shared_target_deg:.2f} deg..."
            )

            # Exactly one target command.
            send_camera_position_target_once(
                bus,
                motor,
                target_rad,
                step_speed_rad_s,
            )

            snapshot, received, decoded_count, rows = (
                wait_and_collect_nonblocking(
                    duration_s=(
                        args.camera_angle_step / args.camera_speed_deg_s
                        + args.camera_settle_time
                    ),
                    image_index=index,
                    phase="camera_step_move",
                    target_deg=shared_target_deg,
                )
            )
            write_feedback_rows(rows)

            filename = images_dir / (
                f"img_{index:04d}_angle_{shared_target_deg:08.2f}.png"
            )
            trigger_wall_time = now_iso()
            trigger_time = software_trigger_and_save(
                camera,
                filename,
            )

            post_snapshot, post_received, post_decoded, post_rows = (
                collect_feedback_nonblocking(
                    image_index=index,
                    phase="camera_post_trigger",
                    target_deg=shared_target_deg,
                    max_messages=64,
                )
            )
            write_feedback_rows(post_rows)

            if post_snapshot is not None:
                snapshot = post_snapshot

            write_capture(
                image_index=index,
                local_angle_deg=local_angle,
                target_deg=shared_target_deg,
                move_start_elapsed_s=move_start,
                trigger_perf=trigger_time,
                trigger_wall_time=trigger_wall_time,
                filename=filename,
                snapshot=snapshot,
                frames_received=received + post_received,
                frames_decoded=decoded_count + post_decoded,
            )

        write_event(
            event_writer,
            start_perf,
            "camera_capture_finished",
            f"images={total_images}, feedback_mode=bounded_nonblocking_no_thread",
        )
        print(f"Camera phase finished. Images saved in: {images_dir}")
        print(f"Camera real feedback saved in: {feedback_csv_path}")

    finally:
        # No receiver thread exists, so there is nothing that can remain alive
        # and steal laser feedback on a later run.
        capture_handle.flush()
        capture_handle.close()
        feedback_handle.flush()
        feedback_handle.close()

        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
        except Exception:
            pass

    return camera_dir

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared-zero laser first, then camera capture workflow.")
    parser.add_argument("--scan", action="store_true", help="Scan gs_usb adapters and exit.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--channel", type=int, default=CAN_CHANNEL)
    parser.add_argument("--bitrate", type=int, default=CAN_BITRATE)
    parser.add_argument("--can-id", type=parse_int, default=MOTOR_CAN_ID)
    parser.add_argument("--master-id", type=parse_int, default=MASTER_ID)
    parser.add_argument("--direction", type=int, choices=(-1, 1), default=DIRECTION)

    parser.add_argument("--laser-speed-deg-s", type=float, default=LASER_SPEED_DEG_PER_SEC)
    parser.add_argument("--laser-rotation", type=float, default=LASER_ROTATION_DEG)
    parser.add_argument("--laser-sample-hz", type=float, default=LASER_SAMPLE_HZ)
    parser.add_argument("--laser-target-profiles", type=int, default=LASER_TARGET_PROFILES)
    parser.add_argument("--laser-device-id", type=int, default=LASER_DEVICE_ID)
    parser.add_argument("--laser-ip", default=LASER_IP)
    parser.add_argument("--laser-port", type=int, default=LASER_PORT)
    parser.add_argument("--laser-high-speed-port", type=int, default=LASER_HIGH_SPEED_PORT)
    parser.add_argument("--laser-chunk-profiles", type=int, default=LASER_CHUNK_PROFILES)
    parser.add_argument("--no-height-csv", action="store_false", dest="export_height_csv", default=True)
    parser.add_argument("--mit-kp", type=float, default=MIT_KP)
    parser.add_argument("--mit-kd", type=float, default=MIT_KD)
    parser.add_argument("--mit-tau", type=float, default=MIT_TAU)
    parser.add_argument("--mit-hz", type=float, default=MIT_COMMAND_HZ)
    parser.add_argument("--mit-stop-early-deg", type=float, default=MIT_STOP_EARLY_DEG)
    parser.add_argument("--mit-hold-after-motion", type=float, default=MIT_HOLD_AFTER_MOTION_S)

    parser.add_argument("--camera-offset-deg", type=float, default=CAMERA_OFFSET_DEG)
    parser.add_argument("--camera-speed-deg-s", type=float, default=CAMERA_SPEED_DEG_PER_SEC)
    parser.add_argument("--camera-rotation", type=float, default=CAMERA_ROTATION_DEG)
    parser.add_argument("--camera-angle-step", type=float, default=CAMERA_ANGLE_STEP_DEG)
    parser.add_argument("--camera-settle-time", type=float, default=CAMERA_SETTLE_TIME_S)
    parser.add_argument("--skip-preview", action="store_true")
    return parser


def main() -> int:
    global STOP_REQUESTED
    STOP_REQUESTED = False
    signal.signal(signal.SIGINT, handle_ctrl_c)
    args = build_arg_parser().parse_args()

    if args.scan:
        return scan_adapters()

    if args.laser_speed_deg_s <= 0 or args.camera_speed_deg_s <= 0:
        raise ValueError("speed must be > 0")
    if args.laser_rotation != 360.0:
        print("Warning: this workflow assumes laser_rotation is exactly 360 deg before re-zeroing.")
    if args.camera_offset_deg < 0:
        print("Warning: --camera-offset-deg is treated as magnitude; use --direction to change motor sign.")
        args.camera_offset_deg = abs(args.camera_offset_deg)

    run_dir = next_run_dir(args.output_root)
    event_handle, event_writer = open_csv(run_dir / "events.csv", ["wall_time_iso", "time_s", "event", "detail"])
    start_perf = time.perf_counter()
    write_event(event_writer, start_perf, "run_created", str(run_dir))

    camera = None
    bus = None
    motor_enabled = False
    try:
        print("\nShared-zero laser + camera experiment")
        print(f"Output folder: {run_dir}")
        print(f"Direction: {args.direction}")
        print(f"Laser: {args.laser_rotation:.1f} deg at {args.laser_speed_deg_s:.3f} deg/s, {args.laser_sample_hz:.1f} Hz")
        print(f"Camera offset from laser: {args.camera_offset_deg:.3f} deg")
        print(f"Camera: {args.camera_rotation:.1f} deg, step {args.camera_angle_step:.3f} deg, settle {args.camera_settle_time:.3f} s")

        print("Opening CAN bus...")
        bus, ctrl, motor = open_motor_bus(args.channel, args.bitrate, args.can_id, args.master_id)
        write_event(event_writer, start_perf, "motor_bus_open", f"channel={args.channel}, bitrate={args.bitrate}")

        set_shared_zero_and_enable(bus, motor, "initial shared zero at laser position", mode="mit")
        motor_enabled = True
        write_event(event_writer, start_perf, "motor_zero_enabled", "initial shared zero")

        laser_dir = run_laser_phase(args, run_dir, bus, ctrl, motor, start_perf, event_writer)
        motor_enabled = False

        # After a complete 360 deg laser revolution, the wheel is physically back at the shared zero.
        # Re-zero here prevents the next camera sweep from exceeding the motor driver's +/-12.5 rad limit.
        set_shared_zero_and_enable(bus, motor, "after laser full revolution; same physical zero", mode="posvel")
        motor_enabled = True
        write_event(event_writer, start_perf, "motor_rezero_enabled", "after laser 360 deg")

        print("Opening camera preview before stop-and-shoot capture...")
        camera = setup_camera()
        if not live_preview(camera, args.skip_preview):
            print("Cancelled in camera preview. Laser phase was completed; camera capture was not started.")
            disable_best_effort(bus, motor)
            motor_enabled = False
            return 0

        camera_dir = run_camera_phase(args, run_dir, bus, ctrl, motor, camera, start_perf, event_writer)

        disable_best_effort(bus, motor)
        motor_enabled = False
        write_event(event_writer, start_perf, "motor_disabled_after_camera")

        metadata = {
            "created_at": now_iso(),
            "run_dir": str(run_dir),
            "laser_dir": str(laser_dir),
            "camera_dir": str(camera_dir),
            "shared_zero_policy": "Laser phase uses MIT. After exact 360 deg laser scan, re-zero at same physical zero, then camera phase uses position-speed stop-and-shoot.",
            "motor_control_mode": "laser MIT, camera POS_VEL",
            "camera_offset_deg_magnitude": args.camera_offset_deg,
            "camera_offset_motor_target_deg": args.direction * args.camera_offset_deg,
            "recommended_ronghe_2d_angle_offset_deg": args.direction * args.camera_offset_deg,
            "direction": args.direction,
            "laser": {
                "rotation_deg": args.laser_rotation,
                "speed_deg_s": args.laser_speed_deg_s,
                "sample_hz": args.laser_sample_hz,
                "ip": args.laser_ip,
            },
            "camera": {
                "rotation_deg": args.camera_rotation,
                "angle_step_deg": args.camera_angle_step,
                "speed_deg_s": args.camera_speed_deg_s,
                "settle_time_s": args.camera_settle_time,
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\nExperiment complete.")
        print(f"Run folder: {run_dir}")
        print(f"Laser data: {laser_dir}")
        print(f"Camera data: {camera_dir}")
        print(f"Use this initial fusion offset: {args.direction * args.camera_offset_deg:.6f} deg")
        return 0

    finally:
        if motor_enabled and bus is not None:
            print("Disabling motor after exit...")
            disable_best_effort(bus, motor)
        if bus is not None:
            try:
                bus.shutdown()
                write_event(event_writer, start_perf, "motor_bus_closed")
            except Exception:
                pass
        if camera is not None:
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
            except Exception:
                pass
            try:
                if camera.IsOpen():
                    camera.Close()
            except Exception:
                pass
        event_handle.flush()
        event_handle.close()
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nEmergency stop requested. Verify motor is stationary, laser is stopped, and camera is closed.")
        raise SystemExit(130)
