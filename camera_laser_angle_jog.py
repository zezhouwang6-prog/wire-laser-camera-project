r"""Manual camera-laser angle calibration jog tool.

Use case
--------
1. Put the cone at the laser reference position.
2. Start this program. It sets the current motor position as 0 deg.
3. The Basler live preview opens with a center cross.
4. Jog/auto-rotate the motor until the cone center is aligned with the preview center.
5. Press S to save the current encoder angle. That angle is the camera-laser offset.

Default controls
----------------
R       start/stop automatic jog at 1 deg/s
Space   stop automatic jog and hold current position
D/A     jog +1 / -1 deg
C/Z     jog +0.1 / -0.1 deg
V/X     jog +0.05 / -0.05 deg
S       save current encoder angle and exit
Q/Esc   quit without saving

Output
------
D:\project\ronghe\angle_calibration\angle_mark1
D:\project\ronghe\angle_calibration\angle_mark2
...
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from config import SCRIPTS_DIR, ANGLE_CALIBRATION_DIR


# =========================
# USER SETTINGS
# =========================

MOTOR_CAMERA_SCRIPT = SCRIPTS_DIR / "motor_camera.py"
OUTPUT_ROOT = ANGLE_CALIBRATION_DIR

CAN_CHANNEL = 0
CAN_BITRATE = 1_000_000
MOTOR_CAN_ID = 0x01
MASTER_ID = 0x00

AUTO_SPEED_DEG_PER_SEC = 1.0
AUTO_DIRECTION = -1  # -1 means the default auto motion is negative angle.

# First coarse move after zeroing. This sends one finite target, waits for it,
# then you fine tune in the camera preview. Set to 0 or use --no-initial-move
# if you want to start manual adjustment from 0 deg.
INITIAL_MOVE_DEG = 45.0
INITIAL_MOVE_SPEED_DEG_PER_SEC = 1.0
INITIAL_MOVE_SETTLE_S = 0.8

COMMAND_INTERVAL_S = 0.10
FEEDBACK_INTERVAL_S = 0.10
SAMPLE_LOG_INTERVAL_S = 0.20

MANUAL_BIG_STEP_DEG = 1.0
MANUAL_FINE_STEP_DEG = 0.1
MANUAL_MICRO_STEP_DEG = 0.05
MANUAL_SPEED_DEG_PER_SEC = 1.0
MANUAL_SETTLE_S = 0.15

POSITION_LIMIT_DEG = 650.0
PREVIEW_WINDOW_NAME = "Camera laser angle calibration"

STOP_REQUESTED = False


# =========================
# Import existing camera/motor helper code
# =========================

def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mc = load_module("motor_camera_lib_for_angle_jog", MOTOR_CAMERA_SCRIPT)


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
    raise KeyboardInterrupt


def next_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("angle_mark"):
            suffix = item.name[len("angle_mark") :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    out = root / f"angle_mark{max_index + 1}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def open_csv(path: Path, header: list[str]):
    handle = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(handle)
    writer.writerow(header)
    return handle, writer


def open_motor(channel: int, bitrate: int, can_id: int, master_id: int):
    bus = mc.can.Bus(interface="gs_usb", channel=channel, bitrate=bitrate)
    motor = mc.Motor(mc.DM_Motor_Type.DM4310, can_id, master_id)
    ctrl = mc.GsUsbDmControl(bus)
    ctrl.add_motor(motor)
    return bus, ctrl, motor


def drain_bus(bus, max_messages: int = 512) -> int:
    count = 0
    for _ in range(max_messages):
        msg = bus.recv(timeout=0.0)
        if msg is None:
            break
        count += 1
    return count


def set_current_position_zero(bus, motor) -> None:
    print("Sending startup disable...")
    mc.send_disable_best_effort(bus, motor)
    time.sleep(0.2)
    drain_bus(bus)

    print("Setting current motor position as 0 deg...")
    mc.send_control_command_no_feedback(bus, motor, 0xFE)
    time.sleep(0.3)
    drain_bus(bus)

    print("Switching motor to position-speed mode...")
    mc.switch_to_position_velocity_no_feedback(bus, motor)
    time.sleep(0.2)
    drain_bus(bus)

    print("Enabling motor...")
    mc.send_control_command_no_feedback(bus, motor, 0xFC)
    time.sleep(0.4)
    drain_bus(bus)


def safe_target_deg(value_deg: float) -> float:
    if abs(value_deg) > POSITION_LIMIT_DEG:
        raise RuntimeError(f"Refusing target {value_deg:.3f} deg outside +/-{POSITION_LIMIT_DEG:.1f} deg")
    return value_deg


def send_target(bus, motor, target_deg: float, speed_deg_s: float) -> None:
    target_deg = safe_target_deg(target_deg)
    mc.send_position_speed_no_feedback(
        bus,
        motor,
        deg_to_rad(target_deg),
        abs(deg_to_rad(speed_deg_s)),
    )


def wait_motion_with_drain(bus, duration_s: float) -> None:
    end = time.perf_counter() + max(0.0, duration_s)
    next_drain = time.perf_counter() + 0.05
    while True:
        if STOP_REQUESTED:
            raise KeyboardInterrupt
        now = time.perf_counter()
        if now >= end:
            break
        if now >= next_drain:
            drain_bus(bus)
            next_drain = now + 0.05
        time.sleep(min(0.02, end - now))
    drain_bus(bus)


def coarse_initial_move(bus, ctrl, motor, experiment_start: float, args) -> tuple[float, dict[str, float | str | bool]]:
    if args.no_initial_move or args.initial_move_deg <= 0:
        snapshot = mc.read_motor_snapshot(ctrl, motor, experiment_start)
        return float(snapshot["feedback_position_deg"]), snapshot

    target_deg = args.auto_direction * abs(args.initial_move_deg)
    print(
        f"Initial coarse move: target {target_deg:.3f} deg "
        f"at {args.initial_move_speed_deg_s:.3f} deg/s"
    )
    send_target(bus, motor, target_deg, args.initial_move_speed_deg_s)
    wait_motion_with_drain(
        bus,
        abs(args.initial_move_deg) / args.initial_move_speed_deg_s + args.initial_move_settle_s,
    )
    snapshot = mc.read_motor_snapshot(ctrl, motor, experiment_start)
    actual_deg = float(snapshot["feedback_position_deg"])
    print(f"Initial coarse move done. Feedback angle: {actual_deg:.4f} deg")
    # Hold actual position so later manual jog starts from feedback, not stale target.
    try:
        send_target(bus, motor, actual_deg, args.manual_speed_deg_s)
    except Exception as exc:
        print(f"Warning: failed to hold after initial move: {exc}")
    return actual_deg, snapshot


def hold_current_position(bus, ctrl, motor, experiment_start: float) -> tuple[float, dict[str, float | str | bool]]:
    snapshot = mc.read_motor_snapshot(ctrl, motor, experiment_start)
    current_deg = float(snapshot["feedback_position_deg"])
    try:
        send_target(bus, motor, current_deg, MANUAL_SPEED_DEG_PER_SEC)
    except Exception as exc:
        print(f"Warning: failed to send hold target: {exc}")
    return current_deg, snapshot


def draw_status(frame, snapshot, *, target_deg: float, auto_running: bool, saved: bool) -> None:
    h, w = frame.shape[:2]
    # Center guides, same idea as motor_camera.py preview.
    cx = w // 2
    cy = h // 2
    green = (0, 255, 0)
    mc.cv2.line(frame, (cx, 0), (cx, h), green, 2)
    mc.cv2.line(frame, (0, cy), (w, cy), green, 2)
    mc.cv2.circle(frame, (cx, cy), 24, green, 2)
    mc.cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    angle = float(snapshot.get("feedback_position_deg", 0.0))
    angle_mod = float(snapshot.get("feedback_position_deg_mod360", angle % 360.0))
    ok = bool(snapshot.get("feedback_read_ok", False))
    vel = float(snapshot.get("feedback_velocity_deg_s", 0.0))
    mode = "AUTO" if auto_running else "MANUAL"

    panel_h = 150
    overlay = frame.copy()
    mc.cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    mc.cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"Mode: {mode}    feedback ok: {int(ok)}",
        f"Angle: {angle:.4f} deg    mod360: {angle_mod:.4f} deg    velocity: {vel:.4f} deg/s",
        f"Target: {target_deg:.4f} deg    S saves this angle as camera-laser offset",
        "R auto 1deg/s | Space stop | D/A +/-1deg | C/Z +/-0.1deg | V/X +/-0.05deg | S save | Q quit",
    ]
    if saved:
        lines.append("Saved. Closing...")

    y = 28
    for line in lines:
        mc.cv2.putText(frame, line, (18, y), mc.cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, mc.cv2.LINE_AA)
        y += 30


def save_result(out_dir: Path, *, snapshot, target_deg: float, samples_path: Path, args) -> Path:
    result = {
        "created_at": now_iso(),
        "meaning": "Current encoder angle after zeroing at laser reference position. Use as camera-laser angle offset.",
        "camera_laser_offset_deg": float(snapshot["feedback_position_deg"]),
        "camera_laser_offset_deg_mod360": float(snapshot["feedback_position_deg_mod360"]),
        "target_deg_at_save": float(target_deg),
        "feedback": {
            key: (float(value) if isinstance(value, (int, float)) and key != "feedback_read_ok" else value)
            for key, value in snapshot.items()
        },
        "settings": {
            "auto_speed_deg_s": args.auto_speed_deg_s,
            "auto_direction": args.auto_direction,
            "manual_big_step_deg": args.big_step,
            "manual_fine_step_deg": args.fine_step,
            "manual_micro_step_deg": args.micro_step,
            "can_channel": args.channel,
            "can_bitrate": args.bitrate,
            "can_id": args.can_id,
            "master_id": args.master_id,
        },
        "samples_csv": str(samples_path),
        "recommended_ronghe_2d_setting": f"ANGLE_OFFSET_DEG = {float(snapshot['feedback_position_deg']):.9f}",
    }
    json_path = out_dir / "camera_laser_angle_result.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    readme_path = out_dir / "readme.txt"
    readme_path.write_text(
        "\n".join(
            [
                "Camera-laser angle calibration result",
                f"camera_laser_offset_deg={float(snapshot['feedback_position_deg']):.9f}",
                f"camera_laser_offset_deg_mod360={float(snapshot['feedback_position_deg_mod360']):.9f}",
                f"target_deg_at_save={target_deg:.9f}",
                "",
                "Put into ronghe_2d.py initially:",
                f"ANGLE_OFFSET_DEG = {float(snapshot['feedback_position_deg']):.9f}",
                "",
                f"JSON: {json_path}",
                f"Samples: {samples_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return json_path


def run(args) -> int:
    out_dir = next_output_dir(args.output_root)
    samples_path = out_dir / "angle_feedback_samples.csv"
    sample_handle, sample_writer = open_csv(
        samples_path,
        [
            "wall_time_iso",
            "elapsed_s",
            "target_deg",
            "auto_running",
            "feedback_read_ok",
            "feedback_position_rad",
            "feedback_position_deg",
            "feedback_position_deg_mod360",
            "feedback_velocity_rad_s",
            "feedback_velocity_deg_s",
            "feedback_torque_nm",
        ],
    )

    camera = None
    bus = None
    motor_enabled = False
    saved = False
    result_path = None
    experiment_start = time.perf_counter()

    print("\nCamera-laser angle jog tool")
    print(f"Output folder: {out_dir}")
    print("Before pressing S, align the cone center with the green preview center.")

    try:
        print("Opening camera...")
        camera = mc.setup_camera()
        mc.configure_camera_for_preview(camera)

        print("Opening CAN bus...")
        bus, ctrl, motor = open_motor(args.channel, args.bitrate, args.can_id, args.master_id)
        set_current_position_zero(bus, motor)
        motor_enabled = True

        # Do not read encoder feedback before the first move. On some CAN states
        # that feedback request can block before the 45 deg coarse move is sent.
        target_deg = 0.0
        snapshot = {
            "feedback_read_ok": False,
            "feedback_wall_time_iso": now_iso(),
            "feedback_elapsed_s": time.perf_counter() - experiment_start,
            "feedback_position_rad": 0.0,
            "feedback_position_deg": 0.0,
            "feedback_position_deg_mod360": 0.0,
            "feedback_velocity_rad_s": 0.0,
            "feedback_velocity_deg_s": 0.0,
            "feedback_torque_nm": 0.0,
        }
        target_deg, snapshot = coarse_initial_move(bus, ctrl, motor, experiment_start, args)

        converter = mc.pylon.ImageFormatConverter()
        converter.OutputPixelFormat = mc.pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = mc.pylon.OutputBitAlignment_MsbAligned

        preview_width, preview_height = mc.get_preview_window_size()
        mc.cv2.namedWindow(PREVIEW_WINDOW_NAME, mc.cv2.WINDOW_NORMAL | mc.cv2.WINDOW_KEEPRATIO)
        mc.cv2.resizeWindow(PREVIEW_WINDOW_NAME, preview_width, preview_height)
        camera.StartGrabbing(mc.pylon.GrabStrategy_LatestImageOnly)

        auto_running = False
        last_command_t = time.perf_counter()
        last_feedback_t = 0.0
        last_sample_t = 0.0

        print("\nControls:")
        print("  R       auto run / pause at 1 deg/s")
        print("  Space   stop auto and hold current position")
        print("  D/A     jog +1 / -1 deg")
        print("  C/Z     jog +0.1 / -0.1 deg")
        print("  V/X     jog +0.05 / -0.05 deg")
        print("  S       save current encoder angle and exit")
        print("  Q/Esc   quit without saving\n")

        while camera.IsGrabbing():
            if STOP_REQUESTED:
                raise KeyboardInterrupt

            now = time.perf_counter()
            if now - last_feedback_t >= FEEDBACK_INTERVAL_S:
                snapshot = mc.read_motor_snapshot(ctrl, motor, experiment_start)
                last_feedback_t = now

            if auto_running and now - last_command_t >= COMMAND_INTERVAL_S:
                dt = now - last_command_t
                target_deg += args.auto_direction * args.auto_speed_deg_s * dt
                try:
                    send_target(bus, motor, target_deg, args.auto_speed_deg_s)
                except Exception as exc:
                    auto_running = False
                    print(f"Auto stopped: {exc}")
                last_command_t = now

            if now - last_sample_t >= SAMPLE_LOG_INTERVAL_S:
                sample_writer.writerow(
                    [
                        now_iso(),
                        f"{now - experiment_start:.6f}",
                        f"{target_deg:.9f}",
                        int(auto_running),
                    ]
                    + mc.motor_snapshot_csv_values(snapshot)
                )
                sample_handle.flush()
                last_sample_t = now

            grab = camera.RetrieveResult(1000, mc.pylon.TimeoutHandling_Return)
            if grab.IsValid():
                try:
                    if grab.GrabSucceeded():
                        frame = converter.Convert(grab).GetArray()
                        draw_status(frame, snapshot, target_deg=target_deg, auto_running=auto_running, saved=saved)
                        mc.cv2.imshow(PREVIEW_WINDOW_NAME, frame)
                finally:
                    grab.Release()

            key = mc.cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key in (27, ord("q"), ord("Q")):
                print("Quit without saving.")
                break
            if key in (ord("r"), ord("R")):
                auto_running = not auto_running
                if auto_running:
                    # Start from current feedback to avoid a stale target jump.
                    target_deg = float(snapshot["feedback_position_deg"])
                    last_command_t = time.perf_counter()
                    print(f"Auto running from {target_deg:.4f} deg...")
                else:
                    target_deg, snapshot = hold_current_position(bus, ctrl, motor, experiment_start)
                    print(f"Auto paused, holding {target_deg:.4f} deg")
            elif key == 32:
                auto_running = False
                target_deg, snapshot = hold_current_position(bus, ctrl, motor, experiment_start)
                print(f"Stopped and holding {target_deg:.4f} deg")
            elif key in (ord("d"), ord("D"), ord("a"), ord("A"), ord("c"), ord("C"), ord("z"), ord("Z"), ord("v"), ord("V"), ord("x"), ord("X")):
                auto_running = False
                sign = 1.0 if key in (ord("d"), ord("D"), ord("c"), ord("C"), ord("v"), ord("V")) else -1.0
                if key in (ord("d"), ord("D"), ord("a"), ord("A")):
                    step = args.big_step
                elif key in (ord("c"), ord("C"), ord("z"), ord("Z")):
                    step = args.fine_step
                else:
                    step = args.micro_step
                # Base manual jog on feedback so repeated clicks do not accumulate old target error.
                target_deg = float(snapshot["feedback_position_deg"]) + sign * step
                print(f"Manual jog to {target_deg:.4f} deg")
                send_target(bus, motor, target_deg, args.manual_speed_deg_s)
                time.sleep(args.manual_settle_s)
                snapshot = mc.read_motor_snapshot(ctrl, motor, experiment_start)
            elif key in (ord("s"), ord("S")):
                auto_running = False
                target_deg, snapshot = hold_current_position(bus, ctrl, motor, experiment_start)
                result_path = save_result(out_dir, snapshot=snapshot, target_deg=target_deg, samples_path=samples_path, args=args)
                saved = True
                print("\nSaved angle calibration:")
                print(f"  {result_path}")
                print(f"  camera_laser_offset_deg = {float(snapshot['feedback_position_deg']):.9f}")
                print(f"  Put into ronghe_2d.py: ANGLE_OFFSET_DEG = {float(snapshot['feedback_position_deg']):.9f}")
                time.sleep(0.5)
                break

        return 0

    finally:
        try:
            sample_handle.flush()
            sample_handle.close()
        except Exception:
            pass
        if motor_enabled and bus is not None:
            print("Disabling motor...")
            mc.send_disable_best_effort(bus, motor)
        if bus is not None:
            try:
                bus.shutdown()
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
        try:
            mc.cv2.destroyAllWindows()
            mc.cv2.waitKey(1)
        except Exception:
            pass


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jog motor while viewing Basler preview to measure camera-laser angle.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--channel", type=int, default=CAN_CHANNEL)
    parser.add_argument("--bitrate", type=int, default=CAN_BITRATE)
    parser.add_argument("--can-id", type=parse_int, default=MOTOR_CAN_ID)
    parser.add_argument("--master-id", type=parse_int, default=MASTER_ID)
    parser.add_argument("--auto-speed-deg-s", type=float, default=AUTO_SPEED_DEG_PER_SEC)
    parser.add_argument("--auto-direction", type=int, choices=(-1, 1), default=AUTO_DIRECTION)
    parser.add_argument("--initial-move-deg", type=float, default=INITIAL_MOVE_DEG)
    parser.add_argument("--initial-move-speed-deg-s", type=float, default=INITIAL_MOVE_SPEED_DEG_PER_SEC)
    parser.add_argument("--initial-move-settle-s", type=float, default=INITIAL_MOVE_SETTLE_S)
    parser.add_argument("--no-initial-move", action="store_true")
    parser.add_argument("--big-step", type=float, default=MANUAL_BIG_STEP_DEG)
    parser.add_argument("--fine-step", type=float, default=MANUAL_FINE_STEP_DEG)
    parser.add_argument("--micro-step", type=float, default=MANUAL_MICRO_STEP_DEG)
    parser.add_argument("--manual-speed-deg-s", type=float, default=MANUAL_SPEED_DEG_PER_SEC)
    parser.add_argument("--manual-settle-s", type=float, default=MANUAL_SETTLE_S)
    return parser


def main() -> int:
    signal.signal(signal.SIGINT, handle_ctrl_c)
    args = build_arg_parser().parse_args()
    if args.auto_speed_deg_s <= 0 or args.manual_speed_deg_s <= 0 or args.initial_move_speed_deg_s <= 0:
        raise ValueError("speed must be > 0")
    if args.big_step <= 0 or args.fine_step <= 0 or args.micro_step <= 0:
        raise ValueError("step sizes must be > 0")
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Verify the motor is stationary.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"Angle jog error: {exc}")
        raise SystemExit(1)
