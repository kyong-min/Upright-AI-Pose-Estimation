import argparse
import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from mediapipe_util import (
    POSE_LANDMARKS,
    create_pose_landmarker,
    default_model_path,
    detect_pose,
    draw_overlay,
    open_camera,
)

import cv2

from posture import compute_metrics

LOG_EVERY_N_FRAMES = 60
MODE_CYCLE = ("auto", "front", "side")
LOGGER = logging.getLogger("upright.main")


def parse_args():
    parser = argparse.ArgumentParser(description="Local webcam posture test runner")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(default_model_path()),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=MODE_CYCLE,
        default="auto",
        help="자세 평가 모드: auto, front, side",
    )
    return parser.parse_args()


def next_mode(mode: str) -> str:
    return MODE_CYCLE[(MODE_CYCLE.index(mode) + 1) % len(MODE_CYCLE)]


def run(camera_id: int, width: int, height: int, model_path: str, mode: str) -> None:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    capture = open_camera(camera_id, width, height)
    if not capture.isOpened():
        LOGGER.error("Failed to open camera: %s", camera_id)
        return

    LOGGER.info("Camera ready: %s", camera_id)
    LOGGER.info("Mode: %s (press m to switch, q/esc to quit)", mode)
    try:
        landmarker = create_pose_landmarker(model_path)
    except RuntimeError as exc:
        capture.release()
        raise RuntimeError(
            "MediaPipe PoseLandmarker 초기화에 실패했습니다. "
            "데스크톱 그래픽 접근이 가능한 Python 환경에서 실행해주세요."
        ) from exc

    try:
        with landmarker:
            timestamp_ms = 0
            frame_count = 0
            active_mode = mode
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    LOGGER.debug("Frame capture failed")
                    continue

                frame_count += 1
                frame = cv2.flip(frame, 1)
                results = detect_pose(landmarker, frame, timestamp_ms)
                timestamp_ms += 33

                pose_landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
                metrics = compute_metrics(
                    pose_landmarks=pose_landmarks,
                    frame_width=frame.shape[1],
                    frame_height=frame.shape[0],
                    landmark_indexes=POSE_LANDMARKS,
                    preferred_mode=active_mode,
                )
                draw_overlay(frame, metrics, requested_mode=active_mode)

                if metrics and frame_count % LOG_EVERY_N_FRAMES == 0:
                    LOGGER.info(
                        "status=%s score=%s neck=%.1f shoulder=%.1f trunk=%.1f view=%s",
                        metrics.classification,
                        metrics.total_score,
                        metrics.neck_angle_deg or 0.0,
                        metrics.shoulder_tilt_deg or 0.0,
                        metrics.upper_body_tilt_deg or 0.0,
                        metrics.view_mode,
                    )

                cv2.imshow("상체 자세 모니터", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("m"):
                    active_mode = next_mode(active_mode)
                    LOGGER.info("Mode switched to: %s", active_mode)
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        LOGGER.info("Shutting down")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    run(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        model_path=args.model_path,
        mode=args.mode,
    )
