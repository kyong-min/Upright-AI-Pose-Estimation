import argparse
import os
from pathlib import Path

from mediapipe_util import (
    POSE_LANDMARKS,
    create_pose_landmarker,
    default_cache_dir,
    default_model_path,
    detect_pose,
    draw_overlay,
    open_camera,
)

os.environ.setdefault("MPLCONFIGDIR", str(default_cache_dir()))

import cv2

from posture import compute_metrics, extract_upper_body_coordinates

LOG_EVERY_N_FRAMES = 10
MODE_CYCLE = ("auto", "front", "side")


def parse_args():
    parser = argparse.ArgumentParser(description="MediaPipe posture demo")
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
        print(f"카메라를 열지 못했습니다: {camera_id}", flush=True)
        return

    print(f"카메라 연결 완료: {camera_id}", flush=True)
    print(f"평가 모드: {mode} (m 키로 전환, q/esc 종료)", flush=True)
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
                    print("프레임을 읽지 못했습니다", flush=True)
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
                    print(
                        {
                            "classification": metrics.classification,
                            "view_mode": metrics.view_mode,
                            "mode_source": metrics.mode_source,
                            "reference_side": metrics.reference_side,
                            "turtle_neck": metrics.turtle_neck,
                            "asymmetry_detected": metrics.asymmetry_detected,
                            "total_score": metrics.total_score,
                            "neck_score": metrics.neck_score,
                            "shoulder_score": metrics.shoulder_score,
                            "trunk_score": metrics.trunk_score,
                            "tracking_score": metrics.tracking_score,
                            "turtle_neck_score": metrics.turtle_neck_score,
                            "neck_angle_deg": metrics.neck_angle_deg,
                            "shoulder_tilt_deg": metrics.shoulder_tilt_deg,
                            "upper_body_tilt_deg": metrics.upper_body_tilt_deg,
                            "head_forward_offset_z": metrics.head_forward_offset_z,
                            "ear_shoulder_distance_px": metrics.ear_shoulder_distance_px,
                            "feedback_message": metrics.feedback_message,
                            "upper_body_coordinates": extract_upper_body_coordinates(metrics),
                        },
                        flush=True,
                    )

                cv2.imshow("상체 자세 모니터", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("m"):
                    active_mode = next_mode(active_mode)
                    print(f"평가 모드 전환: {active_mode}", flush=True)
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        print("정상 종료합니다...", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
        model_path=args.model_path,
        mode=args.mode,
    )
