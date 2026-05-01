from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from posture import PostureMetrics


PROJECT_ROOT = Path(__file__).resolve().parent
POSE_LANDMARKS: Dict[str, int] = {
    "nose": vision.PoseLandmark.NOSE,
    "left_eye": vision.PoseLandmark.LEFT_EYE,
    "right_eye": vision.PoseLandmark.RIGHT_EYE,
    "left_ear": vision.PoseLandmark.LEFT_EAR,
    "right_ear": vision.PoseLandmark.RIGHT_EAR,
    "mouth_left": vision.PoseLandmark.MOUTH_LEFT,
    "mouth_right": vision.PoseLandmark.MOUTH_RIGHT,
    "left_shoulder": vision.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": vision.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": vision.PoseLandmark.LEFT_ELBOW,
    "right_elbow": vision.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": vision.PoseLandmark.LEFT_WRIST,
    "right_wrist": vision.PoseLandmark.RIGHT_WRIST,
    "left_hip": vision.PoseLandmark.LEFT_HIP,
    "right_hip": vision.PoseLandmark.RIGHT_HIP,
}

SKELETON_EDGES = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_ear", "left_shoulder"),
    ("right_ear", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
)

OVERLAY_ORIGIN_X = 20
OVERLAY_ORIGIN_Y = 30
VISIBILITY_THRESHOLD = 0.55
LANDMARK_LABELS = {
    "nose": "코",
    "left_eye": "왼눈",
    "right_eye": "오른눈",
    "left_ear": "왼귀",
    "right_ear": "오른귀",
    "mouth_left": "입왼쪽",
    "mouth_right": "입오른쪽",
    "left_shoulder": "왼어깨",
    "right_shoulder": "오른어깨",
    "left_elbow": "왼팔꿈치",
    "right_elbow": "오른팔꿈치",
    "left_wrist": "왼손목",
    "right_wrist": "오른손목",
    "left_hip": "왼골반",
    "right_hip": "오른골반",
    "face_center": "얼굴중심",
    "chest_center": "가슴중심",
    "hip_center": "골반중심",
}
FONT_CANDIDATES = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/bad9b4bf17cf1669dde54184ba4431c22dcad27b.asset/AssetData/NanumGothic.ttc",
)


def default_model_path() -> Path:
    return PROJECT_ROOT / "model_assets" / "pose_landmarker_lite.task"


def default_cache_dir() -> Path:
    return PROJECT_ROOT / ".cache" / "matplotlib"


def create_pose_landmarker(
    model_path: str,
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
) -> vision.PoseLandmarker:
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def open_camera(camera_id: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_id)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def frame_to_mp_image(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def detect_pose(landmarker: vision.PoseLandmarker, frame, timestamp_ms: int):
    mp_image = frame_to_mp_image(frame)
    return landmarker.detect_for_video(mp_image, timestamp_ms)


def detect_pose_image(landmarker: vision.PoseLandmarker, frame):
    mp_image = frame_to_mp_image(frame)
    return landmarker.detect(mp_image)


@lru_cache(maxsize=8)
def _load_korean_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_texts(image, text_items) -> None:
    if not text_items:
        return
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    for text, position, color, font_size in text_items:
        x, y = position
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text(
                (x + dx, y + dy),
                text,
                font=_load_korean_font(font_size),
                fill=(255, 255, 255),
            )
        draw.text(
            position,
            text,
            font=_load_korean_font(font_size),
            fill=(color[2], color[1], color[0]),
        )
    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_upper_body(image, metrics: "PostureMetrics") -> None:
    for start_name, end_name in SKELETON_EDGES:
        start = metrics.points[start_name]
        end = metrics.points[end_name]
        if (
            start.visibility >= VISIBILITY_THRESHOLD
            and end.visibility >= VISIBILITY_THRESHOLD
        ):
            cv2.line(image, (start.x, start.y), (end.x, end.y), (0, 220, 255), 2)

    for name, point in metrics.points.items():
        color = (0, 220, 0) if point.visibility >= VISIBILITY_THRESHOLD else (0, 0, 255)
        cv2.circle(image, (point.x, point.y), 5, color, -1)

    derived_points = {
        "face_center": metrics.face_center,
        "chest_center": metrics.chest_center,
        "hip_center": metrics.hip_center,
    }
    for name, point in derived_points.items():
        if point is None:
            continue
        color = (255, 200, 0)
        cv2.circle(image, (point.x, point.y), 6, color, -1)

    if metrics.face_center is not None and metrics.chest_center is not None:
        cv2.line(
            image,
            (metrics.face_center.x, metrics.face_center.y),
            (metrics.chest_center.x, metrics.chest_center.y),
            (255, 180, 0),
            2,
        )
    if metrics.chest_center is not None and metrics.hip_center is not None:
        cv2.line(
            image,
            (metrics.chest_center.x, metrics.chest_center.y),
            (metrics.hip_center.x, metrics.hip_center.y),
            (255, 180, 0),
            2,
        )


def draw_overlay(image, metrics: Optional["PostureMetrics"], requested_mode: str = "auto") -> None:
    if metrics is None:
        return

    draw_upper_body(image, metrics)
    status_color = (0, 180, 0) if metrics.classification == "Good" else (0, 190, 255) if metrics.classification == "Warning" else (0, 0, 255)
    overlay_lines = (
        f"Status: {metrics.classification}",
        f"Score: {metrics.total_score}",
    )
    draw_texts(
        image,
        [
            (text, (OVERLAY_ORIGIN_X, OVERLAY_ORIGIN_Y + index * 24), status_color if index == 0 else (0, 0, 0), 20)
            for index, text in enumerate(overlay_lines)
        ],
    )
