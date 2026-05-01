import math
import os
import threading
from dataclasses import dataclass
from collections import deque
from functools import lru_cache
from typing import Deque, Dict, Iterable, Optional, Tuple

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from angle import distance, line_tilt_degrees
from mediapipe.tasks.python import vision

from mediapipe_util import (
    POSE_LANDMARKS,
    create_pose_landmarker,
    default_model_path,
    detect_pose_image,
)


VISIBILITY_THRESHOLD = 0.55
SHOULDER_Y_LIFT_RATIO = 0.06
SHOULDER_Y_LIFT_MIN_PX = 4
NECK_NORMAL_THRESHOLD = 45.0
NECK_WARNING_THRESHOLD = 35.0
SHOULDER_NORMAL_THRESHOLD = 5.0
SHOULDER_WARNING_THRESHOLD = 10.0
TRUNK_NORMAL_THRESHOLD = 7.0
TRUNK_WARNING_THRESHOLD = 15.0
HEAD_FORWARD_Z_WARNING_THRESHOLD = -0.02
HEAD_FORWARD_Z_BAD_THRESHOLD = -0.05
FORWARD_RATIO_WARNING_THRESHOLD = 0.05
FORWARD_RATIO_BAD_THRESHOLD = 0.10
SIDE_VIEW_SCORE_MARGIN = 1.5
TRACKING_SCORE_STEP = 8
TRACKING_SCORE_FHP_BONUS = 12
TOTAL_SCORE_WEIGHTS = {"neck": 0.4, "shoulder": 0.3, "trunk": 0.3}
PENALTY_STRENGTH = {"neck": 0.6, "shoulder": 0.8, "trunk": 0.8}
SMOOTHING_ALPHA = 0.25
ROLLING_WINDOW_SIZE = 12
_IMAGE_LANDMARKER_LOCK = threading.Lock()


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: float
    visibility: float


@dataclass(frozen=True)
class PostureMetrics:
    view_mode: str
    mode_source: str
    mode_confidence: float
    points: Dict[str, Point]
    reference_side: Optional[str]
    face_center: Optional[Point]
    chest_center: Optional[Point]
    hip_center: Optional[Point]
    neck_angle_deg: Optional[float]
    cva_proxy_deg: Optional[float]
    shoulder_tilt_deg: float
    upper_body_tilt_deg: Optional[float]
    asymmetry_deg: float
    asymmetry_detected: bool
    arm_balance_deg: float
    ear_shoulder_distance_px: Optional[float]
    ear_shoulder_ratio: Optional[float]
    head_forward_offset_z: Optional[float]
    forward_ratio: Optional[float]
    side_head_forward_ratio: Optional[float]
    neck_score: int
    shoulder_score: int
    trunk_score: int
    tracking_score: int
    turtle_neck_score: Optional[int]
    turtle_neck: bool
    total_score: int
    classification: str
    feedback_message: str
    visibility_ok: bool
    forward_head_measurement_ok: bool


class MetricSmoother:
    def __init__(self, alpha: float = SMOOTHING_ALPHA, window_size: int = ROLLING_WINDOW_SIZE):
        self.alpha = alpha
        self.window_size = window_size
        self._lock = threading.Lock()
        self._ema: Dict[str, Optional[float]] = {}
        self._windows: Dict[str, Deque[float]] = {}

    def smooth(self, key: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return self._ema.get(key)
        with self._lock:
            previous = self._ema.get(key)
            ema_value = value if previous is None else (self.alpha * value) + ((1.0 - self.alpha) * previous)
            self._ema[key] = ema_value
            window = self._windows.setdefault(key, deque(maxlen=self.window_size))
            window.append(ema_value)
            return sum(window) / len(window)


_METRIC_SMOOTHER = MetricSmoother()


def to_point(landmark, width: int, height: int) -> Point:
    return Point(
        x=int(landmark.x * width),
        y=int(landmark.y * height),
        z=float(landmark.z),
        visibility=float(getattr(landmark, "visibility", 0.0)),
    )


def penalty_from_thresholds(value: float, thresholds, penalties) -> int:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return penalties[index]
    return penalties[-1]


def penalty_to_score(penalty: int, max_penalty: int) -> int:
    if max_penalty <= 0:
        return 100
    return max(0, int(round(100 * (1 - penalty / max_penalty))))


def weighted_score(score_items) -> int:
    weighted_sum = 0.0
    total_weight = 0.0
    for score, weight in score_items:
        if score is None:
            continue
        weighted_sum += score * weight
        total_weight += weight
    if total_weight == 0:
        return 0
    return int(round(weighted_sum / total_weight))


def midpoint(a: Point, b: Point) -> Point:
    return Point(
        x=int((a.x + b.x) / 2),
        y=int((a.y + b.y) / 2),
        z=(a.z + b.z) / 2.0,
        visibility=min(a.visibility, b.visibility),
    )


def weighted_midpoint(a: Point, b: Point, ratio_from_a: float) -> Point:
    ratio = min(max(ratio_from_a, 0.0), 1.0)
    return Point(
        x=int(a.x + (b.x - a.x) * ratio),
        y=int(a.y + (b.y - a.y) * ratio),
        z=a.z + (b.z - a.z) * ratio,
        visibility=min(a.visibility, b.visibility),
    )


def adjust_shoulder_points(points: Dict[str, Point]) -> Dict[str, Point]:
    adjusted_points = dict(points)
    for side in ("left", "right"):
        shoulder_name = f"{side}_shoulder"
        hip_name = f"{side}_hip"
        shoulder = adjusted_points[shoulder_name]
        hip = adjusted_points[hip_name]
        torso_height = abs(hip.y - shoulder.y)
        lift_px = max(SHOULDER_Y_LIFT_MIN_PX, int(round(torso_height * SHOULDER_Y_LIFT_RATIO)))
        adjusted_points[shoulder_name] = Point(
            x=shoulder.x,
            y=max(0, shoulder.y - lift_px),
            z=shoulder.z,
            visibility=shoulder.visibility,
        )
    return adjusted_points


def average_points(points: Iterable[Point]) -> Optional[Point]:
    point_list = list(points)
    if not point_list:
        return None
    return Point(
        x=int(sum(point.x for point in point_list) / len(point_list)),
        y=int(sum(point.y for point in point_list) / len(point_list)),
        z=sum(point.z for point in point_list) / len(point_list),
        visibility=min(point.visibility for point in point_list),
    )


def vertical_deviation_degrees(start: Point, end: Point) -> Optional[float]:
    dx = end.x - start.x
    dy = end.y - start.y
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), max(abs(dy), 1e-6)))


def horizontal_angle_degrees(start: Point, end: Point) -> Optional[float]:
    dx = abs(end.x - start.x)
    dy = abs(end.y - start.y)
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dy, max(dx, 1e-6)))


def pick_reference_side(points: Dict[str, Point]) -> Optional[str]:
    left_visibility = sum(
        points[f"left_{name}"].visibility for name in ("ear", "shoulder", "elbow", "wrist")
    )
    right_visibility = sum(
        points[f"right_{name}"].visibility for name in ("ear", "shoulder", "elbow", "wrist")
    )

    best_side = "left" if left_visibility >= right_visibility else "right"
    if all(
        points[f"{best_side}_{name}"].visibility < VISIBILITY_THRESHOLD
        for name in ("ear", "shoulder", "elbow", "wrist")
    ):
        return None
    return best_side


def detect_view_mode(points: Dict[str, Point], reference_side: Optional[str]) -> Tuple[str, float]:
    left_side_visibility = sum(
        points[f"left_{name}"].visibility for name in ("ear", "shoulder", "hip")
    )
    right_side_visibility = sum(
        points[f"right_{name}"].visibility for name in ("ear", "shoulder", "hip")
    )
    frontal_score = sum(
        min(points[f"left_{name}"].visibility, points[f"right_{name}"].visibility)
        for name in ("ear", "shoulder", "hip")
    )
    side_score = abs(left_side_visibility - right_side_visibility)
    if reference_side is None:
        return "front", 0.5
    opposite_side = "right" if reference_side == "left" else "left"
    opposite_ear_hidden = points[f"{opposite_side}_ear"].visibility < VISIBILITY_THRESHOLD
    if side_score >= SIDE_VIEW_SCORE_MARGIN and opposite_ear_hidden:
        return "side", min(1.0, side_score / 3.0)
    if frontal_score >= 2.0:
        return "front", min(1.0, frontal_score / 3.0)
    return ("side", min(1.0, side_score / 3.0)) if side_score > frontal_score else ("front", min(1.0, frontal_score / 3.0))


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1e-6)


def symmetric_angle(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    angle = abs(float(value))
    return min(angle, abs(180.0 - angle))


def classify_metric(value: Optional[float], normal_threshold: float, warning_threshold: float, higher_is_better: bool) -> str:
    if value is None:
        return "Warning"
    if higher_is_better:
        if value > normal_threshold:
            return "Good"
        if value >= warning_threshold:
            return "Warning"
        return "Needs Correction"
    if value < normal_threshold:
        return "Good"
    if value <= warning_threshold:
        return "Warning"
    return "Needs Correction"


def clamp_score(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def soften_penalty(score: int, strength: float) -> int:
    return clamp_score(100.0 - ((100.0 - score) * strength))


def score_cva(value: Optional[float]) -> int:
    if value is None:
        return 85
    if value >= 55.0:
        return 100
    if value >= NECK_NORMAL_THRESHOLD:
        span = max(55.0 - NECK_NORMAL_THRESHOLD, 1e-6)
        return clamp_score(90.0 + ((value - NECK_NORMAL_THRESHOLD) / span) * 10.0)
    if value >= NECK_WARNING_THRESHOLD:
        span = max(NECK_NORMAL_THRESHOLD - NECK_WARNING_THRESHOLD, 1e-6)
        return clamp_score(72.0 + ((value - NECK_WARNING_THRESHOLD) / span) * 18.0)
    return clamp_score(72.0 - ((NECK_WARNING_THRESHOLD - value) * 1.2))


def score_inverse_threshold(value: Optional[float], normal_threshold: float, warning_threshold: float, severe_step: float) -> int:
    if value is None:
        return 85
    if value <= 0:
        return 100
    if value < normal_threshold:
        span = max(normal_threshold, 1e-6)
        return clamp_score(100.0 - (value / span) * 12.0)
    if value <= warning_threshold:
        span = max(warning_threshold - normal_threshold, 1e-6)
        return clamp_score(88.0 - ((value - normal_threshold) / span) * 23.0)
    return clamp_score(65.0 - (value - warning_threshold) * severe_step)


def classify_head_forward_z(value: Optional[float]) -> str:
    if value is None:
        return "Warning"
    deviation = abs(value)
    if deviation > abs(HEAD_FORWARD_Z_BAD_THRESHOLD):
        return "Needs Correction"
    if deviation >= abs(HEAD_FORWARD_Z_WARNING_THRESHOLD):
        return "Warning"
    return "Good"


def classify_forward_ratio(value: Optional[float]) -> str:
    if value is None:
        return "Warning"
    deviation = abs(value)
    if deviation > FORWARD_RATIO_BAD_THRESHOLD:
        return "Needs Correction"
    if deviation >= FORWARD_RATIO_WARNING_THRESHOLD:
        return "Warning"
    return "Good"


def score_head_forward_z(value: Optional[float]) -> int:
    if value is None:
        return 85
    deviation = abs(value)
    if deviation < abs(HEAD_FORWARD_Z_WARNING_THRESHOLD):
        return 100
    if deviation <= abs(HEAD_FORWARD_Z_BAD_THRESHOLD):
        span = max(abs(HEAD_FORWARD_Z_BAD_THRESHOLD) - abs(HEAD_FORWARD_Z_WARNING_THRESHOLD), 1e-6)
        return clamp_score(85.0 - ((deviation - abs(HEAD_FORWARD_Z_WARNING_THRESHOLD)) / span) * 25.0)
    return clamp_score(60.0 - ((deviation - abs(HEAD_FORWARD_Z_BAD_THRESHOLD)) * 250.0))


def score_forward_ratio(value: Optional[float]) -> int:
    if value is None:
        return 85
    deviation = abs(value)
    if deviation < FORWARD_RATIO_WARNING_THRESHOLD:
        span = max(FORWARD_RATIO_WARNING_THRESHOLD, 1e-6)
        return clamp_score(100.0 - (deviation / span) * 12.0)
    if deviation <= FORWARD_RATIO_BAD_THRESHOLD:
        span = max(FORWARD_RATIO_BAD_THRESHOLD - FORWARD_RATIO_WARNING_THRESHOLD, 1e-6)
        return clamp_score(85.0 - ((deviation - FORWARD_RATIO_WARNING_THRESHOLD) / span) * 25.0)
    return clamp_score(60.0 - ((deviation - FORWARD_RATIO_BAD_THRESHOLD) * 200.0))


def combine_statuses(*statuses: str) -> str:
    if any(status == "Needs Correction" for status in statuses):
        return "Needs Correction"
    if any(status == "Warning" for status in statuses):
        return "Warning"
    return "Good"


def classify_total_score(total_score: int) -> str:
    if total_score >= 90:
        return "Excellent"
    if total_score >= 75:
        return "Good"
    if total_score >= 60:
        return "Warning"
    return "Critical"


def build_feedback_message(
    neck_status: str,
    shoulder_status: str,
    trunk_status: str,
    neck_angle_deg: Optional[float],
    shoulder_tilt_deg: float,
    upper_body_tilt_deg: Optional[float],
    total_score: int,
) -> str:
    feedback_parts = []
    if neck_status != "Good":
        if neck_status == "Warning":
            feedback_parts.append("Head is slightly forward. Try pulling your chin back.")
        else:
            feedback_parts.append("Forward head posture detected. Pull your chin back and align your ear over your shoulder.")
    if shoulder_status != "Good":
        if shoulder_status == "Warning":
            feedback_parts.append("Shoulder balance is slightly off. Relax and level both shoulders.")
        else:
            feedback_parts.append("Shoulder imbalance detected. Adjust posture evenly.")
    if trunk_status != "Good":
        if trunk_status == "Warning":
            feedback_parts.append("Upper body is starting to lean. Re-center your torso.")
        else:
            feedback_parts.append("Upper body leaning forward. Straighten your spine.")
    if not feedback_parts and total_score < 95:
        feedback_parts.append("Posture is mostly stable. Maintain a neutral head and shoulder position.")
    if not feedback_parts:
        return "Posture is stable."
    return " ".join(feedback_parts[:3])


def build_face_center(points: Dict[str, Point]) -> Optional[Point]:
    face_point_names = (
        "nose",
        "left_eye",
        "right_eye",
        "mouth_left",
        "mouth_right",
        "left_ear",
        "right_ear",
    )
    visible_points = [
        points[name]
        for name in face_point_names
        if points[name].visibility >= VISIBILITY_THRESHOLD
    ]
    if len(visible_points) < 3:
        return None
    return average_points(visible_points)


def average_visible_distances(points: Dict[str, Point], pairs) -> Optional[float]:
    distances = []
    for first_name, second_name in pairs:
        first = points[first_name]
        second = points[second_name]
        if (
            first.visibility >= VISIBILITY_THRESHOLD
            and second.visibility >= VISIBILITY_THRESHOLD
        ):
            distances.append(distance(first, second))
    if not distances:
        return None
    return sum(distances) / len(distances)


def shoulder_height_asymmetry_ratio(points: Dict[str, Point]) -> float:
    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    shoulder_span = max(abs(left_shoulder.x - right_shoulder.x), 1.0)
    return abs(left_shoulder.y - right_shoulder.y) / shoulder_span


def build_impact_factor(
    turtle_neck: bool,
    shoulder_imbalance: bool,
    upper_body_tilt_bad: bool,
) -> str:
    impacts = []
    if turtle_neck:
        impacts.append("Forward head posture may increase neck loading and fatigue over time.")
    if shoulder_imbalance:
        impacts.append("Shoulder imbalance can increase uneven muscle loading around the neck and upper back.")
    if upper_body_tilt_bad:
        impacts.append("Upper body tilt can add strain to the thoracic spine and back muscles.")
    if not impacts:
        impacts.append("No major postural loading risk is indicated by the current frame.")
    return " ".join(impacts[:3])


def build_api_result(metrics: PostureMetrics) -> dict:
    neck_angle_deg = round(float(metrics.neck_angle_deg or 0.0), 2)
    shoulder_tilt_deg = round(float(metrics.shoulder_tilt_deg or 0.0), 2)
    upper_body_tilt_deg = round(float(metrics.upper_body_tilt_deg or 0.0), 2)
    head_forward_z = round(float(metrics.head_forward_offset_z or 0.0), 4)
    forward_ratio = round(float(metrics.forward_ratio or 0.0), 4)

    return {
        "status": metrics.classification,
        "turtle_neck": bool(metrics.turtle_neck),
        "neck_angle_deg": neck_angle_deg,
        "shoulder_tilt_deg": shoulder_tilt_deg,
        "upper_body_tilt_deg": upper_body_tilt_deg,
        "head_forward_z": head_forward_z,
        "forward_ratio": forward_ratio,
        "total_score": int(metrics.total_score),
    }


def compute_metrics(
    pose_landmarks,
    frame_width: int,
    frame_height: int,
    landmark_indexes: Dict[str, int],
    preferred_mode: str = "auto",
) -> Optional[PostureMetrics]:
    if not pose_landmarks:
        return None

    landmarks = (
        pose_landmarks.landmark
        if hasattr(pose_landmarks, "landmark")
        else pose_landmarks
    )
    points = {
        name: to_point(landmarks[index], frame_width, frame_height)
        for name, index in landmark_indexes.items()
    }
    points = adjust_shoulder_points(points)

    reference_side = pick_reference_side(points)
    auto_view_mode, mode_confidence = detect_view_mode(points, reference_side)
    view_mode = auto_view_mode if preferred_mode == "auto" else preferred_mode
    mode_source = "auto" if preferred_mode == "auto" else "manual"
    face_center = build_face_center(points)
    frontal_visibility_ok = all(
        points[name].visibility >= VISIBILITY_THRESHOLD
        for name in (
            "nose",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        )
    ) and reference_side is not None and face_center is not None
    side_visibility_ok = reference_side is not None and all(
        points[f"{reference_side}_{name}"].visibility >= VISIBILITY_THRESHOLD
        for name in ("ear", "shoulder", "hip")
    )
    visibility_ok = frontal_visibility_ok if view_mode == "front" else side_visibility_ok

    shoulder_mid = midpoint(points["left_shoulder"], points["right_shoulder"])
    hip_mid = midpoint(points["left_hip"], points["right_hip"])
    chest_center = weighted_midpoint(shoulder_mid, hip_mid, 0.35)

    shoulder_tilt_deg = symmetric_angle(line_tilt_degrees(points["left_shoulder"], points["right_shoulder"])) or 0.0

    upper_body_tilt_deg = symmetric_angle(vertical_deviation_degrees(hip_mid, shoulder_mid))

    cva_proxy_deg = None
    neck_angle_deg = None
    ear_shoulder_distance_px = None
    ear_shoulder_ratio = None
    head_forward_offset_z = None
    forward_ratio = None
    side_head_forward_ratio = None
    turtle_neck = False
    ear_tilt_deg = symmetric_angle(line_tilt_degrees(points["left_ear"], points["right_ear"])) or 0.0
    mouth_tilt_deg = symmetric_angle(line_tilt_degrees(points["mouth_left"], points["mouth_right"])) or 0.0
    elbow_tilt_deg = symmetric_angle(line_tilt_degrees(points["left_elbow"], points["right_elbow"])) or 0.0
    wrist_tilt_deg = symmetric_angle(line_tilt_degrees(points["left_wrist"], points["right_wrist"])) or 0.0
    arm_balance_deg = max(elbow_tilt_deg, wrist_tilt_deg)
    asymmetry_deg = max(shoulder_tilt_deg, ear_tilt_deg, mouth_tilt_deg, arm_balance_deg)
    asymmetry_detected = asymmetry_deg >= SHOULDER_WARNING_THRESHOLD

    if view_mode == "front":
        cva_candidates = []
        ratio_candidates = []
        distance_candidates = []
        shoulder_width = distance(points["left_shoulder"], points["right_shoulder"])
        for side in ("left", "right"):
            ear_point = points[f"{side}_ear"]
            shoulder_point = points[f"{side}_shoulder"]
            if ear_point.visibility < VISIBILITY_THRESHOLD or shoulder_point.visibility < VISIBILITY_THRESHOLD:
                continue
            current_distance = distance(ear_point, shoulder_point)
            distance_candidates.append(current_distance)
            cva_candidate = horizontal_angle_degrees(shoulder_point, ear_point)
            if cva_candidate is not None:
                cva_candidates.append(cva_candidate)
            if shoulder_width > 1.0:
                ratio_candidates.append(current_distance / shoulder_width)
        ear_shoulder_distance_px = sum(distance_candidates) / len(distance_candidates) if distance_candidates else None
        ear_shoulder_ratio = sum(ratio_candidates) / len(ratio_candidates) if ratio_candidates else None
        neck_angle_deg = sum(cva_candidates) / len(cva_candidates) if cva_candidates else None
        cva_proxy_deg = neck_angle_deg
        forward_head_measurement_ok = (
            points["nose"].visibility >= VISIBILITY_THRESHOLD
            and points["left_ear"].visibility >= VISIBILITY_THRESHOLD
            and points["right_ear"].visibility >= VISIBILITY_THRESHOLD
            and points["left_shoulder"].visibility >= VISIBILITY_THRESHOLD
            and points["right_shoulder"].visibility >= VISIBILITY_THRESHOLD
        )
        if forward_head_measurement_ok:
            head_forward_offset_z = points["nose"].z - shoulder_mid.z
            ear_mid_x = (points["left_ear"].x + points["right_ear"].x) / 2.0
            shoulder_mid_x = (points["left_shoulder"].x + points["right_shoulder"].x) / 2.0
            shoulder_width = max(abs(points["left_shoulder"].x - points["right_shoulder"].x), 1.0)
            forward_ratio = (ear_mid_x - shoulder_mid_x) / shoulder_width
        required_names = (
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
        )
    else:
        forward_head_measurement_ok = side_visibility_ok
        if reference_side is not None:
            ear_point = points[f"{reference_side}_ear"]
            shoulder_point = points[f"{reference_side}_shoulder"]
            hip_point = points[f"{reference_side}_hip"]
            torso_height_px = max(abs(hip_point.y - shoulder_point.y), 1.0)
            ear_shoulder_distance_px = distance(ear_point, shoulder_point)
            shoulder_width = distance(points["left_shoulder"], points["right_shoulder"])
            side_head_forward_ratio = safe_ratio(abs(ear_point.x - shoulder_point.x), torso_height_px)
            ear_shoulder_ratio = safe_ratio(ear_shoulder_distance_px, shoulder_width)
            cva_proxy_deg = horizontal_angle_degrees(shoulder_point, ear_point)
            neck_angle_deg = cva_proxy_deg
            upper_body_tilt_deg = symmetric_angle(vertical_deviation_degrees(hip_point, shoulder_point))
            asymmetry_deg = shoulder_tilt_deg
            asymmetry_detected = asymmetry_deg >= SHOULDER_WARNING_THRESHOLD
            head_forward_offset_z = ear_point.z - shoulder_point.z
            forward_ratio = side_head_forward_ratio
        required_names = (
            f"{reference_side}_ear",
            f"{reference_side}_shoulder",
            f"{reference_side}_hip",
        ) if reference_side is not None else ()

    low_visibility_count = sum(
        points[name].visibility < VISIBILITY_THRESHOLD for name in required_names
    )

    tracking_score = max(0, 100 - low_visibility_count * TRACKING_SCORE_STEP)
    if forward_head_measurement_ok:
        tracking_score = min(100, tracking_score + TRACKING_SCORE_FHP_BONUS)

    neck_angle_deg = _METRIC_SMOOTHER.smooth("neck_angle_deg", neck_angle_deg)
    shoulder_tilt_deg = _METRIC_SMOOTHER.smooth("shoulder_tilt_deg", shoulder_tilt_deg) or 0.0
    upper_body_tilt_deg = _METRIC_SMOOTHER.smooth("upper_body_tilt_deg", upper_body_tilt_deg)
    asymmetry_deg = _METRIC_SMOOTHER.smooth("asymmetry_deg", asymmetry_deg) or 0.0
    head_forward_offset_z = _METRIC_SMOOTHER.smooth("head_forward_z", head_forward_offset_z)
    forward_ratio = _METRIC_SMOOTHER.smooth("forward_ratio", forward_ratio)

    head_forward_z_status = classify_head_forward_z(head_forward_offset_z)
    forward_ratio_status = classify_forward_ratio(forward_ratio)
    angle_status = classify_metric(neck_angle_deg, NECK_NORMAL_THRESHOLD, NECK_WARNING_THRESHOLD, higher_is_better=True)
    primary_neck_status = combine_statuses(head_forward_z_status, forward_ratio_status)
    neck_status = primary_neck_status if primary_neck_status != "Good" else ("Warning" if angle_status == "Needs Correction" else angle_status)
    turtle_neck = primary_neck_status != "Good"

    head_forward_z_score = score_head_forward_z(head_forward_offset_z)
    forward_ratio_score = score_forward_ratio(forward_ratio)
    angle_score = score_cva(neck_angle_deg)
    neck_base_score = clamp_score(
        (head_forward_z_score * 0.45)
        + (forward_ratio_score * 0.45)
        + (angle_score * 0.10)
    )
    head_forward_penalty = 0
    if head_forward_offset_z is not None:
        if head_forward_offset_z < -0.05:
            head_forward_penalty += 25
        elif head_forward_offset_z < -0.03:
            head_forward_penalty += 15
        elif head_forward_offset_z < -0.02:
            head_forward_penalty += 8
    z_bad = head_forward_z_status == "Needs Correction"
    ratio_bad = forward_ratio_status == "Needs Correction"
    if z_bad and ratio_bad:
        head_forward_penalty += 15
    elif z_bad or ratio_bad:
        head_forward_penalty += 6
    neck_score = clamp_score(neck_base_score - head_forward_penalty)
    neck_score = soften_penalty(neck_score, PENALTY_STRENGTH["neck"])
    shoulder_score = score_inverse_threshold(
        shoulder_tilt_deg,
        SHOULDER_NORMAL_THRESHOLD,
        SHOULDER_WARNING_THRESHOLD,
        severe_step=3.0,
    )
    shoulder_score = soften_penalty(shoulder_score, PENALTY_STRENGTH["shoulder"])
    trunk_score = score_inverse_threshold(
        upper_body_tilt_deg,
        TRUNK_NORMAL_THRESHOLD,
        TRUNK_WARNING_THRESHOLD,
        severe_step=2.5,
    )
    trunk_score = soften_penalty(trunk_score, PENALTY_STRENGTH["trunk"])
    turtle_neck_score = neck_score if forward_head_measurement_ok else None

    total_score = int(
        round(
            neck_score * TOTAL_SCORE_WEIGHTS["neck"]
            + shoulder_score * TOTAL_SCORE_WEIGHTS["shoulder"]
            + trunk_score * TOTAL_SCORE_WEIGHTS["trunk"]
        )
    )
    if tracking_score < 55:
        total_score = min(total_score, 75)
    shoulder_status = classify_metric(shoulder_tilt_deg, SHOULDER_NORMAL_THRESHOLD, SHOULDER_WARNING_THRESHOLD, higher_is_better=False)
    trunk_status = classify_metric(upper_body_tilt_deg, TRUNK_NORMAL_THRESHOLD, TRUNK_WARNING_THRESHOLD, higher_is_better=False)
    classification = classify_total_score(total_score)
    feedback_message = build_feedback_message(
        neck_status,
        shoulder_status,
        trunk_status,
        neck_angle_deg,
        shoulder_tilt_deg,
        upper_body_tilt_deg,
        total_score,
    )
    return PostureMetrics(
        view_mode=view_mode,
        mode_source=mode_source,
        mode_confidence=mode_confidence,
        points=points,
        reference_side=reference_side,
        face_center=face_center,
        chest_center=chest_center,
        hip_center=hip_mid,
        neck_angle_deg=neck_angle_deg,
        cva_proxy_deg=cva_proxy_deg,
        shoulder_tilt_deg=shoulder_tilt_deg,
        upper_body_tilt_deg=upper_body_tilt_deg,
        asymmetry_deg=asymmetry_deg,
        asymmetry_detected=asymmetry_detected,
        arm_balance_deg=arm_balance_deg,
        ear_shoulder_distance_px=ear_shoulder_distance_px,
        ear_shoulder_ratio=ear_shoulder_ratio,
        head_forward_offset_z=head_forward_offset_z,
        forward_ratio=forward_ratio,
        side_head_forward_ratio=side_head_forward_ratio,
        neck_score=neck_score,
        shoulder_score=shoulder_score,
        trunk_score=trunk_score,
        tracking_score=tracking_score,
        turtle_neck_score=turtle_neck_score,
        turtle_neck=turtle_neck,
        total_score=total_score,
        classification=classification,
        feedback_message=feedback_message,
        visibility_ok=visibility_ok,
        forward_head_measurement_ok=forward_head_measurement_ok,
    )


def analyze_posture(frame) -> dict:
    with _IMAGE_LANDMARKER_LOCK:
        results = detect_pose_image(_get_image_landmarker(), frame)
    pose_landmarks = results.pose_landmarks[0] if results.pose_landmarks else None
    metrics = compute_metrics(
        pose_landmarks=pose_landmarks,
        frame_width=frame.shape[1],
        frame_height=frame.shape[0],
        landmark_indexes=POSE_LANDMARKS,
        preferred_mode="auto",
    )
    if metrics is None:
        return {
            "status": "Needs Correction",
            "turtle_neck": False,
            "neck_angle_deg": 0.0,
            "shoulder_tilt_deg": 0.0,
            "upper_body_tilt_deg": 0.0,
            "head_forward_z": 0.0,
            "forward_ratio": 0.0,
            "total_score": 0,
        }
    return build_api_result(metrics)


@lru_cache(maxsize=1)
def _get_image_landmarker() -> vision.PoseLandmarker:
    return create_pose_landmarker(
        str(default_model_path()),
        running_mode=vision.RunningMode.IMAGE,
    )


def extract_upper_body_coordinates(
    metrics: PostureMetrics,
) -> Dict[str, Tuple[int, int, float]]:
    coordinates = {
        name: (point.x, point.y, point.z) for name, point in metrics.points.items()
    }
    if metrics.face_center is not None:
        coordinates["face_center"] = (
            metrics.face_center.x,
            metrics.face_center.y,
            metrics.face_center.z,
        )
    if metrics.chest_center is not None:
        coordinates["chest_center"] = (
            metrics.chest_center.x,
            metrics.chest_center.y,
            metrics.chest_center.z,
        )
    if metrics.hip_center is not None:
        coordinates["hip_center"] = (
            metrics.hip_center.x,
            metrics.hip_center.y,
            metrics.hip_center.z,
        )
    if "mouth_left" in metrics.points and "mouth_right" in metrics.points:
        mouth_center = midpoint(metrics.points["mouth_left"], metrics.points["mouth_right"])
        coordinates["mouth_center"] = (mouth_center.x, mouth_center.y, mouth_center.z)
    return coordinates
