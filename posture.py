import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from angle import distance, line_tilt_degrees


# Literature-inspired frontal-camera rubric:
# - forward head proxy: face/chest depth offset (CVA surrogate)
# - shoulder balance: shoulder tilt + arm line balance
# - trunk balance: torso verticality + global asymmetry
# A webcam frontal view cannot reproduce a true sagittal CVA, so the neck score
# intentionally weights MediaPipe z-depth more heavily than 2D frontal angle.
VISIBILITY_THRESHOLD = 0.55
SHOULDER_Y_LIFT_RATIO = 0.06
SHOULDER_Y_LIFT_MIN_PX = 4
NECK_ANGLE_THRESHOLDS = (6.0, 10.0, 14.0)
NECK_ANGLE_PENALTIES = (0, 8, 16, 24)
FORWARD_HEAD_Z_THRESHOLDS = (0.05, 0.08, 0.11)
FORWARD_HEAD_Z_PENALTIES = (0, 12, 22, 35)
HEAD_BACK_Z_THRESHOLD = -0.035
SIDE_HEAD_OFFSET_RATIO_THRESHOLDS = (0.08, 0.14, 0.22)
SIDE_HEAD_OFFSET_RATIO_PENALTIES = (0, 12, 24, 36)
SIDE_NECK_ANGLE_THRESHOLDS = (8.0, 14.0, 20.0)
SIDE_NECK_ANGLE_PENALTIES = (0, 8, 18, 28)
SHOULDER_TILT_THRESHOLDS = (3.0, 6.0, 9.0)
SHOULDER_TILT_PENALTIES = (0, 6, 12, 18)
UPPER_BODY_TILT_THRESHOLDS = (4.0, 7.0, 10.0)
UPPER_BODY_TILT_PENALTIES = (0, 6, 12, 20)
ASYMMETRY_THRESHOLDS = (3.0, 6.0, 9.0)
ASYMMETRY_PENALTIES = (0, 6, 12, 20)
ARM_BALANCE_THRESHOLDS = (4.0, 8.0, 12.0)
ARM_BALANCE_PENALTIES = (0, 4, 8, 12)
TURTLE_NECK_NECK_ANGLE_THRESHOLD = 7.0
TURTLE_NECK_Z_THRESHOLD = 0.08
SIDE_TURTLE_NECK_RATIO_THRESHOLD = 0.14
SIDE_VIEW_SCORE_MARGIN = 1.5
ASYMMETRY_DETECTION_THRESHOLD = 6.0
TRACKING_SCORE_STEP = 8
TRACKING_SCORE_FHP_BONUS = 12
STRETCH_SCORE_THRESHOLD = 45
NECK_SCORE_WEIGHTS = (0.35, 0.65)  # frontal neck alignment, forward-head depth proxy
SHOULDER_SCORE_WEIGHTS = (0.7, 0.3)  # shoulder height balance, arm line balance
TRUNK_SCORE_WEIGHTS = (0.7, 0.3)  # trunk verticality, global left-right asymmetry
TOTAL_SCORE_WEIGHTS = {
    "neck": 0.4,
    "shoulder": 0.25,
    "trunk": 0.25,
    "tracking": 0.10,
}


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


def classify_posture(
    view_mode: str,
    total_score: int,
    tracking_score: int,
    turtle_neck: bool,
    asymmetry_detected: bool,
    visibility_ok: bool,
) -> str:
    if tracking_score < 55:
        return "트래킹 불안정"
    if turtle_neck or asymmetry_detected or total_score < 70:
        return "교정 필요"
    if total_score < 85:
        return "주의 필요"
    return "정상"


def build_feedback_message(
    view_mode: str,
    visibility_ok: bool,
    tracking_score: int,
    total_score: int,
    turtle_neck: bool,
    asymmetry_detected: bool,
    neck_angle_deg: Optional[float],
    shoulder_tilt_deg: float,
    upper_body_tilt_deg: Optional[float],
    head_forward_offset_z: Optional[float],
    side_head_forward_ratio: Optional[float],
    points: Dict[str, Point],
) -> str:
    if tracking_score < 55:
        return "어깨와 얼굴이 화면 안에 또렷하게 보이도록 위치를 맞춰주세요."

    feedback_parts = []

    if turtle_neck:
        if view_mode == "side" and side_head_forward_ratio is not None:
            if side_head_forward_ratio >= SIDE_HEAD_OFFSET_RATIO_THRESHOLDS[1]:
                feedback_parts.append(
                    "머리가 몸통보다 너무 앞으로 나와 있습니다. 턱을 당기고 귀가 어깨 바로 위에 오도록 맞춰주세요."
                )
            else:
                feedback_parts.append(
                    "측면에서 거북목 경향이 보입니다. 턱을 살짝 당기고 목 뒤를 길게 세워주세요."
                )
        elif head_forward_offset_z is not None and head_forward_offset_z >= FORWARD_HEAD_Z_THRESHOLDS[1]:
            feedback_parts.append(
                "머리가 상체보다 너무 앞으로 나와 있습니다. 턱을 당기고 뒤통수를 위로 끌어올리듯 정렬해주세요."
            )
        else:
            feedback_parts.append(
                "거북목 경향이 있습니다. 턱을 살짝 당기고 귀와 어깨를 수직에 가깝게 맞춰주세요."
            )
    elif head_forward_offset_z is not None and head_forward_offset_z <= HEAD_BACK_Z_THRESHOLD:
        feedback_parts.append(
            "머리가 몸통보다 뒤로 빠져 있습니다. 턱을 과하게 들지 말고 머리를 몸통 중심 위로 되돌려주세요."
        )

    if upper_body_tilt_deg is not None and upper_body_tilt_deg >= UPPER_BODY_TILT_THRESHOLDS[1]:
        feedback_parts.append(
            "상체가 한쪽으로 기울어져 있습니다. 골반과 어깨 중심을 다시 수직으로 맞춰주세요."
        )

    left_shoulder = points["left_shoulder"]
    right_shoulder = points["right_shoulder"]
    if shoulder_tilt_deg >= SHOULDER_TILT_THRESHOLDS[0]:
        if left_shoulder.y > right_shoulder.y:
            feedback_parts.append(
                "어깨 높이가 틀어졌습니다. 왼쪽 어깨를 조금 올리고 오른쪽 어깨 힘을 빼주세요."
            )
        elif right_shoulder.y > left_shoulder.y:
            feedback_parts.append(
                "어깨 높이가 틀어졌습니다. 오른쪽 어깨를 조금 올리고 왼쪽 어깨 힘을 빼주세요."
            )

    if not feedback_parts and asymmetry_detected:
        feedback_parts.append("고개와 어깨의 좌우 균형이 무너졌습니다. 정면에서 중심을 다시 맞춰주세요.")

    mild_posture_issue = (
        (neck_angle_deg is not None and neck_angle_deg >= NECK_ANGLE_THRESHOLDS[0])
        or shoulder_tilt_deg >= SHOULDER_TILT_THRESHOLDS[0]
        or (
            head_forward_offset_z is not None
            and head_forward_offset_z >= FORWARD_HEAD_Z_THRESHOLDS[0]
        )
    )
    if not feedback_parts and mild_posture_issue:
        feedback_parts.append("자세가 조금 흐트러졌습니다. 턱과 어깨를 편안하게 다시 정렬해보세요.")

    if total_score <= STRETCH_SCORE_THRESHOLD:
        feedback_parts.append("점수가 많이 떨어졌습니다. 잠깐 일어나 20초 정도 목과 어깨를 스트레칭해주세요.")

    if not feedback_parts:
        return "자세가 안정적입니다."
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

    shoulder_tilt_deg = line_tilt_degrees(
        points["left_shoulder"], points["right_shoulder"]
    )
    shoulder_penalty = penalty_from_thresholds(
        min(shoulder_tilt_deg, 90.0),
        SHOULDER_TILT_THRESHOLDS,
        SHOULDER_TILT_PENALTIES,
    )

    upper_body_tilt_deg = vertical_deviation_degrees(hip_mid, shoulder_mid)
    upper_body_penalty = penalty_from_thresholds(
        upper_body_tilt_deg if upper_body_tilt_deg is not None else 90.0,
        UPPER_BODY_TILT_THRESHOLDS,
        UPPER_BODY_TILT_PENALTIES,
    )

    neck_penalty = 0
    forward_head_penalty = 0
    cva_proxy_deg = None
    neck_angle_deg = None
    ear_shoulder_distance_px = None
    ear_shoulder_ratio = None
    head_forward_offset_z = None
    side_head_forward_ratio = None
    turtle_neck_score = None
    turtle_neck = False
    ear_tilt_deg = line_tilt_degrees(points["left_ear"], points["right_ear"])
    mouth_tilt_deg = line_tilt_degrees(points["mouth_left"], points["mouth_right"])
    elbow_tilt_deg = line_tilt_degrees(points["left_elbow"], points["right_elbow"])
    wrist_tilt_deg = line_tilt_degrees(points["left_wrist"], points["right_wrist"])
    arm_balance_deg = max(elbow_tilt_deg, wrist_tilt_deg)
    asymmetry_deg = max(shoulder_tilt_deg, ear_tilt_deg, mouth_tilt_deg, arm_balance_deg)
    asymmetry_penalty = penalty_from_thresholds(
        asymmetry_deg,
        ASYMMETRY_THRESHOLDS,
        ASYMMETRY_PENALTIES,
    )
    arm_balance_penalty = penalty_from_thresholds(
        arm_balance_deg,
        ARM_BALANCE_THRESHOLDS,
        ARM_BALANCE_PENALTIES,
    )
    asymmetry_detected = asymmetry_deg >= ASYMMETRY_DETECTION_THRESHOLD

    if view_mode == "front":
        neck_angle_deg = (
            vertical_deviation_degrees(chest_center, face_center)
            if face_center is not None
            else None
        )
        if neck_angle_deg is not None:
            neck_penalty = penalty_from_thresholds(
                neck_angle_deg,
                NECK_ANGLE_THRESHOLDS,
                NECK_ANGLE_PENALTIES,
            )

        ear_shoulder_distance_px = average_visible_distances(
            points,
            (
                ("left_ear", "left_shoulder"),
                ("right_ear", "right_shoulder"),
            ),
        )
        forward_head_measurement_ok = (
            face_center is not None
            and chest_center is not None
            and ear_shoulder_distance_px is not None
        )
        if forward_head_measurement_ok:
            shoulder_width = max(distance(points["left_shoulder"], points["right_shoulder"]), 1.0)
            ear_shoulder_ratio = ear_shoulder_distance_px / shoulder_width
            head_forward_offset_z = chest_center.z - face_center.z
            forward_head_penalty = penalty_from_thresholds(
                max(head_forward_offset_z, 0.0),
                FORWARD_HEAD_Z_THRESHOLDS,
                FORWARD_HEAD_Z_PENALTIES,
            )
            turtle_neck = (
                head_forward_offset_z >= TURTLE_NECK_Z_THRESHOLD
                or (
                    neck_angle_deg is not None
                    and neck_angle_deg >= TURTLE_NECK_NECK_ANGLE_THRESHOLD
                )
            )
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
            side_head_forward_ratio = safe_ratio(abs(ear_point.x - shoulder_point.x), torso_height_px)
            ear_shoulder_ratio = side_head_forward_ratio
            cva_proxy_deg = vertical_deviation_degrees(shoulder_point, ear_point)
            neck_angle_deg = cva_proxy_deg
            neck_penalty = penalty_from_thresholds(
                cva_proxy_deg if cva_proxy_deg is not None else 90.0,
                SIDE_NECK_ANGLE_THRESHOLDS,
                SIDE_NECK_ANGLE_PENALTIES,
            )
            forward_head_penalty = penalty_from_thresholds(
                side_head_forward_ratio,
                SIDE_HEAD_OFFSET_RATIO_THRESHOLDS,
                SIDE_HEAD_OFFSET_RATIO_PENALTIES,
            )
            upper_body_tilt_deg = vertical_deviation_degrees(hip_point, shoulder_point)
            upper_body_penalty = penalty_from_thresholds(
                upper_body_tilt_deg if upper_body_tilt_deg is not None else 90.0,
                UPPER_BODY_TILT_THRESHOLDS,
                UPPER_BODY_TILT_PENALTIES,
            )
            shoulder_tilt_deg = vertical_deviation_degrees(hip_point, shoulder_point) or 0.0
            shoulder_penalty = penalty_from_thresholds(
                min(shoulder_tilt_deg, 90.0),
                UPPER_BODY_TILT_THRESHOLDS,
                UPPER_BODY_TILT_PENALTIES,
            )
            asymmetry_deg = shoulder_tilt_deg
            asymmetry_penalty = penalty_from_thresholds(
                asymmetry_deg,
                UPPER_BODY_TILT_THRESHOLDS,
                UPPER_BODY_TILT_PENALTIES,
            )
            asymmetry_detected = asymmetry_deg >= UPPER_BODY_TILT_THRESHOLDS[1]
            arm_balance_deg = 0.0
            arm_balance_penalty = 0
            turtle_neck = (
                side_head_forward_ratio >= SIDE_TURTLE_NECK_RATIO_THRESHOLD
                or (
                    cva_proxy_deg is not None
                    and cva_proxy_deg >= SIDE_NECK_ANGLE_THRESHOLDS[1]
                )
            )
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

    neck_angle_score = (
        penalty_to_score(
            neck_penalty,
            SIDE_NECK_ANGLE_PENALTIES[-1] if view_mode == "side" else NECK_ANGLE_PENALTIES[-1],
        )
        if neck_angle_deg is not None
        else None
    )
    forward_head_score = (
        penalty_to_score(
            forward_head_penalty,
            SIDE_HEAD_OFFSET_RATIO_PENALTIES[-1] if view_mode == "side" else FORWARD_HEAD_Z_PENALTIES[-1],
        )
        if forward_head_measurement_ok
        else None
    )
    shoulder_tilt_score = penalty_to_score(
        shoulder_penalty,
        UPPER_BODY_TILT_PENALTIES[-1] if view_mode == "side" else SHOULDER_TILT_PENALTIES[-1],
    )
    arm_balance_score = penalty_to_score(
        arm_balance_penalty,
        ARM_BALANCE_PENALTIES[-1],
    )
    trunk_tilt_score = (
        penalty_to_score(upper_body_penalty, UPPER_BODY_TILT_PENALTIES[-1])
        if upper_body_tilt_deg is not None
        else None
    )
    asymmetry_score = penalty_to_score(
        asymmetry_penalty,
        ASYMMETRY_PENALTIES[-1],
    )

    neck_score = weighted_score(
        (
            (neck_angle_score, NECK_SCORE_WEIGHTS[0]),
            (forward_head_score, NECK_SCORE_WEIGHTS[1]),
        )
    )
    shoulder_score = weighted_score(
        (
            (shoulder_tilt_score, SHOULDER_SCORE_WEIGHTS[0]),
            (arm_balance_score, SHOULDER_SCORE_WEIGHTS[1]),
        )
    )
    trunk_score = weighted_score(
        (
            (trunk_tilt_score, TRUNK_SCORE_WEIGHTS[0]),
            (asymmetry_score, TRUNK_SCORE_WEIGHTS[1]),
        )
    )
    turtle_neck_score = neck_score if forward_head_measurement_ok else None

    total_score = int(
        round(
            neck_score * TOTAL_SCORE_WEIGHTS["neck"]
            + shoulder_score * TOTAL_SCORE_WEIGHTS["shoulder"]
            + trunk_score * TOTAL_SCORE_WEIGHTS["trunk"]
            + tracking_score * TOTAL_SCORE_WEIGHTS["tracking"]
        )
    )
    classification = classify_posture(
        view_mode,
        total_score,
        tracking_score,
        turtle_neck,
        asymmetry_detected,
        visibility_ok,
    )
    feedback_message = build_feedback_message(
        view_mode,
        visibility_ok,
        tracking_score,
        total_score,
        turtle_neck,
        asymmetry_detected,
        neck_angle_deg,
        shoulder_tilt_deg,
        upper_body_tilt_deg,
        head_forward_offset_z,
        side_head_forward_ratio,
        points,
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
