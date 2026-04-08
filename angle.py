import math
from typing import Optional, Protocol


class XYPoint(Protocol):
    x: int
    y: int


def distance(a: XYPoint, b: XYPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def calculate_angle(a: XYPoint, b: XYPoint, c: XYPoint) -> Optional[float]:
    ab_x = a.x - b.x
    ab_y = a.y - b.y
    cb_x = c.x - b.x
    cb_y = c.y - b.y

    ab_mag = math.hypot(ab_x, ab_y)
    cb_mag = math.hypot(cb_x, cb_y)
    if ab_mag == 0 or cb_mag == 0:
        return None

    cosine = max(-1.0, min(1.0, (ab_x * cb_x + ab_y * cb_y) / (ab_mag * cb_mag)))
    return math.degrees(math.acos(cosine))


def line_tilt_degrees(start: XYPoint, end: XYPoint) -> float:
    return abs(math.degrees(math.atan2(end.y - start.y, end.x - start.x)))
