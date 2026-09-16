"""只读可视化模块。"""

import cv2


def draw_pipeline_result(frame, detection, track, command, compensation):
    """绘制候选框、跟踪状态、预测点和 dry-run PTZ 命令。"""
    view = frame.copy()
    for candidate in detection.detections:
        x, y, width, height = candidate.bbox
        cv2.rectangle(
            view, (x, y), (x + width, y + height), (0, 180, 0), 1
        )

    if track.active and track.bbox is not None:
        x, y, width, height = track.bbox
        color = (0, 165, 255) if track.prediction_only else (0, 0, 255)
        cv2.rectangle(
            view, (x, y), (x + width, y + height), color, 2
        )
        if track.measured_center is not None:
            point = tuple(int(round(value)) for value in track.measured_center)
            cv2.circle(view, point, 4, (0, 255, 0), -1)
        if track.filtered_center is not None:
            point = tuple(int(round(value)) for value in track.filtered_center)
            cv2.circle(view, point, 4, (255, 0, 0), -1)
        if track.predicted_center is not None:
            point = tuple(int(round(value)) for value in track.predicted_center)
            cv2.drawMarker(
                view, point, (0, 255, 255), cv2.MARKER_CROSS, 14, 2
            )

    height, width = view.shape[:2]
    cv2.drawMarker(
        view, (width // 2, height // 2), (255, 255, 255),
        cv2.MARKER_CROSS, 20, 1,
    )
    lines = [
        "frame={} detections={} track={}".format(
            detection.source_frame_id,
            len(detection.detections),
            track.track_id if track.track_id is not None else "-",
        ),
        "visible={} miss={} conf={:.2f}".format(
            not track.prediction_only and track.active,
            track.miss_count,
            track.confidence,
        ),
        "PTZ pan={} tilt={} reason={}".format(
            command.pan, command.tilt, command.reason
        ),
        "comp={} ok={}".format(
            compensation.quality.get("mode", ""),
            compensation.compensation_ok,
        ),
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            view, text, (10, 22 + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
            cv2.LINE_AA,
        )
    return view

