"""目标居中后的简化抓拍模块。"""

from pathlib import Path

import cv2

from motion_app.models import SnapshotResult


class CenterSnapshotManager:
    """连续居中、画面清晰且云台近似停止时，每条轨迹抓拍一次。"""

    def __init__(self, output_dir, enabled=True, center_x_ratio=0.12,
                 center_y_ratio=0.15, stable_frames=5,
                 min_sharpness=80.0, max_control_speed=3):
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self.center_x_ratio = float(center_x_ratio)
        self.center_y_ratio = float(center_y_ratio)
        self.stable_frames = int(stable_frames)
        self.min_sharpness = float(min_sharpness)
        self.max_control_speed = int(max_control_speed)
        self._track_id = None
        self._centered_frames = 0
        self._captured_track_ids = set()

    @staticmethod
    def _sharpness(frame, bbox):
        height, width = frame.shape[:2]
        x, y, box_width, box_height = bbox
        x1, y1 = max(0, x), max(0, y)
        x2 = min(width, x + box_width)
        y2 = min(height, y + box_height)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _source_bbox(processed, bbox, source_shape):
        source_height, source_width = source_shape[:2]
        x, y, width, height = bbox
        mapped = (
            int(round(x * processed.scale_x)),
            int(round(y * processed.scale_y)),
            int(round(width * processed.scale_x)),
            int(round(height * processed.scale_y)),
        )
        x, y, width, height = mapped
        x = max(0, min(source_width - 1, x))
        y = max(0, min(source_height - 1, y))
        width = max(1, min(source_width - x, width))
        height = max(1, min(source_height - y, height))
        return x, y, width, height

    def update(self, packet, processed, observation, track, command):
        if not self.enabled:
            return SnapshotResult(False, None, 0.0, 0, "disabled")
        if not track.active or track.track_id is None or track.bbox is None:
            self._track_id = None
            self._centered_frames = 0
            return SnapshotResult(False, None, 0.0, 0, "no_track")
        if track.track_id != self._track_id:
            self._track_id = track.track_id
            self._centered_frames = 0
        if track.track_id in self._captured_track_ids:
            return SnapshotResult(
                False, None, 0.0, self._centered_frames, "already_captured"
            )
        if observation is None or not observation.visible:
            self._centered_frames = 0
            return SnapshotResult(False, None, 0.0, 0, "prediction_only")

        frame_height, frame_width = processed.display_frame.shape[:2]
        center = track.filtered_center
        centered = (
            abs(center[0] - frame_width / 2.0) <=
            frame_width * self.center_x_ratio and
            abs(center[1] - frame_height / 2.0) <=
            frame_height * self.center_y_ratio
        )
        slow = max(abs(command.pan), abs(command.tilt)) <= self.max_control_speed
        if not centered or not slow:
            self._centered_frames = 0
            reason = "not_centered" if not centered else "camera_moving"
            return SnapshotResult(False, None, 0.0, 0, reason)

        self._centered_frames += 1
        if self._centered_frames < self.stable_frames:
            return SnapshotResult(
                False, None, 0.0, self._centered_frames, "stabilizing"
            )

        source_bbox = self._source_bbox(
            processed, track.bbox, packet.bgr_frame.shape
        )
        sharpness = self._sharpness(packet.bgr_frame, source_bbox)
        if sharpness < self.min_sharpness:
            return SnapshotResult(
                False, None, sharpness, self._centered_frames, "not_sharp"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "track_{:03d}_frame_{:06d}.jpg".format(
            track.track_id, packet.frame_id
        )
        saved = cv2.imwrite(str(path), packet.bgr_frame)
        if not saved:
            return SnapshotResult(
                False, None, sharpness, self._centered_frames, "write_failed"
            )
        self._captured_track_ids.add(track.track_id)
        return SnapshotResult(
            True, str(path), sharpness, self._centered_frames, "captured"
        )

