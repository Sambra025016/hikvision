"""目标位置到 PTZ 速度命令的映射与执行适配器。"""

import time

from motion_app.models import ControlCommand, TrackState


class PTZDecisionController:
    """根据预测位置与画面中心误差产生带有效期的速度命令。"""

    def __init__(self, deadband_x_ratio=0.08, deadband_y_ratio=0.10,
                 pan_gain=35.0, tilt_gain=35.0, minimum_speed=3,
                 maximum_speed=20, command_ttl=0.30,
                 allow_prediction_seconds=0.30):
        self.deadband_x_ratio = float(deadband_x_ratio)
        self.deadband_y_ratio = float(deadband_y_ratio)
        self.pan_gain = float(pan_gain)
        self.tilt_gain = float(tilt_gain)
        self.minimum_speed = int(minimum_speed)
        self.maximum_speed = int(maximum_speed)
        self.command_ttl = float(command_ttl)
        self.allow_prediction_seconds = float(allow_prediction_seconds)

    def _axis_speed(self, error_ratio, deadband, gain):
        if abs(error_ratio) <= deadband:
            return 0
        magnitude = int(round(abs(error_ratio) * gain))
        magnitude = max(self.minimum_speed, magnitude)
        magnitude = min(self.maximum_speed, magnitude)
        return magnitude if error_ratio > 0 else -magnitude

    def stop_command(self, now, track_id=None, reason="lost"):
        return ControlCommand(
            command_time=now,
            track_id=track_id,
            pan=0,
            tilt=0,
            zoom=0,
            reason=reason,
            valid_until=now + self.command_ttl,
        )

    def compute(self, track: TrackState, frame_size, now=None):
        """计算命令；无可靠轨迹时显式输出停止，而不是沿用旧命令。"""
        now = time.monotonic() if now is None else float(now)
        if not track.active or track.predicted_center is None:
            return self.stop_command(now, track.track_id, "lost")
        if (track.prediction_only and
                track.prediction_age > self.allow_prediction_seconds):
            return self.stop_command(now, track.track_id, "prediction_stale")

        width, height = frame_size
        center_x, center_y = width / 2.0, height / 2.0
        error_x = (track.predicted_center[0] - center_x) / center_x
        error_y = (track.predicted_center[1] - center_y) / center_y
        pan = self._axis_speed(
            error_x, self.deadband_x_ratio, self.pan_gain
        )
        # 图像 y 轴向下，而海康 PTZ tilt 正值代表向上。
        tilt = -self._axis_speed(
            error_y, self.deadband_y_ratio, self.tilt_gain
        )
        reason = "deadband" if pan == 0 and tilt == 0 else "tracking"
        return ControlCommand(
            command_time=now,
            track_id=track.track_id,
            pan=pan,
            tilt=tilt,
            zoom=0,
            reason=reason,
            valid_until=now + self.command_ttl,
        )


class DryRunPTZExecutor:
    """不连接球机，仅保存命令，用于离线打通控制数据流。"""

    def __init__(self):
        self.last_command = None
        self.command_count = 0

    def execute(self, command: ControlCommand):
        self.last_command = command
        self.command_count += 1
        return True

    def stop(self):
        return None


class HikvisionPTZExecutor:
    """真实球机执行适配器；第一版默认不启用。"""

    def __init__(self, controller, minimum_interval=0.10):
        self.controller = controller
        self.minimum_interval = float(minimum_interval)
        self._last_sent_at = 0.0
        self._last_vector = None

    def execute(self, command: ControlCommand):
        now = time.monotonic()
        if now > command.valid_until:
            self.controller.stop()
            self._last_vector = (0, 0, 0)
            return False
        vector = (command.pan, command.tilt, command.zoom)
        if (vector == self._last_vector and
                now - self._last_sent_at < self.minimum_interval):
            return True
        self.controller.continuous_move(
            pan=command.pan, tilt=command.tilt, zoom=command.zoom
        )
        self._last_vector = vector
        self._last_sent_at = now
        return True

    def stop(self):
        self.controller.stop()
        self._last_vector = (0, 0, 0)

