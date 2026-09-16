"""实验数据记录模块。"""

import csv
from datetime import datetime
from pathlib import Path


METRIC_FIELDS = (
    "frame_id", "capture_time", "processing_ms", "detection_ok",
    "detection_count", "changed_pixel_ratio", "compensation_mode",
    "compensation_ok", "inlier_ratio", "overlap_ratio", "track_id",
    "visible", "prediction_only", "measured_x", "measured_y",
    "filtered_x", "filtered_y", "predicted_x", "predicted_y",
    "center_error_x", "center_error_y", "velocity_x", "velocity_y",
    "miss_count", "confidence",
    "pan", "tilt", "zoom", "command_reason", "captured",
    "sharpness",
)


class MetricsRecorder:
    """每帧写入一行 CSV，作为误差曲线和验收统计的统一数据源。"""

    def __init__(self, output_root):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = Path(output_root) / stamp
        self.log_dir = self.session_dir / "logs"
        self.capture_dir = self.session_dir / "captures"
        self.plot_dir = self.session_dir / "plots"
        for directory in (self.log_dir, self.capture_dir, self.plot_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "tracking.csv"
        self._stream = self.csv_path.open("w", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._stream, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()
        self._plot_rows = []
        self._first_capture_time = None

    @staticmethod
    def _xy(point):
        return ("", "") if point is None else point

    def record(self, result, processing_ms):
        measured_x, measured_y = self._xy(result.track.measured_center)
        filtered_x, filtered_y = self._xy(result.track.filtered_center)
        predicted_x, predicted_y = self._xy(result.track.predicted_center)
        quality = result.compensation.quality
        visible = bool(result.observation and result.observation.visible)
        height, width = result.processed.display_frame.shape[:2]
        if result.track.predicted_center is None:
            error_x, error_y = "", ""
        else:
            error_x = result.track.predicted_center[0] - width / 2.0
            error_y = result.track.predicted_center[1] - height / 2.0
        row = {
            "frame_id": result.packet.frame_id,
            "capture_time": "{:.6f}".format(result.packet.capture_time),
            "processing_ms": "{:.3f}".format(processing_ms),
            "detection_ok": int(result.detection.detection_ok),
            "detection_count": len(result.detection.detections),
            "changed_pixel_ratio": "{:.6f}".format(
                result.detection.quality.changed_pixel_ratio
            ),
            "compensation_mode": quality.get("mode", ""),
            "compensation_ok": int(result.compensation.compensation_ok),
            "inlier_ratio": quality.get("inlier_ratio", ""),
            "overlap_ratio": quality.get("overlap_ratio", ""),
            "track_id": result.track.track_id or "",
            "visible": int(visible),
            "prediction_only": int(result.track.prediction_only),
            "measured_x": measured_x,
            "measured_y": measured_y,
            "filtered_x": filtered_x,
            "filtered_y": filtered_y,
            "predicted_x": predicted_x,
            "predicted_y": predicted_y,
            "center_error_x": error_x,
            "center_error_y": error_y,
            "velocity_x": result.track.velocity[0],
            "velocity_y": result.track.velocity[1],
            "miss_count": result.track.miss_count,
            "confidence": result.track.confidence,
            "pan": result.command.pan,
            "tilt": result.command.tilt,
            "zoom": result.command.zoom,
            "command_reason": result.command.reason,
            "captured": int(result.snapshot.captured),
            "sharpness": result.snapshot.sharpness,
        }
        self._writer.writerow(row)
        self._stream.flush()
        if self._first_capture_time is None:
            self._first_capture_time = result.packet.capture_time
        self._plot_rows.append((
            result.packet.capture_time - self._first_capture_time,
            None if error_x == "" else error_x,
            None if error_y == "" else error_y,
            result.command.pan,
            result.command.tilt,
        ))

    def _save_plot(self):
        valid = [row for row in self._plot_rows if row[1] is not None]
        if not valid:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        times = [row[0] for row in valid]
        error_x = [row[1] for row in valid]
        error_y = [row[2] for row in valid]
        pan = [row[3] for row in valid]
        tilt = [row[4] for row in valid]
        figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(times, error_x, label="horizontal error")
        axes[0].plot(times, error_y, label="vertical error")
        axes[0].set_ylabel("center error (pixel)")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[1].plot(times, pan, label="pan command")
        axes[1].plot(times, tilt, label="tilt command")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("PTZ speed")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(self.plot_dir / "tracking_error.png", dpi=130)
        plt.close(figure)

    def close(self):
        if not self._stream.closed:
            self._stream.close()
        self._save_plot()
