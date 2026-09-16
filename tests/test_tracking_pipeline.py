"""测试目标关联、卡尔曼跟踪、控制决策和完整合成数据流。"""

import tempfile
import unittest

from motion_app.config import load_config
from motion_app.controller import PTZDecisionController
from motion_app.models import Detection
from motion_app.pipeline import TrackingPipeline
from motion_app.selector import TargetSelector
from motion_app.sources import SyntheticFrameSource
from motion_app.tracker import KalmanTargetTracker


def make_detection(frame_id, center_x, center_y, width=30, height=60):
    bbox = (
        int(center_x - width / 2),
        int(center_y - height / 2),
        width,
        height,
    )
    return Detection(
        bbox=bbox,
        center=(float(center_x), float(center_y)),
        contour_area=float(width * height * 0.7),
        bbox_area=width * height,
        fill_ratio=0.7,
        motion_energy=0.8,
        source_frame_id=frame_id,
    )


class TestTrackingCore(unittest.TestCase):
    def test_tracker_updates_velocity_and_coasts_after_one_miss(self):
        selector = TargetSelector(association_gate=80)
        tracker = KalmanTargetTracker(selector, max_misses=3)

        _, first = tracker.update(1, 1.0, [make_detection(1, 100, 100)])
        observation, second = tracker.update(
            2, 1.1, [make_detection(2, 110, 100)]
        )
        predicted_observation, third = tracker.update(3, 1.2, [])

        self.assertTrue(first.active)
        self.assertTrue(observation.visible)
        self.assertGreater(second.velocity[0], 0)
        self.assertTrue(third.active)
        self.assertTrue(third.prediction_only)
        self.assertFalse(predicted_observation.visible)

    def test_controller_moves_toward_predicted_position(self):
        selector = TargetSelector(association_gate=80)
        tracker = KalmanTargetTracker(selector)
        _, state = tracker.update(1, 1.0, [make_detection(1, 500, 180)])
        command = PTZDecisionController().compute(
            state, frame_size=(640, 360), now=1.0
        )

        self.assertGreater(command.pan, 0)
        self.assertEqual(0, command.tilt)
        self.assertEqual("tracking", command.reason)


class TestSyntheticPipeline(unittest.TestCase):
    def test_synthetic_frames_cross_all_pipeline_modules(self):
        config = load_config()
        config["preprocess"]["width"] = 320
        config["preprocess"]["height"] = 180
        config["detector"]["min_area"] = 20
        config["detector"]["min_area_ratio"] = 0.0001
        config["snapshot"]["enabled"] = False
        source = SyntheticFrameSource(
            width=320, height=180, fps=15, duration_seconds=2
        )

        with tempfile.TemporaryDirectory() as directory:
            pipeline = TrackingPipeline(config, capture_dir=directory)
            results = []
            while True:
                packet = source.read()
                if packet is None:
                    break
                result = pipeline.process(packet)
                if result is not None:
                    results.append(result)
            pipeline.close()
            source.close()

        self.assertGreater(len(results), 10)
        self.assertTrue(any(item.detection.detection_ok for item in results))
        self.assertTrue(any(item.track.active for item in results))
        self.assertTrue(any(item.command.reason == "tracking" for item in results))


if __name__ == "__main__":
    unittest.main()
