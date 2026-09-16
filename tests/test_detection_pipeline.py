"""测试 RTSP 地址、预处理和两帧差分的接口契约。"""

import time
import unittest

import cv2
import numpy as np

from motion_app.models import FramePacket
from motion_app.preprocess import FramePreprocessor
from motion_app.stream import build_hikvision_rtsp_url
from motion_app.temporal_difference.detector import (
    IdentityFramePairBuilder,
    MotionDetector,
)


class TestDetectionPipeline(unittest.TestCase):
    def test_rtsp_url_uses_substream_and_escapes_credentials(self):
        url = build_hikvision_rtsp_url(
            "192.168.1.64", "user@example", "p@ss:word", stream=2
        )
        self.assertEqual(
            "rtsp://user%40example:p%40ss%3Aword@192.168.1.64:554/"
            "Streaming/Channels/102",
            url,
        )

    def test_preprocessor_preserves_frame_identity_and_scale(self):
        source = np.zeros((720, 1280, 3), dtype=np.uint8)
        packet = FramePacket(7, time.monotonic(), source, 1280, 720, True)

        result = FramePreprocessor(target_size=(640, 360)).process(packet)

        self.assertEqual(7, result.frame_id)
        self.assertEqual((360, 640), result.gray_frame.shape)
        self.assertEqual(2.0, result.scale_x)
        self.assertEqual(2.0, result.scale_y)
        self.assertEqual(640 * 360, cv2.countNonZero(result.valid_roi))

    def test_two_frame_difference_returns_structured_detection(self):
        preprocessor = FramePreprocessor(target_size=(160, 90))
        pair_builder = IdentityFramePairBuilder()
        detector = MotionDetector(
            diff_thresh=10,
            min_area=20,
            min_area_ratio=0,
            min_fill=0.05,
            open_kernel=(1, 1),
            close_kernel=(3, 3),
            dilate_iterations=0,
        )
        first = np.zeros((90, 160, 3), dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(second, (50, 20), (90, 70), (255, 255, 255), -1)

        first_packet = FramePacket(1, 1.0, first, 160, 90, True)
        second_packet = FramePacket(2, 1.1, second, 160, 90, True)
        warmup = detector.detect(
            pair_builder.build(preprocessor.process(first_packet))
        )
        result = detector.detect(
            pair_builder.build(preprocessor.process(second_packet))
        )

        self.assertFalse(warmup.detection_ok)
        self.assertEqual("waiting_for_previous_frame", warmup.reason)
        self.assertTrue(result.detection_ok)
        self.assertEqual(2, result.source_frame_id)
        self.assertGreaterEqual(len(result.detections), 1)
        self.assertEqual(2, result.detections[0].source_frame_id)
        self.assertGreater(result.quality.changed_pixel_ratio, 0)


if __name__ == "__main__":
    unittest.main()
