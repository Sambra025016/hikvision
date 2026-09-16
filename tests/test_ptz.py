"""测试 PTZ 能力解析和连续运动请求。"""

import unittest
from unittest.mock import Mock

from hikvision.ptz import PTZController


CAPABILITIES_XML = """
<PTZChanelCap xmlns="http://www.hikvision.com/ver20/XMLSchema">
  <ContinuousPanTiltSpace>
    <XRange><Min>-100</Min><Max>100</Max></XRange>
    <YRange><Min>-100</Min><Max>100</Max></YRange>
  </ContinuousPanTiltSpace>
  <ContinuousZoomSpace>
    <ZRange><Min>-100</Min><Max>100</Max></ZRange>
  </ContinuousZoomSpace>
</PTZChanelCap>
"""


class TestPTZController(unittest.TestCase):
    """验证云台控制器生成和解析的数据。"""

    def setUp(self):
        self.controller = PTZController(
            "192.168.1.64", "admin", "password"
        )

    def test_get_capabilities(self):
        """带命名空间的能力 XML 应能正确解析。"""
        self.controller._session.get = Mock(
            return_value=Mock(status_code=200, text=CAPABILITIES_XML)
        )

        result = self.controller.get_capabilities()

        self.assertEqual((-100, 100), result["pan"])
        self.assertEqual((-100, 100), result["tilt"])
        self.assertEqual((-100, 100), result["zoom"])
        self.assertFalse(result["continuous_not_supported"])

    def test_continuous_move(self):
        """连续运动请求应包含水平、垂直和变焦速度。"""
        self.controller._session.put = Mock(
            return_value=Mock(status_code=200, text="OK")
        )

        self.controller.continuous_move(pan=-10, tilt=20, zoom=0)

        payload = self.controller._session.put.call_args[1]["data"]
        payload = payload.decode("utf-8")
        self.assertIn("<pan>-10</pan>", payload)
        self.assertIn("<tilt>20</tilt>", payload)
        self.assertIn("<zoom>0</zoom>", payload)

    def test_rejects_invalid_speed(self):
        """超出设备范围的速度不能发送。"""
        with self.assertRaises(ValueError):
            self.controller.continuous_move(pan=101)


if __name__ == "__main__":
    unittest.main()
