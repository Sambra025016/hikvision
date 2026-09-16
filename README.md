# 运动目标检测与球机控制项目

本目录提供海康威视球机 ISAPI 控制，以及“视频采集、预处理、全局运动补偿、两帧差分、目标关联、卡尔曼跟踪、PTZ 决策、居中抓拍和指标记录”的第一版完整链路。

已在海康威视 `DS-2DC2402IW-DE3 S6B` 上验证 Digest 认证、云台能力查询、连续运动和停止命令。

## 环境与启动

Windows 下可直接使用 `start.bat`。脚本会创建或复用 `.venv`，并在缺少依赖时安装当前项目。

```powershell
# 查看命令帮助
.\start.bat --help

# 只读取云台能力，不会转动摄像机
.\start.bat ptz-check --host 192.168.1.64 --username admin

# 诊断 ISAPI 和账号认证
.\start.bat diagnose --host 192.168.1.64 --username admin
```

未提供密码时，程序会使用安全密码输入提示；输入过程中终端不会显示字符，这是正常行为。

也可以通过环境变量设置常用连接参数：

```powershell
$env:HIKVISION_HOST = "192.168.1.64"
$env:HIKVISION_USERNAME = "admin"
$env:HIKVISION_PASSWORD = "摄像头密码"
.\start.bat ptz-check
```

## 手动云台控制

速度范围为 `1～100`，单次动作持续时间限制为 `0.1～3.0` 秒。动作完成或程序中断时，程序会尽力发送停止命令。

```powershell
.\start.bat left --host 192.168.1.64 --username admin --speed 5 --duration 0.2
.\start.bat right --host 192.168.1.64 --username admin --speed 5 --duration 0.2
.\start.bat up --host 192.168.1.64 --username admin --speed 5 --duration 0.2
.\start.bat down --host 192.168.1.64 --username admin --speed 5 --duration 0.2
.\start.bat zoom-in --host 192.168.1.64 --username admin --speed 5 --duration 0.2
.\start.bat zoom-out --host 192.168.1.64 --username admin --speed 5 --duration 0.2
.\start.bat stop --host 192.168.1.64 --username admin
```

## 目标跟踪链路

没有摄像机时，先使用内置合成目标验证整条数据流。默认只生成 PTZ 命令，不会连接或转动摄像机：

```powershell
.\start_detection.bat --synthetic
```

也可以使用本地视频：

```powershell
.\start_detection.bat --video D:\videos\test.mp4
```

球机恢复连接后，先保持 dry-run，只读取通道 `102` 子码流并记录控制命令：

```powershell
.\start_detection.bat --rtsp --host 192.168.1.64 --username admin
```

只有验证检测、跟踪和方向均正确后，才显式加入 `--enable-ptz`：

```powershell
.\start_detection.bat --rtsp --enable-ptz --host 192.168.1.64 --username admin
```

详细步骤及参数说明见上一级目录的《第一版代码使用与验证说明.md》。

## 当前目录职责

- `hikvision/`：摄像机 ISAPI 云台控制库。
- `main.py`：设备诊断及手动云台控制入口。
- `motion_app/stream.py`：RTSP 最新帧采集、断线状态和自动重连。
- `motion_app/preprocess.py`：缩放、灰度化、降噪和有效 ROI。
- `motion_app/models.py`：模块间统一数据对象。
- `motion_app/compensation.py`：恒等或 ORB 全局运动补偿。
- `motion_app/temporal_difference/`：两帧差分候选提取。
- `motion_app/selector.py`、`tracker.py`：候选关联与卡尔曼跟踪。
- `motion_app/controller.py`：PTZ 决策、dry-run 和真实执行适配器。
- `motion_app/snapshot.py`、`metrics.py`：居中抓拍和实验记录。
- `configs/tracking.json`：集中存放可实时调整的算法参数。
- `outputs/`：按时间创建实验目录，保存视频、抓拍、CSV 和曲线。
- `tests/`：不连接真实摄像机的单元测试。

完整设计见上一级目录中的《课题B_运动目标检测与球机跟踪实现方案.md》和《球机运动场景下的帧间差分补偿实现.md》。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

测试使用合成图像和模拟 HTTP 响应，不会连接或转动真实摄像机。
