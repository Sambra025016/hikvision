"""项目的 Python 打包配置。"""

from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).parent


setup(
    name="motion-detect-ptz",
    version="0.1.0",
    description="用于运动目标跟踪项目的海康威视云台控制模块",
    long_description=(PROJECT_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "requests>=2.21.0,<3",
        "numpy==1.21.6",
        "opencv-python==4.8.1.78",
        "matplotlib==3.5.3",
    ],
    python_requires=">=3.7",
    license="MIT",
    zip_safe=False,
)
