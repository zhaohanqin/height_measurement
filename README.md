# 智能身高测量系统

这是一个基于计算机视觉的智能身高测量系统，使用 YOLOv8 进行人体检测，结合摄像头实现实时身高测量。

## 功能特点

- 实时人体检测和身高测量
- 自动校准功能
- 测量数据记录和导出
- 实时显示和统计
- 支持截图功能
- 数据可视化（图表显示）
- 调试模式

## 系统要求

- Python 3.8 或更高版本
- 摄像头设备
- CUDA 支持（可选，用于 GPU 加速）

## 安装说明

1. 克隆或下载本项目到本地

2. 安装依赖包：
```bash
pip install -r requirements.txt
```

3. 下载 YOLOv8 模型：
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
```

## 使用方法

1. 运行主程序：
```bash
python height_measurement_advanced.py
```

2. 首次使用时需要进行校准：
   - 输入一个已知身高（厘米）
   - 确保画面中只有一个人，且身体完全可见
   - 系统会自动完成校准

3. 主要功能：
   - 开始/停止记录：记录测量数据
   - 截图：保存当前画面
   - 导出数据：将测量数据导出为 CSV 文件
   - 调试模式：显示更多技术信息

## 注意事项

- 确保摄像头画面清晰，光线充足
- 测量时保持适当距离（建议 2-3 米）
- 校准值会保存在 calibration.txt 文件中
- 配置信息保存在 config.json 文件中

## 文件说明

- `height_measurement_advanced.py`: 主程序文件
- `requirements.txt`: 依赖包列表
- `calibration.txt`: 校准数据
- `config.json`: 配置文件

## 技术栈

- OpenCV (cv2)
- YOLOv8
- NumPy
- Tkinter
- Matplotlib 