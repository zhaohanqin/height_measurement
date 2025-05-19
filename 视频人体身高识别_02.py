import cv2
import os
import numpy as np
from collections import deque
from ultralytics import YOLO

# 配置参数
CONF_THRESH = 0.4  # 从0.2提高到0.4
MAX_HISTORY = 5
CALIB_RANGE = (1.2, 3.5)  # 可适当放宽范围

# 模型加载
model = YOLO("yolov8n.pt")

# 校准文件路径
calibration_file = "calibration.txt"

# 读取或初始化换算比例
if os.path.exists(calibration_file):
    with open(calibration_file, "r", encoding="utf-8") as f:
        pixel_to_cm_ratio = float(f.read())
    print(f"已加载标定值：1 像素 ≈ {pixel_to_cm_ratio:.4f} cm")
    calibrated = True
else:
    pixel_to_cm_ratio = None
    calibrated = False
    real_height_cm = float(input("请输入用于标定的真实身高（单位 cm）："))
    print("请确保画面中只有一个完整人像，系统将自动完成标定。")

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 身高历史记录
height_history = deque(maxlen=MAX_HISTORY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头，请检查连接。")
        break

    # 增加调试窗口显示原始画面
    cv2.imshow("Original View", frame)

    results = model(frame)[0]
    frame_height = frame.shape[0]

    persons = []
    for result in results.boxes:
        conf = float(result.conf)  # 转换为float

        # 打印检测信息（调试用）
        print(f"检测到: {model.names[int(result.cls)]} 置信度: {conf:.2f}")

        if conf < CONF_THRESH:
            continue
            
        cls_id = int(result.cls)
        if model.names[cls_id] != "person":
            continue

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        aspect_ratio = bbox_height / bbox_width

        # 放宽检测条件
        if y2 >= frame_height - 50 and aspect_ratio > 1.5:  # 从frame_height-20改为-50，aspect_ratio从1.8改为1.5
            persons.append((x1, y1, x2, y2, bbox_height))
            print(f"符合条件的人像: 高度={bbox_height}, 宽高比={aspect_ratio:.2f}")

    # 标定逻辑
    if not calibrated and len(persons) == 1:
        _, _, _, _, bbox_height = persons[0]
        pixel_to_cm_ratio = real_height_cm / bbox_height
        
        # 标定值验证
        if CALIB_RANGE[0] <= pixel_to_cm_ratio <= CALIB_RANGE[1]:
            with open(calibration_file, "w", encoding="utf-8") as f:
                f.write(str(pixel_to_cm_ratio))
            print(f"标定成功：1 像素 ≈ {pixel_to_cm_ratio:.4f} cm（已保存）")
            calibrated = True
        else:
            print(f"标定值异常：{pixel_to_cm_ratio:.4f}，请重新标定")

    # 身高估算与显示
    if calibrated:
        for x1, y1, x2, y2, bbox_height in persons:
            estimated_height = bbox_height * pixel_to_cm_ratio
            height_history.append(estimated_height)
            avg_height = np.mean(height_history) if height_history else estimated_height
            
            label = f"{avg_height:.1f} cm"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示统计信息
            cv2.putText(frame, f"当前: {estimated_height:.1f}cm", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"平均: {avg_height:.1f}cm", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Height Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
