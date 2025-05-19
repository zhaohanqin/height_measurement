import cv2
import os

from ultralytics import YOLO

# 模型加载
model = YOLO("yolov8n.pt")

# 校准文件路径
calibration_file = "calibration.txt"

# 读取或初始化换算比例
if os.path.exists(calibration_file):
    with open(calibration_file, "r", encoding="utf-8") as f:
        pixel_to_cm_ratio = float(f.read())
    print(f"📂 已加载标定值：1 像素 ≈ {pixel_to_cm_ratio:.4f} cm")
    calibrated = True
else:
    pixel_to_cm_ratio = None
    calibrated = False
    real_height_cm = float(input("请输入用于标定的真实身高（单位 cm）："))
    print("⚠️ 请确保画面中只有一个完整人像，系统将自动完成标定。")

# 打开摄像头
# cap = cv2.VideoCapture(0)
cap2=cv2.VideoCapture(r"C:\Users\30907\OneDrive\桌面\1.mp4")

while True:
    ret, frame = cap2.read()
    if not ret:
        print("无法读取摄像头，请检查连接。")
        break

    results = model(frame)[0]
    frame_height = frame.shape[0]

    persons = []
    for result in results.boxes:
        cls_id = int(result.cls)
        if model.names[cls_id] != "person":
            continue

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        aspect_ratio = bbox_height / bbox_width

        # 判断是否是完整人像
        if y2 >= frame_height - 20 and aspect_ratio > 1.8:
            persons.append((x1, y1, x2, y2, bbox_height))

    # 如果未标定，且只检测到一个完整人像
    if not calibrated and len(persons) == 1:
        _, _, _, _, bbox_height = persons[0]
        pixel_to_cm_ratio = real_height_cm / bbox_height
        with open(calibration_file, "w", encoding="utf-8") as f:
            f.write(str(pixel_to_cm_ratio))
        print(f"✅ 标定成功：1 像素 ≈ {pixel_to_cm_ratio:.4f} cm（已保存）")
        calibrated = True

    # 如果已标定，执行身高估算并绘图
    if calibrated:
        for x1, y1, x2, y2, bbox_height in persons:
            estimated_height_cm = bbox_height * pixel_to_cm_ratio
            label = f"{estimated_height_cm:.1f} cm"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示画面
    cv2.imshow("Height Estimation", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap2.release()
cv2.destroyAllWindows()
