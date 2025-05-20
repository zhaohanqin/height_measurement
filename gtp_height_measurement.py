import cv2
import face_recognition
from ultralytics import YOLO

# -----------------------------
# 配置和初始化
# -----------------------------
model = YOLO("yolov8n.pt")

# 输入你脸部的真实高度（从下巴到头顶）单位 cm
my_face_real_height_cm = float(input("请输入你脸部的实际高度（下巴到头顶，单位 cm）："))

# 加载你的人脸图像并提取编码
known_face_image = face_recognition.load_image_file("my_face.jpg")
known_face_encoding = face_recognition.face_encodings(known_face_image)[0]

# 打开摄像头
cap = cv2.VideoCapture(0)
pixel_to_cm_ratio = None  # 实时更新

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]  # 转为 RGB 用于人脸识别
    frame_height = frame.shape[0]

    # -----------------------------
    # Step 1: 人脸识别 -> 实时标定
    # -----------------------------
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_found = False
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match = face_recognition.compare_faces([known_face_encoding], face_encoding, tolerance=0.5)[0]
        if match:
            face_found = True
            face_height_px = bottom - top
            pixel_to_cm_ratio = my_face_real_height_cm / face_height_px
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)
            cv2.putText(frame, "我", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 255), 2)
            break  # 只取一张匹配到的脸

    # -----------------------------
    # Step 2: YOLO 检测人
    # -----------------------------
    results = model(frame)[0]
    persons = []

    for result in results.boxes:
        cls_id = int(result.cls)
        if model.names[cls_id] != "person":
            continue

        x1, y1, x2, y2 = map(int, result.xyxy[0])
        bbox_height = y2 - y1
        bbox_width = x2 - x1
        aspect_ratio = bbox_height / bbox_width

        # 只保留完整人像
        if y2 >= frame_height - 20 and aspect_ratio > 1.8:
            persons.append((x1, y1, x2, y2, bbox_height))

    # -----------------------------
    # Step 3: 显示结果
    # -----------------------------
    for x1, y1, x2, y2, bbox_height in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if pixel_to_cm_ratio is not None:
            height_cm = bbox_height * pixel_to_cm_ratio
            label = f"{height_cm:.1f} cm"
        else:
            label = "标定中..."

        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 显示人数统计
    cv2.putText(frame, f"人数: {len(persons)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    # 显示窗口
    cv2.imshow("动态标定身高估算系统", frame)

    # 按键控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
