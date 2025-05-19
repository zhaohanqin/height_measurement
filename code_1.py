# main_project.py

import cv2
import numpy as np
import face_recognition  # 用于自我识别
import os

# --- 配置 ---
# 1. 人物检测
# 初始化HOG描述符/人物检测器
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# 2. 身高测量
# 重要提示：需要校准此值！
# 在距离相机已知且典型的距离处测量一个已知高度的物体（例如，你自己）。
# 每厘米像素数 = （图像中物体的像素高度） / （物体的实际高度（厘米））
# 示例：如果你身高170厘米，并且在2米的距离处你的图像高度为340像素：
# 每厘米像素数 = 340 / 170 = 2.0（在该距离下每厘米的像素数）
# 此值至关重要，并且高度依赖于你的相机、分辨率和距离。
# 你必须为你的设置进行校准。
PIXELS_PER_CM = 2.0  # 占位值 - 需要校准！
REFERENCE_DISTANCE_FOR_CALIBRATION_CM = 200  # 示例：在2米处进行校准

# （可选）用于稍微更高级的身高估计，尝试考虑距离因素：
# 假设一个普通人的肩宽（或脸的高度）用于距离估计。
# 这仍然是一个近似值。
KNOWN_PERSON_WIDTH_CM = 40.0  # 平均肩宽（厘米，近似值）
FOCAL_LENGTH_PIXELS = 700  # 近似值，如果你使用此方法，可能也需要校准此值

# 3. 自我识别
KNOWN_SELF_IMAGE_PATH = "self.jpg"  # 将一张名为 "self.jpg" 的你自己的照片放在同一目录下
known_self_encodings = []
known_self_names = ["你的名字"]  # 或者简单地用 "自己"

if os.path.exists(KNOWN_SELF_IMAGE_PATH):
    try:
        self_image = face_recognition.load_image_file(KNOWN_SELF_IMAGE_PATH)
        # 确保在参考图像中找到人脸
        self_face_encodings_list = face_recognition.face_encodings(self_image)
        if self_face_encodings_list:
            known_self_encodings.append(self_face_encodings_list[0])
            print(f"成功加载并编码参考图像: {KNOWN_SELF_IMAGE_PATH}")
        else:
            print(f"错误: 在 {KNOWN_SELF_IMAGE_PATH} 中未找到人脸。自我识别将无法工作。")
            KNOWN_SELF_IMAGE_PATH = None  # 禁用自我识别
    except Exception as e:
        print(f"错误: 无法加载或处理 {KNOWN_SELF_IMAGE_PATH}: {e}")
        KNOWN_SELF_IMAGE_PATH = None  # 禁用自我识别
else:
    print(f"警告: 未找到自我图像 '{KNOWN_SELF_IMAGE_PATH}'。自我识别将被禁用。")
    KNOWN_SELF_IMAGE_PATH = None  # 禁用自我识别


# --- 主要处理函数 ---
def process_frame(frame):
    # 为了提高性能，可以调整帧的大小，但这会影响像素测量
    frame_display = frame.copy()  # 在副本上进行绘制操作
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # HOG在灰度图像上可能效果更好

    # 1. 人物检测和计数
    # detectMultiScale(图像, 窗口步长, 填充, 缩放比例, 是否使用均值漂移分组)
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    persons = []
    person_count = 0
    detected_persons_info = []

    for (x, y, w, h) in rects:
        # 过滤掉不太可能是人的小检测结果
        # 以及如果参数设置不当可能是整个场景的非常大的检测结果
        if h > 50 and h < frame.shape[0] * 0.90:  # 最小高度50像素，最大高度为帧高度的90%
            persons.append((x, y, w, h))
            person_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 2. 身高测量
            pixel_height = h

            # 简单方法（使用在参考距离处预先校准的每厘米像素数）
            estimated_height_cm_simple = pixel_height / PIXELS_PER_CM

            # # 稍微更高级的方法（基于假设宽度的三角形相似性）
            # # 这需要 FOCAL_LENGTH_PIXELS 有一定的准确性
            # # 并假设检测到的宽度 'w' 对应于 KNOWN_PERSON_WIDTH_CM
            # estimated_distance_cm = (KNOWN_PERSON_WIDTH_CM * FOCAL_LENGTH_PIXELS) / w if w > 0 else REFERENCE_DISTANCE_FOR_CALIBRATION_CM
            # estimated_height_cm_advanced = (pixel_height * estimated_distance_cm) / FOCAL_LENGTH_PIXELS if FOCAL_LENGTH_PIXELS > 0 else estimated_height_cm_simple

            # 选择要显示的身高（简单方法更易于上手）
            final_estimated_height_cm = estimated_height_cm_simple

            height_text = f"{final_estimated_height_cm:.1f} 厘米"
            cv2.putText(frame, height_text, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            detected_persons_info.append({
                "bbox": (x, y, w, h),
                "height_cm": final_estimated_height_cm
            })

    # 显示人物数量
    cv2.putText(frame, f"人物数量: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 3. 自我识别（如果参考图像已加载）
    if KNOWN_SELF_IMAGE_PATH and known_self_encodings:
        # 将帧从BGR（OpenCV默认格式）转换为RGB（face_recognition使用的格式）
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在当前帧中找到所有的人脸位置和人脸编码
        # 模型可以是 "hog"（速度快，准确性较低）或 "cnn"（速度慢，准确性较高，需要带有CUDA的dlib以使用GPU）
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings_in_frame = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings_in_frame, face_locations):
            # 查看人脸是否与已知人脸匹配
            # matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            # 容差：值越低意味着更严格。默认值为0.6。
            if known_self_encodings:  # 检查列表是否不为空
                matches = face_recognition.compare_faces(known_self_encodings, face_encoding, tolerance=0.55)
                name = "未知"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_self_names[first_match_index]  # 应该是 "你的名字" 或 "自己"

                    # 在人脸周围画一个框并标注
                    top, right, bottom, left = face_location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # 如果这张脸属于检测到的人物之一，则在检测到的人物信息上标记
                    # 这是一个简单的空间检查：人脸的中心是否在人物的边界框内？
                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    for p_info in detected_persons_info:
                        px, py, pw, ph = p_info["bbox"]
                        if px < face_center_x < px + pw and py < face_center_y < py + ph:
                            p_info["is_self"] = True
                            cv2.putText(frame, "<- 自己", (px + pw, py + ph // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                            break  # 假设只有一个自己

    return frame, person_count, detected_persons_info

# --- 主执行部分 ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0表示默认网络摄像头

    if not cap.isOpened():
        print("错误: 无法打开网络摄像头。")
        exit()

    print("按 'q' 键退出。")
    print("---")
    print("身高校准说明:")
    print(f"当前每厘米像素数为 {PIXELS_PER_CM}。")
    print("校准方法：让一个人站在已知距离（例如，2米）处。")
    print("在视频画面中，估计他们的像素高度（绿色框的 'h' 值）。")
    print("然后，每厘米像素数 = （像素高度） / （实际身高（厘米））。")
    print("在脚本中更新 PIXELS_PER_CM 变量。")
    print("身高测量的准确性在很大程度上取决于此值和一致的距离。")
    print("---")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法接收帧（流结束？）。正在退出...")
            break

        processed_frame, p_count, p_info = process_frame(frame.copy())  # 传递副本

        cv2.imshow('数字图像处理作业', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()