import cv2
import os
import numpy as np
from ultralytics import YOLO

# 模型加载
model = YOLO("yolov8n.pt")

# 参考文件路径
reference_face_file = "my_face.jpg"
calibration_file = "calibration.txt"

# 参考参数
REFERENCE_HEIGHT = 172  # 参考身高（厘米）

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 初始化特征检测器
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# 参考人脸特征
reference_face_features = None

def load_or_create_reference_face():
    """加载或创建参考人脸特征"""
    global reference_face_features
    
    if os.path.exists(reference_face_file):
        # 加载参考图片
        ref_img = cv2.imread(reference_face_file)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        # 检测人脸
        faces = face_cascade.detectMultiScale(ref_gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = ref_gray[y:y+h, x:x+w]
            # 提取特征
            keypoints, descriptors = sift.detectAndCompute(face_roi, None)
            if descriptors is not None:
                reference_face_features = descriptors
                print("已加载参考人脸特征")
                return True
    else:
        print("⚠️ 未找到参考人脸图片，请提供一张清晰的正面照片")
        # 打开摄像头拍照
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 显示预览
            cv2.imshow("Take Reference Photo (Press SPACE to capture)", frame)
            
            # 按空格键拍照
            if cv2.waitKey(1) & 0xFF == ord(' '):
                # 保存图片
                cv2.imwrite(reference_face_file, frame)
                print("已保存参考照片")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 处理新保存的照片
        ref_img = cv2.imread(reference_face_file)
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(ref_gray, 1.1, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = ref_gray[y:y+h, x:x+w]
            keypoints, descriptors = sift.detectAndCompute(face_roi, None)
            if descriptors is not None:
                reference_face_features = descriptors
                print("已提取参考人脸特征")
                return True
    
    print("无法提取参考人脸特征")
    return False

def detect_face(frame):
    """检测人脸并返回位置"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces

def is_reference_face(face_roi):
    """判断是否匹配参考人脸"""
    if reference_face_features is None:
        return False
        
    # 提取当前人脸特征
    keypoints, descriptors = sift.detectAndCompute(face_roi, None)
    if descriptors is None:
        return False
        
    # 特征匹配
    matches = bf.knnMatch(reference_face_features, descriptors, k=2)
    
    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    # 如果好的匹配点足够多，认为是同一个人
    return len(good_matches) > 10

def calibrate_with_reference_height(person_height_pixels):
    """使用参考身高进行标定"""
    if person_height_pixels > 0:
        # 计算像素到厘米的比例
        pixel_to_cm_ratio = REFERENCE_HEIGHT / person_height_pixels
        
        # 保存标定结果
        with open(calibration_file, "w", encoding="utf-8") as f:
            f.write(str(pixel_to_cm_ratio))
            
        return True, pixel_to_cm_ratio
    
    return False, None

# 读取或初始化换算比例
if os.path.exists(calibration_file):
    with open(calibration_file, "r", encoding="utf-8") as f:
        pixel_to_cm_ratio = float(f.read())
    print(f"已加载标定值:1 像素 ≈ {pixel_to_cm_ratio:.4f} cm")
    calibrated = True
else:
    pixel_to_cm_ratio = None
    calibrated = False
    print("等待识别参考人脸进行标定...")

# 加载参考人脸特征
face_recognized = load_or_create_reference_face()

# 打开摄像头
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r"C:\Users\30907\OneDrive\桌面\1.mp4")

# 创建窗口
cv2.namedWindow("Height Estimation", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头，请检查连接。")
        break

    # 执行人体检测
    results = model(frame)[0]
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

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
            persons.append((x1, y1, x2, y2, bbox_height, bbox_width))

    # 显示人数统计
    cv2.putText(frame, f"检测到 {len(persons)} 人", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 检测人脸
    faces = detect_face(frame)
    
    # 在每个人体检测框内查找人脸
    for i, (x1, y1, x2, y2, bbox_height, bbox_width) in enumerate(persons, 1):
        # 检查是否有人脸在这个人体框内
        has_face = False
        is_you = False
        for (fx, fy, fw, fh) in faces:
            # 如果人脸中心点在人体框内
            face_center_x = fx + fw//2
            face_center_y = fy + fh//2
            if (x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2):
                has_face = True
                # 提取人脸区域
                face_roi = cv2.cvtColor(frame[fy:fy+fh, fx:fx+fw], cv2.COLOR_BGR2GRAY)
                # 判断是否是参考人脸
                if face_recognized and is_reference_face(face_roi):
                    is_you = True
                    # 如果未标定，使用这个人的身高进行标定
                    if not calibrated:
                        success, ratio = calibrate_with_reference_height(bbox_height)
                        if success:
                            pixel_to_cm_ratio = ratio
                            calibrated = True
                            print(f"标定成功:1 像素 ≈ {pixel_to_cm_ratio:.4f} cm(已保存)")
                    # 绘制人脸框
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                    cv2.putText(frame, "You", (fx, fy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 绘制其他人脸框
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 2)
                    cv2.putText(frame, "Face", (fx, fy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                break

        # 绘制人体边界框和标签
        if is_you:
            color = (0, 0, 255)  # 红色表示是你
            # 计算并显示身高
            if calibrated:
                estimated_height_cm = bbox_height * pixel_to_cm_ratio
                label = f"Person {i}: {estimated_height_cm:.1f} cm [You]"
            else:
                label = f"Person {i} [You]"
        elif has_face:
            color = (0, 255, 255)  # 黄色表示其他人
            if calibrated:
                estimated_height_cm = bbox_height * pixel_to_cm_ratio
                label = f"Person {i}: {estimated_height_cm:.1f} cm [Face]"
            else:
                label = f"Person {i} [Face]"
        else:
            color = (0, 255, 0)  # 绿色表示未检测到人脸
            if calibrated:
                estimated_height_cm = bbox_height * pixel_to_cm_ratio
                label = f"Person {i}: {estimated_height_cm:.1f} cm"
            else:
                label = f"Person {i}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 确保标签不会超出图像边界
        label_y = max(y1 - 10, 30)
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示画面
    cv2.imshow("Height Estimation", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 