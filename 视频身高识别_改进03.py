import cv2
import os
import numpy as np
from ultralytics import YOLO

# 模型加载
model = YOLO("yolov8n.pt")

# 校准文件路径
calibration_file = "calibration.txt"
reference_face_file = "reference_face.jpg"

# A4纸标准尺寸（厘米）
A4_WIDTH = 21.0
A4_HEIGHT = 29.7

# 参考参数
REFERENCE_HEIGHT = 170  # 参考身高（厘米）
REFERENCE_DISTANCE = 200  # 参考距离（厘米）

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
                print("✅ 已加载参考人脸特征")
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
                print("✅ 已保存参考照片")
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
                print("✅ 已提取参考人脸特征")
                return True
    
    print("❌ 无法提取参考人脸特征")
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

def detect_a4_paper(frame):
    """检测A4纸并返回其轮廓和变换矩阵"""
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 按面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        if area < 1000:  # 忽略太小的轮廓
            continue
            
        # 多边形近似
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # 确保是四边形
        if len(approx) == 4:
            # 计算长宽比
            rect = cv2.minAreaRect(contour)
            width = min(rect[1])
            height = max(rect[1])
            aspect_ratio = height / width
            
            # A4纸的长宽比约为1.414
            if 1.3 < aspect_ratio < 1.5:
                # 获取四个角点
                points = approx.reshape(4, 2)
                
                # 计算目标尺寸（像素）
                target_width = int(width)
                target_height = int(height)
                
                # 定义目标点（用于透视变换）
                dst_points = np.array([
                    [0, 0],
                    [target_width - 1, 0],
                    [target_width - 1, target_height - 1],
                    [0, target_height - 1]
                ], dtype="float32")
                
                # 计算透视变换矩阵
                M = cv2.getPerspectiveTransform(points.astype("float32"), dst_points)
                
                return contour, M, target_width, target_height
    
    return None, None, None, None

def calibrate_with_a4(frame):
    """使用A4纸进行标定"""
    contour, M, target_width, target_height = detect_a4_paper(frame)
    
    if contour is not None:
        # 计算像素到厘米的比例
        pixel_to_cm_ratio_width = A4_WIDTH / target_width
        pixel_to_cm_ratio_height = A4_HEIGHT / target_height
        
        # 使用平均值作为最终比例
        pixel_to_cm_ratio = (pixel_to_cm_ratio_width + pixel_to_cm_ratio_height) / 2
        
        # 保存标定结果
        with open(calibration_file, "w", encoding="utf-8") as f:
            f.write(str(pixel_to_cm_ratio))
            
        return True, pixel_to_cm_ratio, contour
    
    return False, None, None

# 读取或初始化换算比例
if os.path.exists(calibration_file):
    with open(calibration_file, "r", encoding="utf-8") as f:
        pixel_to_cm_ratio = float(f.read())
    print(f"📂 已加载标定值：1 像素 ≈ {pixel_to_cm_ratio:.4f} cm")
    calibrated = True
else:
    pixel_to_cm_ratio = None
    calibrated = False
    print("⚠️ 请将A4纸放在画面中，系统将自动完成标定。")

# 加载参考人脸特征
face_recognized = load_or_create_reference_face()

def estimate_distance(bbox_width):
    """根据检测框宽度估算距离"""
    if bbox_width == 0:
        return None
    # 使用相似三角形原理估算距离
    estimated_distance = (REFERENCE_HEIGHT * REFERENCE_DISTANCE) / (pixel_to_cm_ratio * bbox_width)
    return estimated_distance

def adjust_height_ratio(distance):
    """根据距离调整身高比例"""
    if distance is None:
        return pixel_to_cm_ratio
    # 距离越远，比例越小
    return pixel_to_cm_ratio * (REFERENCE_DISTANCE / distance)

# 打开摄像头
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"C:\Users\30907\OneDrive\桌面\1.mp4")

# 创建窗口
cv2.namedWindow("Height Estimation", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头，请检查连接。")
        break

    # 如果未标定，尝试使用A4纸进行标定
    if not calibrated:
        success, ratio, contour = calibrate_with_a4(frame)
        if success:
            pixel_to_cm_ratio = ratio
            calibrated = True
            print(f"✅ 标定成功:1 像素 ≈ {pixel_to_cm_ratio:.4f} cm(已保存)")
            # 在图像上绘制A4纸轮廓
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, "A4纸检测成功", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "请将A4纸放在画面中", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 如果已标定，执行人体检测和身高估算
    if calibrated:
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
            # 估算距离
            distance = estimate_distance(bbox_width)
            # 根据距离调整比例
            adjusted_ratio = adjust_height_ratio(distance)
            # 计算身高
            estimated_height_cm = bbox_height * adjusted_ratio

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
            elif has_face:
                color = (0, 255, 255)  # 黄色表示其他人
            else:
                color = (0, 255, 0)  # 绿色表示未检测到人脸
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 显示详细信息
            label = f"Person {i}: {estimated_height_cm:.1f} cm"
            if distance:
                label += f" (距离: {distance:.1f} cm)"
            if is_you:
                label += " [You]"
            elif has_face:
                label += " [Face]"
            
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