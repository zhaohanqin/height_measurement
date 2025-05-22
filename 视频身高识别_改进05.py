import cv2
import os
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# 模型加载
model = YOLO("yolov8n.pt")
facenet = FaceNet()

# 参考文件路径
reference_face_file = "my_face.jpg"
calibration_file = "calibration.txt"

# 参考参数
REFERENCE_HEIGHT = 172  # 参考身高（厘米）
FACE_SIMILARITY_THRESHOLD = 0.7  # 人脸相似度阈值

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 参考人脸特征
reference_face_embedding = None
# 标定状态跟踪
last_calibration_time = 0  # 上次标定时间
calibration_interval = 1.0  # 标定间隔（秒）

# 人体检测参数
MIN_CONFIDENCE = 0.3  # 置信度阈值
MIN_HEIGHT_RATIO = 0.3  # 最小高度比例
MIN_ASPECT_RATIO = 1.2  # 最小宽高比
MAX_ASPECT_RATIO = 3.0  # 最大宽高比
MIN_BOTTOM_RATIO = 0.5  # 最小底部比例

def preprocess_face(face_img):
    """预处理人脸图像"""
    # 调整大小为FaceNet所需的输入尺寸
    face_img = cv2.resize(face_img, (160, 160))
    # 转换为float32并归一化
    face_img = face_img.astype('float32')
    # 扩展维度以匹配模型输入
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

def get_face_embedding(face_img):
    """获取人脸特征向量"""
    # 预处理人脸图像
    preprocessed_face = preprocess_face(face_img)
    # 使用FaceNet获取特征向量
    embedding = facenet.embeddings(preprocessed_face)
    return embedding[0]

def load_or_create_reference_face():
    """加载或创建参考人脸特征"""
    global reference_face_embedding
    
    if os.path.exists(reference_face_file):
        try:
            # 加载参考图片
            ref_img = cv2.imread(reference_face_file)
            if ref_img is None:
                raise Exception("无法读取参考图片")
                
            # 检测人脸
            gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = ref_img[y:y+h, x:x+w]
                # 获取人脸特征向量
                reference_face_embedding = get_face_embedding(face_roi)
                print("已加载参考人脸特征")
                return True
            else:
                print("在参考图片中未检测到人脸")
        except Exception as e:
            print(f"处理参考图片时出错: {e}")
    else:
        print("未找到参考人脸图片，请提供一张清晰的正面照片")
        try:
            # 打开摄像头拍照
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("无法打开摄像头")
                
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 显示预览
                cv2.imshow("Take Reference Photo (Press SPACE to capture)", frame)
                
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    # 保存图片
                    cv2.imwrite(reference_face_file, frame)
                    print("已保存参考照片")
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # 处理新保存的照片
            ref_img = cv2.imread(reference_face_file)
            gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = ref_img[y:y+h, x:x+w]
                reference_face_embedding = get_face_embedding(face_roi)
                print("已提取参考人脸特征")
                return True
            else:
                print("在新拍摄的照片中未检测到人脸")
        except Exception as e:
            print(f"拍照过程出错: {e}")
    
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
    if reference_face_embedding is None:
        return False
    try:
        # 获取当前人脸的特征向量
        current_embedding = get_face_embedding(face_roi)
        # 计算余弦相似度
        similarity = cosine_similarity(
            reference_face_embedding.reshape(1, -1),
            current_embedding.reshape(1, -1)
        )[0][0]
        # 根据相似度阈值判断是否为同一个人
        return similarity > FACE_SIMILARITY_THRESHOLD
    except Exception as e:
        print(f"人脸匹配出错: {e}")
        return False

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

def detect_persons(frame, model):
    """改进的人体检测函数"""
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    # 使用多个尺度进行检测
    scales = [1.0, 0.8, 1.2]  # 原始尺寸、缩小、放大
    all_persons = []
    
    for scale in scales:
        # 调整图像大小
        if scale != 1.0:
            scaled_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))
        else:
            scaled_frame = frame
            
        # 执行检测
        results = model(scaled_frame)[0]
        
        for result in results.boxes:
            cls_id = int(result.cls)
            if model.names[cls_id] != "person":
                continue
                
            confidence = float(result.conf)
            if confidence < MIN_CONFIDENCE:
                continue
                
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            
            # 如果使用了缩放，需要将坐标转换回原始尺寸
            if scale != 1.0:
                x1 = int(x1 / scale)
                y1 = int(y1 / scale)
                x2 = int(x2 / scale)
                y2 = int(y2 / scale)
            
            # 计算边界框属性
            bbox_height = y2 - y1
            bbox_width = x2 - x1
            aspect_ratio = bbox_height / bbox_width
            bottom_ratio = y2 / frame_height
            
            # 放宽检测条件
            is_valid_person = (
                bottom_ratio >= MIN_BOTTOM_RATIO and  # 降低底部比例要求
                MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO and  # 放宽宽高比范围
                bbox_height >= frame_height * MIN_HEIGHT_RATIO  # 降低最小高度要求
            )
            
            if is_valid_person:
                # 检查是否与已检测到的人体重叠
                overlap = False
                for existing_person in all_persons:
                    ex1, ey1, ex2, ey2 = existing_person[:4]
                    # 计算IoU
                    iou = calculate_iou((x1, y1, x2, y2), (ex1, ey1, ex2, ey2))
                    if iou > 0.5:  # 如果重叠度大于50%
                        overlap = True
                        # 保留置信度更高的检测结果
                        if confidence > existing_person[6]:
                            all_persons.remove(existing_person)
                            all_persons.append((x1, y1, x2, y2, bbox_height, bbox_width, confidence))
                        break
                
                if not overlap:
                    all_persons.append((x1, y1, x2, y2, bbox_height, bbox_width, confidence))
    
    return all_persons

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

# 读取上次的换算比例（如果存在）
if os.path.exists(calibration_file):
    with open(calibration_file, "r", encoding="utf-8") as f:
        pixel_to_cm_ratio = float(f.read())
    print(f"已加载上次标定值:1 像素 ≈ {pixel_to_cm_ratio:.4f} cm")
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

    # 使用改进的人体检测函数
    persons = detect_persons(frame, model)

    # 显示人数统计和检测置信度
    cv2.putText(frame, f"the number of people:{len(persons)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 检测人脸
    faces = detect_face(frame)
    
    # 获取当前时间（用于控制标定频率）
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    # 在每个人体检测框内查找人脸
    for i, (x1, y1, x2, y2, bbox_height, bbox_width, confidence) in enumerate(persons, 1):
        # 检查是否有人脸在这个人体框内
        has_face = False
        is_you = False
        face_scale = 1.0  # 初始化人脸比例因子

        for (fx, fy, fw, fh) in faces:
            # 如果人脸中心点在人体框内或接近人体框
            face_center_x = fx + fw//2
            face_center_y = fy + fh//2
            
            # 放宽人脸位置判断条件
            margin = 0.1  # 允许10%的误差
            x_margin = bbox_width * margin
            y_margin = bbox_height * margin
            
            if (x1 - x_margin <= face_center_x <= x2 + x_margin and 
                y1 - y_margin <= face_center_y <= y2 + y_margin):
                has_face = True
                # 计算人脸在人体中的相对位置
                face_scale = fw / bbox_width  # 计算人脸宽度与人体宽度的比例
                
                # 提取人脸区域
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                # 判断是否是参考人脸
                if face_recognized and is_reference_face(face_roi):
                    is_you = True
                    # 每次检测到参考人脸都进行标定（但控制频率）
                    if current_time - last_calibration_time > calibration_interval:
                        # 根据人脸比例调整身高计算
                        adjusted_height = bbox_height * (1 + (0.15 - face_scale))  # 微调系数
                        success, ratio = calibrate_with_reference_height(adjusted_height)
                        if success:
                            pixel_to_cm_ratio = ratio
                            calibrated = True
                            last_calibration_time = current_time
                            print(f"重新标定:1 像素 ≈ {pixel_to_cm_ratio:.4f} cm(已保存)")
                    # 绘制人脸框
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                    cv2.putText(frame, f"You ({confidence:.2f})", (fx, fy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 绘制其他人脸框
                    cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 2)
                    cv2.putText(frame, f"Face ({confidence:.2f})", (fx, fy-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                break

        # 绘制人体边界框和标签
        if is_you:
            color = (0, 0, 255)  # 红色表示是你
            # 计算并显示身高
            if calibrated:
                # 根据人脸比例调整身高计算
                adjusted_height = bbox_height * (1 + (0.15 - face_scale))
                estimated_height_cm = adjusted_height * pixel_to_cm_ratio
                label = f"Person {i}: {estimated_height_cm:.1f} cm [You] ({confidence:.2f})"
            else:
                label = f"Person {i} [You] ({confidence:.2f})"
        elif has_face:
            color = (0, 255, 255)  # 黄色表示其他人
            if calibrated:
                # 根据人脸比例调整身高计算
                adjusted_height = bbox_height * (1 + (0.15 - face_scale))
                estimated_height_cm = adjusted_height * pixel_to_cm_ratio
                label = f"Person {i}: {estimated_height_cm:.1f} cm [Face] ({confidence:.2f})"
            else:
                label = f"Person {i} [Face] ({confidence:.2f})"
        else:
            color = (0, 255, 0)  # 绿色表示未检测到人脸
            if calibrated:
                estimated_height_cm = bbox_height * pixel_to_cm_ratio
                label = f"Person {i}: {estimated_height_cm:.1f} cm ({confidence:.2f})"
            else:
                label = f"Person {i} ({confidence:.2f})"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # 确保标签不会超出图像边界
        label_y = max(y1 - 10, 30)
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 显示标定状态
    if calibrated:
        cv2.putText(frame, f"标定值: 1像素={pixel_to_cm_ratio:.4f}cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        last_calibration_text = f"上次标定: {int(current_time - last_calibration_time)}秒前"
        cv2.putText(frame, last_calibration_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 显示画面
    cv2.imshow("Height Estimation", frame)

    # 按 q 键退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()