import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                             QDoubleSpinBox, QComboBox, QGroupBox, QMessageBox)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
import os
import time
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

class HeightMeasurementGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("身高测量系统")
        self.setMinimumSize(1200, 800)

        # 初始化模型和参数
        self.init_models()
        self.init_parameters()
        
        # 创建界面
        self.create_ui()
        
        # 初始化视频捕获
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def init_models(self):
        """初始化所需的模型"""
        try:
            self.model = YOLO("yolov8n.pt")
            self.facenet = FaceNet()
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.reference_face_embedding = None
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败: {str(e)}")
            sys.exit(1)

    def init_parameters(self):
        """初始化参数"""
        self.reference_face_file = "my_face.jpg"
        self.calibration_file = "calibration.txt"
        self.REFERENCE_HEIGHT = 172
        self.FACE_SIMILARITY_THRESHOLD = 0.7
        self.calibrated = False
        self.pixel_to_cm_ratio = None
        self.face_recognized = False
        # 新增：标定状态跟踪
        self.last_calibration_time = 0
        self.calibration_interval = 1.0  # 标定间隔（秒）
        
        # 人体检测参数
        self.MIN_CONFIDENCE = 0.3  # 降低置信度阈值
        self.MIN_HEIGHT_RATIO = 0.3  # 降低最小高度比例
        self.MIN_ASPECT_RATIO = 1.2  # 降低最小宽高比
        self.MAX_ASPECT_RATIO = 3.0  # 增加最大宽高比
        self.MIN_BOTTOM_RATIO = 0.5  # 降低最小底部比例
        
        # 加载上次标定结果（如果存在）
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    self.pixel_to_cm_ratio = float(f.read())
                self.calibrated = True
                self.calibration_label = QLabel(f"标定状态: 已加载上次标定")
            except:
                pass

    def create_ui(self):
        """创建用户界面"""
        # 创建主窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧视频显示区域
        video_group = QGroupBox("视频预览")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group)

        # 右侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()

        # 相机控制
        camera_group = QGroupBox("相机控制")
        camera_layout = QVBoxLayout()
        
        # 相机选择
        camera_select_layout = QHBoxLayout()
        camera_select_layout.addWidget(QLabel("选择相机:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("默认相机", 0)
        self.camera_combo.addItem("相机 1", 1)
        self.camera_combo.addItem("视频文件", -1)
        camera_select_layout.addWidget(self.camera_combo)
        camera_layout.addLayout(camera_select_layout)

        # 开始/停止按钮
        self.start_button = QPushButton("开始")
        self.start_button.clicked.connect(self.toggle_camera)
        camera_layout.addWidget(self.start_button)
        
        # 拍照按钮
        self.capture_button = QPushButton("拍摄参考照片")
        self.capture_button.clicked.connect(self.capture_reference)
        camera_layout.addWidget(self.capture_button)
        
        camera_group.setLayout(camera_layout)
        control_layout.addWidget(camera_group)

        # 参数设置
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout()
        
        # 参考身高设置
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("参考身高(cm):"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(100, 250)
        self.height_spinbox.setValue(self.REFERENCE_HEIGHT)
        self.height_spinbox.valueChanged.connect(self.update_reference_height)
        height_layout.addWidget(self.height_spinbox)
        params_layout.addLayout(height_layout)

        # 人脸相似度阈值
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("人脸相似度阈值:"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.1, 1.0)
        self.threshold_spinbox.setSingleStep(0.05)
        self.threshold_spinbox.setValue(self.FACE_SIMILARITY_THRESHOLD)
        self.threshold_spinbox.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_spinbox)
        params_layout.addLayout(threshold_layout)
        
        # 标定间隔设置
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("标定间隔(秒):"))
        self.interval_spinbox = QDoubleSpinBox()
        self.interval_spinbox.setRange(0.1, 10.0)
        self.interval_spinbox.setSingleStep(0.5)
        self.interval_spinbox.setValue(self.calibration_interval)
        self.interval_spinbox.valueChanged.connect(self.update_calibration_interval)
        interval_layout.addWidget(self.interval_spinbox)
        params_layout.addLayout(interval_layout)
        
        # 人体检测参数设置
        detection_group = QGroupBox("人体检测参数")
        detection_layout = QVBoxLayout()
        
        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 0.9)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(self.MIN_CONFIDENCE)
        self.conf_spinbox.valueChanged.connect(self.update_min_confidence)
        conf_layout.addWidget(self.conf_spinbox)
        detection_layout.addLayout(conf_layout)
        
        # 最小高度比例
        height_ratio_layout = QHBoxLayout()
        height_ratio_layout.addWidget(QLabel("最小高度比例:"))
        self.height_ratio_spinbox = QDoubleSpinBox()
        self.height_ratio_spinbox.setRange(0.1, 0.9)
        self.height_ratio_spinbox.setSingleStep(0.05)
        self.height_ratio_spinbox.setValue(self.MIN_HEIGHT_RATIO)
        self.height_ratio_spinbox.valueChanged.connect(self.update_min_height_ratio)
        height_ratio_layout.addWidget(self.height_ratio_spinbox)
        detection_layout.addLayout(height_ratio_layout)
        
        # 最小宽高比
        min_ar_layout = QHBoxLayout()
        min_ar_layout.addWidget(QLabel("最小宽高比:"))
        self.min_ar_spinbox = QDoubleSpinBox()
        self.min_ar_spinbox.setRange(0.5, 2.0)
        self.min_ar_spinbox.setSingleStep(0.1)
        self.min_ar_spinbox.setValue(self.MIN_ASPECT_RATIO)
        self.min_ar_spinbox.valueChanged.connect(self.update_min_aspect_ratio)
        min_ar_layout.addWidget(self.min_ar_spinbox)
        detection_layout.addLayout(min_ar_layout)
        
        # 最大宽高比
        max_ar_layout = QHBoxLayout()
        max_ar_layout.addWidget(QLabel("最大宽高比:"))
        self.max_ar_spinbox = QDoubleSpinBox()
        self.max_ar_spinbox.setRange(2.0, 5.0)
        self.max_ar_spinbox.setSingleStep(0.1)
        self.max_ar_spinbox.setValue(self.MAX_ASPECT_RATIO)
        self.max_ar_spinbox.valueChanged.connect(self.update_max_aspect_ratio)
        max_ar_layout.addWidget(self.max_ar_spinbox)
        detection_layout.addLayout(max_ar_layout)
        
        # 最小底部比例
        bottom_ratio_layout = QHBoxLayout()
        bottom_ratio_layout.addWidget(QLabel("最小底部比例:"))
        self.bottom_ratio_spinbox = QDoubleSpinBox()
        self.bottom_ratio_spinbox.setRange(0.1, 0.9)
        self.bottom_ratio_spinbox.setSingleStep(0.05)
        self.bottom_ratio_spinbox.setValue(self.MIN_BOTTOM_RATIO)
        self.bottom_ratio_spinbox.valueChanged.connect(self.update_min_bottom_ratio)
        bottom_ratio_layout.addWidget(self.bottom_ratio_spinbox)
        detection_layout.addLayout(bottom_ratio_layout)
        
        detection_group.setLayout(detection_layout)
        params_layout.addWidget(detection_group)

        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)

        # 状态显示
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        
        self.calibration_label = QLabel("标定状态: 未标定")
        self.face_status_label = QLabel("人脸识别状态: 未加载参考人脸")
        self.last_calibration_label = QLabel("上次标定: 无")
        
        status_layout.addWidget(self.calibration_label)
        status_layout.addWidget(self.face_status_label)
        status_layout.addWidget(self.last_calibration_label)
        
        status_group.setLayout(status_layout)
        control_layout.addWidget(status_group)

        # 添加弹性空间
        control_layout.addStretch()
        
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # 加载参考人脸
        self.load_reference_face()

    def load_reference_face(self):
        """尝试加载参考人脸"""
        if os.path.exists(self.reference_face_file):
            self.face_recognized = self.load_or_create_reference_face()
            if self.face_recognized:
                self.face_status_label.setText("人脸识别状态: 已加载参考人脸")

    def toggle_camera(self):
        """切换相机状态"""
        if self.timer.isActive():
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.cap = None
            self.start_button.setText("开始")
            self.video_label.clear()
        else:
            camera_id = self.camera_combo.currentData()
            if camera_id == -1:
                # 打开文件选择对话框
                from PySide6.QtWidgets import QFileDialog
                file_name, _ = QFileDialog.getOpenFileName(
                    self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)"
                )
                if file_name:
                    self.cap = cv2.VideoCapture(file_name)
                else:
                    return
            else:
                self.cap = cv2.VideoCapture(camera_id)

            if not self.cap.isOpened():
                QMessageBox.warning(self, "警告", "无法打开相机或视频文件")
                return

            self.timer.start(30)  # 30ms per frame
            self.start_button.setText("停止")

    def capture_reference(self):
        """拍摄参考照片"""
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "警告", "请先打开相机")
            return

        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(self.reference_face_file, frame)
            self.face_recognized = self.load_or_create_reference_face()
            if self.face_recognized:
                QMessageBox.information(self, "成功", "参考照片保存成功")
                self.face_status_label.setText("人脸识别状态: 已加载参考人脸")
            else:
                QMessageBox.warning(self, "警告", "未能在照片中检测到人脸")

    def update_reference_height(self, value):
        """更新参考身高"""
        self.REFERENCE_HEIGHT = value

    def update_threshold(self, value):
        """更新人脸相似度阈值"""
        self.FACE_SIMILARITY_THRESHOLD = value
        
    def update_calibration_interval(self, value):
        """更新标定间隔"""
        self.calibration_interval = value
        
    def update_min_confidence(self, value):
        """更新最小置信度阈值"""
        self.MIN_CONFIDENCE = value
        
    def update_min_height_ratio(self, value):
        """更新最小高度比例"""
        self.MIN_HEIGHT_RATIO = value
        
    def update_min_aspect_ratio(self, value):
        """更新最小宽高比"""
        self.MIN_ASPECT_RATIO = value
        
    def update_max_aspect_ratio(self, value):
        """更新最大宽高比"""
        self.MAX_ASPECT_RATIO = value
        
    def update_min_bottom_ratio(self, value):
        """更新最小底部比例"""
        self.MIN_BOTTOM_RATIO = value

    def update_frame(self):
        """更新视频帧"""
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.start_button.setText("开始")
            return

        # 处理帧
        processed_frame = self.process_frame(frame)
        
        # 转换为Qt图像
        h, w, ch = processed_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(processed_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整大小以适应标签
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def calculate_iou(self, box1, box2):
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

    def detect_persons(self, frame):
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
            results = self.model(scaled_frame)[0]
            
            for result in results.boxes:
                cls_id = int(result.cls)
                if self.model.names[cls_id] != "person":
                    continue
                    
                confidence = float(result.conf)
                if confidence < self.MIN_CONFIDENCE:
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
                    bottom_ratio >= self.MIN_BOTTOM_RATIO and  # 降低底部比例要求
                    self.MIN_ASPECT_RATIO <= aspect_ratio <= self.MAX_ASPECT_RATIO and  # 放宽宽高比范围
                    bbox_height >= frame_height * self.MIN_HEIGHT_RATIO  # 降低最小高度要求
                )
                
                if is_valid_person:
                    # 检查是否与已检测到的人体重叠
                    overlap = False
                    for existing_person in all_persons:
                        ex1, ey1, ex2, ey2 = existing_person[:4]
                        # 计算IoU
                        iou = self.calculate_iou((x1, y1, x2, y2), (ex1, ey1, ex2, ey2))
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

    def process_frame(self, frame):
        """处理视频帧"""
        # 使用改进的人体检测函数
        persons = self.detect_persons(frame)

        # 显示人数统计
        cv2.putText(frame, f"检测到 {len(persons)} 人", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 检测人脸
        faces = self.detect_face(frame)
        
        # 获取当前时间（用于控制标定频率）
        current_time = time.time()
        
        # 处理每个检测到的人
        for i, (x1, y1, x2, y2, bbox_height, bbox_width, confidence) in enumerate(persons, 1):
            has_face = False
            is_you = False
            face_scale = 1.0

            for (fx, fy, fw, fh) in faces:
                face_center_x = fx + fw//2
                face_center_y = fy + fh//2
                
                margin = 0.1
                x_margin = bbox_width * margin
                y_margin = bbox_height * margin
                
                if (x1 - x_margin <= face_center_x <= x2 + x_margin and 
                    y1 - y_margin <= face_center_y <= y2 + y_margin):
                    has_face = True
                    face_scale = fw / bbox_width
                    
                    face_roi = frame[fy:fy+fh, fx:fx+fw]
                    if self.face_recognized and self.is_reference_face(face_roi):
                        is_you = True
                        # 每次检测到参考人脸都进行标定（但控制频率）
                        if current_time - self.last_calibration_time > self.calibration_interval:
                            adjusted_height = bbox_height * (1 + (0.15 - face_scale))
                            success, ratio = self.calibrate_with_reference_height(adjusted_height)
                            if success:
                                self.pixel_to_cm_ratio = ratio
                                self.calibrated = True
                                self.last_calibration_time = current_time
                                self.calibration_label.setText(f"标定状态: 已标定 (1像素={ratio:.4f}cm)")
                                self.last_calibration_label.setText("上次标定: 刚刚")
                        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)
                        cv2.putText(frame, f"You ({confidence:.2f})", (fx, fy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 255), 2)
                        cv2.putText(frame, f"Face ({confidence:.2f})", (fx, fy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    break

            if is_you:
                color = (0, 0, 255)
                if self.calibrated:
                    adjusted_height = bbox_height * (1 + (0.15 - face_scale))
                    estimated_height_cm = adjusted_height * self.pixel_to_cm_ratio
                    label = f"Person {i}: {estimated_height_cm:.1f} cm [You] ({confidence:.2f})"
                else:
                    label = f"Person {i} [You] ({confidence:.2f})"
            elif has_face:
                color = (0, 255, 255)
                if self.calibrated:
                    adjusted_height = bbox_height * (1 + (0.15 - face_scale))
                    estimated_height_cm = adjusted_height * self.pixel_to_cm_ratio
                    label = f"Person {i}: {estimated_height_cm:.1f} cm [Face] ({confidence:.2f})"
                else:
                    label = f"Person {i} [Face] ({confidence:.2f})"
            else:
                color = (0, 255, 0)
                if self.calibrated:
                    estimated_height_cm = bbox_height * self.pixel_to_cm_ratio
                    label = f"Person {i}: {estimated_height_cm:.1f} cm ({confidence:.2f})"
                else:
                    label = f"Person {i} ({confidence:.2f})"
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_y = max(y1 - 10, 30)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 显示标定状态
        if self.calibrated:
            cv2.putText(frame, f"标定值: 1像素={self.pixel_to_cm_ratio:.4f}cm", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elapsed_time = int(current_time - self.last_calibration_time)
            if elapsed_time > 0:
                self.last_calibration_label.setText(f"上次标定: {elapsed_time}秒前")
                cv2.putText(frame, f"上次标定: {elapsed_time}秒前", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 转换颜色空间
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 从原代码复制其他必要的方法
    def detect_face(self, frame):
        """检测人脸并返回位置"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def preprocess_face(self, face_img):
        """预处理人脸图像"""
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def get_face_embedding(self, face_img):
        """获取人脸特征向量"""
        preprocessed_face = self.preprocess_face(face_img)
        embedding = self.facenet.embeddings(preprocessed_face)
        return embedding[0]

    def is_reference_face(self, face_roi):
        """判断是否匹配参考人脸"""
        if self.reference_face_embedding is None:
            return False
        try:
            current_embedding = self.get_face_embedding(face_roi)
            similarity = cosine_similarity(
                self.reference_face_embedding.reshape(1, -1),
                current_embedding.reshape(1, -1)
            )[0][0]
            return similarity > self.FACE_SIMILARITY_THRESHOLD
        except Exception as e:
            print(f"人脸匹配出错: {e}")
            return False

    def load_or_create_reference_face(self):
        """加载或创建参考人脸特征"""
        if os.path.exists(self.reference_face_file):
            try:
                ref_img = cv2.imread(self.reference_face_file)
                if ref_img is None:
                    raise Exception("无法读取参考图片")
                    
                gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_roi = ref_img[y:y+h, x:x+w]
                    self.reference_face_embedding = self.get_face_embedding(face_roi)
                    return True
            except Exception as e:
                print(f"处理参考图片时出错: {e}")
        return False

    def calibrate_with_reference_height(self, person_height_pixels):
        """使用参考身高进行标定"""
        if person_height_pixels > 0:
            pixel_to_cm_ratio = self.REFERENCE_HEIGHT / person_height_pixels
            with open(self.calibration_file, "w", encoding="utf-8") as f:
                f.write(str(pixel_to_cm_ratio))
            return True, pixel_to_cm_ratio
        return False, None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HeightMeasurementGUI()
    window.show()
    sys.exit(app.exec())