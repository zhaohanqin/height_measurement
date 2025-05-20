import cv2
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import ttk, messagebox
import json
from datetime import datetime
import threading
from PIL import Image, ImageTk
from ultralytics import YOLO  # 添加YOLO导入

class HeightMeasurementApp:
    def __init__(self):
        # 初始化参数
        self.PIXELS_PER_CM = 2.0
        self.REFERENCE_DISTANCE_CM = 200
        self.KNOWN_PERSON_WIDTH_CM = 40.0
        self.FOCAL_LENGTH_PIXELS = 700
        self.ground_level = None
        self.calibration_mode = False
        self.recording = False
        self.measurement_history = []
        
        # 人脸识别相关
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_recognition_mode = False
        self.face_recognition_initialized = False
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("身高测量系统")
        self.setup_gui()
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("无法打开摄像头")
            
        # 初始化YOLO模型
        self.model = YOLO("yolov8n.pt")
        
        # 加载配置
        self.load_config()
        
        # 启动视频处理线程
        self.running = True
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()

    def setup_gui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        self.video_label = ttk.Label(main_frame)
        self.video_label.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板")
        control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # 校准设置
        ttk.Label(control_frame, text="每厘米像素数:").grid(row=0, column=0, padx=5, pady=5)
        self.pixels_per_cm_var = tk.StringVar(value=str(self.PIXELS_PER_CM))
        self.pixels_per_cm_entry = ttk.Entry(control_frame, textvariable=self.pixels_per_cm_var)
        self.pixels_per_cm_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(control_frame, text="参考距离(cm):").grid(row=1, column=0, padx=5, pady=5)
        self.ref_distance_var = tk.StringVar(value=str(self.REFERENCE_DISTANCE_CM))
        self.ref_distance_entry = ttk.Entry(control_frame, textvariable=self.ref_distance_var)
        self.ref_distance_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # 按钮
        self.calibration_btn = ttk.Button(control_frame, text="开始校准", command=self.toggle_calibration)
        self.calibration_btn.grid(row=2, column=0, padx=5, pady=5)
        
        self.ground_level_btn = ttk.Button(control_frame, text="设置地面", command=self.set_ground_level)
        self.ground_level_btn.grid(row=2, column=1, padx=5, pady=5)
        
        self.record_btn = ttk.Button(control_frame, text="开始记录", command=self.toggle_recording)
        self.record_btn.grid(row=3, column=0, padx=5, pady=5)
        
        self.save_btn = ttk.Button(control_frame, text="保存设置", command=self.save_config)
        self.save_btn.grid(row=3, column=1, padx=5, pady=5)
        
        # 添加人脸识别相关按钮
        self.face_recognition_btn = ttk.Button(control_frame, text="添加人脸", command=self.add_face)
        self.face_recognition_btn.grid(row=4, column=0, padx=5, pady=5)
        
        self.toggle_face_recognition_btn = ttk.Button(control_frame, text="开启人脸识别", 
                                                    command=self.toggle_face_recognition)
        self.toggle_face_recognition_btn.grid(row=4, column=1, padx=5, pady=5)
        
        # 测量历史显示
        history_frame = ttk.LabelFrame(main_frame, text="测量历史")
        history_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        self.history_text = tk.Text(history_frame, height=10, width=40)
        self.history_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def add_face(self):
        """添加新的人脸特征"""
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("错误", "无法获取摄像头画面")
            return
            
        # 转换为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测人脸
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            messagebox.showerror("错误", "未检测到人脸")
            return
            
        # 获取人脸编码
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
        
        # 获取用户输入的名字
        name = tk.simpledialog.askstring("输入", "请输入这个人的名字：")
        if name:
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            messagebox.showinfo("成功", f"已添加 {name} 的人脸特征")
            
            # 保存人脸特征
            self.save_face_data()

    def save_face_data(self):
        """保存人脸特征数据"""
        face_data = {
            'encodings': [enc.tolist() for enc in self.known_face_encodings],
            'names': self.known_face_names
        }
        with open('face_data.json', 'w') as f:
            json.dump(face_data, f)

    def load_face_data(self):
        """加载保存的人脸特征数据"""
        try:
            with open('face_data.json', 'r') as f:
                face_data = json.load(f)
                self.known_face_encodings = [np.array(enc) for enc in face_data['encodings']]
                self.known_face_names = face_data['names']
                self.face_recognition_initialized = True
        except FileNotFoundError:
            pass

    def toggle_face_recognition(self):
        """切换人脸识别模式"""
        self.face_recognition_mode = not self.face_recognition_mode
        if self.face_recognition_mode and not self.face_recognition_initialized:
            self.load_face_data()
        self.toggle_face_recognition_btn.configure(
            text="关闭人脸识别" if self.face_recognition_mode else "开启人脸识别"
        )

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # 处理帧
            processed_frame, measurements = self.process_frame(frame)
            
            # 转换为tkinter可显示的格式
            image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image=image)
            
            # 更新显示
            self.video_label.configure(image=photo)
            self.video_label.image = photo
            
            # 记录测量结果
            if self.recording and measurements:
                self.record_measurements(measurements)

    def process_frame(self, frame):
        frame_display = frame.copy()
        
        # 使用YOLO进行人体检测
        results = self.model(frame)[0]
        measurements = []
        
        # 获取图像尺寸
        frame_height, frame_width = frame.shape[:2]
        
        for result in results.boxes:
            cls_id = int(result.cls)
            confidence = float(result.conf[0])  # 获取检测置信度
            
            # 只处理"person"类别且置信度大于0.5的检测结果
            if self.model.names[cls_id] != "person" or confidence < 0.5:
                continue
                
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            
            # 计算边界框的宽高比
            aspect_ratio = h / w
            
            # 计算边界框底部到图像底部的距离
            bottom_margin = frame_height - y2
            
            # 增强的过滤条件：
            # 1. 高度在合理范围内（大于50像素且小于图像高度的90%）
            # 2. 宽高比在合理范围内（正常人体比例约为2.0-3.5）
            # 3. 边界框底部接近图像底部（允许20像素的误差）
            # 4. 边界框不能太靠近图像左右边缘（至少20像素）
            # 5. 边界框顶部不能太靠近图像顶部（至少20像素）
            if (h > 50 and h < frame_height * 0.90 and  # 高度条件
                1.8 < aspect_ratio < 3.5 and  # 宽高比条件
                bottom_margin < 20 and  # 底部接近图像底部
                x1 > 20 and x2 < frame_width - 20 and  # 左右边缘条件
                y1 > 20):  # 顶部条件
                
                # 绘制检测框
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 计算身高
                if self.ground_level:
                    pixel_height = self.ground_level - y1
                else:
                    pixel_height = h
                
                # 使用高级身高估计方法
                estimated_distance = (self.KNOWN_PERSON_WIDTH_CM * self.FOCAL_LENGTH_PIXELS) / w
                height_cm = (pixel_height * estimated_distance) / self.FOCAL_LENGTH_PIXELS
                
                # 显示结果（添加置信度信息）
                height_text = f"{height_cm:.1f} cm (conf: {confidence:.2f})"
                cv2.putText(frame_display, height_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                measurements.append({
                    'height': height_cm,
                    'distance': estimated_distance,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat()
                })
        
        # 人脸识别
        if self.face_recognition_mode and self.face_recognition_initialized:
            # 将摄像头捕获的帧转换为RGB格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸位置
            face_locations = face_recognition.face_locations(rgb_frame)
            # 获取人脸编码
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # 识别每个人脸
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # 与已知人脸比较
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                name = "未知"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                # 绘制人脸框和名字
                cv2.rectangle(frame_display, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame_display, name, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示校准模式和地面线
        if self.calibration_mode:
            cv2.putText(frame_display, "校准模式", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.ground_level:
            cv2.line(frame_display, (0, self.ground_level),
                    (frame_display.shape[1], self.ground_level), (255, 0, 0), 2)
        
        return frame_display, measurements

    def toggle_calibration(self):
        self.calibration_mode = not self.calibration_mode
        self.calibration_btn.configure(text="结束校准" if self.calibration_mode else "开始校准")

    def set_ground_level(self):
        # 在下一次点击视频帧时设置地面线
        def on_click(event):
            self.ground_level = event.y
            self.video_label.unbind('<Button-1>')
        
        self.video_label.bind('<Button-1>', on_click)

    def toggle_recording(self):
        self.recording = not self.recording
        self.record_btn.configure(text="停止记录" if self.recording else "开始记录")

    def record_measurements(self, measurements):
        for m in measurements:
            self.measurement_history.append(m)
            self.history_text.insert(tk.END, 
                f"身高: {m['height']:.1f}cm, 距离: {m['distance']:.1f}cm\n")
            self.history_text.see(tk.END)

    def save_config(self):
        config = {
            'PIXELS_PER_CM': float(self.pixels_per_cm_var.get()),
            'REFERENCE_DISTANCE_CM': float(self.ref_distance_var.get()),
            'KNOWN_PERSON_WIDTH_CM': self.KNOWN_PERSON_WIDTH_CM,
            'FOCAL_LENGTH_PIXELS': self.FOCAL_LENGTH_PIXELS
        }
        
        with open('config.json', 'w') as f:
            json.dump(config, f)

    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.PIXELS_PER_CM = config.get('PIXELS_PER_CM', self.PIXELS_PER_CM)
                self.REFERENCE_DISTANCE_CM = config.get('REFERENCE_DISTANCE_CM', self.REFERENCE_DISTANCE_CM)
                self.KNOWN_PERSON_WIDTH_CM = config.get('KNOWN_PERSON_WIDTH_CM', self.KNOWN_PERSON_WIDTH_CM)
                self.FOCAL_LENGTH_PIXELS = config.get('FOCAL_LENGTH_PIXELS', self.FOCAL_LENGTH_PIXELS)
                
                self.pixels_per_cm_var.set(str(self.PIXELS_PER_CM))
                self.ref_distance_var.set(str(self.REFERENCE_DISTANCE_CM))
        except FileNotFoundError:
            pass

    def run(self):
        self.root.mainloop()
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    app = HeightMeasurementApp()
    app.run() 