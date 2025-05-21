import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import json
from datetime import datetime
import threading
from PIL import Image, ImageTk
from ultralytics import YOLO

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
        
        # 人脸标记相关
        self.face_marking_mode = False
        self.marked_faces = {}  # 存储标记的人脸位置和名称
        
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
        
        # 人脸标记相关按钮
        self.mark_face_btn = ttk.Button(control_frame, text="开始标记人脸", command=self.toggle_face_marking)
        self.mark_face_btn.grid(row=4, column=0, padx=5, pady=5)
        
        self.clear_marks_btn = ttk.Button(control_frame, text="清除所有标记", command=self.clear_face_marks)
        self.clear_marks_btn.grid(row=4, column=1, padx=5, pady=5)
        
        # 测量历史显示
        history_frame = ttk.LabelFrame(main_frame, text="测量历史")
        history_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        self.history_text = tk.Text(history_frame, height=10, width=40)
        self.history_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def toggle_face_marking(self):
        """切换人脸标记模式"""
        self.face_marking_mode = not self.face_marking_mode
        if self.face_marking_mode:
            self.mark_face_btn.configure(text="取消标记")
            messagebox.showinfo("提示", "请点击要标记的人脸，然后输入名称")
            self.video_label.bind("<Button-1>", self.on_face_click)
        else:
            self.mark_face_btn.configure(text="开始标记人脸")
            self.video_label.unbind("<Button-1>")

    def on_face_click(self, event):
        """处理人脸点击事件"""
        if not self.face_marking_mode:
            return
            
        # 获取点击位置
        click_x, click_y = event.x, event.y
        
        # 在标记的人脸中查找点击位置
        for face_id, (x1, y1, x2, y2, name) in self.marked_faces.items():
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                # 如果点击了已标记的人脸，询问是否要修改名称
                new_name = simpledialog.askstring("修改名称", 
                                                f"当前名称: {name}\n请输入新的名称:",
                                                initialvalue=name)
                if new_name:
                    self.marked_faces[face_id] = (x1, y1, x2, y2, new_name)
                return
        
        # 如果没有点击已标记的人脸，询问是否要添加新标记
        name = simpledialog.askstring("添加标记", "请输入这个人的名称:")
        if name:
            # 在点击位置附近查找人脸
            for result in self.model(self.current_frame, verbose=False)[0].boxes:
                cls_id = int(result.cls)
                if self.model.names[cls_id] == "face":
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    # 检查点击位置是否在人脸框内
                    if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                        face_id = f"face_{len(self.marked_faces)}"
                        self.marked_faces[face_id] = (x1, y1, x2, y2, name)
                        messagebox.showinfo("成功", f"已标记: {name}")
                        return
            
            messagebox.showinfo("提示", "未在点击位置检测到人脸")

    def clear_face_marks(self):
        """清除所有标记的人脸"""
        self.marked_faces.clear()
        messagebox.showinfo("提示", "已清除所有标记")

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IOU（交并比）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def process_frame(self, frame):
        frame_display = frame.copy()
        self.current_frame = frame.copy()  # 保存当前帧用于人脸标记
        
        # 使用YOLO进行人体检测
        results = self.model(frame)[0]
        measurements = []
        
        # 获取图像尺寸
        frame_height, frame_width = frame.shape[:2]
        
        for result in results.boxes:
            cls_id = int(result.cls)
            confidence = float(result.conf[0])  # 获取检测置信度
            class_name = self.model.names[cls_id]
            
            # 处理人体检测
            if class_name == "person" and confidence >= 0.5:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                
                # 计算边界框的宽高比
                aspect_ratio = h / w
                
                # 计算边界框底部到图像底部的距离
                bottom_margin = frame_height - y2
                
                # 增强的过滤条件
                if (h > 50 and h < frame_height * 0.90 and
                    1.8 < aspect_ratio < 3.5 and
                    bottom_margin < 20 and
                    x1 > 20 and x2 < frame_width - 20 and
                    y1 > 20):
                    
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
                    
                    # 显示结果
                    height_text = f"{height_cm:.1f} cm (conf: {confidence:.2f})"
                    cv2.putText(frame_display, height_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    measurements.append({
                        'height': height_cm,
                        'distance': estimated_distance,
                        'confidence': confidence,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # 处理人脸检测
            elif class_name == "face":
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                
                # 检查是否是被标记的人脸
                face_id = None
                for fid, (fx1, fy1, fx2, fy2, name) in self.marked_faces.items():
                    # 使用IOU（交并比）来判断是否是同一个人脸
                    iou = self.calculate_iou((x1, y1, x2, y2), (fx1, fy1, fx2, fy2))
                    if iou > 0.5:  # 如果重叠度大于50%
                        face_id = fid
                        break
                
                if face_id:
                    # 显示标记的名称
                    name = self.marked_faces[face_id][4]
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_display, name, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                else:
                    # 显示未标记的人脸
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame_display, "Face", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
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

    def run(self):
        self.root.mainloop()
        self.running = False
        self.cap.release()

if __name__ == "__main__":
    app = HeightMeasurementApp()
    app.run() 