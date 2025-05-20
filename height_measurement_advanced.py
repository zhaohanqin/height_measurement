import cv2
import os
import numpy as np
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from datetime import datetime
import threading
from PIL import Image, ImageTk
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO

class AdvancedHeightMeasurementApp:
    def __init__(self):
        # 初始化参数
        self.CONF_THRESH = 0.4
        self.MAX_HISTORY = 5
        self.CALIB_RANGE = (1.2, 3.5)
        self.pixel_to_cm_ratio = None
        self.calibrated = False
        self.ground_level = None
        self.real_height_cm = 170  # 默认值
        self.height_history = deque(maxlen=self.MAX_HISTORY)
        self.measurement_history = []
        self.recording = False
        self.show_debug = False
        self.running = True
        self.current_frame = None
        self.processed_frame = None
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("高级身高测量系统")
        self.root.geometry("1200x800")
        
        # 设置样式
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 10))
        
        # 创建GUI
        self.setup_gui()
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开摄像头，请检查连接")
            self.root.destroy()
            return
            
        # 加载YOLO模型
        try:
            self.model = YOLO("yolov8n.pt")
            self.status_var.set("模型加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"无法加载YOLO模型: {str(e)}")
            self.status_var.set(f"模型加载失败: {str(e)}")
        
        # 加载配置
        self.load_config()
        
        # 加载校准数据
        self.load_calibration()
        
        # 启动视频处理线程
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # 启动图表更新线程
        self.chart_thread = threading.Thread(target=self.update_chart)
        self.chart_thread.daemon = True
        self.chart_thread.start()

    def setup_gui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧面板
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        video_frame = ttk.LabelFrame(left_frame, text="视频监控")
        video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        status_frame = ttk.Frame(left_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="准备就绪")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                 font=("Arial", 10, "italic"))
        status_label.pack(side=tk.LEFT)
        
        # 右侧控制面板
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # 控制面板 - 标定
        calibration_frame = ttk.LabelFrame(right_frame, text="标定")
        calibration_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(calibration_frame, text="真实身高 (cm):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.height_var = tk.StringVar(value=str(self.real_height_cm))
        height_entry = ttk.Entry(calibration_frame, textvariable=self.height_var, width=10)
        height_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(calibration_frame, text="像素-厘米比例:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ratio_var = tk.StringVar(value="未标定")
        ratio_label = ttk.Label(calibration_frame, textvariable=self.ratio_var)
        ratio_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.calibrate_btn = ttk.Button(calibration_frame, text="开始标定", command=self.start_calibration)
        self.calibrate_btn.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.ground_btn = ttk.Button(calibration_frame, text="设置地面线", command=self.set_ground_level)
        self.ground_btn.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # 控制面板 - 操作
        action_frame = ttk.LabelFrame(right_frame, text="操作")
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.record_btn = ttk.Button(action_frame, text="开始记录", command=self.toggle_recording)
        self.record_btn.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.screenshot_btn = ttk.Button(action_frame, text="截图", command=self.take_screenshot)
        self.screenshot_btn.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.export_btn = ttk.Button(action_frame, text="导出数据", command=self.export_data)
        self.export_btn.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.debug_var = tk.BooleanVar(value=self.show_debug)
        self.debug_chk = ttk.Checkbutton(action_frame, text="调试模式", 
                                        variable=self.debug_var, command=self.toggle_debug)
        self.debug_chk.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # 控制面板 - 参数设置
        settings_frame = ttk.LabelFrame(right_frame, text="参数设置")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(settings_frame, text="置信度阈值:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.conf_var = tk.StringVar(value=str(self.CONF_THRESH))
        conf_entry = ttk.Entry(settings_frame, textvariable=self.conf_var, width=10)
        conf_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(settings_frame, text="历史记录长度:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.history_var = tk.StringVar(value=str(self.MAX_HISTORY))
        history_entry = ttk.Entry(settings_frame, textvariable=self.history_var, width=10)
        history_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        save_settings_btn = ttk.Button(settings_frame, text="保存设置", command=self.save_config)
        save_settings_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
        
        # 测量历史区域
        history_frame = ttk.LabelFrame(right_frame, text="测量历史")
        history_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建选项卡
        tab_control = ttk.Notebook(history_frame)
        tab_control.pack(fill=tk.BOTH, expand=True)
        
        # 列表选项卡
        list_tab = ttk.Frame(tab_control)
        tab_control.add(list_tab, text="列表")
        
        self.history_text = tk.Text(list_tab, height=10, width=30)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        history_scroll = ttk.Scrollbar(self.history_text, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scroll.set)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 图表选项卡
        chart_tab = ttk.Frame(tab_control)
        tab_control.add(chart_tab, text="图表")
        
        self.fig, self.ax = plt.subplots(figsize=(4, 3), dpi=100)
        self.ax.set_title("身高记录")
        self.ax.set_xlabel("时间")
        self.ax.set_ylabel("身高 (cm)")
        self.ax.grid(True)
        
        self.chart_canvas = FigureCanvasTkAgg(self.fig, chart_tab)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def video_loop(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_var.set("无法读取摄像头画面")
                    time.sleep(0.1)
                    continue
                
                self.current_frame = frame.copy()
                processed, measurements = self.process_frame(frame)
                self.processed_frame = processed
                
                # 转换为Tkinter可显示格式
                img = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
                
                # 更新UI（线程安全方式）
                self.root.after(1, lambda: self._update_image(img))
                
                # 更新测量历史
                if self.recording and measurements:
                    self.root.after(1, lambda m=measurements: self._update_history(m))
                
                time.sleep(0.03)  # 约30FPS
            except Exception as e:
                self.status_var.set(f"处理错误: {str(e)}")
                time.sleep(0.1)
    
    def _update_image(self, img):
        """线程安全的图像更新函数"""
        self.video_label.configure(image=img)
        self.video_label.image = img
    
    def _update_history(self, measurements):
        """线程安全的历史记录更新函数"""
        for m in measurements:
            self.measurement_history.append(m)
            self.history_text.insert(tk.END, 
                f"[{m['timestamp']}] 身高: {m['height']:.1f}cm\n")
            self.history_text.see(tk.END)

    def process_frame(self, frame):
        """处理视频帧，返回处理后的帧和测量结果"""
        frame_display = frame.copy()
        
        # 运行YOLO检测
        results = self.model(frame, verbose=False)[0]
        frame_height = frame.shape[0]
        
        persons = []
        measurements = []
        
        # 检测人体
        for result in results.boxes:
            conf = float(result.conf)
            
            # 调试模式显示所有检测结果
            if self.show_debug:
                cls_id = int(result.cls)
                class_name = self.model.names[cls_id]
                self.status_var.set(f"检测到: {class_name} 置信度: {conf:.2f}")
            
            # 过滤低置信度
            if conf < self.CONF_THRESH:
                continue
                
            cls_id = int(result.cls)
            if self.model.names[cls_id] != "person":
                continue
            
            # 获取边界框
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            bbox_height = y2 - y1
            bbox_width = x2 - x1
            aspect_ratio = bbox_height / bbox_width
            
            # 过滤不合理的人像
            if y2 >= frame_height - 50 and aspect_ratio > 1.5:
                persons.append((x1, y1, x2, y2, bbox_height, bbox_width))
                
                # 计算身高
                if self.calibrated:
                    if self.ground_level:
                        pixel_height = self.ground_level - y1
                    else:
                        pixel_height = bbox_height
                        
                    estimated_height = pixel_height * self.pixel_to_cm_ratio
                    self.height_history.append(estimated_height)
                    avg_height = np.mean(self.height_history) if self.height_history else estimated_height
                    
                    # 距离估计（基于宽度）
                    estimated_distance = None
                    if self.pixel_to_cm_ratio:
                        estimated_distance = (170 * 400) / (self.pixel_to_cm_ratio * bbox_height)
                    
                    # 记录结果
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    measurements.append({
                        'height': avg_height,
                        'raw_height': estimated_height,
                        'distance': estimated_distance,
                        'timestamp': timestamp
                    })
                    
                    # 显示结果
                    label = f"{avg_height:.1f} cm"
                    if estimated_distance:
                        label += f", {estimated_distance:.1f} cm"
                        
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 显示统计信息
                    cv2.putText(frame_display, f"当前: {estimated_height:.1f}cm", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame_display, f"平均: {avg_height:.1f}cm", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if self.recording:
                        cv2.putText(frame_display, "记录中", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 标定逻辑
        if not self.calibrated and len(persons) == 1 and self.real_height_cm:
            _, _, _, _, bbox_height, _ = persons[0]
            ratio = self.real_height_cm / bbox_height
            
            if self.CALIB_RANGE[0] <= ratio <= self.CALIB_RANGE[1]:
                self.pixel_to_cm_ratio = ratio
                self.ratio_var.set(f"{ratio:.4f} cm/pixel")
                self.calibrated = True
                self.save_calibration()
                self.status_var.set(f"标定成功: 1像素 ≈ {ratio:.4f}厘米")
                
                # 更新UI状态
                self.calibrate_btn.configure(text="重新标定")
            else:
                self.status_var.set(f"标定值异常: {ratio:.4f}，请重新标定")
        
        # 绘制地面线
        if self.ground_level:
            cv2.line(frame_display, (0, self.ground_level),
                    (frame_display.shape[1], self.ground_level), (255, 0, 0), 2)
        
        # 调试信息
        if self.show_debug:
            cv2.putText(frame_display, f"检测: {len(persons)}人", (frame_display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if self.calibrated:
                cv2.putText(frame_display, f"比例: {self.pixel_to_cm_ratio:.4f}", (frame_display.shape[1] - 150, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return frame_display, measurements

    def start_calibration(self):
        """开始标定过程"""
        try:
            self.real_height_cm = float(self.height_var.get())
            self.calibrated = False
            self.height_history.clear()
            self.status_var.set("请确保画面中只有一个人，且身体完全可见")
        except ValueError:
            messagebox.showerror("错误", "请输入有效的身高数值")

    def set_ground_level(self):
        """设置地面线位置"""
        self.status_var.set("请点击视频中的地面位置")
        
        def on_click(event):
            self.ground_level = event.y
            self.status_var.set(f"地面线设置在y={self.ground_level}")
            self.video_label.unbind("<Button-1>")
        
        self.video_label.bind("<Button-1>", on_click)

    def toggle_recording(self):
        """切换记录状态"""
        self.recording = not self.recording
        self.record_btn.configure(text="停止记录" if self.recording else "开始记录")
        self.status_var.set("开始记录测量数据" if self.recording else "停止记录")

    def take_screenshot(self):
        """保存当前画面截图"""
        if self.processed_frame is None:
            messagebox.showinfo("提示", "没有可用的画面")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, self.processed_frame)
        self.status_var.set(f"截图已保存: {filename}")

    def export_data(self):
        """导出测量数据"""
        if not self.measurement_history:
            messagebox.showinfo("提示", "没有可导出的数据")
            return
            
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
                title="保存测量数据"
            )
            
            if not filename:
                return
                
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["时间戳", "身高(cm)", "原始身高(cm)", "距离(cm)"])
                
                for m in self.measurement_history:
                    writer.writerow([
                        m.get("timestamp", ""),
                        m.get("height", ""),
                        m.get("raw_height", ""),
                        m.get("distance", "")
                    ])
                    
            self.status_var.set(f"数据导出成功: {filename}")
        except Exception as e:
            messagebox.showerror("导出错误", str(e))

    def toggle_debug(self):
        """切换调试模式"""
        self.show_debug = self.debug_var.get()
        self.status_var.set(f"调试模式: {'开启' if self.show_debug else '关闭'}")

    def update_chart(self):
        """更新图表数据"""
        while self.running:
            if self.measurement_history:
                try:
                    # 取最近20条记录
                    data = self.measurement_history[-20:]
                    times = [d.get("timestamp", "") for d in data]
                    heights = [d.get("height", 0) for d in data]
                    
                    # 更新图表
                    self.root.after(1, lambda t=times, h=heights: self._draw_chart(t, h))
                except Exception as e:
                    print(f"图表更新错误: {e}")
                    
            time.sleep(1)  # 每秒更新一次
    
    def _draw_chart(self, times, heights):
        """线程安全的图表绘制"""
        try:
            self.ax.clear()
            self.ax.plot(times, heights, 'o-', color='green')
            self.ax.set_title("身高记录")
            self.ax.set_xlabel("时间")
            self.ax.set_ylabel("身高 (cm)")
            self.ax.tick_params(axis='x', rotation=45)
            self.ax.grid(True)
            
            # 注意时间轴标签处理
            if len(times) > 6:
                # 只显示部分时间轴标签
                step = len(times) // 5
                self.ax.set_xticks(times[::step])
                
            self.fig.tight_layout()
            self.chart_canvas.draw()
        except Exception as e:
            print(f"绘图错误: {e}")

    def save_config(self):
        """保存配置"""
        try:
            self.CONF_THRESH = float(self.conf_var.get())
            self.MAX_HISTORY = int(self.history_var.get())
            self.height_history = deque(list(self.height_history)[-self.MAX_HISTORY:], maxlen=self.MAX_HISTORY)
            
            config = {
                "CONF_THRESH": self.CONF_THRESH,
                "MAX_HISTORY": self.MAX_HISTORY,
                "CALIB_RANGE": self.CALIB_RANGE,
                "real_height_cm": self.real_height_cm,
                "show_debug": self.show_debug
            }
            
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
                
            self.status_var.set("配置已保存")
        except Exception as e:
            messagebox.showerror("保存错误", str(e))

    def load_config(self):
        """加载配置"""
        try:
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    
                self.CONF_THRESH = config.get("CONF_THRESH", self.CONF_THRESH)
                self.MAX_HISTORY = config.get("MAX_HISTORY", self.MAX_HISTORY)
                self.CALIB_RANGE = tuple(config.get("CALIB_RANGE", self.CALIB_RANGE))
                self.real_height_cm = config.get("real_height_cm", self.real_height_cm)
                self.show_debug = config.get("show_debug", self.show_debug)
                
                # 更新UI
                self.height_var.set(str(self.real_height_cm))
                self.conf_var.set(str(self.CONF_THRESH))
                self.history_var.set(str(self.MAX_HISTORY))
                self.debug_var.set(self.show_debug)
                
                self.status_var.set("配置加载成功")
        except Exception as e:
            self.status_var.set(f"配置加载错误: {str(e)}")

    def save_calibration(self):
        """保存校准数据"""
        if self.calibrated and self.pixel_to_cm_ratio:
            try:
                with open("calibration.txt", "w", encoding="utf-8") as f:
                    f.write(str(self.pixel_to_cm_ratio))
            except Exception as e:
                self.status_var.set(f"保存校准数据失败: {str(e)}")

    def load_calibration(self):
        """加载校准数据"""
        try:
            if os.path.exists("calibration.txt"):
                with open("calibration.txt", "r", encoding="utf-8") as f:
                    self.pixel_to_cm_ratio = float(f.read())
                    
                if self.pixel_to_cm_ratio and self.CALIB_RANGE[0] <= self.pixel_to_cm_ratio <= self.CALIB_RANGE[1]:
                    self.calibrated = True
                    self.ratio_var.set(f"{self.pixel_to_cm_ratio:.4f} cm/pixel")
                    self.status_var.set(f"已加载校准数据: 1像素 ≈ {self.pixel_to_cm_ratio:.4f}厘米")
                    
                    # 更新UI状态
                    self.calibrate_btn.configure(text="重新标定")
        except Exception as e:
            self.status_var.set(f"加载校准数据失败: {str(e)}")

    def run(self):
        """运行应用程序"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()

    def on_close(self):
        """关闭程序时的清理操作"""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = AdvancedHeightMeasurementApp()
    app.run() 