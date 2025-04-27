import tkinter as tk
import torch
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from torchvision import transforms
from datetime import datetime
from threading import Thread, Event
from queue import Queue
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param

class VideoDetectionApp:
    def __init__(self, root):
        """初始化视频检测应用程序界面和相关组件"""
        self.root = root
        self.root.title("红外小目标视频检测系统")
        self.root.geometry("1280x720")
        
        # 视频相关变量
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 0
        self.playing = False
        self.stop_event = Event()
        self.frame_queue = Queue(maxsize=5)  # 帧缓冲队列
        
        # 模型相关变量
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = None
        self.model_loaded = False
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        
        # 创建界面布局
        self._create_ui()
        
    def _create_ui(self):
        """创建用户界面布局"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 右侧显示区域
        display_frame = tk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 视频显示区域
        self.video_frame = tk.Frame(display_frame, bg="black")
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)
        
        # 原始视频与结果视频并排显示
        self.original_canvas = tk.Canvas(self.video_frame, bg="black")
        self.original_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.original_canvas.config(width=400, height=300)
        
        self.result_canvas = tk.Canvas(self.video_frame, bg="black")
        self.result_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.result_canvas.config(width=400, height=300)
        
        # 帧控制区域
        control_bar = tk.Frame(display_frame)
        control_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = tk.Scale(control_bar, from_=0, to=100, 
                                     orient=tk.HORIZONTAL, variable=self.progress_var,
                                     command=self.seek_video)
        self.progress_bar.pack(fill=tk.X, padx=10)
        
        # 播放控制按钮区域
        btn_frame = tk.Frame(control_bar)
        btn_frame.pack(pady=5)
        
        self.prev_btn = tk.Button(btn_frame, text="上一帧", command=self.previous_frame, state=tk.DISABLED)
        self.prev_btn.grid(row=0, column=0, padx=5)
        
        self.play_btn = tk.Button(btn_frame, text="播放", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.grid(row=0, column=1, padx=5)
        
        self.next_btn = tk.Button(btn_frame, text="下一帧", command=self.next_frame, state=tk.DISABLED)
        self.next_btn.grid(row=0, column=2, padx=5)
        
        # 左侧控制按钮
        self.model_btn = tk.Button(control_frame, text="选择模型权重", command=self.select_model)
        self.model_btn.pack(fill=tk.X, pady=5)
        
        self.video_btn = tk.Button(control_frame, text="选择视频文件", command=self.select_video)
        self.video_btn.pack(fill=tk.X, pady=5)
        
        self.process_btn = tk.Button(control_frame, text="开始检测", command=self.start_detection, state=tk.DISABLED)
        self.process_btn.pack(fill=tk.X, pady=5)
        
        self.save_btn = tk.Button(control_frame, text="保存结果视频", command=self.save_results, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=5)
        
        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("等待加载模型和视频...")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def select_model(self):
        """选择模型权重文件并加载模型"""
        self.model_path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch模型", "*.pth.tar"), ("所有文件", "*.*")]
        )
        
        if self.model_path:
            self.status_var.set(f"正在加载模型: {os.path.basename(self.model_path)}...")
            try:
                self.load_model()
                self.status_var.set(f"模型加载成功: {os.path.basename(self.model_path)}")
                messagebox.showinfo("成功", "模型加载成功")
            except Exception as e:
                self.status_var.set("模型加载失败")
                messagebox.showerror("错误", f"模型加载失败: {str(e)}")
    
    def load_model(self):
        """加载DNANet模型"""
        if not self.model_path:
            messagebox.showerror("错误", "请先选择模型权重文件")
            return
            
        # 加载模型参数
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, 
                      num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        self.model = model
        self.model_loaded = True
        
        # 启用检测按钮
        if self.video_path:
            self.process_btn.config(state=tk.NORMAL)
    
    def select_video(self):
        """选择视频文件并加载基本信息"""
        self.video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
        )
        
        if self.video_path:
            try:
                self.status_var.set(f"正在加载视频: {os.path.basename(self.video_path)}...")
                self.load_video()
                self.status_var.set(f"视频加载成功: {os.path.basename(self.video_path)}")
                
                # 启用检测按钮
                if self.model_loaded:
                    self.process_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.status_var.set("视频加载失败")
                messagebox.showerror("错误", f"视频加载失败: {str(e)}")
    
    def load_video(self):
        """加载视频文件并获取基本信息"""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        # 获取视频基本信息
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        
        # 配置进度条
        self.progress_bar.config(to=self.frame_count-1)
        self.progress_var.set(0)
        
        # 显示第一帧
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame, self.original_canvas)
        
        self.prev_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.NORMAL)
    
    def display_frame(self, frame, canvas):
        """在指定的画布上显示帧图像"""
        # 转换BGR为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸以适应画布
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # 确保画布已初始化
            h, w = rgb_frame.shape[:2]
            ratio = min(canvas_width/w, canvas_height/h)
            new_size = (int(w*ratio), int(h*ratio))
            rgb_frame = cv2.resize(rgb_frame, new_size)
        
        # 转换为PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # 保留引用以防止垃圾回收
        canvas.image = img_tk
        
        # 显示图像
        canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    
    def start_detection(self):
        """开始检测过程"""
        if not self.model_loaded:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        if not self.video_path or not self.cap:
            messagebox.showerror("错误", "请先加载视频")
            return
        
        # 停止当前播放
        self.stop_playback()
        
        self.status_var.set("开始处理视频...")
        # 重置视频到第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        self.progress_var.set(0)
        
        # 开始检测线程
        self.detection_thread = Thread(target=self.process_video)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        # 启用保存按钮
        self.save_btn.config(state=tk.NORMAL)
    
    def process_video(self):
        """在单独的线程中处理视频帧"""
        result_frames = []  # 存储处理结果帧
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 更新进度条和状态
                self.current_frame += 1
                frame_percentage = (self.current_frame / self.frame_count) * 100
                self.root.after(0, lambda p=self.current_frame: self.progress_var.set(p))
                self.root.after(0, lambda p=frame_percentage: 
                               self.status_var.set(f"处理中: {p:.1f}% ({self.current_frame}/{self.frame_count})"))
                
                # 处理当前帧
                result_frame = self.predict_frame(frame)
                result_frames.append(result_frame)
                
                # 显示原始帧和结果帧
                self.root.after(0, lambda f=frame: self.display_frame(f, self.original_canvas))
                self.root.after(0, lambda f=result_frame: self.display_frame(f, self.result_canvas))
                
            self.processed_frames = result_frames
            self.root.after(0, lambda: self.status_var.set("视频处理完成！"))
            self.root.after(0, lambda: messagebox.showinfo("完成", "视频处理完成"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"处理出错: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理视频时出错: {str(e)}"))
    
    def predict_frame(self, frame):
        """对单个帧进行预测和处理"""
        # 转换为PIL图像
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        
        # 预处理
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            
            # 如果使用深度监督，取最后一个输出
            if isinstance(output, tuple):
                output = output[0]
                
            # 将输出转换为概率图
            pred = torch.sigmoid(output[0, 0]).cpu().numpy()
            
            # 阈值处理得到二值图
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            
            # 调整为原始尺寸
            orig_h, orig_w = frame.shape[:2]
            pred_resized = cv2.resize(pred_binary, (orig_w, orig_h))
            
            # 创建可视化结果
            # 为每个检测到的目标绘制红色轮廓
            contours, _ = cv2.findContours(pred_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result_frame = frame.copy()
            cv2.drawContours(result_frame, contours, -1, (0, 0, 255), 2)
            
            # 添加目标计数文本
            target_count = len(contours)
            cv2.putText(result_frame, f"目标数: {target_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return result_frame
    
    def toggle_play(self):
        """切换视频播放/暂停状态"""
        if self.playing:
            self.stop_playback()
            self.play_btn.config(text="播放")
        else:
            self.start_playback()
            self.play_btn.config(text="暂停")
    
    def start_playback(self):
        """开始视频播放"""
        self.playing = True
        self.stop_event.clear()
        
        playback_thread = Thread(target=self.playback_loop)
        playback_thread.daemon = True
        playback_thread.start()
    
    def stop_playback(self):
        """停止视频播放"""
        self.playing = False
        self.stop_event.set()
    
    def playback_loop(self):
        """视频播放循环"""
        try:
            while self.playing and not self.stop_event.is_set():
                # 检查是否到达视频末尾
                if self.current_frame >= self.frame_count - 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                
                # 读取当前帧
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 显示帧
                self.display_frame(frame, self.original_canvas)
                
                # 如果有处理过的帧，也显示
                if hasattr(self, 'processed_frames') and len(self.processed_frames) > self.current_frame:
                    result_frame = self.processed_frames[self.current_frame]
                    self.display_frame(result_frame, self.result_canvas)
                
                # 更新进度条
                self.progress_var.set(self.current_frame)
                self.current_frame += 1
                
                # 控制帧率
                self.stop_event.wait(1/self.fps)
                
            if not self.stop_event.is_set():  # 如果是自然结束
                self.root.after(0, lambda: self.play_btn.config(text="播放"))
                self.playing = False
                
        except Exception as e:
            self.playing = False
            self.root.after(0, lambda: self.status_var.set(f"播放错误: {str(e)}"))
    
    def seek_video(self, value):
        """跳转到视频的特定位置"""
        if not self.cap:
            return
            
        frame_idx = int(float(value))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.current_frame = frame_idx
        
        # 显示当前帧
        ret, frame = self.cap.read()
        if ret:
            self.display_frame(frame, self.original_canvas)
            
            # 如果有处理过的帧，也显示
            if hasattr(self, 'processed_frames') and len(self.processed_frames) > self.current_frame:
                result_frame = self.processed_frames[self.current_frame]
                self.display_frame(result_frame, self.result_canvas)
    
    def next_frame(self):
        """显示下一帧"""
        if not self.cap:
            return
            
        if self.current_frame < self.frame_count - 1:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.progress_var.set(self.current_frame)
                self.display_frame(frame, self.original_canvas)
                
                # 如果有处理过的帧，也显示
                if hasattr(self, 'processed_frames') and len(self.processed_frames) > self.current_frame - 1:
                    result_frame = self.processed_frames[self.current_frame - 1]
                    self.display_frame(result_frame, self.result_canvas)
    
    def previous_frame(self):
        """显示上一帧"""
        if not self.cap:
            return
            
        if self.current_frame > 0:
            self.current_frame -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            if ret:
                self.progress_var.set(self.current_frame)
                self.display_frame(frame, self.original_canvas)
                
                # 如果有处理过的帧，也显示
                if hasattr(self, 'processed_frames') and len(self.processed_frames) > self.current_frame:
                    result_frame = self.processed_frames[self.current_frame]
                    self.display_frame(result_frame, self.result_canvas)
    
    def save_results(self):
        """保存处理结果到视频文件"""
        if not hasattr(self, 'processed_frames') or not self.processed_frames:
            messagebox.showerror("错误", "没有可保存的处理结果")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="保存结果视频",
            defaultextension=".mp4",
            filetypes=[("MP4视频", "*.mp4"), ("AVI视频", "*.avi"), ("所有文件", "*.*")]
        )
        
        if save_path:
            try:
                self.status_var.set("正在保存结果视频...")
                
                # 获取第一帧确定尺寸
                h, w = self.processed_frames[0].shape[:2]
                
                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或使用'XVID'
                out = cv2.VideoWriter(save_path, fourcc, self.fps, (w, h))
                
                # 写入每一帧
                for frame in self.processed_frames:
                    out.write(frame)
                
                out.release()
                
                self.status_var.set(f"结果视频已保存至: {save_path}")
                messagebox.showinfo("成功", f"结果视频已保存至:\n{save_path}")
            except Exception as e:
                self.status_var.set("保存失败")
                messagebox.showerror("错误", f"保存视频失败: {str(e)}")
    
    def on_close(self):
        """窗口关闭时的清理操作"""
        self.stop_playback()
        if self.cap:
            self.cap.release()
        self.root.destroy()

# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()