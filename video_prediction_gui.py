import tkinter as tk
import torch
import cv2
import os
import time
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
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
        self.frame_queue = Queue(maxsize=10)  # 帧缓冲队列
        
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
        
        # 添加标签
        self.label_frame = tk.Frame(self.video_frame)
        self.label_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.original_label = tk.Label(self.label_frame, text="预测标签")
        self.original_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.result_label = tk.Label(self.label_frame, text="预测结果")
        self.result_label.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
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
        """优化的模型加载函数 - 添加模型压缩和优化"""
        if not self.model_path:
            messagebox.showerror("错误", "请先选择模型权重文件")
            return
        
        # 如果已经加载了相同的模型，避免重复加载
        if self.model_loaded and hasattr(self, 'loaded_model_path') and self.loaded_model_path == self.model_path:
            return
        
        # 显示进度对话框
        progress_window = tk.Toplevel(self.root)
        progress_window.title("加载模型")
        progress_window.geometry("300x100")
        tk.Label(progress_window, text=f"正在加载模型...").pack(pady=10)
        progress = ttk.Progressbar(progress_window, mode="indeterminate")
        progress.pack(fill=tk.X, padx=20, pady=10)
        progress.start()
        progress_window.update()
        
        try:
            # 加载模型参数
            nb_filter, num_blocks = load_param('three', 'resnet_18')
            model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, 
                          num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
            
            # 使用内存映射加载大模型文件
            import mmap
            with open(self.model_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                    # 使用 torch.load 从内存映射对象加载
                    checkpoint = torch.load(m, map_location=self.device)
            
            model.load_state_dict(checkpoint['state_dict'])
            model.to(self.device)
            model.eval()
            
            # 优化内存使用 - 删除不需要的变量
            del checkpoint
            
            # 将模型转换为优化格式（需要 PyTorch 2.0+）
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, backend="inductor")
                except Exception as e:
                    print(f"模型编译失败，使用原始模型: {e}")
            
            # 将模型设置为评估模式并启用内存优化
            with torch.no_grad():
                model.eval()
            
            # 释放CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 关闭进度窗口
            progress_window.destroy()
            
            # 更新模型和状态
            self.model = model
            self.model_loaded = True
            self.loaded_model_path = self.model_path
            
            # 启用检测按钮
            if self.video_path:
                self.process_btn.config(state=tk.NORMAL)
                
            return model
            
        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            return None
    
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
            self.display_frame(frame, self.result_canvas)
        
        self.prev_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL)
        self.next_btn.config(state=tk.NORMAL)
    
    def display_frame(self, frame, canvas):
        """优化的帧显示函数 - 使用缓存和跳过不必要的转换"""
        if frame is None:
            return
            
        # 避免在没有可见变化时重绘
        current_frame_id = id(frame)
        cache_attr = f"{canvas.winfo_name()}_last_frame_id"
        
        if hasattr(self, cache_attr) and getattr(self, cache_attr) == current_frame_id:
            return  # 相同的帧，跳过重绘
        
        setattr(self, cache_attr, current_frame_id)
        
        # 转换BGR为RGB (如果需要)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if not hasattr(frame, 'converted_to_rgb') or not frame.converted_to_rgb:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.converted_to_rgb = True  # 标记已转换
            else:
                rgb_frame = frame
        else:
            rgb_frame = frame
        
        # 获取画布尺寸
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # 仅在必要时调整大小
        if canvas_width > 1 and canvas_height > 1:  # 确保画布已初始化
            h, w = rgb_frame.shape[:2]
            
            # 计算缩放比例
            ratio = min(canvas_width/w, canvas_height/h)
            new_size = (int(w*ratio), int(h*ratio))
            
            # 检查是否需要调整大小
            if w != new_size[0] or h != new_size[1]:
                # 使用更高效的调整大小方法
                rgb_frame = cv2.resize(rgb_frame, new_size, interpolation=cv2.INTER_AREA)
        
        # 转换为PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # 保留引用以防止垃圾回收
        canvas.image = img_tk
        
        # 清除画布和显示图像
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
    
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
        """优化的视频处理函数 - 减少内存使用并提高性能"""
        # 创建结果视频的临时存储
        temp_result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              f"temp_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # 创建预测标签视频的临时存储
        temp_label_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             f"temp_label_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        
        # 获取视频尺寸
        ret, sample_frame = self.cap.read()
        if not ret:
            self.root.after(0, lambda: messagebox.showerror("错误", "无法读取视频帧"))
            return
        
        h, w = sample_frame.shape[:2]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
        
        # 使用临时视频写入器而不是内存列表
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_result_writer = cv2.VideoWriter(temp_result_path, fourcc, self.fps, (w, h))
        temp_label_writer = cv2.VideoWriter(temp_label_path, fourcc, self.fps, (w, h))
        
        # 批处理参数 - 根据可用内存调整
        batch_size = 5  # 每次处理的帧数
        total_processed = 0
        
        try:
            while True:
                # 读取一批帧
                frames_batch = []
                for _ in range(batch_size):
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frames_batch.append(frame)
                    
                if not frames_batch:
                    break  # 没有更多帧
                    
                # 批量预测和处理
                results_batch, labels_batch = self.predict_frames_batch(frames_batch)
                
                # 写入处理结果和预测标签并更新UI
                for i in range(len(results_batch)):
                    result_frame = results_batch[i]
                    label_frame = labels_batch[i]
                    
                    temp_result_writer.write(result_frame)
                    temp_label_writer.write(label_frame)
                    
                    # 更新计数和UI
                    total_processed += 1
                    self.root.after(0, lambda p=total_processed: self.progress_var.set(p))
                    progress_pct = (total_processed / self.frame_count) * 100
                    self.root.after(0, lambda p=progress_pct: 
                                   self.status_var.set(f"处理中: {p:.1f}% ({total_processed}/{self.frame_count})"))
                    
                    # 周期性更新显示 (减少UI更新频率)
                    if total_processed % 10 == 0 or total_processed == self.frame_count:
                        self.root.after(0, lambda f=label_frame: self.display_frame(f, self.original_canvas))
                        self.root.after(0, lambda f=result_frame: self.display_frame(f, self.result_canvas))
            
            # 完成处理
            temp_result_writer.release()
            temp_label_writer.release()
            
            # 保存路径
            self.processed_result_path = temp_result_path
            self.processed_label_path = temp_label_path
            
            # 加载处理后的视频用于播放
            self.load_processed_videos(temp_label_path, temp_result_path)
            self.root.after(0, lambda: self.status_var.set("视频处理完成！"))
            self.root.after(0, lambda: messagebox.showinfo("完成", "视频处理完成"))
            
        except Exception as e:
            if 'temp_result_writer' in locals():
                temp_result_writer.release()
            if 'temp_label_writer' in locals():
                temp_label_writer.release()
                
            # 清理临时文件
            for path in [temp_result_path, temp_label_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
                        
            self.root.after(0, lambda: self.status_var.set(f"处理出错: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理视频时出错: {str(e)}"))
    
    def predict_frames_batch(self, frames):
        """批量处理多个帧，提高GPU利用率"""
        # 预处理所有帧
        input_tensors = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            input_tensor = self.transform(pil_img)
            input_tensors.append(input_tensor)
        
        # 批量推理
        batch_tensor = torch.stack(input_tensors).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            
            # 处理输出
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # 处理每一帧的预测结果
            result_frames = []
            label_frames = []
            
            for i, frame in enumerate(frames):
                # 获取当前帧的预测
                pred = torch.sigmoid(outputs[i, 0]).cpu().numpy()
                
                # 阈值处理得到二值图
                pred_binary = (pred > 0.5).astype(np.uint8) * 255
                
                # 创建预测标签图像 (左侧显示)
                label_frame = np.zeros_like(frame)
                # 使用蓝色通道显示预测标签
                label_frame[:,:,0] = pred_binary
                label_frames.append(label_frame)
                
                # 调整为原始尺寸
                orig_h, orig_w = frame.shape[:2]
                pred_resized = cv2.resize(pred_binary, (orig_w, orig_h))
                
                # 查找轮廓并绘制
                contours, _ = cv2.findContours(pred_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                result_frame = frame.copy()
                cv2.drawContours(result_frame, contours, -1, (0, 0, 255), 2)
                
                # 添加目标计数文本
                target_count = len(contours)
                cv2.putText(result_frame, f"目标数: {target_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                result_frames.append(result_frame)
            
            return result_frames, label_frames
    
    def load_processed_videos(self, label_path, result_path):
        """加载处理后的标签和结果视频用于播放"""
        # 释放之前的视频捕获
        if hasattr(self, 'label_cap') and self.label_cap:
            self.label_cap.release()
        if hasattr(self, 'result_cap') and self.result_cap:
            self.result_cap.release()
        
        # 加载新的视频捕获
        self.label_cap = cv2.VideoCapture(label_path)
        if not self.label_cap.isOpened():
            raise ValueError("无法打开处理后的标签视频")
            
        self.result_cap = cv2.VideoCapture(result_path)
        if not self.result_cap.isOpened():
            raise ValueError("无法打开处理后的结果视频")
        
        # 保存路径供后续使用
        self.processed_label_path = label_path
        self.processed_result_path = result_path
    
    def toggle_play(self):
        """切换视频播放/暂停状态"""
        if self.playing:
            self.stop_playback()
            self.play_btn.config(text="播放")
        else:
            self.start_playback()
            self.play_btn.config(text="暂停")
    
    def start_playback(self):
        """改进的视频播放启动函数 - 添加预缓冲"""
        self.playing = True
        self.stop_event.clear()
        
        # 启动预缓冲线程
        buffer_thread = Thread(target=self.buffer_frames)
        buffer_thread.daemon = True
        buffer_thread.start()
        
        # 启动播放线程
        playback_thread = Thread(target=self.playback_loop)
        playback_thread.daemon = True
        playback_thread.start()

    def buffer_frames(self):
        """预先加载帧到缓冲区"""
        buffer_size = 10  # 预缓冲10帧
        
        try:
            while self.playing and not self.stop_event.is_set():
                # 如果队列已满，等待
                if self.frame_queue.qsize() >= buffer_size:
                    time.sleep(0.01)
                    continue
                
                # 计算下一帧索引
                next_frame_idx = self.current_frame + self.frame_queue.qsize()
                if next_frame_idx >= self.frame_count:
                    next_frame_idx = 0  # 循环播放
                
                # 加载原始帧
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
                ret, frame = self.cap.read()
                
                # 加载标签和结果帧
                label_frame = None
                result_frame = None
                
                if hasattr(self, 'label_cap') and self.label_cap:
                    self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
                    label_ret, label_frame = self.label_cap.read()
                    
                if hasattr(self, 'result_cap') and self.result_cap:
                    self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
                    result_ret, result_frame = self.result_cap.read()
                    
                # 添加到队列
                if ret:
                    self.frame_queue.put((frame, label_frame, result_frame))
        except Exception as e:
            print(f"缓冲错误: {str(e)}")

    def playback_loop(self):
        """改进的视频播放循环 - 使用缓冲区并优化同步"""
        frame_time = 1.0 / self.fps if self.fps > 0 else 0.033  # 默认30fps
        
        try:
            while self.playing and not self.stop_event.is_set():
                start_time = time.time()
                
                # 从队列获取帧
                if not self.frame_queue.empty():
                    frame, label_frame, result_frame = self.frame_queue.get()
                    
                    # 显示预测标签帧（左侧）
                    if label_frame is not None:
                        self.display_frame(label_frame, self.original_canvas)
                    else:
                        self.display_frame(frame, self.original_canvas)
                    
                    # 显示预测结果帧（右侧）
                    if result_frame is not None:
                        self.display_frame(result_frame, self.result_canvas)
                    else:
                        self.display_frame(frame, self.result_canvas)
                    
                    # 更新进度条
                    self.progress_var.set(self.current_frame)
                    self.current_frame += 1
                    
                    # 检查是否到达视频末尾
                    if self.current_frame >= self.frame_count:
                        self.current_frame = 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if hasattr(self, 'label_cap') and self.label_cap:
                            self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        if hasattr(self, 'result_cap') and self.result_cap:
                            self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # 计算需要等待的时间以保持帧率
                elapsed = time.time() - start_time
                wait_time = max(0, frame_time - elapsed)
                self.stop_event.wait(wait_time)
                    
            if not self.stop_event.is_set():  # 如果是自然结束
                self.root.after(0, lambda: self.play_btn.config(text="播放"))
                self.playing = False
                
        except Exception as e:
            self.playing = False
            self.root.after(0, lambda: self.status_var.set(f"播放错误: {str(e)}"))
    
    def stop_playback(self):
        """停止视频播放"""
        self.playing = False
        self.stop_event.set()
        
        # 清空帧队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
    
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
            # 回退一帧，因为读取操作会前进一帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # 显示预测标签
            if hasattr(self, 'label_cap') and self.label_cap:
                self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                label_ret, label_frame = self.label_cap.read()
                if label_ret:
                    self.display_frame(label_frame, self.original_canvas)
                else:
                    self.display_frame(frame, self.original_canvas)
            else:
                self.display_frame(frame, self.original_canvas)
            
            # 显示预测结果
            if hasattr(self, 'result_cap') and self.result_cap:
                self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                result_ret, result_frame = self.result_cap.read()
                if result_ret:
                    self.display_frame(result_frame, self.result_canvas)
                else:
                    self.display_frame(frame, self.result_canvas)
            else:
                self.display_frame(frame, self.result_canvas)

    def next_frame(self):
        """显示下一帧"""
        if not self.cap:
            return
            
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            frame_idx = self.current_frame
            
            # 设置所有视频到该帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.progress_var.set(frame_idx)
                
                # 显示预测标签
                if hasattr(self, 'label_cap') and self.label_cap:
                    self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    label_ret, label_frame = self.label_cap.read()
                    if label_ret:
                        self.display_frame(label_frame, self.original_canvas)
                    else:
                        self.display_frame(frame, self.original_canvas)
                else:
                    self.display_frame(frame, self.original_canvas)
                
                # 显示预测结果
                if hasattr(self, 'result_cap') and self.result_cap:
                    self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    result_ret, result_frame = self.result_cap.read()
                    if result_ret:
                        self.display_frame(result_frame, self.result_canvas)
                    else:
                        self.display_frame(frame, self.result_canvas)
                else:
                    self.display_frame(frame, self.result_canvas)
    
    def previous_frame(self):
        """显示上一帧"""
        if not self.cap:
            return
            
        if self.current_frame > 0:
            self.current_frame -= 1
            frame_idx = self.current_frame
            
            # 设置所有视频到该帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.progress_var.set(frame_idx)
                
                # 显示预测标签
                if hasattr(self, 'label_cap') and self.label_cap:
                    self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    label_ret, label_frame = self.label_cap.read()
                    if label_ret:
                        self.display_frame(label_frame, self.original_canvas)
                    else:
                        self.display_frame(frame, self.original_canvas)
                else:
                    self.display_frame(frame, self.original_canvas)
                
                # 显示预测结果
                if hasattr(self, 'result_cap') and self.result_cap:
                    self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    result_ret, result_frame = self.result_cap.read()
                    if result_ret:
                        self.display_frame(result_frame, self.result_canvas)
                    else:
                        self.display_frame(frame, self.result_canvas)
                else:
                    self.display_frame(frame, self.result_canvas)
    
    def save_results(self):
        """优化的视频保存功能 - 保存预测结果和标签视频"""
        if (not hasattr(self, 'processed_result_path') or not os.path.exists(self.processed_result_path) or
            not hasattr(self, 'processed_label_path') or not os.path.exists(self.processed_label_path)):
            messagebox.showerror("错误", "没有可保存的处理结果")
            return
            
        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        
        try:
            self.status_var.set("正在保存结果视频...")
            
            # 构造文件名
            basename = os.path.basename(self.video_path).split('.')[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 复制预测结果文件
            result_save_path = os.path.join(save_dir, f"{basename}_result_{timestamp}.mp4")
            label_save_path = os.path.join(save_dir, f"{basename}_label_{timestamp}.mp4")
            
            # 复制文件
            import shutil
            shutil.copy2(self.processed_result_path, result_save_path)
            shutil.copy2(self.processed_label_path, label_save_path)
            
            self.status_var.set(f"结果视频已保存至: {save_dir}")
            messagebox.showinfo("成功", f"预测结果已保存至:\n{result_save_path}\n\n预测标签已保存至:\n{label_save_path}")
        except Exception as e:
            self.status_var.set("保存失败")
            messagebox.showerror("错误", f"保存视频失败: {str(e)}")
    
    def on_close(self):
        """改进的资源清理函数"""
        # 停止所有线程
        self.stop_playback()
        self.stop_event.set()
        
        # 释放视频资源
        if self.cap:
            self.cap.release()
        
        if hasattr(self, 'label_cap') and self.label_cap:
            self.label_cap.release()
            
        if hasattr(self, 'result_cap') and self.result_cap:
            self.result_cap.release()
        
        # 清理临时文件
        for attr in ['processed_result_path', 'processed_label_path', 'processed_video_path']:
            if hasattr(self, attr) and getattr(self, attr) and os.path.exists(getattr(self, attr)):
                try:
                    os.remove(getattr(self, attr))
                except:
                    pass
        
        # 释放GPU内存
        if self.model is not None:
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 强制进行垃圾收集
        import gc
        gc.collect()
        
        # 销毁窗口
        self.root.destroy()

# 主程序入口
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()