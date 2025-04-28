import cv2  # 第一位置导入
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageTk, ImageDraw
import threading
import time
from datetime import datetime
import tempfile
from skimage import measure

# 导入模型相关组件
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param

class VideoModeFrame:
    def __init__(self, parent, status_var, model_path=None, update_model_callback=None):
        self.parent = parent
        self.status_var = status_var
        self.model_path = model_path
        self.update_model_callback = update_model_callback
        
        # 初始化变量
        self.selected_video = None
        self.model = None
        self.model_loaded = False
        self.processing = False
        self.playing = False
        self.cap = None
        self.label_video_path = None
        self.result_video_path = None
        
        # 视频处理参数
        self.frame_skip = 1  # 每隔几帧处理一次
        self.detection_threshold = 0.3
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建UI
        self.create_widgets()
        
        # 初始化图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        
        # 如果有模型路径，尝试加载模型
        if self.model_path and os.path.exists(self.model_path):
            self.load_model()
    
    def create_widgets(self):
        """创建视频模式界面"""
        # 创建工具栏
        self.toolbar = ttk.Frame(self.parent)
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 创建按钮
        self.select_model_btn = ttk.Button(
            self.toolbar, 
            text="选择模型权重", 
            command=self.select_model
        )
        self.select_model_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_video_btn = ttk.Button(
            self.toolbar, 
            text="选择视频", 
            command=self.select_video
        )
        self.select_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = ttk.Button(
            self.toolbar, 
            text="开始检测", 
            command=self.process_video,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(
            self.toolbar, 
            text="播放结果", 
            command=self.toggle_playback,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            self.toolbar, 
            text="保存结果", 
            command=self.save_results,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建视频显示区域
        self.display_frame = ttk.Frame(self.parent)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 标签视频区域(左)
        self.label_frame = ttk.LabelFrame(self.display_frame, text="预测标签")
        self.label_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        
        self.label_canvas = tk.Canvas(self.label_frame, width=640, height=480, bg="black")
        self.label_canvas.pack(padx=5, pady=5)
        
        # 结果视频区域(右)
        self.result_frame = ttk.LabelFrame(self.display_frame, text="检测结果")
        self.result_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        
        self.result_canvas = tk.Canvas(self.result_frame, width=640, height=480, bg="black")
        self.result_canvas.pack(padx=5, pady=5)
        
        # 设置网格权重
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
        
        # 进度条
        self.progress_frame = ttk.Frame(self.parent)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.progress_frame, 
            orient=tk.HORIZONTAL, 
            length=100, 
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def select_model(self):
        """选择模型权重文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch模型", "*.pth;*.pth.tar"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.model_path = file_path
            self.status_var.set(f"正在加载模型: {os.path.basename(file_path)}...")
            
            # 在后台线程中加载模型，避免UI冻结
            threading.Thread(target=self._load_model_thread, args=(file_path,), daemon=True).start()
    
    def _load_model_thread(self, file_path):
        """在后台线程中加载模型"""
        try:
            self.load_model()
            
            # 更新UI（在主线程中）
            self.parent.after(0, lambda: self.status_var.set(f"模型加载成功: {os.path.basename(file_path)}"))
            self.parent.after(0, lambda: messagebox.showinfo("成功", "模型加载成功"))
            
            # 更新模型信息
            if self.update_model_callback:
                self.parent.after(0, lambda: self.update_model_callback(file_path))
            
            # 启用检测按钮
            if self.selected_video:
                self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
        except Exception as e:
            self.parent.after(0, lambda: self.status_var.set("模型加载失败"))
            self.parent.after(0, lambda: messagebox.showerror("错误", f"模型加载失败: {str(e)}"))
    
    def load_model(self):
        """加载模型"""
        if not self.model_path:
            return
        
        # 加载模型参数
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        model = DNANet(
            num_classes=1, 
            input_channels=3, 
            block=Res_CBAM_block, 
            num_blocks=num_blocks, 
            nb_filter=nb_filter, 
            deep_supervision=True
        )
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        # 如果有支持的优化方法，尝试应用
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, backend="inductor")
            except:
                pass
        
        self.model = model
        self.model_loaded = True
    
    def select_video(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.selected_video = file_path
            self.status_var.set(f"已选择视频: {os.path.basename(file_path)}")
            
            # 初始化视频捕获
            self.initialize_video(file_path)
            
            # 如果模型已加载，启用检测按钮
            if self.model_loaded:
                self.process_btn.config(state=tk.NORMAL)
    
    def initialize_video(self, file_path):
        """初始化视频捕获"""
        try:
            # 释放之前的视频捕获
            if self.cap is not None:
                self.cap.release()
            
            # 打开视频
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            # 获取视频信息
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 显示第一帧
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                
                # 显示第一帧
                self.display_frame_on_canvas(frame_resized, self.result_canvas)
                
                # 清空标签画布
                self.label_canvas.delete("all")
                
                # 禁用播放和保存按钮
                self.play_btn.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.DISABLED)
                
                return True
            else:
                messagebox.showerror("错误", "无法读取视频帧")
                return False
            
        except Exception as e:
            messagebox.showerror("错误", f"初始化视频失败: {str(e)}")
            return False
    
    def display_frame_on_canvas(self, frame, canvas):
        """在画布上显示帧"""
        # 将NumPy数组转换为PIL图像
        img = Image.fromarray(frame)
        
        # 转换为PhotoImage
        photo = ImageTk.PhotoImage(image=img)
        
        # 在画布上显示
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # 保留引用，防止垃圾回收
        canvas.image = photo
    
    def process_video(self):
        """处理视频"""
        if not self.selected_video:
            messagebox.showwarning("警告", "请先选择视频文件")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        if self.processing:
            messagebox.showinfo("提示", "正在处理中，请等待...")
            return
        
        # 开始处理
        self.processing = True
        self.status_var.set("正在处理视频...")
        self.process_btn.config(state=tk.DISABLED)
        self.select_video_btn.config(state=tk.DISABLED)
        self.select_model_btn.config(state=tk.DISABLED)
        
        # 在后台线程中处理视频
        threading.Thread(target=self._process_video_thread, daemon=True).start()
    
    def _process_video_thread(self):
        """在后台线程中处理视频"""
        try:
            # 创建临时目录用于存储处理后的视频
            temp_dir = tempfile.mkdtemp()
            label_video_path = os.path.join(temp_dir, "label_video.mp4")
            result_video_path = os.path.join(temp_dir, "result_video.mp4")
            
            # 重新打开视频文件
            cap = cv2.VideoCapture(self.selected_video)
            
            # 设置输出视频参数
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建视频写入器
            label_writer = cv2.VideoWriter(label_video_path, fourcc, fps, (width, height))
            result_writer = cv2.VideoWriter(result_video_path, fourcc, fps, (width, height))
            
            # 处理视频帧
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 更新进度
                progress = (frame_count / total_frames) * 100
                self.parent.after(0, lambda p=progress: self.progress_var.set(p))
                
                # 每隔几帧处理一次
                if frame_count % self.frame_skip == 0:
                    # 处理帧
                    label_frame, result_frame = self.process_frame(frame)
                    
                    # 写入处理后的帧
                    label_writer.write(label_frame)
                    result_writer.write(result_frame)
                    
                    # 更新状态
                    self.parent.after(0, lambda: self.status_var.set(f"处理中: {frame_count}/{total_frames} 帧 ({int(progress)}%)"))
                    
                    # 在UI上显示最新帧
                    if frame_count % (self.frame_skip * 10) == 0:  # 减少UI更新频率
                        label_frame_rgb = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
                        result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                        label_frame_resized = cv2.resize(label_frame_rgb, (640, 480))
                        result_frame_resized = cv2.resize(result_frame_rgb, (640, 480))
                        
                        self.parent.after(0, lambda l=label_frame_resized: self.display_frame_on_canvas(l, self.label_canvas))
                        self.parent.after(0, lambda r=result_frame_resized: self.display_frame_on_canvas(r, self.result_canvas))
                else:
                    # 对于跳过的帧，直接写入上一帧的结果
                    label_writer.write(label_frame)
                    result_writer.write(result_frame)
                
                frame_count += 1
            
            # 完成视频处理
            cap.release()
            label_writer.release()
            result_writer.release()
            
            # 保存处理后的视频路径
            self.label_video_path = label_video_path
            self.result_video_path = result_video_path
            
            # 更新UI
            self.parent.after(0, lambda: self.status_var.set("视频处理完成"))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_video_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_model_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: messagebox.showinfo("成功", "视频处理完成"))
            
        except Exception as e:
            self.parent.after(0, lambda: self.status_var.set("视频处理失败"))
            self.parent.after(0, lambda: messagebox.showerror("错误", f"视频处理失败: {str(e)}"))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_video_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_model_btn.config(state=tk.NORMAL))
        
        finally:
            self.processing = False
    
    def process_frame(self, frame):
        """处理单个视频帧"""
        # 转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(frame_rgb)
        
        # 模型预测
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(input_tensor)
            if isinstance(output, list):
                output = output[-1]
            predicted_result = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 创建标签图像(纯黑背景，目标为白色)
        label_image = np.zeros_like(frame)
        
        # 二值化预测结果
        binary_mask = (predicted_result > self.detection_threshold).astype(np.uint8)
        
        # 提取连通区域
        labeled_array = measure.label(binary_mask)
        regions = measure.regionprops(labeled_array)
        
        # 创建标注图像(原始图像上添加红框)
        result_image = frame.copy()
        
        # 标签图像: 黑色背景，目标为白色区域
        if regions:
            # 调整mask大小为原始图像大小
            mask_resized = cv2.resize(binary_mask * 255, (frame.shape[1], frame.shape[0]))
            # 将mask从灰度转为BGR
            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            label_image = mask_bgr
            
            # 在原始图像上标注目标
            for region in regions:
                # 获取区域边界框
                minr, minc, maxr, maxc = region.bbox
                
                # 调整到原始图像大小
                h_ratio = frame.shape[0] / 256
                w_ratio = frame.shape[1] / 256
                
                minr_orig = int(minr * h_ratio)
                minc_orig = int(minc * w_ratio)
                maxr_orig = int(maxr * h_ratio)
                maxc_orig = int(maxc * w_ratio)
                
                # 扩大边界框
                padding = 3
                minr_orig = max(0, minr_orig - padding)
                minc_orig = max(0, minc_orig - padding)
                maxr_orig = min(frame.shape[0], maxr_orig + padding)
                maxc_orig = min(frame.shape[1], maxc_orig + padding)
                
                # 在结果图像上绘制矩形
                cv2.rectangle(result_image, (minc_orig, minr_orig), (maxc_orig, maxr_orig), (0, 0, 255), 2)
        
        return label_image, result_image
    
    def toggle_playback(self):
        """切换播放/暂停状态"""
        if not self.label_video_path or not self.result_video_path:
            messagebox.showwarning("警告", "请先处理视频")
            return
        
        if self.playing:
            # 暂停播放
            self.playing = False
            self.play_btn.config(text="播放结果")
        else:
            # 开始播放
            self.playing = True
            self.play_btn.config(text="暂停播放")
            
            # 在后台线程中播放视频
            threading.Thread(target=self._playback_thread, daemon=True).start()
    
    def _playback_thread(self):
        """在后台线程中播放视频"""
        try:
            # 打开视频文件
            label_cap = cv2.VideoCapture(self.label_video_path)
            result_cap = cv2.VideoCapture(self.result_video_path)
            
            # 获取帧率
            fps = label_cap.get(cv2.CAP_PROP_FPS)
            delay = int(1000 / fps)  # 毫秒
            
            # 播放视频
            while self.playing and label_cap.isOpened() and result_cap.isOpened():
                # 读取帧
                ret1, label_frame = label_cap.read()
                ret2, result_frame = result_cap.read()
                
                if not ret1 or not ret2:
                    # 到达视频末尾，循环播放
                    label_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    result_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # 转换为RGB
                label_frame_rgb = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # 调整大小以适应画布
                label_frame_resized = cv2.resize(label_frame_rgb, (640, 480))
                result_frame_resized = cv2.resize(result_frame_rgb, (640, 480))
                
                # 在主线程中更新UI
                self.parent.after(0, lambda l=label_frame_resized: self.display_frame_on_canvas(l, self.label_canvas))
                self.parent.after(0, lambda r=result_frame_resized: self.display_frame_on_canvas(r, self.result_canvas))
                
                # 等待一段时间
                time.sleep(delay / 1000.0)
            
            # 关闭视频
            label_cap.release()
            result_cap.release()
            
            # 如果不是用户暂停的，则重置按钮状态
            if not self.playing:
                self.parent.after(0, lambda: self.play_btn.config(text="播放结果"))
            
        except Exception as e:
            self.parent.after(0, lambda: self.status_var.set("视频播放失败"))
            self.parent.after(0, lambda: messagebox.showerror("错误", f"视频播放失败: {str(e)}"))
            self.parent.after(0, lambda: self.play_btn.config(text="播放结果"))
            self.playing = False
    
    def save_results(self):
        """保存处理结果"""
        if not self.label_video_path or not self.result_video_path:
            messagebox.showwarning("警告", "请先处理视频")
            return
        
        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        
        try:
            self.status_var.set("正在保存视频...")
            
            # 构造文件名
            basename = os.path.basename(self.selected_video).split('.')[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存路径
            label_save_path = os.path.join(save_dir, f"{basename}_label_{timestamp}.mp4")
            result_save_path = os.path.join(save_dir, f"{basename}_result_{timestamp}.mp4")
            
            # 复制文件
            import shutil
            shutil.copy2(self.label_video_path, label_save_path)
            shutil.copy2(self.result_video_path, result_save_path)
            
            self.status_var.set("视频保存成功")
            messagebox.showinfo("保存成功", 
                             f"预测标签视频已保存至:\n{label_save_path}\n\n"
                             f"检测结果视频已保存至:\n{result_save_path}")
            
        except Exception as e:
            self.status_var.set("视频保存失败")
            messagebox.showerror("错误", f"保存视频失败: {str(e)}")