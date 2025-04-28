import cv2  # 确保cv2在最前面导入
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
from datetime import datetime
import threading
import time
import traceback
import tempfile
from queue import Queue
from threading import Event
from torchvision import transforms

# 导入模型相关组件
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param

class VideoModeFrame:
    def __init__(self, parent, status_var, model_path=None, update_model_callback=None, logger=None):
        self.parent = parent
        self.status_var = status_var
        self.model_path = model_path
        self.update_model_callback = update_model_callback
        self.logger = logger  # 日志记录器
        
        # 初始化变量
        self.model = None
        self.model_loaded = False
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.current_frame = 0
        self.playing = False
        self.stop_event = Event()
        self.frame_queue = Queue(maxsize=10)
        
        # 临时视频文件路径
        self.label_video_path = None
        self.result_video_path = None
        
        # 预览相关
        self.label_cap = None
        self.result_cap = None
        
        # 图像转换
        self.transform = torch.nn.Sequential(
            transforms.Resize((256, 256)),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        )
        
        # 设备选择
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.logger:
            self.logger.info(f"视频模式初始化，使用设备：{self.device}")
        
        # 创建界面
        self.create_ui()
        
        # 如果已有模型路径，尝试加载
        if self.model_path and os.path.exists(self.model_path):
            self._load_model_thread(self.model_path)
    
    def create_ui(self):
        """创建用户界面"""
        # 工具栏
        self.toolbar = ttk.Frame(self.parent)
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 添加按钮
        self.select_model_btn = ttk.Button(
            self.toolbar, 
            text="选择模型", 
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
            command=self.start_processing,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(
            self.toolbar, 
            text="播放结果", 
            command=self.toggle_play,
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
        
        # 左右两个视频显示区
        # 左侧：预测标签
        self.label_frame = ttk.LabelFrame(self.display_frame, text="预测标签")
        self.label_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        
        self.label_canvas = tk.Canvas(self.label_frame, width=600, height=450, bg="black")
        self.label_canvas.pack(padx=5, pady=5)
        
        # 右侧：检测结果
        self.result_frame = ttk.LabelFrame(self.display_frame, text="检测结果")
        self.result_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        
        self.result_canvas = tk.Canvas(self.result_frame, width=600, height=450, bg="black")
        self.result_canvas.pack(padx=5, pady=5)
        
        # 配置网格权重
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
        
        # 进度控制区域
        self.control_frame = ttk.Frame(self.parent)
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(
            self.control_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            command=self.seek_video
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # 进度信息标签
        self.progress_info = ttk.Label(self.control_frame, text="0/0")
        self.progress_info.pack(side=tk.RIGHT, padx=5)
    
    def select_model(self):
        """选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")],
            initialdir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result")
        )
        
        if file_path:
            self.status_var.set("正在加载模型...")
            if self.logger:
                self.logger.info(f"选择模型文件: {file_path}")
            
            # 在后台线程中加载模型
            threading.Thread(target=self._load_model_thread, args=(file_path,), daemon=True).start()
    
    def _load_model_thread(self, file_path):
        """在后台线程中加载模型"""
        try:
            if self.logger:
                self.logger.info(f"开始加载模型: {file_path}")
            
            # 加载模型参数
            nb_filter, num_blocks = load_param('three', 'resnet_18')
            
            # 创建模型
            model = DNANet(
                num_classes=1, 
                input_channels=3, 
                block=Res_CBAM_block, 
                num_blocks=num_blocks, 
                nb_filter=nb_filter, 
                deep_supervision=True
            )
            
            # 加载权重
            checkpoint = torch.load(file_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(self.device)
            model.eval()
            
            # 更新模型
            self.model = model
            self.model_loaded = True
            self.model_path = file_path
            
            # 更新UI
            self.parent.after(0, lambda: self.status_var.set("模型加载成功"))
            
            # 如果已选择视频，启用检测按钮
            if self.video_path:
                self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            
            # 更新主应用中的模型路径
            if self.update_model_callback:
                self.update_model_callback(file_path)
            
            if self.logger:
                self.logger.info(f"模型加载成功: {file_path}")
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.parent.after(0, lambda: self.status_var.set(error_msg))
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            self.parent.after(0, lambda: messagebox.showerror("错误", error_msg))
    
    def select_video(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"正在加载视频: {os.path.basename(file_path)}...")
            
            if self.logger:
                self.logger.info(f"选择视频文件: {file_path}")
            
            try:
                # 加载视频
                self.load_video(file_path)
                
                # 如果模型已加载，启用检测按钮
                if self.model_loaded:
                    self.process_btn.config(state=tk.NORMAL)
                
                self.status_var.set(f"已加载视频: {os.path.basename(file_path)}")
                
            except Exception as e:
                error_msg = f"加载视频失败: {str(e)}"
                self.status_var.set(error_msg)
                
                if self.logger:
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())
                
                messagebox.showerror("错误", error_msg)
    
    def load_video(self, file_path):
        """加载视频并获取基本信息"""
        try:
            if self.logger:
                self.logger.info(f"开始加载视频: {file_path}")
            
            # 如果有之前的视频捕获对象，释放它
            if self.cap is not None:
                self.cap.release()
            
            # 创建新的视频捕获对象
            self.cap = cv2.VideoCapture(file_path)
            
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频文件: {file_path}")
            
            # 获取视频基本信息
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.current_frame = 0
            
            if self.logger:
                self.logger.info(f"视频信息 - 帧数: {self.frame_count}, FPS: {self.fps}, 分辨率: {self.width}x{self.height}")
            
            # 配置进度条
            self.progress_bar.config(from_=0, to=self.frame_count-1)
            self.progress_var.set(0)
            self.progress_info.config(text=f"0/{self.frame_count}")
            
            # 显示第一帧
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_frame_on_canvas(frame_rgb, self.label_canvas)
                self.display_frame_on_canvas(frame_rgb, self.result_canvas)
            
            # 重置视频到开始
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载视频失败: {str(e)}")
                self.logger.error(traceback.format_exc())
            raise
    
    def display_frame_on_canvas(self, frame, canvas):
        """在画布上显示视频帧"""
        if frame is None:
            if self.logger:
                self.logger.warning("尝试显示空帧")
            return
        
        try:
            # 获取画布尺寸
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # 如果画布还未初始化，使用默认尺寸
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = canvas.winfo_reqwidth()
                canvas_height = canvas.winfo_reqheight()
            
            # 调整图像大小以适应画布
            h, w = frame.shape[:2]
            ratio = min(canvas_width/w, canvas_height/h)
            new_size = (int(w*ratio), int(h*ratio))
            
            # 调整帧大小
            resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            
            # 转换为PIL图像
            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # 在画布中显示
            canvas.delete("all")
            canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
            canvas.image = img_tk  # 保留引用，防止垃圾回收
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"显示帧出错: {str(e)}")
                self.logger.error(traceback.format_exc())
            
            # 显示错误信息
            canvas.delete("all")
            canvas.create_text(canvas_width//2, canvas_height//2, 
                              text="显示错误", fill="red", font=("Arial", 14))
    
    def start_processing(self):
        """开始处理视频"""
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        if not self.video_path or not self.cap:
            messagebox.showwarning("警告", "请先加载视频")
            return
        
        # 停止当前播放
        self.stop_playback()
        
        # 更新状态
        self.status_var.set("开始处理视频...")
        
        if self.logger:
            self.logger.info(f"开始处理视频: {self.video_path}")
        
        # 禁用按钮
        self.process_btn.config(state=tk.DISABLED)
        self.select_video_btn.config(state=tk.DISABLED)
        self.select_model_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        
        # 重置视频到开始
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        self.progress_var.set(0)
        self.progress_info.config(text=f"0/{self.frame_count}")
        
        # 创建处理线程
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def process_video(self):
        """处理视频线程"""
        try:
            if self.logger:
                self.logger.info(f"视频处理线程启动，处理视频: {self.video_path}")
            
            # 创建临时目录保存处理后的视频
            temp_dir = tempfile.mkdtemp()
            temp_label_path = os.path.join(temp_dir, "temp_label.mp4")
            temp_result_path = os.path.join(temp_dir, "temp_result.mp4")
            
            if self.logger:
                self.logger.info(f"创建临时目录: {temp_dir}")
                self.logger.info(f"临时标签视频路径: {temp_label_path}")
                self.logger.info(f"临时结果视频路径: {temp_result_path}")
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            label_writer = cv2.VideoWriter(temp_label_path, fourcc, self.fps, (self.width, self.height))
            result_writer = cv2.VideoWriter(temp_result_path, fourcc, self.fps, (self.width, self.height))
            
            if not label_writer.isOpened() or not result_writer.isOpened():
                error_msg = "无法创建视频写入器"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 处理视频帧
            total_processed = 0
            error_frames = 0
            
            while total_processed < self.frame_count:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    if self.logger:
                        self.logger.warning(f"无法读取帧 {total_processed}/{self.frame_count}")
                    break
                
                try:
                    # 处理帧
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label_frame, result_frame = self.process_frame(frame_rgb)
                    
                    # 转回BGR并写入
                    label_bgr = cv2.cvtColor(label_frame, cv2.COLOR_RGB2BGR)
                    result_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    
                    label_writer.write(label_bgr)
                    result_writer.write(result_bgr)
                    
                except Exception as frame_error:
                    if self.logger:
                        self.logger.error(f"处理帧 {total_processed} 时出错: {str(frame_error)}")
                        self.logger.error(traceback.format_exc())
                    
                    # 写入原始帧作为错误帧
                    label_writer.write(frame)
                    
                    # 创建带有错误信息的结果帧
                    error_frame = frame.copy()
                    cv2.putText(error_frame, f"Error: {str(frame_error)[:30]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    result_writer.write(error_frame)
                    
                    error_frames += 1
                
                # 更新进度
                total_processed += 1
                progress = (total_processed / self.frame_count) * 100
                
                # 更新UI
                self.parent.after(0, lambda p=total_processed: self.progress_var.set(p))
                self.parent.after(0, lambda p=total_processed: 
                                self.progress_info.config(text=f"{p}/{self.frame_count}"))
                self.parent.after(0, lambda p=progress: 
                                self.status_var.set(f"处理中: {p:.1f}% ({total_processed}/{self.frame_count})"))
                
                # 定期记录进度
                if total_processed % 30 == 0 or total_processed == self.frame_count:
                    if self.logger:
                        self.logger.info(f"视频处理进度: {progress:.1f}% ({total_processed}/{self.frame_count})")
                
                # 定期更新显示
                if total_processed % 10 == 0 or total_processed == self.frame_count:
                    self.parent.after(0, lambda l=label_frame: self.display_frame_on_canvas(l, self.label_canvas))
                    self.parent.after(0, lambda r=result_frame: self.display_frame_on_canvas(r, self.result_canvas))
            
            # 完成处理，关闭写入器
            label_writer.release()
            result_writer.release()
            
            # 保存处理后的视频路径
            self.label_video_path = temp_label_path
            self.result_video_path = temp_result_path
            
            if self.logger:
                self.logger.info(f"视频处理完成，共处理 {total_processed}/{self.frame_count} 帧，错误帧: {error_frames}")
            
            # 更新UI
            completion_msg = f"处理完成 - 总帧数: {total_processed}, 错误帧: {error_frames}"
            self.parent.after(0, lambda: self.status_var.set(completion_msg))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_video_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_model_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            
            if error_frames > 0:
                warning_msg = f"视频处理完成，但有 {error_frames} 帧处理出错。详情请查看日志。"
                self.parent.after(0, lambda: messagebox.showwarning("警告", warning_msg))
            else:
                self.parent.after(0, lambda: messagebox.showinfo("成功", "视频处理完成"))
            
        except Exception as e:
            error_msg = f"处理视频时出错: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            # 更新UI
            self.parent.after(0, lambda: self.status_var.set(f"处理失败: {str(e)}"))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_video_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.select_model_btn.config(state=tk.NORMAL))
            
            self.parent.after(0, lambda: messagebox.showerror("错误", f"{error_msg}\n\n详细信息已记录到日志"))
    
    def process_frame(self, frame):
        """处理单个视频帧"""
        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(frame)
            
            # 转换为Tensor
            input_tensor = torch.from_numpy(np.array(pil_image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # 应用预处理
            input_tensor = self.transform(input_tensor).to(self.device)
            
            # 模型推理
            with torch.inference_mode():
                outputs = self.model(input_tensor)
                if isinstance(outputs, list):  # 处理深度监督的情况
                    output = outputs[-1]
                else:
                    output = outputs
                
                # 将输出转换为概率
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # 创建标签图像
            # 二值化预测结果
            threshold = 0.3
            binary_mask = (prob_map > threshold).astype(np.uint8)
            
            # 调整掩码大小为原始帧大小
            mask_resized = cv2.resize(binary_mask * 255, (frame.shape[1], frame.shape[0]))
            
            # 创建标签图像
            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
            label_image = mask_bgr
            
            # 在原始图像上标注目标
            result_image = frame.copy()
            
            # 查找轮廓
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 绘制优化的边界框
            target_count = 0
            for contour in contours:
                # 过滤过小的轮廓（可能是噪声）
                if cv2.contourArea(contour) < 5:
                    continue
                    
                target_count += 1
                
                # 计算轮廓的矩
                M = cv2.moments(contour)
                
                # 计算质心
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    cX = x + w // 2
                    cY = y + h // 2
                
                # 计算轮廓面积和适当的边框大小
                area = cv2.contourArea(contour)
                min_box_size = max(20, int(np.sqrt(area) * 3))
                
                # 确保最小大小不会太小
                min_box_size = max(min_box_size, int(frame.shape[1] * 0.02))  # 至少为图像宽度的2%
                
                # 以质心为中心创建边界框
                half_size = min_box_size // 2
                
                # 确保边界框不超出图像边界
                left = max(0, cX - half_size)
                top = max(0, cY - half_size)
                right = min(frame.shape[1], cX + half_size)
                bottom = min(frame.shape[0], cY + half_size)
                
                # 绘制边界框，使用红色
                cv2.rectangle(result_image, (left, top), (right, bottom), (255, 0, 0), 2)
                
                # 在边界框附近标注目标编号
                cv2.putText(result_image, f"{target_count}", (left, top-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 添加目标计数文本
            cv2.putText(result_image, f"Targets: {target_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return label_image, result_image
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"处理单帧时出错: {str(e)}")
                self.logger.error(traceback.format_exc())
            
            # 处理错误，创建错误信息帧
            error_frame = frame.copy()
            cv2.putText(error_frame, f"Error: {str(e)[:50]}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 返回原始帧和错误帧
            return frame, error_frame
    
    def toggle_play(self):
        """切换播放/暂停"""
        if not hasattr(self, 'label_video_path') or not hasattr(self, 'result_video_path'):
            messagebox.showwarning("警告", "请先处理视频")
            return
        
        if self.playing:
            self.stop_playback()
            self.play_btn.config(text="播放结果")
        else:
            self.start_playback()
            self.play_btn.config(text="暂停")
    
    def start_playback(self):
        """开始播放视频"""
        if self.logger:
            self.logger.info("开始播放处理后的视频")
        
        self.playing = True
        self.stop_event.clear()
        
        try:
            # 打开处理后的视频
            if self.label_cap is not None:
                self.label_cap.release()
            if self.result_cap is not None:
                self.result_cap.release()
            
            self.label_cap = cv2.VideoCapture(self.label_video_path)
            self.result_cap = cv2.VideoCapture(self.result_video_path)
            
            if not self.label_cap.isOpened() or not self.result_cap.isOpened():
                raise ValueError("无法打开处理后的视频")
            
            # 启动播放线程
            self.playback_thread = threading.Thread(target=self.playback_loop, daemon=True)
            self.playback_thread.start()
            
        except Exception as e:
            error_msg = f"启动播放失败: {str(e)}"
            self.status_var.set(error_msg)
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            messagebox.showerror("错误", error_msg)
            self.playing = False
            self.play_btn.config(text="播放结果")
    
    def playback_loop(self):
        """视频播放循环"""
        try:
            frame_time = 1.0 / self.fps if self.fps > 0 else 0.033  # 默认约30fps
            
            while self.playing and not self.stop_event.is_set():
                # 读取帧
                label_ret, label_frame = self.label_cap.read()
                result_ret, result_frame = self.result_cap.read()
                
                if not label_ret or not result_ret:
                    # 播放结束，循环播放
                    if self.logger:
                        self.logger.info("播放结束，重新开始")
                    
                    self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # 转换颜色空间
                label_rgb = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # 显示帧
                self.parent.after(0, lambda l=label_rgb: self.display_frame_on_canvas(l, self.label_canvas))
                self.parent.after(0, lambda r=result_rgb: self.display_frame_on_canvas(r, self.result_canvas))
                
                # 更新进度
                current_frame = int(self.label_cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.parent.after(0, lambda f=current_frame: self.progress_var.set(f))
                self.parent.after(0, lambda f=current_frame: 
                                self.progress_info.config(text=f"{f}/{self.frame_count}"))
                
                # 控制播放速度
                time.sleep(frame_time)
                
        except Exception as e:
            error_msg = f"播放视频时出错: {str(e)}"
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            self.parent.after(0, lambda: self.status_var.set(error_msg))
        finally:
            # 播放结束或出错，更新UI
            self.parent.after(0, lambda: self.play_btn.config(text="播放结果"))
            self.playing = False
    
    def stop_playback(self):
        """停止视频播放"""
        if self.logger:
            self.logger.info("停止视频播放")
        
        self.playing = False
        self.stop_event.set()
        
        # 释放视频资源
        if hasattr(self, 'label_cap') and self.label_cap:
            self.label_cap.release()
            self.label_cap = None
        
        if hasattr(self, 'result_cap') and self.result_cap:
            self.result_cap.release()
            self.result_cap = None
    
    def seek_video(self, value):
        """跳转到视频的特定位置"""
        if not hasattr(self, 'label_cap') or not hasattr(self, 'result_cap'):
            return
            
        try:
            frame_idx = int(float(value))
            
            # 更新进度信息
            self.progress_info.config(text=f"{frame_idx}/{self.frame_count}")
            
            # 设置视频位置
            self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # 读取并显示当前帧
            label_ret, label_frame = self.label_cap.read()
            result_ret, result_frame = self.result_cap.read()
            
            if label_ret and result_ret:
                label_rgb = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                self.display_frame_on_canvas(label_rgb, self.label_canvas)
                self.display_frame_on_canvas(result_rgb, self.result_canvas)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"跳转视频时出错: {str(e)}")
    
    def save_results(self):
        """保存处理结果"""
        if not hasattr(self, 'label_video_path') or not hasattr(self, 'result_video_path'):
            messagebox.showwarning("警告", "请先处理视频")
            return
        
        try:
            self.status_var.set("正在保存视频...")
            
            # 创建保存目录结构
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predicts")
            save_dir = os.path.join(base_dir, "videos")
            os.makedirs(save_dir, exist_ok=True)
            
            if self.logger:
                self.logger.info(f"开始保存处理结果到: {save_dir}")
            
            import shutil
            
            # 构造保存文件名
            basename = os.path.basename(self.video_path).split('.')[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 创建保存路径
            label_save_path = os.path.join(save_dir, f"{basename}_mask_{timestamp}.mp4")
            result_save_path = os.path.join(save_dir, f"{basename}_result_{timestamp}.mp4")
            
            # 复制文件
            shutil.copy2(self.label_video_path, label_save_path)
            shutil.copy2(self.result_video_path, result_save_path)
            
            if self.logger:
                self.logger.info(f"保存掩码视频到: {label_save_path}")
                self.logger.info(f"保存结果视频到: {result_save_path}")
            
            self.status_var.set(f"视频保存成功到 {save_dir}")
            messagebox.showinfo("保存成功", 
                            f"结果已保存到:\n{save_dir}\n\n"
                            f"预测掩码: {os.path.basename(label_save_path)}\n"
                            f"检测结果: {os.path.basename(result_save_path)}")
            
        except Exception as e:
            error_msg = f"保存视频失败: {str(e)}"
            self.status_var.set(error_msg)
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            messagebox.showerror("错误", error_msg)
    
    def on_closing(self):
        """关闭时的清理操作"""
        if self.logger:
            self.logger.info("清理视频模式资源")
        
        # 停止播放
        self.stop_playback()
        
        # 释放视频资源
        if self.cap:
            self.cap.release()