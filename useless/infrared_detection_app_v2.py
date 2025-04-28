import tkinter as tk
import torch
import cv2  # 注意：与video_prediction_gui.py保持相同的导入顺序
import os
import time
import json
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from torchvision import transforms
from datetime import datetime
from threading import Thread, Event
from queue import Queue

# 导入模型相关组件
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param

class InfraredDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("红外小目标检测系统")
        self.root.geometry("1280x800")
        
        # 检查GPU是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 初始化变量
        self.model = None
        self.model_loaded = False
        self.model_path = None
        self.current_mode = "image"  # 默认图像模式
        
        # 视频模式相关变量
        self.video_path = None
        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.fps = 0
        self.playing = False
        self.stop_event = Event()
        self.frame_queue = Queue(maxsize=10)
        
        # 图像模式相关变量
        self.selected_image = None
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        
        # 加载配置
        self.config = self.load_config()
        self.model_path = self.config.get("model_path", None)
        
        # 创建主UI
        self.create_ui()
        
        # 如果有保存的模型设置，尝试加载
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)
        
        # 关闭窗口时执行
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_config(self):
        """加载配置文件"""
        config = {}
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
        except Exception as e:
            print(f"加载配置文件出错: {e}")
        
        return config
    
    def save_config(self):
        """保存配置到文件"""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        try:
            config = {
                "model_path": self.model_path,
                "last_mode": self.current_mode
            }
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"保存配置文件出错: {e}")
    
    def create_ui(self):
        """创建用户界面"""
        # 主框架
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部模式选择区域
        self.mode_frame = ttk.LabelFrame(self.main_frame, text="检测模式")
        self.mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模式选择按钮
        self.image_mode_btn = ttk.Button(
            self.mode_frame, 
            text="图像模式", 
            command=self.switch_to_image_mode
        )
        self.image_mode_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.video_mode_btn = ttk.Button(
            self.mode_frame, 
            text="视频模式", 
            command=self.switch_to_video_mode
        )
        self.video_mode_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 模型信息标签
        if self.model_path and os.path.exists(self.model_path):
            model_name = os.path.basename(self.model_path)
            self.model_label = ttk.Label(
                self.mode_frame, 
                text=f"当前模型: {model_name}", 
                foreground="green"
            )
        else:
            self.model_label = ttk.Label(
                self.mode_frame, 
                text="未加载模型", 
                foreground="red"
            )
        self.model_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 内容区域
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
        
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=5)
        
        # 默认显示图像模式
        self.switch_to_image_mode()
    
    def switch_to_image_mode(self):
        """切换到图像模式"""
        # 清空内容区域
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self.current_mode = "image"
        self.image_mode_btn.state(['disabled'])
        self.video_mode_btn.state(['!disabled'])
        
        # 创建图像模式UI
        self.create_image_mode_ui()
        
        self.status_var.set("图像模式已加载")
    
    def create_image_mode_ui(self):
        """创建图像模式界面"""
        # 工具栏
        toolbar = ttk.Frame(self.content_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 工具栏按钮
        self.select_model_btn = ttk.Button(
            toolbar, 
            text="选择模型权重", 
            command=self.select_model
        )
        self.select_model_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_image_btn = ttk.Button(
            toolbar, 
            text="选择图片", 
            command=self.select_image
        )
        self.select_image_btn.pack(side=tk.LEFT, padx=5)
        
        self.image_detect_btn = ttk.Button(
            toolbar, 
            text="开始检测", 
            command=self.start_image_detection,
            state=tk.DISABLED
        )
        self.image_detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.image_save_btn = ttk.Button(
            toolbar, 
            text="保存结果", 
            command=self.save_image_results,
            state=tk.DISABLED
        )
        self.image_save_btn.pack(side=tk.LEFT, padx=5)
        
        # 图像显示区域
        display_frame = ttk.Frame(self.content_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 四个显示区域
        # 原始图像
        self.orig_frame = ttk.LabelFrame(display_frame, text="原始图像")
        self.orig_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        
        self.orig_canvas = tk.Canvas(self.orig_frame, width=300, height=300, bg="black")
        self.orig_canvas.pack(padx=5, pady=5)
        
        # 真实标签
        self.true_frame = ttk.LabelFrame(display_frame, text="真实标签")
        self.true_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        
        self.true_canvas = tk.Canvas(self.true_frame, width=300, height=300, bg="black")
        self.true_canvas.pack(padx=5, pady=5)
        
        # 预测标签
        self.pred_frame = ttk.LabelFrame(display_frame, text="预测标签")
        self.pred_frame.grid(row=0, column=2, padx=5, pady=5, sticky=tk.NSEW)
        
        self.pred_canvas = tk.Canvas(self.pred_frame, width=300, height=300, bg="black")
        self.pred_canvas.pack(padx=5, pady=5)
        
        # 预测结果
        self.result_frame = ttk.LabelFrame(display_frame, text="预测结果")
        self.result_frame.grid(row=0, column=3, padx=5, pady=5, sticky=tk.NSEW)
        
        self.result_canvas = tk.Canvas(self.result_frame, width=300, height=300, bg="black")
        self.result_canvas.pack(padx=5, pady=5)
        
        # 设置网格权重
        for i in range(4):
            display_frame.columnconfigure(i, weight=1)
        display_frame.rowconfigure(0, weight=1)
    
    def switch_to_video_mode(self):
        """切换到视频模式"""
        # 清空内容区域
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self.current_mode = "video"
        self.image_mode_btn.state(['!disabled'])
        self.video_mode_btn.state(['disabled'])
        
        # 创建视频模式UI
        self.create_video_mode_ui()
        
        self.status_var.set("视频模式已加载")
    
    def create_video_mode_ui(self):
        """创建视频模式界面"""
        # 工具栏
        toolbar = ttk.Frame(self.content_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # 工具栏按钮
        self.select_model_btn = ttk.Button(
            toolbar, 
            text="选择模型权重", 
            command=self.select_model
        )
        self.select_model_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_video_btn = ttk.Button(
            toolbar, 
            text="选择视频文件", 
            command=self.select_video
        )
        self.select_video_btn.pack(side=tk.LEFT, padx=5)
        
        self.process_btn = ttk.Button(
            toolbar, 
            text="开始检测", 
            command=self.start_video_detection,
            state=tk.DISABLED
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(
            toolbar, 
            text="播放结果", 
            command=self.toggle_play,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            toolbar, 
            text="保存结果", 
            command=self.save_video_results,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 视频显示区域
        display_frame = ttk.Frame(self.content_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧显示处理标签
        self.label_frame = ttk.LabelFrame(display_frame, text="预测标签")
        self.label_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        
        self.label_canvas = tk.Canvas(self.label_frame, width=600, height=450, bg="black")
        self.label_canvas.pack(padx=5, pady=5)
        
        # 右侧显示处理结果
        self.video_result_frame = ttk.LabelFrame(display_frame, text="检测结果")
        self.video_result_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        
        self.video_result_canvas = tk.Canvas(self.video_result_frame, width=600, height=450, bg="black")
        self.video_result_canvas.pack(padx=5, pady=5)
        
        # 设置网格权重
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # 进度控制条
        control_frame = ttk.Frame(self.content_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Scale(
            control_frame, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL,
            variable=self.progress_var,
            command=self.seek_video
        )
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def select_model(self):
        """选择模型权重文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch模型", "*.pth;*.pth.tar"), ("所有文件", "*.*")],
            initialdir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")  # 从result目录选择模型
        )
        
        if file_path:
            self.status_var.set(f"正在加载模型: {os.path.basename(file_path)}...")
            
            # 在后台线程中加载模型，避免UI冻结
            Thread(target=self._load_model_thread, args=(file_path,), daemon=True).start()
    
    def _load_model_thread(self, file_path):
        """在后台线程中加载模型"""
        try:
            self.load_model(file_path)
            
            # 更新UI（在主线程中）
            self.root.after(0, lambda: self.status_var.set(f"模型加载成功: {os.path.basename(file_path)}"))
            self.root.after(0, lambda: messagebox.showinfo("成功", "模型加载成功"))
            
            # 启用相关按钮
            if self.current_mode == "image" and self.selected_image:
                self.root.after(0, lambda: self.image_detect_btn.config(state=tk.NORMAL))
            elif self.current_mode == "video" and self.video_path:
                self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("模型加载失败"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"模型加载失败: {str(e)}"))
    
    def load_model(self, model_path):
        """加载模型"""
        if not model_path or not os.path.exists(model_path):
            return False
        
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
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        # 更新模型和状态
        self.model = model
        self.model_loaded = True
        self.model_path = model_path
        
        # 更新模型标签
        model_name = os.path.basename(model_path)
        self.model_label.config(text=f"当前模型: {model_name}", foreground="green")
        
        # 保存配置
        self.save_config()
        
        return True
    
    #---------------------------------
    # 图像模式相关方法
    #---------------------------------
    def select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.selected_image = file_path
            self.load_and_display_image(file_path)
            self.status_var.set(f"已选择图片: {os.path.basename(file_path)}")
            
            # 如果模型已加载，启用检测按钮
            if self.model_loaded:
                self.image_detect_btn.config(state=tk.NORMAL)
    
    def load_and_display_image(self, file_path):
        """加载并显示图片"""
        try:
            # 打开原图并调整大小
            original_image = Image.open(file_path).convert("RGB")
            original_image_resized = original_image.resize((300, 300))
            
            # 显示原图
            self.orig_photo = ImageTk.PhotoImage(original_image_resized)
            self.orig_canvas.create_image(0, 0, anchor=tk.NW, image=self.orig_photo)
            
            # 尝试加载对应的真实标签（如果存在）
            self._try_load_ground_truth(file_path)
            
            # 清除其他画布
            self.pred_canvas.delete("all")
            self.result_canvas.delete("all")
            
            # 禁用保存按钮
            self.image_save_btn.config(state=tk.DISABLED)
            
            return True
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            return False
    
    def _try_load_ground_truth(self, image_path):
        """尝试加载真实标签（如果存在）"""
        # 清空画布
        self.true_canvas.delete("all")
        
        try:
            # 获取图像文件名（不含路径）
            image_filename = os.path.basename(image_path)
            base_filename = os.path.splitext(image_filename)[0]
            
            # 构建掩码路径
            mask_path = os.path.join("dataset", "NUDT-SIRST", "masks", f"{base_filename}.png")
            absolute_mask_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mask_path)
            
            # 检查文件是否存在
            if os.path.exists(absolute_mask_path):
                print(f"找到真实标签: {absolute_mask_path}")
                mask_image = Image.open(absolute_mask_path).convert("L")
                mask_image_resized = mask_image.resize((300, 300))
                self.true_photo = ImageTk.PhotoImage(mask_image_resized)
                self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_photo)
            else:
                print(f"未找到图像的真实标签，尝试路径: {absolute_mask_path}")
        except Exception as e:
            print(f"加载真实标签失败: {e}")
            import traceback
            traceback.print_exc()
    
    def start_image_detection(self):
        """开始图像检测"""
        if not self.selected_image:
            messagebox.showwarning("警告", "请先选择图片文件")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        self.status_var.set("正在检测...")
        
        # 在后台线程中执行检测，避免UI冻结
        Thread(target=self._image_detection_thread, daemon=True).start()
    
    def _image_detection_thread(self):
        """在后台线程中执行图像检测"""
        try:
            # 执行检测
            true_label, predicted_result = self.predict_image(self.selected_image)
            
            # 生成标注图像
            annotated_image = self.generate_annotated_image(predicted_result, self.selected_image)
            
            # 保存结果
            self.current_image_results = {
                "predicted_result": predicted_result,
                "annotated_image": annotated_image,
                "original_image_path": self.selected_image
            }
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self._update_image_detection_ui(true_label, predicted_result, annotated_image))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set("检测失败"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"检测失败: {str(e)}"))
    
    def _update_image_detection_ui(self, true_label, predicted_result, annotated_image):
        """更新图像检测结果UI"""
        # 显示真实标签(如果有)
        if true_label is not None:
            true_label_image = Image.fromarray((true_label * 255).astype(np.uint8)).resize((300, 300))
            self.true_photo = ImageTk.PhotoImage(true_label_image)
            self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_photo)
        
        # 显示预测标签
        predicted_image = Image.fromarray((predicted_result * 255).astype(np.uint8)).resize((300, 300))
        self.predicted_photo = ImageTk.PhotoImage(predicted_image)
        self.pred_canvas.create_image(0, 0, anchor=tk.NW, image=self.predicted_photo)
        
        # 显示标注结果
        annotated_image_resized = annotated_image.resize((300, 300))
        self.annotated_photo = ImageTk.PhotoImage(annotated_image_resized)
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.annotated_photo)
        
        # 启用保存按钮
        self.image_save_btn.config(state=tk.NORMAL)
        
        # 更新状态
        self.status_var.set("检测完成")
    
    def predict_image(self, image_path):
        """执行图像预测"""
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 模型预测
        with torch.inference_mode():
            output = self.model(input_tensor)
            if isinstance(output, list):
                output = output[-1]
            predicted_result = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 尝试加载真实标签(如果有)
        true_label = None
        return true_label, predicted_result
    
    def generate_annotated_image(self, predicted_result, original_file_path):
        """生成带标注的图像，确保目标在框中居中且不被框覆盖"""
        # 加载原始图像
        original_image = Image.open(original_file_path).convert("RGB")
        original_image = original_image.resize((256, 256))
        
        # 创建副本用于绘制
        annotated_image = original_image.copy()
        
        # 二值化
        threshold = 0.3
        binary_mask = (predicted_result > threshold).astype(np.uint8)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建可以绘制的图像
        img_array = np.array(annotated_image)
        
        # 绘制优化的边界框
        for contour in contours:
            # 计算轮廓的矩
            M = cv2.moments(contour)
            
            # 计算质心
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:  # 如果轮廓面积为0，则使用边界框中心
                x, y, w, h = cv2.boundingRect(contour)
                cX = x + w // 2
                cY = y + h // 2
                
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 根据目标大小设置边界框大小
            # 对于小目标，设置最小边框大小
            min_box_size = max(10, int(np.sqrt(area) * 3))
            
            # 以质心为中心创建边界框
            half_size = min_box_size // 2
            
            # 确保边界框不超出图像边界
            left = max(0, cX - half_size)
            top = max(0, cY - half_size)
            right = min(binary_mask.shape[1], cX + half_size)
            bottom = min(binary_mask.shape[0], cY + half_size)
            
            # 绘制边界框，使用红色(RGB:255,0,0)
            cv2.rectangle(img_array, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # 可选：在边界框附近标注目标序号
            # cv2.putText(img_array, f"{i+1}", (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 转回PIL格式
        annotated_image = Image.fromarray(img_array)
        return annotated_image
    
    def save_image_results(self):
        """保存图像检测结果"""
        if not hasattr(self, 'current_image_results'):
            messagebox.showinfo("提示", "请先执行检测!")
            return
        
        try:
            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   "predicts", "images")
            os.makedirs(save_dir, exist_ok=True)
            
            # 获取当前结果
            predicted_result = self.current_image_results["predicted_result"]
            annotated_image = self.current_image_results["annotated_image"]
            original_file_path = self.current_image_results["original_image_path"]
            
            # 构造保存文件名
            file_name = os.path.basename(original_file_path).split(".")[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存预测掩码
            mask_save_path = os.path.join(save_dir, f"{file_name}_mask_{timestamp}.png")
            mask_image = Image.fromarray((predicted_result * 255).astype(np.uint8))
            mask_image.save(mask_save_path)
            
            # 保存带标注的图像
            annotated_save_path = os.path.join(save_dir, f"{file_name}_annotated_{timestamp}.png")
            annotated_image.save(annotated_save_path)
            
            messagebox.showinfo("保存成功", 
                             f"预测掩码已保存至: {mask_save_path}\n"
                             f"标注结果已保存至: {annotated_save_path}")
            
            self.status_var.set("结果已保存")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存结果失败: {str(e)}")
    
    #---------------------------------
    # 视频模式相关方法
    #---------------------------------
    def select_video(self):
        """选择视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.video_path = file_path
            self.status_var.set(f"正在加载视频: {os.path.basename(file_path)}...")
            
            try:
                self.load_video(file_path)
                self.status_var.set(f"视频加载成功: {os.path.basename(file_path)}")
                
                # 如果模型已加载，启用检测按钮
                if self.model_loaded:
                    self.process_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.status_var.set("视频加载失败")
                messagebox.showerror("错误", f"视频加载失败: {str(e)}")
    
    def load_video(self, file_path):
        """加载视频文件并获取基本信息"""
        if self.cap is not None:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(file_path)
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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_frame(frame_rgb, self.video_result_canvas)
            self.display_frame(frame_rgb, self.label_canvas)
        
        # 重置视频到开始
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def display_frame(self, frame, canvas):
        """在画布上显示帧"""
        if frame is None:
            return
        
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
    
    def start_video_detection(self):
        """开始视频检测过程"""
        if not self.model_loaded:
            messagebox.showerror("错误", "请先加载模型")
            return
            
        if not self.video_path or not self.cap:
            messagebox.showerror("错误", "请先加载视频")
            return
        
        # 停止当前播放
        self.stop_playback()
        
        self.status_var.set("开始处理视频...")
        # 禁用按钮
        self.process_btn.config(state=tk.DISABLED)
        self.select_video_btn.config(state=tk.DISABLED)
        self.select_model_btn.config(state=tk.DISABLED)
        
        # 重置视频到第一帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        self.progress_var.set(0)
        
        # 开始检测线程
        self.detection_thread = Thread(target=self.process_video)
        self.detection_thread.daemon = True
        self.detection_thread.start()
    
    def process_video(self):
        """处理视频线程"""
        try:
            # 创建临时目录保存处理后的视频
            import tempfile
            temp_dir = tempfile.mkdtemp()
            temp_label_path = os.path.join(temp_dir, "temp_label.mp4")
            temp_result_path = os.path.join(temp_dir, "temp_result.mp4")
            
            # 获取视频尺寸
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            label_writer = cv2.VideoWriter(temp_label_path, fourcc, self.fps, (width, height))
            result_writer = cv2.VideoWriter(temp_result_path, fourcc, self.fps, (width, height))
            
            # 批处理参数
            batch_size = 1  # 每次处理的帧数
            total_processed = 0
            
            # 处理视频
            while total_processed < self.frame_count:
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 处理帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label_frame, result_frame = self.process_frame(frame_rgb)
                
                # 转回BGR并写入
                label_bgr = cv2.cvtColor(label_frame, cv2.COLOR_RGB2BGR)
                result_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                
                label_writer.write(label_bgr)
                result_writer.write(result_bgr)
                
                # 更新进度
                total_processed += 1
                progress = (total_processed / self.frame_count) * 100
                self.root.after(0, lambda p=total_processed: self.progress_var.set(p))
                self.root.after(0, lambda p=progress: 
                              self.status_var.set(f"处理中: {p:.1f}% ({total_processed}/{self.frame_count})"))
                
                # 定期更新显示
                if total_processed % 10 == 0 or total_processed == self.frame_count:
                    self.root.after(0, lambda l=label_frame: self.display_frame(l, self.label_canvas))
                    self.root.after(0, lambda r=result_frame: self.display_frame(r, self.video_result_canvas))
            
            # 完成处理
            label_writer.release()
            result_writer.release()
            
            # 保存视频路径
            self.label_video_path = temp_label_path
            self.result_video_path = temp_result_path
            
            # 更新UI
            self.root.after(0, lambda: self.status_var.set("视频处理完成"))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.select_video_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.select_model_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.play_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("成功", "视频处理完成"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"处理出错: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"处理视频时出错: {str(e)}"))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.select_video_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.select_model_btn.config(state=tk.NORMAL))
    
    def process_frame(self, frame):
        """处理单个视频帧，优化目标框显示"""
        # 转换为PIL图像
        pil_image = Image.fromarray(frame)
        
        # 预处理并进行预测
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = self.model(input_tensor)
            if isinstance(output, list):
                output = output[-1]
            predicted_result = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # 创建标签图像
        label_image = np.zeros_like(frame)
        
        # 二值化预测结果
        threshold = 0.3
        binary_mask = (predicted_result > threshold).astype(np.uint8)
        
        # 调整掩码大小为原始帧大小
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(binary_mask * 255, (w, h))
        
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
            min_box_size = max(min_box_size, int(w * 0.02))  # 至少为图像宽度的2%
            
            # 以质心为中心创建边界框
            half_size = min_box_size // 2
            
            # 确保边界框不超出图像边界
            left = max(0, cX - half_size)
            top = max(0, cY - half_size)
            right = min(w, cX + half_size)
            bottom = min(h, cY + half_size)
            
            # 绘制边界框，使用红色
            cv2.rectangle(result_image, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # 可选：在边界框附近标注目标编号
            cv2.putText(result_image, f"{target_count}", (left, top-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 添加目标计数文本
        cv2.putText(result_image, f"目标数: {target_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return label_image, result_image
    
    def toggle_play(self):
        """切换视频播放/暂停状态"""
        if not hasattr(self, 'label_video_path') or not hasattr(self, 'result_video_path'):
            messagebox.showinfo("提示", "请先处理视频")
            return
            
        if self.playing:
            self.stop_playback()
            self.play_btn.config(text="播放结果")
        else:
            self.start_playback()
            self.play_btn.config(text="暂停")
    
    def start_playback(self):
        """开始播放视频"""
        self.playing = True
        self.stop_event.clear()
        
        # 加载处理后的视频
        self.label_cap = cv2.VideoCapture(self.label_video_path)
        self.result_cap = cv2.VideoCapture(self.result_video_path)
        
        # 启动播放线程
        playback_thread = Thread(target=self.playback_loop)
        playback_thread.daemon = True
        playback_thread.start()
    
    def playback_loop(self):
        """视频播放循环"""
        frame_time = 1.0 / self.fps if self.fps > 0 else 0.033
        
        try:
            while self.playing and not self.stop_event.is_set():
                # 读取标签帧
                label_ret, label_frame = self.label_cap.read()
                # 读取结果帧
                result_ret, result_frame = self.result_cap.read()
                
                if not label_ret or not result_ret:
                    # 循环播放
                    self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # 转换颜色
                label_rgb = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
                result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # 显示帧
                self.root.after(0, lambda l=label_rgb: self.display_frame(l, self.label_canvas))
                self.root.after(0, lambda r=result_rgb: self.display_frame(r, self.video_result_canvas))
                
                # 更新进度条
                current_pos = self.label_cap.get(cv2.CAP_PROP_POS_FRAMES)
                self.root.after(0, lambda p=current_pos: self.progress_var.set(p))
                
                # 延时
                time.sleep(frame_time)
                
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"播放错误: {str(e)}"))
        finally:
            self.playing = False
            self.root.after(0, lambda: self.play_btn.config(text="播放结果"))
    
    def stop_playback(self):
        """停止视频播放"""
        self.playing = False
        self.stop_event.set()
        
        # 释放视频资源
        if hasattr(self, 'label_cap') and self.label_cap:
            self.label_cap.release()
        if hasattr(self, 'result_cap') and self.result_cap:
            self.result_cap.release()
    
    def seek_video(self, value):
        """跳转到视频的特定位置"""
        if not hasattr(self, 'label_cap') or not hasattr(self, 'result_cap'):
            return
            
        frame_idx = int(float(value))
        
        # 设置视频帧位置
        self.label_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        self.result_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取并显示当前帧
        label_ret, label_frame = self.label_cap.read()
        result_ret, result_frame = self.result_cap.read()
        
        if label_ret and result_ret:
            label_rgb = cv2.cvtColor(label_frame, cv2.COLOR_BGR2RGB)
            result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            
            self.display_frame(label_rgb, self.label_canvas)
            self.display_frame(result_rgb, self.video_result_canvas)
    
    def save_video_results(self):
        """保存处理结果视频"""
        if not hasattr(self, 'label_video_path') or not hasattr(self, 'result_video_path'):
            messagebox.showwarning("警告", "请先处理视频")
            return
        
        # 选择保存目录
        save_dir = filedialog.askdirectory(title="选择保存目录")
        if not save_dir:
            return
        
        try:
            import shutil
            self.status_var.set("正在保存视频...")
            
            # 构造文件名
            basename = os.path.basename(self.video_path).split('.')[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存路径
            label_save_path = os.path.join(save_dir, f"{basename}_label_{timestamp}.mp4")
            result_save_path = os.path.join(save_dir, f"{basename}_result_{timestamp}.mp4")
            
            # 复制文件
            shutil.copy2(self.label_video_path, label_save_path)
            shutil.copy2(self.result_video_path, result_save_path)
            
            self.status_var.set("视频保存成功")
            messagebox.showinfo("保存成功", 
                             f"预测标签视频已保存至:\n{label_save_path}\n\n"
                             f"检测结果视频已保存至:\n{result_save_path}")
            
        except Exception as e:
            self.status_var.set("视频保存失败")
            messagebox.showerror("错误", f"保存视频失败: {str(e)}")
    
    def on_closing(self):
        """关闭应用程序"""
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
        
        # 保存配置
        self.save_config()
        
        # 关闭窗口
        self.root.destroy()

# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = InfraredDetectionApp(root)
    root.mainloop()