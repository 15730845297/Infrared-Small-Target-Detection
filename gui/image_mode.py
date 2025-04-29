import cv2  # 确保cv2在最前面导入
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
from torchvision import transforms
import numpy as np
from datetime import datetime
import threading
import traceback

# 导入自定义模块
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param

class ImageModeFrame:
    def __init__(self, parent, status_var, model_path=None, update_model_callback=None, logger=None):
        self.parent = parent
        self.status_var = status_var
        self.model_path = model_path
        self.update_model_callback = update_model_callback
        self.logger = logger  # 添加日志记录器
        
        # 初始化变量
        self.model = None
        self.model_loaded = False
        self.selected_image = None
        self.canvas_size = 300
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])
        
        # 创建界面
        self.create_ui()
        
        # 如果提供了模型路径，尝试加载模型
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
        
        self.select_image_btn = ttk.Button(
            self.toolbar, 
            text="选择图片", 
            command=self.select_image
        )
        self.select_image_btn.pack(side=tk.LEFT, padx=5)
        
        self.detect_btn = ttk.Button(
            self.toolbar, 
            text="开始检测", 
            command=self.detect_image,
            state=tk.DISABLED
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            self.toolbar, 
            text="保存结果", 
            command=self.save_results,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建显示区域
        self.display_frame = ttk.Frame(self.parent)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 四个画布
        # 原始图像
        self.orig_frame = ttk.LabelFrame(self.display_frame, text="原始图像")
        self.orig_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        
        self.orig_canvas = tk.Canvas(self.orig_frame, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.orig_canvas.pack(padx=5, pady=5)
        
        # 真实标签
        self.true_frame = ttk.LabelFrame(self.display_frame, text="真实标签")
        self.true_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        
        self.true_canvas = tk.Canvas(self.true_frame, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.true_canvas.pack(padx=5, pady=5)
        
        # 预测标签
        self.pred_frame = ttk.LabelFrame(self.display_frame, text="预测标签")
        self.pred_frame.grid(row=0, column=2, padx=5, pady=5, sticky=tk.NSEW)
        
        self.pred_canvas = tk.Canvas(self.pred_frame, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.pred_canvas.pack(padx=5, pady=5)
        
        # 检测结果
        self.result_frame = ttk.LabelFrame(self.display_frame, text="检测结果")
        self.result_frame.grid(row=0, column=3, padx=5, pady=5, sticky=tk.NSEW)
        
        self.result_canvas = tk.Canvas(self.result_frame, width=self.canvas_size, height=self.canvas_size, bg="black")
        self.result_canvas.pack(padx=5, pady=5)
        
        # 配置网格权重
        for i in range(4):
            self.display_frame.columnconfigure(i, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
    
    def select_model(self):
        """选择模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.tar"), ("所有文件", "*.*")],
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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(file_path, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            model.to(device)
            model.eval()
            
            # 更新模型
            self.model = model
            self.model_loaded = True
            self.model_path = file_path
            
            # 更新UI
            self.parent.after(0, lambda: self.status_var.set("模型加载成功"))
            
            # 如果已选择图像，启用检测按钮
            if self.selected_image:
                self.parent.after(0, lambda: self.detect_btn.config(state=tk.NORMAL))
            
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
    
    def select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.selected_image = file_path
            if self.logger:
                self.logger.info(f"选择图片: {file_path}")
            
            try:
                # 显示原始图像
                self.display_image(file_path)
                
                # 立即加载并显示真实标签
                self.load_ground_truth(file_path)
                
                # 如果模型已加载，启用检测按钮
                if self.model_loaded:
                    self.detect_btn.config(state=tk.NORMAL)
                
                self.status_var.set(f"已加载图片: {os.path.basename(file_path)}")
            except Exception as e:
                error_msg = f"加载图片失败: {str(e)}"
                self.status_var.set(error_msg)
                
                if self.logger:
                    self.logger.error(error_msg)
                    self.logger.error(traceback.format_exc())
                
                messagebox.showerror("错误", error_msg)
    
    def display_image(self, file_path):
        """显示图像到原始图像画布"""
        try:
            # 加载并调整图像大小
            img = Image.open(file_path).convert("RGB")
            img_resized = img.resize((self.canvas_size, self.canvas_size))
            
            # 显示在画布上
            self.orig_photo = ImageTk.PhotoImage(img_resized)
            self.orig_canvas.create_image(0, 0, anchor=tk.NW, image=self.orig_photo)
            
            # 清除其他画布
            self.pred_canvas.delete("all")
            self.result_canvas.delete("all")
            
            # 禁用保存按钮
            self.save_btn.config(state=tk.DISABLED)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"显示图像失败: {str(e)}")
                self.logger.error(traceback.format_exc())
            raise
    
    def load_ground_truth(self, image_path):
        """加载并显示真实标签"""
        try:
            # 清空真实标签画布
            self.true_canvas.delete("all")
            
            # 获取图像文件名（不含路径和扩展名）
            image_filename = os.path.basename(image_path)
            base_filename = os.path.splitext(image_filename)[0]
            
            # 构建多个可能的掩码路径
            possible_mask_paths = [
                # 数据集掩码目录 - 使用png扩展名
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "dataset", "NUDT-SIRST", "masks", f"{base_filename}.png"),
                # 数据集掩码目录 - 使用原始扩展名
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "dataset", "NUDT-SIRST", "masks", image_filename),
                # 同目录下同名文件
                os.path.join(os.path.dirname(image_path), image_filename)
            ]
            
            # 尝试每个可能的路径
            mask_found = False
            for mask_path in possible_mask_paths:
                if os.path.exists(mask_path):
                    if self.logger:
                        self.logger.info(f"找到真实标签: {mask_path}")
                    
                    # 加载并转换为灰度图
                    mask_image = Image.open(mask_path).convert("L")
                    mask_image_resized = mask_image.resize((self.canvas_size, self.canvas_size))
                    
                    # 将掩码图像转换为PhotoImage并显示在画布上
                    self.true_photo = ImageTk.PhotoImage(mask_image_resized)
                    self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_photo)
                    
                    mask_found = True
                    break
            
            if not mask_found:
                if self.logger:
                    self.logger.warning(f"未找到图像 {image_filename} 的真实标签")
                # 在画布上显示"无标签"提示
                self.true_canvas.create_text(self.canvas_size//2, self.canvas_size//2, 
                                          text="无真实标签", fill="white", font=("Arial", 14))
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"加载真实标签失败: {str(e)}")
                self.logger.error(traceback.format_exc())
            # 在画布上显示错误信息
            self.true_canvas.create_text(self.canvas_size//2, self.canvas_size//2, 
                                       text="标签加载错误", fill="red", font=("Arial", 14))
    
    def detect_image(self):
        """对图像进行检测"""
        if not self.selected_image:
            messagebox.showwarning("警告", "请先选择图片")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        self.status_var.set("正在检测...")
        
        # 在后台线程中执行检测
        threading.Thread(target=self._detect_thread, daemon=True).start()
    
    def _detect_thread(self):
        """在后台线程中执行检测"""
        try:
            if self.logger:
                self.logger.info(f"开始检测图像: {self.selected_image}")
            
            # 加载图像
            img = Image.open(self.selected_image).convert("RGB")
            
            # 预处理图像
            input_tensor = self.transform(img).unsqueeze(0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = input_tensor.to(device)
            
            # 模型推理
            with torch.inference_mode():
                outputs = self.model(input_tensor)
                if isinstance(outputs, list):  # 处理深度监督的情况
                    output = outputs[-1]
                else:
                    output = outputs
                
                # 将输出转换为概率
                prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # 生成标注结果
            annotated_image = self.generate_annotated_image(prob_map, self.selected_image)
            
            # 保存结果用于之后的保存
            self.detection_results = {
                'prob_map': prob_map,
                'annotated_image': annotated_image
            }
            
            # 在主线程中更新UI
            self.parent.after(0, lambda: self._update_detection_ui(prob_map, annotated_image))
            
            if self.logger:
                self.logger.info(f"图像检测完成: {self.selected_image}")
            
        except Exception as e:
            error_msg = f"检测失败: {str(e)}"
            self.parent.after(0, lambda: self.status_var.set(error_msg))
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            self.parent.after(0, lambda: messagebox.showerror("错误", error_msg))
    
    def _update_detection_ui(self, prob_map, annotated_image):
        """更新UI显示检测结果"""
        # 显示预测标签
        prob_image = Image.fromarray((prob_map * 255).astype(np.uint8))
        prob_image_resized = prob_image.resize((self.canvas_size, self.canvas_size))
        self.pred_photo = ImageTk.PhotoImage(prob_image_resized)
        self.pred_canvas.create_image(0, 0, anchor=tk.NW, image=self.pred_photo)
        
        # 显示检测结果
        annotated_image_resized = annotated_image.resize((self.canvas_size, self.canvas_size))
        self.result_photo = ImageTk.PhotoImage(annotated_image_resized)
        self.result_canvas.create_image(0, 0, anchor=tk.NW, image=self.result_photo)
        
        # 启用保存按钮
        self.save_btn.config(state=tk.NORMAL)
        
        # 更新状态
        self.status_var.set("检测完成")
    
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
        
        # 转回PIL格式
        annotated_image = Image.fromarray(img_array)
        return annotated_image
    
    def save_results(self):
        """保存检测结果"""
        if not hasattr(self, 'detection_results'):
            messagebox.showwarning("警告", "请先进行检测")
            return
        
        try:
            # 创建保存目录结构
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predicts")
            save_dir = os.path.join(base_dir, "images")
            os.makedirs(save_dir, exist_ok=True)
            
            if self.logger:
                self.logger.info(f"保存检测结果到目录: {save_dir}")
            
            # 获取文件名（不含扩展名）
            base_filename = os.path.splitext(os.path.basename(self.selected_image))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存预测标签
            prob_map = self.detection_results['prob_map']
            mask_filename = f"{base_filename}_mask_{timestamp}.png"
            mask_path = os.path.join(save_dir, mask_filename)
            mask_image = Image.fromarray((prob_map * 255).astype(np.uint8))
            mask_image.save(mask_path)
            
            # 保存标注图像
            annotated_image = self.detection_results['annotated_image']
            result_filename = f"{base_filename}_result_{timestamp}.png"
            result_path = os.path.join(save_dir, result_filename)
            annotated_image.save(result_path)
            
            self.status_var.set(f"结果已保存到 {save_dir}")
            
            if self.logger:
                self.logger.info(f"预测掩码已保存: {mask_path}")
                self.logger.info(f"标注结果已保存: {result_path}")
            
            messagebox.showinfo("保存成功", f"结果已保存到:\n{save_dir}\n\n预测掩码: {mask_filename}\n标注结果: {result_filename}")
            
        except Exception as e:
            error_msg = f"保存结果失败: {str(e)}"
            self.status_var.set(error_msg)
            
            if self.logger:
                self.logger.error(error_msg)
                self.logger.error(traceback.format_exc())
            
            messagebox.showerror("错误", error_msg)