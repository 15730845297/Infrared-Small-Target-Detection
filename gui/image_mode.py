import cv2  # 第一位置导入
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
from torchvision import transforms
import numpy as np
from datetime import datetime
import threading

# 导入模型相关组件
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param
from skimage import measure

class ImageModeFrame:
    def __init__(self, parent, status_var, model_path=None, update_model_callback=None):
        self.parent = parent
        self.status_var = status_var
        self.model_path = model_path
        self.update_model_callback = update_model_callback
        
        # 初始化变量
        self.selected_file = None
        self.model = None
        self.model_loaded = False
        self.current_results = None
        
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
        """创建图像模式界面"""
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
        
        self.select_file_btn = ttk.Button(
            self.toolbar, 
            text="选择图片", 
            command=self.select_file
        )
        self.select_file_btn.pack(side=tk.LEFT, padx=5)
        
        self.test_btn = ttk.Button(
            self.toolbar, 
            text="开始检测", 
            command=self.start_test,
            state=tk.DISABLED
        )
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
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
        
        # 创建四个显示区域 (原始图像 -> 真实标签 -> 预测标签 -> 预测结果)
        self.create_display_area()
    
    def create_display_area(self):
        """创建图像显示区域"""
        # 原始图像区域
        self.original_frame = ttk.LabelFrame(self.display_frame, text="原始图像")
        self.original_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        
        self.original_canvas = tk.Canvas(self.original_frame, width=300, height=300, bg="black")
        self.original_canvas.pack(padx=5, pady=5)
        
        # 真实标签区域
        self.true_frame = ttk.LabelFrame(self.display_frame, text="真实标签")
        self.true_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.NSEW)
        
        self.true_canvas = tk.Canvas(self.true_frame, width=300, height=300, bg="black")
        self.true_canvas.pack(padx=5, pady=5)
        
        # 预测标签区域
        self.pred_frame = ttk.LabelFrame(self.display_frame, text="预测标签")
        self.pred_frame.grid(row=0, column=2, padx=5, pady=5, sticky=tk.NSEW)
        
        self.pred_canvas = tk.Canvas(self.pred_frame, width=300, height=300, bg="black")
        self.pred_canvas.pack(padx=5, pady=5)
        
        # 预测结果区域
        self.result_frame = ttk.LabelFrame(self.display_frame, text="预测结果")
        self.result_frame.grid(row=0, column=3, padx=5, pady=5, sticky=tk.NSEW)
        
        self.result_canvas = tk.Canvas(self.result_frame, width=300, height=300, bg="black")
        self.result_canvas.pack(padx=5, pady=5)
        
        # 设置网格权重
        for i in range(4):
            self.display_frame.columnconfigure(i, weight=1)
        self.display_frame.rowconfigure(0, weight=1)
    
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
            if self.selected_file:
                self.parent.after(0, lambda: self.test_btn.config(state=tk.NORMAL))
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
        
        self.model = model
        self.model_loaded = True
    
    def select_file(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.selected_file = file_path
            self.load_and_display_image(file_path)
            self.status_var.set(f"已选择图片: {os.path.basename(file_path)}")
            
            # 如果模型已加载，启用检测按钮
            if self.model_loaded:
                self.test_btn.config(state=tk.NORMAL)
    
    def load_and_display_image(self, file_path):
        """加载并显示图片"""
        try:
            # 打开原图并调整大小
            original_image = Image.open(file_path).convert("RGB")
            original_image_resized = original_image.resize((300, 300))
            
            # 显示原图
            self.original_photo = ImageTk.PhotoImage(original_image_resized)
            self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_photo)
            
            # 清除其他画布
            self.true_canvas.delete("all")
            self.pred_canvas.delete("all")
            self.result_canvas.delete("all")
            
            # 禁用保存按钮
            self.save_btn.config(state=tk.DISABLED)
            
            return True
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
            return False
    
    def start_test(self):
        """开始检测"""
        if not self.selected_file:
            messagebox.showwarning("警告", "请先选择图片文件")
            return
        
        if not self.model_loaded:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        self.status_var.set("正在检测...")
        
        # 在后台线程中执行检测，避免UI冻结
        threading.Thread(target=self._detection_thread, daemon=True).start()
    
    def _detection_thread(self):
        """在后台线程中执行检测"""
        try:
            # 执行检测
            true_label, predicted_result = self.predict(self.selected_file)
            
            # 生成标注图像
            annotated_image = self.generate_annotated_image(predicted_result, self.selected_file)
            
            # 保存结果
            self.current_results = {
                "predicted_result": predicted_result,
                "annotated_image": annotated_image,
                "original_image_path": self.selected_file
            }
            
            # 在主线程中更新UI
            self.parent.after(0, lambda: self._update_detection_ui(true_label, predicted_result, annotated_image))
        except Exception as e:
            self.parent.after(0, lambda: self.status_var.set("检测失败"))
            self.parent.after(0, lambda: messagebox.showerror("错误", f"检测失败: {str(e)}"))
    
    def _update_detection_ui(self, true_label, predicted_result, annotated_image):
        """更新检测结果UI"""
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
        self.save_btn.config(state=tk.NORMAL)
        
        # 更新状态
        self.status_var.set("检测完成")
    
    def predict(self, image_path):
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
        if "dataset/NUDT-SIRST" in image_path or "dataset\\NUDT-SIRST" in image_path:
            mask_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "dataset", "NUDT-SIRST", "masks")
            image_name = os.path.basename(image_path)
            mask_path = os.path.join(mask_dir, image_name)
            if os.path.exists(mask_path):
                true_label = np.array(Image.open(mask_path).convert("L")) / 255.0
        
        return true_label, predicted_result
    
    def generate_annotated_image(self, predicted_result, original_file_path):
        """生成带标注的图像"""
        # 加载原始图像
        original_image = Image.open(original_file_path).convert("RGB")
        original_image = original_image.resize((256, 256))
        
        # 创建副本用于绘制
        annotated_image = original_image.copy()
        
        # 二值化
        threshold = 0.3
        binary_mask = (predicted_result > threshold).astype(np.uint8)
        
        # 提取连通区域
        labeled_array = measure.label(binary_mask)
        regions = measure.regionprops(labeled_array)
        
        # 绘制红色边框
        draw = ImageDraw.Draw(annotated_image)
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            
            # 扩大边界框
            padding = 3
            minr = max(0, minr - padding)
            minc = max(0, minc - padding)
            maxr = min(256, maxr + padding)
            maxc = min(256, maxc + padding)
            
            # 绘制矩形框
            try:
                draw.rectangle([(minc, minr), (maxc, maxr)], outline="red", width=2)
            except TypeError:
                # 兼容旧版PIL
                for i in range(2):
                    draw.rectangle([(minc-i, minr-i), (maxc+i, maxr+i)], outline="red")
        
        return annotated_image
    
    def save_results(self):
        """保存检测结果"""
        if not self.current_results:
            messagebox.showinfo("提示", "请先执行检测!")
            return
        
        try:
            # 创建保存目录
            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "predicts", "images")
            os.makedirs(save_dir, exist_ok=True)
            
            # 获取当前结果
            predicted_result = self.current_results["predicted_result"]
            annotated_image = self.current_results["annotated_image"]
            original_file_path = self.current_results["original_image_path"]
            
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
    
    def select_image(self):
        """选择图片文件"""
        file_path = filedialog.askopenfilename(
            title="选择图片文件",
            filetypes=[("图片文件", "*.png;*.jpg;*.jpeg;*.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.selected_image = file_path
            self.display_image(file_path)
            self.status_var.set(f"已选择图片: {os.path.basename(file_path)}")
            
            # 立即尝试加载并显示真实标签
            self.load_ground_truth(file_path)
            
            # 如果模型已加载，启用检测按钮
            if self.model_loaded:
                self.detect_btn.config(state=tk.NORMAL)

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
                    print(f"找到真实标签: {mask_path}")
                    
                    # 加载并转换为灰度图
                    mask_image = Image.open(mask_path).convert("L")
                    mask_image_resized = mask_image.resize((self.canvas_size, self.canvas_size))
                    
                    # 将掩码图像转换为PhotoImage并显示在画布上
                    self.true_photo = ImageTk.PhotoImage(mask_image_resized)
                    self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_photo)
                    
                    mask_found = True
                    break
            
            if not mask_found:
                print(f"未找到图像 {image_filename} 的真实标签")
                # 可以在画布上显示"无标签"提示
                self.true_canvas.create_text(self.canvas_size//2, self.canvas_size//2, 
                                            text="无真实标签", fill="white", font=("Arial", 14))
                
        except Exception as e:
            print(f"加载真实标签失败: {e}")
            # 在画布上显示错误信息
            self.true_canvas.create_text(self.canvas_size//2, self.canvas_size//2, 
                                        text="标签加载错误", fill="red", font=("Arial", 14))