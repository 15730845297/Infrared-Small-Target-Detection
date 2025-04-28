import cv2  # 最重要的改动：将cv2放在第一个导入
import os
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch

# 强制预加载相关DLL
try:
    # 确保相关DLL已加载
    cv2_path = cv2.__file__
    cv2_dir = os.path.dirname(cv2_path)
    os.environ['PATH'] = cv2_dir + os.pathsep + os.environ['PATH']
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"OpenCV 路径: {cv2_path}")
except Exception as e:
    print(f"OpenCV 预加载失败: {e}")

# 导入图像和视频模式
from gui.image_mode import ImageModeFrame
from gui.video_mode import VideoModeFrame
from utils.file_operations import load_config, save_config

class InfraredDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("红外小目标检测系统")
        self.root.geometry("1280x800")
        
        # 检查GPU是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载配置
        self.config = load_config()
        self.model_path = self.config.get("model_path", None)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建模式选择区域
        self.create_mode_selection()
        
        # 创建状态栏
        self.create_status_bar()
        
        # 默认显示图像模式
        self.show_image_mode()
        
        # 关闭窗口时保存配置
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_mode_selection(self):
        """创建模式选择区域"""
        self.mode_frame = ttk.LabelFrame(self.main_frame, text="检测模式")
        self.mode_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 添加模式按钮
        self.image_mode_btn = ttk.Button(
            self.mode_frame, 
            text="图像模式", 
            command=self.show_image_mode
        )
        self.image_mode_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.video_mode_btn = ttk.Button(
            self.mode_frame, 
            text="视频模式", 
            command=self.show_video_mode
        )
        self.video_mode_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 显示当前加载的模型
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
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=2)
        
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=5)
    
    def show_image_mode(self):
        """显示图像模式"""
        # 清除当前内容
        if hasattr(self, 'current_frame'):
            self.current_frame.destroy()
        
        # 创建图像模式框架
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        # 设置按钮状态
        self.image_mode_btn.state(['disabled'])
        self.video_mode_btn.state(['!disabled'])
        
        # 初始化图像模式
        self.image_mode = ImageModeFrame(
            self.current_frame, 
            self.status_var, 
            self.model_path,
            self.update_model_path
        )
        
        self.status_var.set("图像模式已加载")
    
    def show_video_mode(self):
        """显示视频模式"""
        # 清除当前内容
        if hasattr(self, 'current_frame'):
            self.current_frame.destroy()
        
        # 创建视频模式框架
        self.current_frame = ttk.Frame(self.main_frame)
        self.current_frame.pack(fill=tk.BOTH, expand=True)
        
        # 设置按钮状态
        self.image_mode_btn.state(['!disabled'])
        self.video_mode_btn.state(['disabled'])
        
        # 初始化视频模式
        self.video_mode = VideoModeFrame(
            self.current_frame, 
            self.status_var,
            self.model_path,
            self.update_model_path
        )
        
        self.status_var.set("视频模式已加载")
    
    def update_model_path(self, model_path):
        """更新模型路径并在UI中显示"""
        if model_path and os.path.exists(model_path):
            self.model_path = model_path
            model_name = os.path.basename(model_path)
            self.model_label.config(text=f"当前模型: {model_name}", foreground="green")
            
            # 更新配置
            self.config["model_path"] = model_path
            save_config(self.config)
            
            return True
        return False
    
    def on_closing(self):
        """关闭窗口时执行的操作"""
        # 保存配置
        save_config(self.config)
        
        # 关闭窗口
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = InfraredDetectionApp(root)
    root.mainloop()