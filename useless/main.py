import os
import tkinter as tk
from tkinter import ttk, messagebox
import sys

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from gui.image_mode import ImageModeFrame
from gui.video_mode import VideoModeFrame
from utils.file_operations import load_config, save_config
import config

class InfraredDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("红外小目标检测系统")
        self.root.geometry("1280x720")
        
        # 加载配置
        self.config = load_config()
        
        # 创建顶部菜单栏
        self.menu_frame = tk.Frame(root, height=40, bg="#f0f0f0")
        self.menu_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 模式选择按钮
        self.mode_label = tk.Label(self.menu_frame, text="检测模式:", bg="#f0f0f0")
        self.mode_label.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        
        self.image_btn = ttk.Button(self.menu_frame, text="图片模式", command=self.switch_to_image_mode)
        self.image_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.video_btn = ttk.Button(self.menu_frame, text="视频模式", command=self.switch_to_video_mode)
        self.video_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 模型选择下拉框
        self.model_label = tk.Label(self.menu_frame, text="选择模型:", bg="#f0f0f0")
        self.model_label.pack(side=tk.LEFT, padx=(20, 5), pady=5)
        
        self.model_var = tk.StringVar()
        self.available_models = self._get_available_models()
        
        if self.available_models:
            self.model_var.set(self.config.get("last_model", self.available_models[0]))
        
        self.model_dropdown = ttk.Combobox(self.menu_frame, textvariable=self.model_var, 
                                          values=self.available_models, width=30)
        self.model_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.load_model_btn = ttk.Button(self.menu_frame, text="加载模型", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # 状态显示
        self.status_var = tk.StringVar(value="状态: 未加载模型")
        self.status_label = tk.Label(self.menu_frame, textvariable=self.status_var, 
                                    bg="#f0f0f0", fg="red")
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # 创建主内容区域
        self.content_frame = tk.Frame(root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 当前活动模式
        self.current_mode = None
        self.image_mode_frame = None
        self.video_mode_frame = None
        
        # 默认启动图片模式
        self.switch_to_image_mode()
        
        # 如果有保存的模型设置，自动加载模型
        if "last_model" in self.config and os.path.exists(self.config["last_model"]):
            self.load_model()
        
        # 在窗口关闭时保存配置
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _get_available_models(self):
        """获取可用的模型列表"""
        weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
        models = []
        
        if os.path.exists(weights_dir):
            for file in os.listdir(weights_dir):
                if file.endswith('.pth') or file.endswith('.pth.tar'):
                    models.append(os.path.join(weights_dir, file))
        
        return models
    
    def load_model(self):
        """加载选中的模型"""
        model_path = self.model_var.get()
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("错误", "请选择有效的模型文件!")
            return
        
        try:
            # 将模型路径传递给当前活动的模式
            if self.current_mode:
                self.current_mode.load_model(model_path)
                
                # 更新状态
                model_name = os.path.basename(model_path)
                self.status_var.set(f"状态: 已加载模型 - {model_name}")
                self.status_label.config(fg="green")
                
                # 更新配置
                self.config["last_model"] = model_path
                save_config(self.config)
        
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {str(e)}")
            self.status_var.set("状态: 模型加载失败")
            self.status_label.config(fg="red")
    
    def switch_to_image_mode(self):
        """切换到图片模式"""
        # 清空当前内容
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # 创建图片模式界面
        if not self.image_mode_frame:
            self.image_mode_frame = ImageModeFrame(self.content_frame)
        else:
            self.image_mode_frame = ImageModeFrame(self.content_frame)
        
        self.current_mode = self.image_mode_frame
        
        # 高亮当前模式按钮
        self.image_btn.state(['pressed'])
        self.video_btn.state(['!pressed'])
    
    def switch_to_video_mode(self):
        """切换到视频模式"""
        # 清空当前内容
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # 创建视频模式界面
        if not self.video_mode_frame:
            self.video_mode_frame = VideoModeFrame(self.content_frame)
        else:
            self.video_mode_frame = VideoModeFrame(self.content_frame)
        
        self.current_mode = self.video_mode_frame
        
        # 高亮当前模式按钮
        self.video_btn.state(['pressed'])
        self.image_btn.state(['!pressed'])
    
    def on_closing(self):
        """应用关闭时执行"""
        # 如果视频模式正在运行，需要停止
        if self.current_mode and hasattr(self.current_mode, 'stop_video'):
            self.current_mode.stop_video()
        
        # 保存配置
        save_config(self.config)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = InfraredDetectionApp(root)
    root.mainloop()