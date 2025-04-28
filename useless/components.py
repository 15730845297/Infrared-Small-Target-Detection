import tkinter as tk
from tkinter import Frame, Button, Label, Canvas, ttk

class Components:
    def __init__(self, master):
        self.master = master
        self.create_widgets()

    def create_widgets(self):
        self.frame = Frame(self.master)
        self.frame.pack()

        self.label = Label(self.frame, text="红外小目标检测系统")
        self.label.pack()

        self.select_image_button = Button(self.frame, text="选择图像", command=self.select_image)
        self.select_image_button.pack()

        self.select_video_button = Button(self.frame, text="选择视频", command=self.select_video)
        self.select_video_button.pack()

        self.start_button = Button(self.frame, text="开始检测", command=self.start_detection)
        self.start_button.pack()

        self.save_button = Button(self.frame, text="保存结果", command=self.save_results)
        self.save_button.pack()

        self.canvas = Canvas(self.frame, width=600, height=400)
        self.canvas.pack()

    def select_image(self):
        # 选择图像的逻辑
        pass

    def select_video(self):
        # 选择视频的逻辑
        pass

    def start_detection(self):
        # 开始检测的逻辑
        pass

    def save_results(self):
        # 保存结果的逻辑
        pass

class ProgressDialog:
    """进度对话框"""
    def __init__(self, parent, title="处理中", message="请稍候...", mode="indeterminate"):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("300x100")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)  # 设置为父窗口的临时窗口
        self.dialog.grab_set()  # 模态对话框
        
        # 消息标签
        self.message_label = ttk.Label(self.dialog, text=message)
        self.message_label.pack(pady=(10, 5))
        
        # 进度条
        self.progress = ttk.Progressbar(self.dialog, mode=mode)
        self.progress.pack(fill=tk.X, padx=20, pady=10)
        
        if mode == "indeterminate":
            self.progress.start()
        
        # 居中显示
        self.center_dialog(parent)
    
    def center_dialog(self, parent):
        """居中显示对话框"""
        parent.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    def update_message(self, message):
        """更新消息文本"""
        self.message_label.config(text=message)
    
    def update_progress(self, value):
        """更新进度值 (对于determinate模式)"""
        self.progress["value"] = value
    
    def close(self):
        """关闭对话框"""
        if self.progress["mode"] == "indeterminate":
            self.progress.stop()
        self.dialog.destroy()