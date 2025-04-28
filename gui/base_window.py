import tkinter as tk
from tkinter import ttk
from tkinter import Tk, Frame, Button, Label

class BaseWindow:
    """所有窗口的基类"""
    def __init__(self, root, title="窗口", geometry="800x600"):
        self.root = root
        self.root.title(title)
        self.root.geometry(geometry)
        
        # 创建主框架
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_widgets(self):
        self.title_label = Label(self.main_frame, text="欢迎使用红外小目标检测系统", font=("Arial", 16))
        self.title_label.pack(pady=10)

        self.image_mode_button = Button(self.main_frame, text="图像模式", command=self.open_image_mode)
        self.image_mode_button.pack(pady=5)

        self.video_mode_button = Button(self.main_frame, text="视频模式", command=self.open_video_mode)
        self.video_mode_button.pack(pady=5)

    def open_image_mode(self):
        from image_mode import ImageMode
        self.hide()
        self.image_mode = ImageMode(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def open_video_mode(self):
        from video_mode import VideoMode
        self.hide()
        self.video_mode = VideoMode(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def hide(self):
        self.main_frame.pack_forget()

    def on_closing(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    def set_status(self, text):
        """设置状态栏文本"""
        self.status_var.set(text)