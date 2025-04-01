from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import visulization  # 假设 visulization.py 在同一目录下

class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像显示界面")
        
        self.selected_file = None  # 用于存储选择的文件路径

        # 创建三个显示区
        self.original_label = tk.Label(root, text="原始图像")
        self.original_label.grid(row=0, column=0)
        self.original_canvas = tk.Canvas(root, width=300, height=300)
        self.original_canvas.grid(row=1, column=0)
        
        self.true_label = tk.Label(root, text="真实标签")
        self.true_label.grid(row=0, column=1)
        self.true_canvas = tk.Canvas(root, width=300, height=300)
        self.true_canvas.grid(row=1, column=1)
        
        self.predicted_label = tk.Label(root, text="预测结果")
        self.predicted_label.grid(row=0, column=2)
        self.predicted_canvas = tk.Canvas(root, width=300, height=300)
        self.predicted_canvas.grid(row=1, column=2)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.selected_file = file_path
            self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        # 加载原始图像并调整大小
        original_image = Image.open(file_path).resize((300, 300))
        self.original_image_tk = ImageTk.PhotoImage(original_image)
        self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_image_tk)

    def start_test(self):
        if not self.selected_file:
            print("请先选择一个文件！")
            return
        
        # 调用 visulization.py 的预测逻辑
        true_label, predicted_result = visulization.predict(self.selected_file)

        # 显示真实标签
        true_label_image = Image.fromarray((true_label * 255).astype(np.uint8)).resize((300, 300))
        self.true_image_tk = ImageTk.PhotoImage(true_label_image)
        self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_image_tk)

        # 显示预测结果
        predicted_result_image = Image.fromarray((predicted_result * 255).astype(np.uint8)).resize((300, 300))
        self.predicted_image_tk = ImageTk.PhotoImage(predicted_result_image)
        self.predicted_canvas.create_image(0, 0, anchor=tk.NW, image=self.predicted_image_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDisplayApp(root)
    
    # 添加文件选择按钮
    select_button = tk.Button(root, text="选择文件", command=lambda: app.select_file())
    select_button.grid(row=2, column=0)

    # 添加开始测试按钮
    test_button = tk.Button(root, text="开始测试", command=lambda: app.start_test())
    test_button.grid(row=2, column=2)
    
    root.mainloop()