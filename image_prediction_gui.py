import os  # 添加这一行
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import torch
from torchvision import transforms
import numpy as np
from datetime import datetime  # 用于获取当前时间
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param

class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像显示界面")
        
        self.selected_file = None  # 用于存储选择的文件路径

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

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

    def load_model(self):
        # 加载模型参数
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
        model_path = "result/NUDT-SIRST_DNANet_21_02_2025_23_09_23_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar"
        checkpoint = torch.load(model_path, map_location=self.device)

        # 过滤掉多余的键
        filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model.state_dict()}

        # 加载权重
        model.load_state_dict(filtered_state_dict)
        model.to(self.device)
        model.eval()
        return model

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
        
        # 实时预测
        true_label, predicted_result = self.predict(self.selected_file)

        # 显示真实标签（如果有）
        if true_label is not None:
            true_label_image = Image.fromarray((true_label * 255).astype(np.uint8)).resize((300, 300))
            self.true_image_tk = ImageTk.PhotoImage(true_label_image)
            self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_image_tk)

        # 显示预测结果
        predicted_result_image = Image.fromarray((predicted_result * 255).astype(np.uint8)).resize((300, 300))
        self.predicted_image_tk = ImageTk.PhotoImage(predicted_result_image)
        self.predicted_canvas.create_image(0, 0, anchor=tk.NW, image=self.predicted_image_tk)

        # 保存预测结果
        self.save_prediction(predicted_result, self.selected_file)

    def predict(self, image_path):
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 模型预测
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, list):  # 如果输出是列表，取最后一个
                output = output[-1]
            predicted_result = torch.sigmoid(output).squeeze().cpu().numpy()

        # 加载真实标签
        # 假设真实标签存储在 dataset/NUDT-SIRST/masks 文件夹中，文件名与输入图像相同
        mask_dir = "dataset/NUDT-SIRST/masks"
        image_name = image_path.split("/")[-1]  # 获取图像文件名
        mask_path = f"{mask_dir}/{image_name}"
        if os.path.exists(mask_path):
            true_label = np.array(Image.open(mask_path).convert("L")) / 255.0
        else:
            true_label = None  # 如果找不到真实标签文件，则返回 None

        return true_label, predicted_result

    def save_prediction(self, predicted_result, original_file_path):
        # 创建保存目录
        save_dir = "predicts"
        os.makedirs(save_dir, exist_ok=True)

        # 获取原始文件名和当前时间
        original_file_name = os.path.basename(original_file_path).split(".")[0]
        current_time = datetime.now().strftime("%Y%m%d_%H%M")

        # 构造保存文件路径
        save_path = os.path.join(save_dir, f"{original_file_name}_Pred_{current_time}.png")

        # 保存预测结果
        predicted_image = Image.fromarray((predicted_result * 255).astype(np.uint8))
        predicted_image.save(save_path)
        print(f"预测结果已保存到: {save_path}")

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