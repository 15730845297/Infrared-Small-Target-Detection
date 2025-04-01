import os  # 添加这一行
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog
import torch
from torchvision import transforms
import numpy as np
from datetime import datetime  # 用于获取当前时间
from model.model_DNANet import DNANet, Res_CBAM_block
from model.load_param_data import load_param
from scipy.ndimage import binary_dilation

class ImageDisplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像显示界面")
        
        self.selected_file = None  # 用于存储选择的文件路径
        self.model_path = None  # 用于存储选择的模型权重路径

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # 初始化模型为空

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        # 创建四个显示区
        self.original_label = tk.Label(root, text="原始图像")
        self.original_label.grid(row=0, column=0)
        self.original_canvas = tk.Canvas(root, width=300, height=300)
        self.original_canvas.grid(row=1, column=0)

        self.annotated_label = tk.Label(root, text="预测图像")
        self.annotated_label.grid(row=0, column=1)
        self.annotated_canvas = tk.Canvas(root, width=300, height=300)
        self.annotated_canvas.grid(row=1, column=1)

        self.true_label = tk.Label(root, text="真实标签")
        self.true_label.grid(row=0, column=2)
        self.true_canvas = tk.Canvas(root, width=300, height=300)
        self.true_canvas.grid(row=1, column=2)

        self.predicted_label = tk.Label(root, text="预测结果")
        self.predicted_label.grid(row=0, column=3)
        self.predicted_canvas = tk.Canvas(root, width=300, height=300)
        self.predicted_canvas.grid(row=1, column=3)

    def load_model(self):
        if not self.model_path:
            print("请先选择模型权重文件！")
            return None

        # 加载模型参数
        nb_filter, num_blocks = load_param('three', 'resnet_18')
        model = DNANet(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=True)
        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 加载权重
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        print(f"模型已加载：{self.model_path}")
        return model

    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth;*.pth.tar")])
        if file_path:
            self.model_path = file_path
            self.model = self.load_model()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.selected_file = file_path
            self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        original_image = Image.open(file_path).resize((300, 300))
        self.original_image_tk = ImageTk.PhotoImage(original_image)
        self.original_canvas.create_image(0, 0, anchor=tk.NW, image=self.original_image_tk)

    def start_test(self):
        if not self.selected_file:
            print("请先选择一个文件！")
            return
        if not self.model:
            print("请先选择并加载模型权重！")
            return
        
        true_label, predicted_result = self.predict(self.selected_file)

        # 显示真实标签
        if true_label is not None:
            true_label_image = Image.fromarray((true_label * 255).astype(np.uint8)).resize((300, 300))
            self.true_image_tk = ImageTk.PhotoImage(true_label_image)
            self.true_canvas.create_image(0, 0, anchor=tk.NW, image=self.true_image_tk)

        # 显示预测结果
        predicted_result_image = Image.fromarray((predicted_result * 255).astype(np.uint8)).resize((300, 300))
        self.predicted_image_tk = ImageTk.PhotoImage(predicted_result_image)
        self.predicted_canvas.create_image(0, 0, anchor=tk.NW, image=self.predicted_image_tk)

        # 保存预测结果并显示带红色框的图像
        annotated_image = self.save_prediction(predicted_result, self.selected_file)
        annotated_image = annotated_image.resize((300, 300))
        self.annotated_image_tk = ImageTk.PhotoImage(annotated_image)
        self.annotated_canvas.create_image(0, 0, anchor=tk.NW, image=self.annotated_image_tk)

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
        image_name = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, image_name)
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
        save_annotated_path = os.path.join(save_dir, f"{original_file_name}_Annotated_{current_time}.png")

        # 保存预测结果
        predicted_image = Image.fromarray((predicted_result * 255).astype(np.uint8))
        predicted_image.save(save_path)

        # 加载原始图像
        original_image = Image.open(original_file_path).convert("RGB")
        original_image = original_image.resize((256, 256))  # 确保尺寸一致

        # 计算目标边界
        threshold = 0.5  # 阈值，预测值大于此值的区域被认为是目标
        binary_mask = (predicted_result > threshold).astype(np.uint8)  # 二值化预测结果
        dilated_mask = binary_dilation(binary_mask)  # 扩展目标区域
        boundary_mask = dilated_mask ^ binary_mask  # 边界为扩展区域减去原始区域

        # 绘制红色边界
        draw = ImageDraw.Draw(original_image)
        for y in range(boundary_mask.shape[0]):
            for x in range(boundary_mask.shape[1]):
                if boundary_mask[y, x]:  # 如果是边界像素
                    draw.point([x, y], fill="red")  # 绘制红色像素点

        # 保存带红色边界的图像
        original_image.save(save_annotated_path)
        print(f"预测结果已保存到: {save_path}")
        print(f"标注结果已保存到: {save_annotated_path}")
        return original_image

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageDisplayApp(root)
    
    # 添加选择模型按钮
    select_model_button = tk.Button(root, text="选择模型权重", command=lambda: app.select_model())
    select_model_button.grid(row=2, column=0)

    # 添加文件选择按钮
    select_button = tk.Button(root, text="选择文件", command=lambda: app.select_file())
    select_button.grid(row=2, column=1)

    # 添加开始测试按钮
    test_button = tk.Button(root, text="开始测试", command=lambda: app.start_test())
    test_button.grid(row=2, column=2)
    
    root.mainloop()