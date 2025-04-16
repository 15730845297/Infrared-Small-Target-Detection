import os 
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
        self.root.title("红外小目标检测系统")
        
        self.selected_file = None  # 用于存储选择的文件路径
        self.model_path = None  # 用于存储选择的模型权重路径
        self.current_results = None  # 用于存储当前的预测结果和标记结果

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # 初始化模型为空
        self.model_loaded = False  # 添加标记避免重复加载

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

        # 创建四个显示区 (修改顺序: 原始图像 -> 真实标签 -> 预测标签 -> 预测结果)
        self.original_label = tk.Label(root, text="原始图像")
        self.original_label.grid(row=0, column=0)
        self.original_canvas = tk.Canvas(root, width=300, height=300)
        self.original_canvas.grid(row=1, column=0)

        self.true_label = tk.Label(root, text="真实标签")
        self.true_label.grid(row=0, column=1)
        self.true_canvas = tk.Canvas(root, width=300, height=300)
        self.true_canvas.grid(row=1, column=1)

        self.predicted_label = tk.Label(root, text="预测标签")  # 已修改"预测结果"为"预测标签"
        self.predicted_label.grid(row=0, column=2)
        self.predicted_canvas = tk.Canvas(root, width=300, height=300)
        self.predicted_canvas.grid(row=1, column=2)

        self.annotated_label = tk.Label(root, text="预测结果")  # 已修改"预测图像"为"预测结果"
        self.annotated_label.grid(row=0, column=3)
        self.annotated_canvas = tk.Canvas(root, width=300, height=300)
        self.annotated_canvas.grid(row=1, column=3)

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
            # 加载模型
            if not self.model_loaded:
                self.model = self.load_model()
                self.model_loaded = True
            else:
                # 只有模型路径变更时才重新加载
                if self.model_path != file_path:
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

        # 生成带红色框的预测图像
        annotated_image = self.generate_annotated_image(predicted_result, self.selected_file)
        annotated_image_resized = annotated_image.resize((300, 300))
        self.annotated_image_tk = ImageTk.PhotoImage(annotated_image_resized)
        self.annotated_canvas.create_image(0, 0, anchor=tk.NW, image=self.annotated_image_tk)

        # 存储当前结果
        self.current_results = {
            "predicted_result": predicted_result,
            "annotated_image": annotated_image,
            "original_file_path": self.selected_file
        }

    def predict(self, image_path):
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 模型预测
        with torch.inference_mode():  # 使用inference_mode替代no_grad提高效率
            output = self.model(input_tensor)
            if isinstance(output, list):
                output = output[-1]
            predicted_result = torch.sigmoid(output).squeeze().cpu().numpy()

        # 优化真实标签加载 - 只有在需要时才尝试加载
        true_label = None
        # 只有可能来自数据集的图像才尝试查找对应标签
        if "dataset/NUDT-SIRST" in image_path or "dataset\\NUDT-SIRST" in image_path:
            mask_dir = "dataset/NUDT-SIRST/masks"
            image_name = os.path.basename(image_path)
            mask_path = os.path.join(mask_dir, image_name)
            if os.path.exists(mask_path):
                true_label = np.array(Image.open(mask_path).convert("L")) / 255.0

        return true_label, predicted_result

    def generate_annotated_image(self, predicted_result, original_file_path):
        # 加载原始图像
        original_image = Image.open(original_file_path).convert("RGB")
        original_image = original_image.resize((256, 256))

        # 创建副本用于绘制，避免修改原始图像
        annotated_image = original_image.copy()
        
        # 优化二值化过程 - 直接使用numpy操作而非scipy的binary_dilation
        threshold = 0.3
        binary_mask = (predicted_result > threshold).astype(np.uint8)
        
        # 使用更高效的连通区域查找
        from skimage import measure
        labeled_array = measure.label(binary_mask)
        regions = measure.regionprops(labeled_array)
        
        # 绘制红色边框
        draw = ImageDraw.Draw(annotated_image)
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            
            # 扩大边界框以确保完全框住目标
            padding = 3  # 边界框扩展像素数
            minr = max(0, minr - padding)
            minc = max(0, minc - padding)
            maxr = min(256, maxr + padding)
            maxc = min(256, maxc + padding)
            
            # 绘制矩形框，统一使用width参数
            try:
                draw.rectangle([(minc, minr), (maxc, maxr)], outline="red", width=2)
            except TypeError:
                # 兼容旧版PIL
                for i in range(2):
                    draw.rectangle([(minc-i, minr-i), (maxc+i, maxr+i)], outline="red")

        return annotated_image

    def save_results(self):
        if not self.current_results:
            print("没有可保存的结果，请先运行测试！")
            return

        # 创建保存目录
        save_dir = "predicts"
        os.makedirs(save_dir, exist_ok=True)

        # 获取当前结果
        predicted_result = self.current_results["predicted_result"]
        annotated_image = self.current_results["annotated_image"]
        original_file_path = self.current_results["original_file_path"]

        # 获取原始文件名和当前时间
        original_file_name = os.path.basename(original_file_path).split(".")[0]
        current_time = datetime.now().strftime("%Y%m%d_%H%M")

        # 构造保存文件路径
        save_path = os.path.join(save_dir, f"{original_file_name}_Pred_{current_time}.png")
        save_annotated_path = os.path.join(save_dir, f"{original_file_name}_Annotated_{current_time}.png")

        # 保存预测结果
        predicted_image = Image.fromarray((predicted_result * 255).astype(np.uint8))
        predicted_image.save(save_path)

        # 保存带红色边界的图像
        annotated_image.save(save_annotated_path)

        print(f"预测结果已保存到: {save_path}")
        print(f"标注结果已保存到: {save_annotated_path}")

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

    # 添加保存结果按钮
    save_button = tk.Button(root, text="保存结果", command=lambda: app.save_results())
    save_button.grid(row=2, column=3)
    
    root.mainloop()