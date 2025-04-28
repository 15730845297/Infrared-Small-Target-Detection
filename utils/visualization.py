import numpy as np
import cv2
from PIL import Image, ImageDraw

def visualize_predictions(original_image, predicted_mask, threshold=0.5):
    """
    生成带有红框的标注图像和仅显示标签的图像。
    
    :param original_image: 原始图像 (numpy array)
    :param predicted_mask: 预测的掩码 (numpy array)
    :param threshold: 二值化阈值
    :return: 带红框的图像和仅显示标签的图像
    """
    # 将预测掩码二值化
    binary_mask = (predicted_mask > threshold).astype(np.uint8)

    # 生成仅显示标签的图像
    label_image = np.zeros_like(original_image)
    label_image[binary_mask > 0] = [255, 255, 255]  # 白色标签

    # 创建带红框的图像
    annotated_image = original_image.copy()
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # 过滤小区域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 红色框

    return annotated_image, label_image

def save_visualization_results(annotated_image, label_image, save_path_annotated, save_path_label):
    """
    保存可视化结果到指定路径。
    
    :param annotated_image: 带红框的图像
    :param label_image: 仅显示标签的图像
    :param save_path_annotated: 保存带红框图像的路径
    :param save_path_label: 保存标签图像的路径
    """
    annotated_pil_image = Image.fromarray(annotated_image)
    label_pil_image = Image.fromarray(label_image)

    annotated_pil_image.save(save_path_annotated)
    label_pil_image.save(save_path_label)