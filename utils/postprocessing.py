from torchvision import transforms
import numpy as np
from skimage import measure

def get_transform():
    """获取图像预处理转换"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    return transform

def generate_bounding_boxes(predicted_result, threshold=0.3, padding=3):
    """
    从预测结果中生成边界框
    
    Args:
        predicted_result: 模型预测的概率图
        threshold: 二值化阈值
        padding: 边界框扩展像素数
    
    Returns:
        边界框列表，每个框为 (minc, minr, maxc, maxr)
    """
    # 二值化
    binary_mask = (predicted_result > threshold).astype(np.uint8)
    
    # 标记连通区域
    labeled_array = measure.label(binary_mask)
    regions = measure.regionprops(labeled_array)
    
    boxes = []
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        
        # 扩大边界框以确保完全框住目标
        minr = max(0, minr - padding)
        minc = max(0, minc - padding)
        maxr = min(predicted_result.shape[0], maxr + padding)
        maxc = min(predicted_result.shape[1], maxc + padding)
        
        boxes.append((minc, minr, maxc, maxr))
    
    return boxes