import sys
import os

# 添加项目根目录到sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.utils import *


def parse_args():
    """测试红外小目标检测网络的参数配置
    
    此函数定义了测试和可视化过程中所有可配置的参数，包括：
    - 模型选择与网络架构配置
    - 数据集选择与预处理参数
    - 测试参数（批次大小等）
    - 硬件与评估指标配置
    
    Returns:
        argparse.Namespace: 包含所有配置参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    
    # ---------- 模型选择参数 ----------
    parser.add_argument('--model', type=str, default='DNANet',
                        help='模型名称: DNANet')

    # ---------- DNANet网络架构参数 ----------
    parser.add_argument('--channel_size', type=str, default='three',
                        help='特征通道大小: one(较小), two, three(默认), four(较大)')
    parser.add_argument('--backbone', type=str, default='resnet_18',
                        help='骨干网络: vgg10, resnet_10, resnet_18(默认), resnet_34')
    parser.add_argument('--deep_supervision', type=str, default='True', 
                        help='是否使用深度监督: True(启用多尺度监督) 或 False(仅使用最终输出)')

    # ---------- 数据集与模型权重路径配置 ----------
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='数据集名称: NUDT-SIRST, NUAA-SIRST, NUST-SIRST')
    parser.add_argument('--st_model', type=str, default='NUDT-SIRST_DNANet_31_07_2021_14_50_57_wDS',
                        help='模型存储目录名称，用于结果可视化')
    parser.add_argument('--model_dir', type=str,
                        default = '../result/NUDT-SIRST_DNANet_21_02_2025_23_09_23_wDS/mIoU__DNANet_NUDT-SIRST_epoch.pth.tar',
                        help    = '模型权重文件路径，可以是相对于result/目录的路径')
    
    # ---------- 数据加载与预处理参数 ----------
    parser.add_argument('--mode', type=str, default='TXT', 
                        help='数据分割模式: TXT(使用预定义文件列表), Ratio(按比例随机分割)')
    parser.add_argument('--test_size', type=float, default='0.5', 
                        help='测试集比例，仅在mode=Ratio时生效')
    parser.add_argument('--root', type=str, default='dataset/',
                        help='数据集根目录路径')
    parser.add_argument('--suffix', type=str, default='.png',
                        help='图像文件后缀')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='数据分割方法: 50_50(默认), 10000_100(用于NUST-SIRST)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='数据加载器的工作线程数')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='输入图像通道数，默认为3(RGB)')
    parser.add_argument('--base_size', type=int, default=256,
                        help='基础图像大小')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='裁剪图像大小')

    # ---------- 测试超参数 ----------
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='训练轮数，用于日志记录')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='测试批次大小')

    # ---------- 硬件配置 ----------
    parser.add_argument('--gpus', type=str, default='0',
                        help='使用的GPU编号，例如可指定"0"或"0,1"表示使用多个GPU')

    # ---------- 评估指标配置 ----------
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='ROC曲线阈值数量，用于评估模型性能')

    # 解析参数
    args = parser.parse_args()

    # 返回解析后的参数
    return args