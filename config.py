"""
全局配置文件
"""

# 模型相关配置
MODEL_CONFIG = {
    "backbone": "resnet_18",
    "channel_size": "three",
    "deep_supervision": True,
    "input_channels": 3,
    "num_classes": 1
}

# 数据预处理配置
PREPROCESSING_CONFIG = {
    "image_size": (256, 256),
    "mean": [.485, .456, .406],
    "std": [.229, .224, .225]
}

# 推理配置
INFERENCE_CONFIG = {
    "default_threshold": 0.3,
    "box_padding": 3
}

# 路径配置
PATH_CONFIG = {
    "dataset_dir": "../dataset",
    "predicts_dir": "../predicts",
    "weights_dir": "../weights"
}