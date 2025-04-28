import os
import json

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")

def load_config():
    """加载配置文件"""
    config = {}
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
    except Exception as e:
        print(f"加载配置文件出错: {e}")
    
    return config

def save_config(config):
    """保存配置到文件"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print(f"保存配置文件出错: {e}")