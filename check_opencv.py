import sys
import importlib.util
import os

def check_package(package_name):
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"{package_name} 未安装")
        return False
    else:
        print(f"{package_name} 已安装")
        if package_name == "cv2":
            try:
                import cv2
                print(f"OpenCV 版本: {cv2.__version__}")
                print(f"OpenCV 路径: {cv2.__file__}")
                return True
            except ImportError as e:
                print(f"导入 {package_name} 时出错: {e}")
                return False
        return True

def check_dll_path():
    print("\n系统路径检查:")
    path_var = os.environ.get('PATH', '')
    paths = path_var.split(os.pathsep)
    for p in paths:
        print(f"- {p}")

# 检查Python信息
print(f"Python 版本: {sys.version}")
print(f"Python 路径: {sys.executable}")
print(f"系统平台: {sys.platform}")
print("")

# 检查关键包
packages = ["cv2", "numpy", "PIL"]
for package in packages:
    check_package(package)

# 检查PATH环境变量
check_dll_path()

# 尝试执行OpenCV基本功能
try:
    import cv2
    import numpy as np
    # 创建一个简单的图像
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # 尝试调用OpenCV函数
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("\nOpenCV基本功能测试: 成功")
except Exception as e:
    print(f"\nOpenCV基本功能测试: 失败 - {e}")