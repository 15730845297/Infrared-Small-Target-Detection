import os
import sys
import tkinter as tk

# 设置OpenCV DLL搜索路径
try:
    # 添加可能的OpenCV DLL路径
    import cv2
    cv2_path = cv2.__file__
    cv2_dir = os.path.dirname(cv2_path)
    dll_path = os.path.join(os.path.dirname(cv2_dir), 'Library', 'bin')
    
    if os.path.exists(dll_path):
        # 将DLL路径添加到环境变量
        os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
        print(f"已添加OpenCV DLL路径: {dll_path}")
    
    print(f"OpenCV 版本: {cv2.__version__}")
    print(f"OpenCV 路径: {cv2_path}")
    
except ImportError as e:
    print(f"无法导入OpenCV: {e}")
    input("按任意键退出...")
    sys.exit(1)

# 启动应用程序
try:
    from infrared_detection_app import InfraredDetectionApp 
    
    root = tk.Tk()
    app = InfraredDetectionApp(root)
    root.mainloop()
    
except Exception as e:
    print(f"应用程序启动失败: {e}")
    import traceback
    traceback.print_exc()
    input("按任意键退出...")