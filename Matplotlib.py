import matplotlib.pyplot as plt
import numpy as np
import re
import os

# 数据提取函数
def extract_training_data(log_file):
    epochs = []
    train_losses = []
    test_losses = []
    mious = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 使用正则表达式匹配数据行
            match = re.search(r'- (\d+):\t - train_loss: ([0-9.]+):\t - test_loss: ([0-9.]+):\t mIoU ([0-9.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                test_losses.append(float(match.group(3)))
                mious.append(float(match.group(4)))
                
    return epochs, train_losses, test_losses, mious

# 读取日志文件
log_file = r"result\NUDT-SIRST_DNANet_21_02_2025_23_09_23_wDS\DNANet_NUDT-SIRST_best_IoU_IoU.log"
epochs, train_losses, test_losses, mious = extract_training_data(log_file)

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置全局字体大小
plt.rcParams['font.size'] = 14  # 设置全局默认字体大小

# 创建图表
plt.figure(figsize=(15, 10))

# 绘制损失曲线
plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, 'b-', linewidth=2.0, label='训练损失')
plt.plot(epochs, test_losses, 'r-', linewidth=2.0, label='测试损失')
plt.xlabel('轮次', fontsize=16)
plt.ylabel('损失', fontsize=16)
plt.title('训练和测试损失曲线', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)  # 增加刻度字体大小

# 绘制mIoU曲线
plt.subplot(2, 1, 2)
plt.plot(epochs, mious, 'g-', linewidth=2.0, label='平均交并比')
plt.axhline(y=0.85, color='r', linestyle='--', linewidth=2.0, label='0.85 基准线')
plt.xlabel('轮次', fontsize=16)
plt.ylabel('平均交并比', fontsize=16)
plt.title('模型平均交并比性能曲线', fontsize=20)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)  # 增加刻度字体大小

plt.tight_layout()
plt.savefig('训练可视化结果.png', dpi=300)  # 增加分辨率
plt.show()