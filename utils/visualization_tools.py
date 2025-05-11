import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib
matplotlib.use('Agg')  # 避免需要GUI环境

class TrainingVisualizer:
    def __init__(self, save_dir):
        """
        初始化训练可视化器
        
        Args:
            save_dir: 保存可视化结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(os.path.join(save_dir, 'visualization'), exist_ok=True)
        
        # 存储训练历史数据
        self.history = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'mIoU': [],
            'recall': [],
            'precision': [],
            'learning_rate': []
        }
        
        # 存储预测和真实标签，用于混淆矩阵
        self.all_preds = []
        self.all_labels = []
        
    def update_history(self, epoch, train_loss, test_loss, mIoU, recall, precision, lr):
        """更新训练历史数据"""
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['test_loss'].append(test_loss)
        self.history['mIoU'].append(mIoU)
        self.history['recall'].append(recall[5] if isinstance(recall, list) and len(recall) > 5 else recall)  # 使用中间阈值的召回率
        self.history['precision'].append(precision[5] if isinstance(precision, list) and len(precision) > 5 else precision)
        self.history['learning_rate'].append(lr)
        
    def collect_batch_results(self, preds, labels):
        """收集批次预测结果和真实标签"""
        # 二值化预测
        batch_preds = (torch.sigmoid(preds) > 0.5).cpu().numpy().flatten()
        batch_labels = labels.cpu().numpy().flatten()
        
        self.all_preds.extend(batch_preds)
        self.all_labels.extend(batch_labels)
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(15, 12))
        
        # 损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Training Loss')
        plt.plot(self.history['epoch'], self.history['test_loss'], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # mIoU曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.history['epoch'], self.history['mIoU'], 'g-', label='mIoU')
        plt.axhline(y=max(self.history['mIoU']), color='r', linestyle='--', 
                    label=f'Best mIoU: {max(self.history["mIoU"]):.4f}')
        plt.xlabel('Epoch')
        plt.ylabel('mIoU')
        plt.title('Mean IoU Evolution')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 精确率和召回率曲线
        plt.subplot(2, 2, 3)
        plt.plot(self.history['epoch'], self.history['precision'], 'm-', label='Precision')
        plt.plot(self.history['epoch'], self.history['recall'], 'c-', label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 学习率曲线
        plt.subplot(2, 2, 4)
        plt.plot(self.history['epoch'], self.history['learning_rate'], 'y-')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'visualization', 'results.png'))
        plt.close()
        
    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        if not self.all_preds or not self.all_labels:
            return
            
        # 计算二分类混淆矩阵
        cm = confusion_matrix(self.all_labels, self.all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Background', 'Target'],
                   yticklabels=['Background', 'Target'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'visualization', 'confusion_matrix.png'))
        plt.close()
        
    def plot_pr_curves(self, roc_data):
        """绘制精确率-召回率曲线和F1曲线"""
        thresholds = np.linspace(0, 1, 11)  # 从ROC计算的阈值
        
        if 'precision' not in roc_data or 'recall' not in roc_data:
            return
            
        precision = roc_data['precision']
        recall = roc_data['recall']
        
        # 计算F1分数
        f1_scores = []
        for p, r in zip(precision, recall):
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0
            f1_scores.append(f1)
        
        # 精确率曲线
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, precision, 'b-')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Precision')
        plt.title('Precision vs Threshold')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'visualization', 'P_curve.png'))
        plt.close()
        
        # 召回率曲线
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, recall, 'r-')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Recall')
        plt.title('Recall vs Threshold')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'visualization', 'R_curve.png'))
        plt.close()
        
        # PR曲线
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'g-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'visualization', 'PR_curve.png'))
        plt.close()
        
        # F1曲线
        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, f1_scores, 'm-')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'visualization', 'F1_curve.png'))
        plt.close()
        
    def visualize_predictions(self, images, labels, preds, batch_idx):
        """可视化批次的预测结果和真实标签"""
        # 确保目录存在
        os.makedirs(os.path.join(self.save_dir, 'visualization'), exist_ok=True)
        
        batch_size = images.size(0)
        # 最多显示16张图
        num_images = min(batch_size, 16)
        
        # 创建网格布局
        rows = int(np.ceil(num_images / 4))
        cols = min(4, num_images)
        
        # 真实标签可视化
        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(num_images):
            plt.subplot(rows, cols, i + 1)
            
            # 转换回图像空间
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # 叠加真实标签
            label = labels[i].cpu().numpy()
            
            # 创建彩色显示
            overlay = np.zeros_like(img)
            overlay[..., 0] = label  # 在红色通道上显示标签
            
            # 叠加图像和标签
            blended = img * 0.7 + overlay * 0.3
            
            plt.imshow(blended)
            plt.title(f"True - {i}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'visualization', f'val_batch{batch_idx}_labels.jpg'))
        plt.close()
        
        # 预测结果可视化
        plt.figure(figsize=(cols * 3, rows * 3))
        for i in range(num_images):
            plt.subplot(rows, cols, i + 1)
            
            # 转换回图像空间
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # 获取预测结果并转换为二值
            if isinstance(preds, list):
                pred = torch.sigmoid(preds[-1][i]).cpu().numpy()
            else:
                pred = torch.sigmoid(preds[i]).cpu().numpy()
                
            pred_binary = (pred > 0.5).astype(float)
            
            # 创建彩色显示
            overlay = np.zeros_like(img)
            overlay[..., 2] = pred_binary  # 在蓝色通道上显示预测
            
            # 叠加图像和预测
            blended = img * 0.7 + overlay * 0.3
            
            plt.imshow(blended)
            plt.title(f"Pred - {i}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'visualization', f'val_batch{batch_idx}_pred.jpg'))
        plt.close()