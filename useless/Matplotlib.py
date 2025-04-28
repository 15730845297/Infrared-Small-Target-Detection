import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Data extraction function
def extract_training_data(log_file):
    epochs = []
    train_losses = []
    test_losses = []
    mious = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Match data lines with regex
            match = re.search(r'- (\d+):\t - train_loss: ([0-9.]+):\t - test_loss: ([0-9.]+):\t mIoU ([0-9.]+)', line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                test_losses.append(float(match.group(3)))
                mious.append(float(match.group(4)))
                
    return epochs, train_losses, test_losses, mious

# Read log file
log_file = r"result\NUDT-SIRST_DNANet_21_02_2025_23_09_23_wDS\DNANet_NUDT-SIRST_best_IoU_IoU.log"
epochs, train_losses, test_losses, mious = extract_training_data(log_file)

# Create plot
plt.figure(figsize=(15, 10))

# Plot loss curves
plt.subplot(2, 1, 1)
plt.plot(epochs, train_losses, 'b-', label='Training Loss')
plt.plot(epochs, test_losses, 'r-', label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('DNANet Training and Testing Loss Curves')
plt.legend()
plt.grid(True)

# Plot mIoU curve
plt.subplot(2, 1, 2)
plt.plot(epochs, mious, 'g-', label='mIoU')
plt.axhline(y=0.85, color='r', linestyle='--', label='0.85 Baseline')
plt.xlabel('Epoch')
plt.ylabel('mIoU')
plt.title('DNANet Model mIoU Performance Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_visualization.png')
plt.show()