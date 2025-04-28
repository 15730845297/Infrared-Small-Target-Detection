import torch
print(torch.cuda.is_available())  # 预期输出 True
print(torch.cuda.current_device())  # 预期输出 0
print(torch.cuda.get_device_name(0))  # 预期输出 NVIDIA GeForce RTX 4070
print(torch.rand(3,3).cuda())  # 生成 GPU Tensor
print(torch.version.cuda)  # 预期输出 11.8
print(torch.__version__)          # 应 ≥1.7.0
print(torch.cuda.is_available())  # 应输出True
