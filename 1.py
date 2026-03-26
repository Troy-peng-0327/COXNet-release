import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU型号: {torch.cuda.get_device_name(0)}")
print(f"GPU计算能力: {torch.cuda.get_device_capability(0)}")

# 检查PyTorch编译时支持的架构
print(f"支持的CUDA架构: {torch.cuda.get_arch_list()}")