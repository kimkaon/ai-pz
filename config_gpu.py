# GPU 환경 설정 예시 (config_gpu.py)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch

def setup_gpu(cuda_device=0):
    """
    CUDA 및 PyTorch GPU 환경을 자동으로 설정합니다.
    - cuda_device: 사용할 GPU 번호 (여러 개일 경우)
    """
    if not torch.cuda.is_available():
        print("❗ CUDA GPU를 사용할 수 없습니다. 드라이버, CUDA, cuDNN 설치를 확인하세요.")
        return False
    torch.cuda.set_device(cuda_device)
    print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(cuda_device)} (device {cuda_device})")
    return True

# 사용 예시 (main.py 등에서)
# from config_gpu import setup_gpu
# setup_gpu(cuda_device=0)
