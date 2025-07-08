#!/usr/bin/env python3
"""
GPU 및 환경 설정 상태 확인 스크립트
"""

import sys

def check_environment():
    """환경 상태 전체 점검"""
    print("=== AI PZ2 환경 상태 점검 ===\n")
    
    # 1. PyTorch 및 CUDA
    print("1. PyTorch & CUDA 상태:")
    try:
        import torch
        print(f"   ✅ PyTorch 버전: {torch.__version__}")
        
        if hasattr(torch, 'cuda'):
            if torch.cuda.is_available():
                print(f"   ✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
                print(f"   💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("   ❌ CUDA 사용 불가")
        else:
            print("   ❌ PyTorch가 CPU 전용 버전입니다")
            print("   🔧 해결: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        print("   ❌ PyTorch가 설치되지 않았습니다")
    
    # 2. Transformers
    print("\n2. Transformers 라이브러리:")
    try:
        import transformers
        print(f"   ✅ Transformers 버전: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers 미설치")
        print("   🔧 해결: pip install transformers")
    
    # 3. 파인튜닝 관련 패키지
    print("\n3. 파인튜닝 패키지:")
    
    packages = {
        'datasets': 'datasets',
        'peft': 'peft', 
        'bitsandbytes': 'bitsandbytes',
        'accelerate': 'accelerate'
    }
    
    missing_packages = []
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n   🔧 해결: pip install {' '.join(missing_packages)}")
    
    # 4. GGUF 모델 지원 (llama-cpp-python)
    print("\n4. GGUF 모델 지원:")
    try:
        import llama_cpp
        print(f"   ✅ llama-cpp-python 설치됨")
    except ImportError:
        print("   ❌ llama-cpp-python 미설치")
        print("   🔧 해결: pip install llama-cpp-python")
    
    # 5. 현재 메인 모델 확인
    print("\n5. 메인 프로그램 모델:")
    import os
    model_path = "models/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / 1024**3
        print(f"   ✅ {model_path} ({size_gb:.1f}GB)")
    else:
        print(f"   ❌ {model_path} 찾을 수 없음")
    
    # 6. 파인튜닝 데이터 확인
    print("\n6. 파인튜닝 데이터:")
    data_files = [
        "finetuning/datasets/processed_english/unified_train.jsonl",
        "finetuning/datasets/processed_english/unified_validation.jsonl"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for line in f if line.strip() and not line.startswith('#'))
            print(f"   ✅ {os.path.basename(file_path)}: {lines}개 데이터")
        else:
            print(f"   ❌ {file_path} 없음")
    
    print("\n=== 권장 파인튜닝 방법 ===")
    
    # PyTorch CUDA 여부에 따른 권장사항
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            print("🚀 GPU 파인튜닝 권장:")
            print("   - 모델: NousResearch/Nous-Hermes-2-Mistral-7B-DPO")
            print("   - 방법: LoRA + 4bit 양자화")
            print("   - 예상 시간: 30분-1시간")
            print("   - GPU 메모리: 6-8GB 사용")
        else:
            print("💻 CPU 파인튜닝:")
            print("   - 모델: microsoft/DialoGPT-small")
            print("   - 방법: 전체 파인튜닝")
            print("   - 예상 시간: 2-4시간")
            print("   - RAM: 8-16GB 사용")
    except:
        pass

if __name__ == "__main__":
    check_environment()
