#!/usr/bin/env python3
"""
GGUF 모델 양자화 스크립트
RTX 3070 8GB VRAM에 맞도록 모델 크기를 줄입니다.
"""

import os
import sys
from pathlib import Path

def quantize_gguf_model():
    """GGUF 모델을 Q4_K_M으로 양자화하여 크기를 대폭 줄입니다."""
    
    # 경로 설정
    original_model = "e:/Work/ai pz2/models/rtx3070_final_merged.gguf"
    quantized_model = "e:/Work/ai pz2/models/rtx3070_final_merged_q4km.gguf"
    
    # llama.cpp 빌드 경로 확인
    llama_cpp_path = "e:/Work/ai pz2/llama.cpp"
    
    print("🔄 GGUF 모델 양자화 시작...")
    print(f"📁 원본 모델: {original_model}")
    print(f"📁 양자화 모델: {quantized_model}")
    
    # 원본 모델 존재 확인
    if not os.path.exists(original_model):
        print(f"❌ 원본 모델 파일을 찾을 수 없습니다: {original_model}")
        return False
    
    # 원본 모델 크기 확인
    original_size_gb = os.path.getsize(original_model) / (1024**3)
    print(f"📊 원본 모델 크기: {original_size_gb:.1f}GB")
    
    # 양자화 방법들과 예상 크기
    quantization_options = {
        "Q4_K_M": {"size_ratio": 0.35, "quality": "높음", "desc": "4비트 양자화 (추천)"},
        "Q5_K_M": {"size_ratio": 0.45, "quality": "매우 높음", "desc": "5비트 양자화"},
        "Q3_K_M": {"size_ratio": 0.25, "quality": "보통", "desc": "3비트 양자화 (가장 작음)"},
        "Q6_K": {"size_ratio": 0.55, "quality": "최고", "desc": "6비트 양자화"}
    }
    
    print("\n🎯 양자화 옵션:")
    for i, (qtype, info) in enumerate(quantization_options.items(), 1):
        expected_size = original_size_gb * info["size_ratio"]
        print(f"{i}. {qtype}: ~{expected_size:.1f}GB ({info['desc']}) - 품질: {info['quality']}")
    
    # 기본값으로 Q4_K_M 선택 (RTX 3070에 최적)
    selected_quantization = "Q4_K_M"
    expected_size = original_size_gb * quantization_options[selected_quantization]["size_ratio"]
    
    print(f"\n✅ {selected_quantization} 양자화 선택됨")
    print(f"📉 예상 크기: {original_size_gb:.1f}GB → {expected_size:.1f}GB")
    print(f"💾 VRAM 절약: {original_size_gb - expected_size:.1f}GB")
    
    # llama.cpp quantize 바이너리 찾기
    possible_paths = [
        f"{llama_cpp_path}/build/bin/llama-quantize.exe",
        f"{llama_cpp_path}/build/Release/llama-quantize.exe", 
        f"{llama_cpp_path}/build/Debug/llama-quantize.exe",
        f"{llama_cpp_path}/llama-quantize.exe",
        f"{llama_cpp_path}/quantize.exe"
    ]
    
    quantize_binary = None
    for path in possible_paths:
        if os.path.exists(path):
            quantize_binary = path
            break
    
    if not quantize_binary:
        print("❌ llama-quantize 바이너리를 찾을 수 없습니다.")
        print("💡 해결방법:")
        print("1. llama.cpp 빌드: cd llama.cpp && mkdir build && cd build && cmake .. && cmake --build . --config Release")
        print("2. 또는 HuggingFace에서 이미 양자화된 모델 다운로드")
        print("3. 또는 AutoGPTQ 등 다른 양자화 도구 사용")
        return False
    
    print(f"🔧 양자화 도구 발견: {quantize_binary}")
    
    # 양자화 실행
    import subprocess
    
    print(f"\n🚀 양자화 실행 중... (시간이 오래 걸릴 수 있습니다)")
    
    try:
        cmd = [
            quantize_binary,
            original_model,
            quantized_model, 
            selected_quantization
        ]
        
        print(f"💻 실행 명령어: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1시간 제한
        )
        
        if result.returncode == 0:
            if os.path.exists(quantized_model):
                quantized_size_gb = os.path.getsize(quantized_model) / (1024**3)
                compression_ratio = (1 - quantized_size_gb / original_size_gb) * 100
                
                print(f"\n🎉 양자화 완료!")
                print(f"📊 결과:")
                print(f"   - 원본: {original_size_gb:.1f}GB")
                print(f"   - 양자화: {quantized_size_gb:.1f}GB")
                print(f"   - 압축률: {compression_ratio:.1f}%")
                print(f"   - 절약: {original_size_gb - quantized_size_gb:.1f}GB")
                
                print(f"\n✅ RTX 3070 8GB VRAM에 로드 가능: {'예' if quantized_size_gb < 7 else '아니오'}")
                return True
            else:
                print("❌ 양자화 완료했지만 파일이 생성되지 않았습니다.")
                return False
        else:
            print(f"❌ 양자화 실패: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 양자화 시간 초과 (1시간)")
        return False
    except Exception as e:
        print(f"❌ 양자화 오류: {e}")
        return False

def suggest_alternatives():
    """양자화 대안 방법들 제안"""
    print("\n📋 대안 방법들:")
    print("\n1. **HuggingFace Hub에서 이미 양자화된 모델 다운로드**")
    print("   - TheBloke의 GGUF 모델들: https://huggingface.co/TheBloke")
    print("   - 검색: 'Mistral 7B GGUF Q4_K_M'")
    
    print("\n2. **AutoGPTQ로 양자화**")
    print("   - pip install auto-gptq")
    print("   - 더 정밀한 양자화 제어 가능")
    
    print("\n3. **BitsAndBytes 4bit/8bit 로딩**")
    print("   - transformers와 함께 사용")
    print("   - 실시간 양자화 (메모리 절약)")
    
    print("\n4. **모델 프루닝**")
    print("   - 불필요한 레이어/파라미터 제거")
    print("   - 더 복잡하지만 더 큰 압축 가능")

if __name__ == "__main__":
    print("🎯 RTX 3070 최적화: GGUF 모델 양자화")
    print("=" * 50)
    
    success = quantize_gguf_model()
    
    if not success:
        suggest_alternatives()
    else:
        print("\n💡 다음 단계:")
        print("1. model_manager.py에서 양자화된 모델 경로 업데이트")
        print("2. GPU 레이어 수를 32+ 로 증가 (더 많은 GPU 활용)")
        print("3. 성능 테스트 및 튜닝")
