#!/usr/bin/env python3
"""
파인튜닝된 모델을 GGUF 형식으로 변환하는 스크립트
llama-cpp-python과 호환되도록 변환
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_llama_cpp():
    """llama.cpp 설정 및 설치"""
    logger.info("=== llama.cpp 설정 ===")
    
    # llama.cpp 클론 (이미 있으면 업데이트)
    llama_cpp_dir = Path("llama.cpp")
    if not llama_cpp_dir.exists():
        logger.info("llama.cpp 클론 중...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git"
        ], check=True)
    else:
        logger.info("llama.cpp 이미 존재함")
    
    # 의존성 확인
    try:
        import torch
        import transformers
        logger.info("✅ PyTorch 및 Transformers 확인됨")
    except ImportError as e:
        logger.error(f"❌ 필수 라이브러리 누락: {e}")
        return False
    
    return True

def convert_to_gguf(model_path, output_path, quantization="q5_k_m"):
    """
    HuggingFace 모델을 GGUF로 변환
    
    Args:
        model_path: 파인튜닝된 모델 경로
        output_path: 출력 GGUF 파일 경로
        quantization: 양자화 타입 (q4_0, q4_1, q5_0, q5_1, q5_k_m, q8_0)
    """
    logger.info(f"모델 변환 시작: {model_path} -> {output_path}")
    
    model_path = Path(model_path)
    output_path = Path(output_path)
    
    if not model_path.exists():
        logger.error(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
        return False
    
    # 출력 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: HF to GGML 변환
        temp_ggml = output_path.with_suffix('.ggml')
        logger.info("Step 1: HuggingFace -> GGML 변환")
        
        convert_cmd = [
            sys.executable, "llama.cpp/convert.py",
            str(model_path),
            "--outtype", "f16",
            "--outfile", str(temp_ggml)
        ]
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"GGML 변환 실패: {result.stderr}")
            return False
        
        # Step 2: GGML to GGUF 양자화
        logger.info(f"Step 2: GGML -> GGUF 양자화 ({quantization})")
        
        # llama.cpp 빌드 (Windows)
        if os.name == 'nt':  # Windows
            quantize_exe = "llama.cpp/quantize.exe"
            if not Path(quantize_exe).exists():
                logger.info("quantize.exe 빌드 중...")
                build_cmd = ["cmake", "-B", "llama.cpp/build", "-S", "llama.cpp"]
                subprocess.run(build_cmd)
                make_cmd = ["cmake", "--build", "llama.cpp/build", "--config", "Release"]
                subprocess.run(make_cmd)
        else:
            quantize_exe = "llama.cpp/quantize"
        
        quantize_cmd = [
            quantize_exe,
            str(temp_ggml),
            str(output_path),
            quantization
        ]
        
        result = subprocess.run(quantize_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"GGUF 양자화 실패: {result.stderr}")
            return False
        
        # 임시 파일 정리
        if temp_ggml.exists():
            temp_ggml.unlink()
        
        logger.info(f"✅ 변환 완료: {output_path}")
        
        # 파일 크기 확인
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"📁 파일 크기: {size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 변환 중 오류: {e}")
        return False

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="파인튜닝된 모델을 GGUF로 변환")
    parser.add_argument("--model_path", required=True, help="파인튜닝된 모델 경로")
    parser.add_argument("--output_path", required=True, help="출력 GGUF 파일 경로")
    parser.add_argument("--quantization", default="q5_k_m", 
                       choices=["q4_0", "q4_1", "q5_0", "q5_1", "q5_k_m", "q8_0"],
                       help="양자화 타입")
    
    args = parser.parse_args()
    
    # llama.cpp 설정
    if not setup_llama_cpp():
        logger.error("llama.cpp 설정 실패")
        return 1
    
    # 변환 실행
    success = convert_to_gguf(args.model_path, args.output_path, args.quantization)
    
    if success:
        logger.info("🎉 모델 변환이 완료되었습니다!")
        logger.info(f"메인 프로그램에서 사용하려면:")
        logger.info(f"  - 파일 경로: {args.output_path}")
        logger.info(f"  - nous_hermes2_mistral_loader.py에서 model_path 수정")
        return 0
    else:
        logger.error("❌ 모델 변환에 실패했습니다.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
