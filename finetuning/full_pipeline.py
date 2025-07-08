#!/usr/bin/env python3
"""
전체 파인튜닝 및 모델 변환 파이프라인
메인 프로그램과의 완전한 호환성을 위한 통합 스크립트
"""

import os
import sys
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# 현재 디렉토리를 finetuning으로 설정
os.chdir(Path(__file__).parent)

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_to_file(message, log_file="pipeline.log"):
    """로그를 파일에도 기록"""
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()}: {message}\n")

def check_environment():
    """환경 확인"""
    logger.info("=== 환경 상태 확인 ===")
    
    # GPU 확인
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return True, "gpu"
        else:
            logger.warning("⚠️  CUDA 미사용 - CPU 모드로 진행")
            return True, "cpu"
    except ImportError:
        logger.error("❌ PyTorch가 설치되지 않았습니다")
        return False, "none"

def install_packages():
    """필요한 패키지 자동 설치"""
    logger.info("=== 패키지 설치 확인 ===")
    
    required_packages = [
        "transformers>=4.30.0",
        "datasets>=2.10.0", 
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.41.0"
    ]
    
    for package in required_packages:
        try:
            __import__(package.split('>=')[0])
            logger.info(f"✅ {package.split('>=')[0]} 이미 설치됨")
        except ImportError:
            logger.info(f"📦 {package} 설치 중...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            logger.info(f"✅ {package} 설치 완료")

def run_finetuning():
    """파인튜닝 실행"""
    logger.info("=== 파인튜닝 시작 ===")
    log_to_file("파인튜닝 시작")
    
    try:
        # 파인튜닝 스크립트 실행
        result = subprocess.run([
            sys.executable, "mistral_finetuning.py"
        ], capture_output=True, text=True, timeout=3600)  # 1시간 타임아웃
        
        if result.returncode == 0:
            logger.info("✅ 파인튜닝 완료")
            log_to_file("파인튜닝 성공")
            return True
        else:
            logger.error(f"❌ 파인튜닝 실패: {result.stderr}")
            log_to_file(f"파인튜닝 실패: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ 파인튜닝 타임아웃 (1시간 초과)")
        log_to_file("파인튜닝 타임아웃")
        return False
    except Exception as e:
        logger.error(f"❌ 파인튜닝 오류: {e}")
        log_to_file(f"파인튜닝 오류: {e}")
        return False

def convert_model():
    """파인튜닝된 모델을 GGUF로 변환"""
    logger.info("=== 모델 변환 ===")
    log_to_file("GGUF 변환 시작")
    
    # 파인튜닝된 모델 경로 찾기
    model_dir = Path("models")
    finetuned_models = list(model_dir.glob("*finetuned*"))
    
    if not finetuned_models:
        logger.error("❌ 파인튜닝된 모델을 찾을 수 없습니다")
        log_to_file("파인튜닝된 모델 없음")
        return False
    
    latest_model = max(finetuned_models, key=lambda x: x.stat().st_mtime)
    logger.info(f"📁 파인튜닝된 모델: {latest_model}")
    
    # GGUF 출력 경로
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gguf_output = Path("../models") / f"Nous-Hermes-2-Mistral-7B-DPO-Finetuned-{timestamp}.Q5_K_M.gguf"
    
    try:
        result = subprocess.run([
            sys.executable, "convert_to_gguf.py",
            "--model_path", str(latest_model),
            "--output_path", str(gguf_output),
            "--quantization", "q5_k_m"
        ], capture_output=True, text=True, timeout=1800)  # 30분 타임아웃
        
        if result.returncode == 0:
            logger.info(f"✅ GGUF 변환 완료: {gguf_output}")
            log_to_file(f"GGUF 변환 성공: {gguf_output}")
            return str(gguf_output)
        else:
            logger.error(f"❌ GGUF 변환 실패: {result.stderr}")
            log_to_file(f"GGUF 변환 실패: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ GGUF 변환 타임아웃")
        log_to_file("GGUF 변환 타임아웃")
        return False
    except Exception as e:
        logger.error(f"❌ GGUF 변환 오류: {e}")
        log_to_file(f"GGUF 변환 오류: {e}")
        return False

def update_main_loader(gguf_path):
    """메인 프로그램의 모델 로더 업데이트"""
    logger.info("=== 메인 프로그램 연동 ===")
    
    loader_file = Path("../nous_hermes2_mistral_loader.py")
    
    try:
        # 백업 생성
        backup_file = loader_file.with_suffix('.py.backup')
        with open(loader_file, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 새 모델 경로로 업데이트
        new_content = content.replace(
            'model_path="models/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"',
            f'model_path="{gguf_path}"  # 파인튜닝된 모델 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        )
        
        with open(loader_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"✅ 메인 로더 업데이트 완료: {gguf_path}")
        logger.info(f"📁 백업 파일: {backup_file}")
        log_to_file(f"메인 로더 업데이트: {gguf_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 메인 로더 업데이트 실패: {e}")
        log_to_file(f"메인 로더 업데이트 실패: {e}")
        return False

def generate_test_script():
    """파인튜닝 결과 테스트 스크립트 생성"""
    test_script = """#!/usr/bin/env python3
\"\"\"
파인튜닝된 모델 테스트 스크립트
\"\"\"

import sys
sys.path.append('..')

from nous_hermes2_mistral_loader import load_nous_hermes2_mistral, chat_nous_hermes2_mistral

def test_finetuned_model():
    print("=== 파인튜닝된 모델 테스트 ===")
    
    # 모델 로드
    llm = load_nous_hermes2_mistral()
    
    # 테스트 질문들
    test_prompts = [
        "안녕하세요! 어떻게 도와드릴까요?",
        "날씨에 대해 알려주세요.",
        "Python 프로그래밍에 대해 설명해주세요.",
        "추천해주고 싶은 영화가 있나요?"
    ]
    
    for prompt in test_prompts:
        print(f"\\n질문: {prompt}")
        response = chat_nous_hermes2_mistral(llm, prompt)
        print(f"답변: {response}")
        print("-" * 50)

if __name__ == "__main__":
    test_finetuned_model()
"""
    
    with open("test_finetuned_model.py", 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    logger.info("✅ 테스트 스크립트 생성 완료: test_finetuned_model.py")

def main():
    """전체 파이프라인 실행"""
    start_time = datetime.now()
    logger.info("🚀 파인튜닝 파이프라인 시작")
    log_to_file("파이프라인 시작")
    
    # 1. 환경 확인
    env_ok, device_type = check_environment()
    if not env_ok:
        logger.error("❌ 환경 설정이 올바르지 않습니다")
        return 1
    
    # 2. 패키지 설치
    try:
        install_packages()
    except Exception as e:
        logger.error(f"❌ 패키지 설치 실패: {e}")
        return 1
    
    # 3. 파인튜닝 실행
    if not run_finetuning():
        logger.error("❌ 파인튜닝 실패")
        return 1
    
    # 4. GGUF 변환
    gguf_path = convert_model()
    if not gguf_path:
        logger.error("❌ GGUF 변환 실패")
        return 1
    
    # 5. 메인 프로그램 연동
    if not update_main_loader(gguf_path):
        logger.warning("⚠️  메인 프로그램 연동 실패 (수동으로 경로 변경 필요)")
    
    # 6. 테스트 스크립트 생성
    generate_test_script()
    
    # 완료
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("🎉 파인튜닝 파이프라인 완료!")
    logger.info(f"⏱️  총 소요 시간: {duration}")
    logger.info(f"📁 새 모델: {gguf_path}")
    logger.info(f"🧪 테스트: python test_finetuned_model.py")
    
    log_to_file(f"파이프라인 완료 - 소요시간: {duration}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
