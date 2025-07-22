"""
초기화 모듈
main.py에서 분리된 초기화 관련 기능
"""

import torch
import warnings
import os
import logging

# 전역 LLM 인스턴스
_global_llm = None

def setup_environment():
    """환경 설정 초기화"""
    # 경고 메시지 숨기기
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['PYTORCH_WARNINGS'] = '0'
    
    # 오프라인 모드 설정
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # faster_whisper 로그 숨기기
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("ctranslate2").setLevel(logging.WARNING)
    
    # transformers 경고 숨기기
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
    
    print("main.py에서 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

def get_global_llm():
    """전역 LLM 인스턴스 가져오기"""
    return _global_llm

def set_global_llm(llm_instance):
    """전역 LLM 인스턴스 설정"""
    global _global_llm
    _global_llm = llm_instance

def initialize_system():
    """시스템 초기화"""
    global _global_llm
    
    # 환경 설정
    setup_environment()
    
    # GPU 설정
    print("⚡ GPU 설정 중...")
    from config_gpu import setup_gpu
    setup_gpu(cuda_device=0)
    
    # 기본 LLM 로딩
    print("� 기본 LLM 로딩 중...")
    from nous_hermes2_mistral_loader import load_nous_hermes2_mistral
    _global_llm = load_nous_hermes2_mistral()
    print("✅ 기본 모델 로딩 완료!")
    
    # 하이브리드 파인튜닝 시스템 초기화
    try:
        from finetuning_integration import get_finetuning_stats
        from hybrid_finetuning_integration import initialize_hybrid_system
        initialize_hybrid_system(_global_llm)
        print("✅ 하이브리드 파인튜닝 시스템 초기화 완료")
    except ImportError:
        print("⚠️ 하이브리드 파인튜닝 시스템 없음")
    
    # 전역 model_manager 인스턴스를 직접 사용
    import models.model_manager as mm
    model_manager = mm.model_manager
    model_manager.detect_available_models()
    
    print("🎯 시스템 초기화 완료!")

def get_global_llm():
    """전역 LLM 인스턴스 반환"""
    return _global_llm

def cleanup_system():
    """시스템 정리"""
    try:
        from models.model_manager import FINETUNING_ENABLED
        if FINETUNING_ENABLED:
            from finetuning_integration import cleanup_finetuning_system
            cleanup_finetuning_system()
    except:
        pass
