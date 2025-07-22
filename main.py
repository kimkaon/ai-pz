"""
AI 어시스턴트 메인 애플리케이션
리팩터링된 모듈 기반으로 실행되는 간단한 메인 파일
"""

import torch
import warnings
import os

# 경고 메시지 및 로그 제어
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTORCH_WARNINGS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import logging
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("ctranslate2").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

print("main.py에서 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

# 분리된 모듈들 import
from core.initialization import initialize_system, cleanup_system, get_global_llm, set_global_llm
from interfaces.menu_system import MenuSystem
from processing.response_handler import ResponseHandler
from utils.logging_utils import log_print

def main():
    """메인 실행 함수 - 모듈화된 시스템 실행"""
    print("🤖 AI 어시스턴트를 시작합니다...")
    
    try:
        # 1. 시스템 초기화
        print("⚡ 시스템 초기화 중...")
        initialize_system()
        
        # 2. 모델 매니저 초기화 (전역 인스턴스 사용)
        print("🔍 모델 관리자 초기화 중...")
        import models.model_manager as mm
        model_manager = mm.model_manager
        model_manager.detect_available_models()
        
        # 3. 모델 미리 로딩 (기본 LLM 로드 후 수행)
        print("🚀 모델 미리 로딩 중...")
        model_manager.preload_models()
        
        # 4. 응답 처리기 초기화
        print("📝 응답 처리기 초기화 중...")
        response_handler = ResponseHandler()
        
        # 5. 메뉴 시스템 초기화 및 실행
        print("🎯 메뉴 시스템 시작...")
        menu_system = MenuSystem(model_manager, response_handler)
        menu_system.run_main_menu()
        
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 종료되었습니다.")
    except Exception as e:
        log_print(f"❌ 시스템 오류: {e}", "error")
        print(f"오류가 발생했습니다: {e}")
    finally:
        # 6. 시스템 정리
        print("🧹 시스템 정리 중...")
        cleanup_system()
        print("✅ 프로그램이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
