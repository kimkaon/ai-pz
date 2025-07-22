"""
응답 처리 모듈 - 리팩터링된 버전
main.py에서 분리된 응답 생성 및 처리 기능
"""

import os
from utils.logging_utils import log_print, clean_llm_response, settings_manager
from core.initialization import get_global_llm

# 파인튜닝 기능 사용 가능 여부 확인
try:
    from finetuning_integration import process_with_finetuned_models_main, get_finetuning_stats
    FINETUNING_ENABLED = True
except ImportError:
    FINETUNING_ENABLED = False

class ResponseHandler:
    """응답 처리 클래스"""
    
    def __init__(self):
        self.response_stats = {
            'total_responses': 0,
            'finetuned_responses': 0,
            'fallback_responses': 0
        }
    
    def generate_response(self, prompt, conversation_history=None):
        """통합 응답 생성"""
        try:
            self.response_stats['total_responses'] += 1
            
            # 1. 파인튜닝 모델 우선 시도
            response, used_finetuned, model_info = self.try_finetuned_response(prompt)
            if used_finetuned and response:
                self.response_stats['finetuned_responses'] += 1
                log_print(f"파인튜닝 응답 사용: {model_info.get('model_used', 'unknown')}", "model_loading")
                return clean_llm_response(response)
            
            # 2. 기본 모델 사용
            response = self._generate_basic_response(prompt, conversation_history)
            if response:
                self.response_stats['fallback_responses'] += 1
                log_print("기본 모델 응답 사용", "model_loading")
                return clean_llm_response(response)
            
            return "죄송합니다. 응답을 생성할 수 없습니다."
            
        except Exception as e:
            log_print(f"응답 생성 오류: {e}", "error")
            return f"오류가 발생했습니다: {e}"
    
    def try_finetuned_response(self, prompt):
        """파인튜닝 응답 시도"""
        current_model = settings_manager.get('model.current', 'original')
        
        # 하이브리드 파인튜닝 모델
        if FINETUNING_ENABLED and current_model in ['hybrid', 'english_unified']:
            try:
                response_data = process_with_finetuned_models_main(prompt)
                if response_data and response_data.get('response'):
                    model_info = {
                        'model_used': current_model,
                        'category': response_data.get('category', 'unknown'),
                        'confidence': response_data.get('confidence', 0.0),
                        'quality_level': response_data.get('quality_level', 'unknown')
                    }
                    return response_data['response'], True, model_info
            except Exception as e:
                log_print(f"파인튜닝 모델 오류: {e}", "model_loading")
        
        # RTX 3070 특화 모델들
        elif current_model in ['rtx3070_unfiltered', 'rtx3070_language_limited', 'rtx3070_gguf']:
            try:
                # 전역 model_manager 인스턴스를 직접 사용
                import models.model_manager as mm
                model_manager = mm.model_manager
                
                log_print(f"🔍 ResponseHandler에서 모델 매니저 상태 확인: {id(model_manager)}", "model_loading")
                
                if current_model == 'rtx3070_unfiltered':
                    response = model_manager.rtx3070_unfiltered_model.generate_response(prompt)
                elif current_model == 'rtx3070_language_limited':
                    response = model_manager.rtx3070_language_limited_model.generate_response(prompt)
                elif current_model == 'rtx3070_gguf':
                    response = model_manager.rtx3070_gguf_model.generate_response(prompt)
                
                if response:
                    model_info = {
                        'model_used': current_model,
                        'model_type': 'rtx3070_finetuned'
                    }
                    return response, True, model_info
                    
            except Exception as e:
                log_print(f"RTX 3070 모델 오류: {e}", "model_loading")
        
        # 영어 통합 모델
        elif current_model == 'english_unified':
            try:
                from models.model_manager import model_manager
                response = model_manager.english_finetuned_model.generate_response(prompt)
                if response:
                    model_info = {
                        'model_used': current_model,
                        'model_type': 'english_finetuned'
                    }
                    return response, True, model_info
            except Exception as e:
                log_print(f"영어 통합 모델 오류: {e}", "model_loading")
        
        return None, False, {}
    
    def _generate_basic_response(self, prompt, conversation_history=None):
        """기본 모델 응답 생성"""
        llm = get_global_llm()
        if not llm:
            return None
        
        try:
            from prompt_templates import make_default_prompt
            formatted_prompt = make_default_prompt(prompt)
            
            from nous_hermes2_mistral_loader import chat_nous_hermes2_mistral
            response = chat_nous_hermes2_mistral(llm, formatted_prompt)
            return response
            
        except Exception as e:
            log_print(f"기본 모델 응답 생성 오류: {e}", "error")
            return None
    
    def get_stats(self):
        """응답 통계 반환"""
        return self.response_stats.copy()
