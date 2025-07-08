#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 파인튜닝 통합 시스템 (메인 인터페이스)
Hybrid Fine-tuning Integration System (Main Interface)

통합형 모델 기본 + 전문모델 동적 로딩 방식
Unified model as primary + dynamic specialist model loading
"""

import sys
from pathlib import Path
import logging

# 하이브리드 시스템 임포트
try:
    from hybrid_finetuning_integration import process_with_finetuned_models, get_system_stats
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 하이브리드 시스템 임포트 실패 / Failed to import hybrid system: {e}")
    HYBRID_AVAILABLE = False

# 기존 프롬프트 템플릿 fallback
try:
    from prompt_templates import make_role_prompt
    PROMPT_TEMPLATES_AVAILABLE = True
except ImportError:
    PROMPT_TEMPLATES_AVAILABLE = False

def process_with_finetuned_models_main(user_input: str, enable_specialist: bool = True) -> dict:
    """
    메인 파인튜닝 처리 함수 (하이브리드 방식)
    Main fine-tuning processing function (hybrid approach)
    
    Args:
        user_input: 사용자 입력
        enable_specialist: 전문모델 사용 허용 여부
        
    Returns:
        dict: 응답 결과
    """
    
    if not HYBRID_AVAILABLE:
        return {
            'response': f"하이브리드 시스템을 사용할 수 없습니다. 기본 응답: {user_input}",
            'category': 'fallback',
            'model_used': 'none',
            'quality_level': 'basic',
            'can_upgrade': False,
            'error': 'Hybrid system not available'
        }
    
    try:
        # 하이브리드 시스템으로 처리
        result = process_with_finetuned_models(
            user_input, 
            force_specialist=False  # 자동 결정 방식 사용
        )
        
        # 전문모델 비활성화 시 통합모델만 사용
        if not enable_specialist and result.get('model_used', '').endswith('_specialist'):
            # 통합모델로 재처리
            result = process_with_finetuned_models(user_input, force_specialist=False)
            # 강제로 통합모델 결과로 변경
            result['model_used'] = 'unified'
            result['quality_level'] = 'standard'
            result['response'] = result['response'].replace('[QNA SPECIALIST]', '[UNIFIED]')
            # 하이브리드 시스템에서는 daily_chat이 통합모델에서 처리되므로 UNIFIED로 표시
            result['response'] = result['response'].replace('[DAILY_CHAT SPECIALIST]', '[UNIFIED]')
            result['response'] = result['response'].replace('[TECHNICAL SPECIALIST]', '[UNIFIED]')
        
        return result
        
    except Exception as e:
        logging.error(f"하이브리드 처리 오류 / Hybrid processing error: {e}")
        
        # Fallback to prompt templates
        if PROMPT_TEMPLATES_AVAILABLE:
            try:
                prompt = make_role_prompt(user_input)
                return {
                    'response': f"[FALLBACK] {user_input}에 대한 기본 응답입니다.",
                    'category': 'fallback',
                    'model_used': 'prompt_template',
                    'quality_level': 'basic',
                    'can_upgrade': True,
                    'error': f'Hybrid error: {str(e)}'
                }
            except Exception as fallback_error:
                logging.error(f"Fallback 오류 / Fallback error: {fallback_error}")
        
        # Final fallback
        return {
            'response': f"죄송합니다. 현재 파인튜닝 시스템에 문제가 있습니다: {user_input}",
            'category': 'error',
            'model_used': 'none',
            'quality_level': 'error',
            'can_upgrade': False,
            'error': f'All systems failed: {str(e)}'
        }

def get_finetuning_stats() -> dict:
    """파인튜닝 시스템 통계 조회 / Get fine-tuning system statistics"""
    
    if not HYBRID_AVAILABLE:
        return {
            'status': 'unavailable',
            'error': 'Hybrid system not available'
        }
    
    try:
        stats = get_system_stats()
        stats['status'] = 'active'
        stats['system_type'] = 'hybrid'
        return stats
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'system_type': 'hybrid'
        }

def request_specialist_mode(user_input: str, category: str = None) -> dict:
    """
    전문가 모드 명시적 요청
    Explicit request for specialist mode
    
    Args:
        user_input: 사용자 입력
        category: 선택적 카테고리 지정
        
    Returns:
        dict: 전문가 응답 결과
    """
    
    if not HYBRID_AVAILABLE:
        return process_with_finetuned_models_main(user_input, enable_specialist=False)
    
    try:
        # 전문가 모드로 강제 처리
        result = process_with_finetuned_models(user_input, force_specialist=True)
        result['specialist_requested'] = True
        return result
        
    except Exception as e:
        logging.error(f"전문가 모드 오류 / Specialist mode error: {e}")
        return process_with_finetuned_models_main(user_input, enable_specialist=False)

def cleanup_finetuning_system():
    """파인튜닝 시스템 정리 / Cleanup fine-tuning system"""
    
    if HYBRID_AVAILABLE:
        try:
            # 하이브리드 시스템 정리 (필요시 구현)
            pass
        except Exception as e:
            logging.error(f"시스템 정리 오류 / System cleanup error: {e}")

# 호환성을 위한 기존 함수명 유지
def integrate_finetuned_response(user_input: str) -> str:
    """
    기존 호환성을 위한 래퍼 함수
    Wrapper function for backward compatibility
    """
    result = process_with_finetuned_models_main(user_input)
    return result.get('response', user_input)

if __name__ == "__main__":
    # 테스트 코드
    print("🚀 하이브리드 파인튜닝 통합 시스템 테스트")
    print("=" * 50)
    
    test_cases = [
        ("안녕하세요!", "일상 인사"),
        ("파이썬 프로그래밍이 뭐예요?", "기술 질문"),
        ("지구의 나이는?", "사실 질문"), 
        ("SketchUp 전문가 모드로 고급 모델링 알려주세요", "전문가 요청"),
        ("요즘 너무 우울해요...", "감정 상담")
    ]
    
    for user_input, description in test_cases:
        print(f"\n📝 테스트: {description}")
        print(f"입력: {user_input}")
        
        # 일반 모드
        result = process_with_finetuned_models_main(user_input)
        print(f"🤖 [{result['model_used']}] {result['response'][:80]}...")
        print(f"📊 카테고리: {result['category']}, 품질: {result['quality_level']}")
        
        # 전문가 모드 (업그레이드 가능한 경우)
        if result.get('can_upgrade'):
            print("🎯 전문가 모드 시도...")
            specialist_result = request_specialist_mode(user_input)
            print(f"🎓 [{specialist_result['model_used']}] {specialist_result['response'][:80]}...")
    
    # 시스템 통계
    print(f"\n📈 시스템 통계:")
    stats = get_finetuning_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
