#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
하이브리드 파인튜닝 통합 시스템
Hybrid Fine-tuning Integration System

통합형 모델을 기본으로 하되, 필요시 전문모델을 동적 로딩하는 하이브리드 시스템
Uses unified model as primary, with dynamic loading of specialist models when needed
"""

import json
import os
import sys
from pathlib import Path
import logging
from typing import Dict, Optional, Tuple, List
import threading
import time
import gc

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from prompt_templates import make_role_prompt, ROLE_TYPES

# 메인 LLM 모듈 임포트
try:
    from nous_hermes2_mistral_loader import chat_nous_hermes2_mistral
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    chat_nous_hermes2_mistral = None

# 전역 LLM 인스턴스 (메인에서 전달받음)
GLOBAL_LLM = None

def set_global_llm(llm_instance):
    """메인에서 LLM 인스턴스 설정"""
    global GLOBAL_LLM
    GLOBAL_LLM = llm_instance
import gc

# 현재 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from prompt_templates import make_role_prompt, ROLE_TYPES

class HybridFinetunedSystem:
    """하이브리드 파인튜닝 시스템 / Hybrid fine-tuned system"""
    
    def __init__(self):
        self.finetuning_dir = Path(__file__).parent / "finetuning"
        self.models_dir = self.finetuning_dir / "models"
        
        # 모델 상태 / Model states
        self.unified_model = None
        self.specialist_models = {
            'qna': None,
            'technical': None
        }
        
        # 모델 로딩 상태 / Model loading states
        self.model_loading = {
            'qna': False,
            'technical': False
        }
        
        # 성능 통계 / Performance statistics
        self.stats = {
            'unified_responses': 0,
            'specialist_responses': 0,
            'model_switches': 0,
            'avg_response_time': 0.0
        }
        
        # 로깅 설정 / Logging setup
        self.setup_logging()
        
    def setup_logging(self):
        """로깅 설정 / Setup logging"""
        log_dir = self.finetuning_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "hybrid_system.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("HybridSystem")
        
    def initialize_unified_model(self) -> bool:
        """통합형 모델 초기화 / Initialize unified model"""
        try:
            self.logger.info("🚀 통합형 모델 초기화 중... / Initializing unified model...")
            
            # 통합형 모델 체크포인트 확인 / Check unified model checkpoint
            unified_model_path = self.models_dir / "unified_model" / "model.safetensors"
            
            if unified_model_path.exists():
                self.logger.info("✅ 파인튜닝된 통합형 모델 발견 / Found fine-tuned unified model")
                # 실제 모델 로딩 코드는 구현에 따라 추가
                # self.unified_model = load_model(unified_model_path)
                self.unified_model = "unified_finetuned"  # 플레이스홀더
            else:
                self.logger.info("📝 기본 LLM 사용 (통합형 미파인튜닝) / Using base LLM (unified not fine-tuned)")
                self.unified_model = "base_llm"  # 플레이스홀더
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 통합형 모델 초기화 실패 / Failed to initialize unified model: {e}")
            return False
    
    def classify_and_assess_confidence(self, user_input: str) -> Tuple[str, float, str]:
        """
        입력 분류 및 신뢰도 평가 / Classify input and assess confidence
        
        Returns:
            tuple: (category, confidence, reason)
        """
        try:
            # 통합형 모델을 사용한 분류 및 응답 생성
            # 실제로는 모델 추론을 통해 분류와 신뢰도를 얻음
            
            # 간단한 키워드 기반 분류 (실제로는 모델 추론으로 대체)
            user_lower = user_input.lower()
            
            # QnA 패턴 감지
            qna_keywords = ['what', 'how', 'why', 'when', 'where', 'who', 'explain', 'define', '?']
            if any(keyword in user_lower for keyword in qna_keywords):
                category = 'qna'
                # 복잡도 기반 신뢰도 계산
                if len(user_input.split()) > 10 or 'complex' in user_lower or 'advanced' in user_lower:
                    confidence = 0.6  # 복잡한 질문은 낮은 신뢰도
                    reason = "Complex question detected"
                else:
                    confidence = 0.8  # 간단한 질문은 높은 신뢰도
                    reason = "Simple factual question"
                    
            # 감정/일상 대화 패턴 감지 (통합모델에서 처리)
            elif any(word in user_lower for word in ['feel', 'emotion', 'sad', 'happy', 'stress', 'tired', 'good morning', 'how are you']):
                category = 'general'
                if any(word in user_lower for word in ['depressed', 'anxiety', 'therapy', 'counseling']):
                    confidence = 0.7  # 감정 상담도 통합모델에서 처리 (하이브리드 시스템)
                    reason = "Emotional conversation handled by unified model"
                else:
                    confidence = 0.9  # 일반 일상대화는 통합모델로 충분
                    reason = "General daily conversation"
                    
            # 기술/프로그램 관련 패턴 감지
            elif any(word in user_lower for word in ['sketchup', 'cad', 'programming', 'code', 'software', 'technical', 'algorithm']):
                category = 'technical'
                if any(word in user_lower for word in ['advanced', 'professional', 'expert', 'complex']):
                    confidence = 0.4  # 고급 기술 질문은 전문모델 필요
                    reason = "Advanced technical consultation required"
                else:
                    confidence = 0.7  # 기본 기술 질문은 통합모델로 처리 가능
                    reason = "Basic technical question"
                    
            else:
                category = 'general'
                confidence = 0.9  # 일반 대화는 통합모델로 충분
                reason = "General conversation"
                
            return category, confidence, reason
            
        except Exception as e:
            self.logger.error(f"❌ 분류 오류 / Classification error: {e}")
            return 'general', 0.5, "Error in classification"
    
    def generate_unified_response(self, user_input: str, category: str) -> str:
        """통합형 모델로 응답 생성 / Generate response using unified model"""
        try:
            self.logger.info(f"🤖 통합형 모델 응답 생성 중... / Generating unified response...")
            
            # 실제 LLM 호출
            if GLOBAL_LLM and LLM_AVAILABLE:
                if self.unified_model == "unified_finetuned":
                    # 파인튜닝된 모델 사용
                    prompt = f"""You are a versatile AI assistant. Based on the category '{category}', respond appropriately to the user's question.

User: {user_input}
Assistant:"""
                else:
                    # 기본 LLM 사용 (프롬프트 템플릿 활용)
                    prompt = make_role_prompt(user_input)
                
                # 실제 LLM 호출
                response = chat_nous_hermes2_mistral(GLOBAL_LLM, prompt)
                
            else:
                # Fallback: LLM이 없을 때
                response = f"[Base LLM] {user_input}에 대한 기본 응답입니다."
            
            self.stats['unified_responses'] += 1
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ 통합형 응답 생성 실패 / Failed to generate unified response: {e}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"
    
    def load_specialist_model(self, category: str) -> bool:
        """전문모델 동적 로딩 / Dynamic loading of specialist model"""
        if self.model_loading[category]:
            self.logger.info(f"⏳ {category} 모델 로딩 중... / Loading {category} model...")
            return False
            
        try:
            self.model_loading[category] = True
            self.logger.info(f"🔄 {category} 전문모델 로딩 시작 / Starting to load {category} specialist model...")
            
            # 전문모델 경로 확인 (모의 모델)
            specialist_path = self.models_dir / f"{category}_specialist" / "config.json"
            
            if specialist_path.exists():
                # 모의 모델 로딩
                with open(specialist_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                
                # 모델 로딩 시뮬레이션
                time.sleep(1)  # 로딩 시간 시뮬레이션
                
                self.specialist_models[category] = model_config
                self.logger.info(f"✅ {category} 전문모델 로딩 완료 / {category} specialist model loaded")
                return True
            else:
                self.logger.warning(f"⚠️ {category} 전문모델 파일 없음 / {category} specialist model file not found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ {category} 모델 로딩 실패 / Failed to load {category} model: {e}")
            return False
        finally:
            self.model_loading[category] = False
    
    def generate_specialist_response(self, user_input: str, category: str) -> str:
        """전문모델로 고품질 응답 생성 / Generate high-quality response using specialist model"""
        try:
            self.logger.info(f"🎯 {category} 전문모델 응답 생성 중... / Generating specialist response...")
            
            if not self.specialist_models[category]:
                if not self.load_specialist_model(category):
                    return "전문모델을 로딩할 수 없습니다. 통합모델 응답을 사용합니다."
            
            # 모의 전문모델 응답과 실제 LLM 결합
            model_config = self.specialist_models[category]
            responses = model_config.get('responses', {})
            
            # 키워드 매칭으로 전문 응답 템플릿 선택
            user_lower = user_input.lower()
            specialist_context = responses.get('default', f"전문적인 {category} 응답")
            
            for keyword, template_response in responses.items():
                if keyword != 'default' and keyword in user_lower:
                    specialist_context = template_response
                    break
            
            # 실제 LLM을 사용해 전문가 응답 생성
            if GLOBAL_LLM and LLM_AVAILABLE:
                # 전문가 모드 프롬프트 구성
                if category == 'qna':
                    expert_prompt = f"""You are a knowledgeable QnA specialist. Provide accurate, detailed, and well-structured answers to questions.

User question: {user_input}

Provide a comprehensive, informative response:"""
                elif category == 'technical':
                    expert_prompt = f"""You are a technical specialist with expertise in software, programming, CAD, and technical consultation. Provide professional, detailed technical guidance.

Technical question: {user_input}

Provide expert technical guidance:"""
                else:
                    expert_prompt = f"""You are a specialist assistant for {category}. Provide expert-level responses.

User input: {user_input}

Response:"""
                
                response = chat_nous_hermes2_mistral(GLOBAL_LLM, expert_prompt)
                
            else:
                # Fallback: 모의 응답 사용
                response = specialist_context
            
            self.stats['specialist_responses'] += 1
            self.stats['model_switches'] += 1
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"❌ 전문모델 응답 생성 실패 / Failed to generate specialist response: {e}")
            return self.generate_unified_response(user_input, category)
    
    def should_use_specialist(self, confidence: float, user_input: str, category: str) -> bool:
        """전문모델 사용 여부 결정 / Decide whether to use specialist model"""
        
        # 명시적 전문가 모드 요청
        expert_keywords = ['expert', 'professional', 'specialist', '전문가', '전문적', '고급', 'advanced']
        if any(keyword in user_input.lower() for keyword in expert_keywords):
            self.logger.info("🎯 사용자가 전문가 모드 요청 / User requested expert mode")
            return True
            
        # 신뢰도 기반 결정
        confidence_threshold = 0.7
        if confidence < confidence_threshold:
            self.logger.info(f"📊 낮은 신뢰도로 전문모델 사용 / Using specialist due to low confidence: {confidence}")
            return True
            
        # 카테고리별 복잡도 임계값 (daily_chat 제거)
        complexity_indicators = {
            'qna': ['complex', 'detailed', 'comprehensive', 'research'],
            'technical': ['advanced', 'professional', 'enterprise', 'optimization']
        }
        
        if category in complexity_indicators:
            if any(indicator in user_input.lower() for indicator in complexity_indicators[category]):
                self.logger.info(f"🔬 복잡도 임계값 초과로 전문모델 사용 / Using specialist due to complexity")
                return True
        
        return False
    
    def generate_response(self, user_input: str, force_specialist: bool = False) -> Dict:
        """
        메인 응답 생성 함수 / Main response generation function
        
        Args:
            user_input: 사용자 입력
            force_specialist: 강제로 전문모델 사용
            
        Returns:
            dict: 응답 정보
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"📝 입력 처리 시작 / Starting input processing: {user_input[:50]}...")
            
            # 1. 분류 및 신뢰도 평가
            category, confidence, reason = self.classify_and_assess_confidence(user_input)
            
            # 2. 전문모델 사용 여부 결정
            use_specialist = force_specialist or self.should_use_specialist(confidence, user_input, category)
            
            # 3. 응답 생성
            if use_specialist and category in self.specialist_models:
                response = self.generate_specialist_response(user_input, category)
                model_used = f"{category}_specialist"
                quality_level = "high"
            else:
                response = self.generate_unified_response(user_input, category)
                model_used = "unified"
                quality_level = "standard"
            
            # 4. 응답 시간 계산
            response_time = time.time() - start_time
            self.stats['avg_response_time'] = (self.stats['avg_response_time'] + response_time) / 2
            
            # 5. 결과 반환
            result = {
                'response': response,
                'category': category,
                'confidence': confidence,
                'reason': reason,
                'model_used': model_used,
                'quality_level': quality_level,
                'response_time': response_time,
                'can_upgrade': not use_specialist and category in self.specialist_models
            }
            
            self.logger.info(f"✅ 응답 완료 / Response completed: {model_used} ({response_time:.2f}s)")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 응답 생성 중 오류 / Error during response generation: {e}")
            return {
                'response': "죄송합니다. 처리 중 오류가 발생했습니다.",
                'category': 'error',
                'confidence': 0.0,
                'reason': str(e),
                'model_used': 'error',
                'quality_level': 'error',
                'response_time': time.time() - start_time,
                'can_upgrade': False
            }
    
    def get_stats(self) -> Dict:
        """시스템 통계 반환 / Return system statistics"""
        return {
            **self.stats,
            'loaded_specialists': [k for k, v in self.specialist_models.items() if v is not None],
            'unified_model_loaded': self.unified_model is not None
        }
    
    def unload_specialist(self, category: str):
        """전문모델 언로드 (메모리 절약) / Unload specialist model (save memory)"""
        if self.specialist_models[category]:
            self.specialist_models[category] = None
            gc.collect()
            self.logger.info(f"🗑️ {category} 전문모델 언로드 / {category} specialist model unloaded")
    
    def cleanup(self):
        """시스템 정리 / System cleanup"""
        self.logger.info("🧹 시스템 정리 중... / Cleaning up system...")
        for category in self.specialist_models:
            self.unload_specialist(category)
        self.unified_model = None
        gc.collect()
        self.logger.info("✅ 정리 완료 / Cleanup completed")

# 전역 시스템 인스턴스
_hybrid_system = None

def get_hybrid_system():
    """하이브리드 시스템 싱글톤 가져오기 / Get hybrid system singleton"""
    global _hybrid_system
    if _hybrid_system is None:
        _hybrid_system = HybridFinetunedSystem()
        _hybrid_system.initialize_unified_model()
    return _hybrid_system

def process_with_finetuned_models(user_input: str, force_specialist: bool = False) -> Dict:
    """
    하이브리드 파인튜닝 시스템으로 입력 처리
    Process input with hybrid fine-tuning system
    
    Args:
        user_input: 사용자 입력
        force_specialist: 강제로 전문모델 사용
        
    Returns:
        dict: 응답 정보
    """
    system = get_hybrid_system()
    return system.generate_response(user_input, force_specialist)

def get_system_stats() -> Dict:
    """시스템 통계 조회 / Get system statistics"""
    system = get_hybrid_system()
    return system.get_stats()

def initialize_hybrid_system(llm_instance=None):
    """하이브리드 시스템 초기화 / Initialize hybrid system"""
    if llm_instance:
        set_global_llm(llm_instance)
    
    system = get_hybrid_system()
    return system

def unload_specialists():
    """전문모델 언로드 / Unload specialist models"""
    system = get_hybrid_system()
    for category in ['qna', 'technical']:
        system.unload_specialist(category)

if __name__ == "__main__":
    # 테스트 코드
    print("🚀 하이브리드 파인튜닝 시스템 테스트 / Hybrid Fine-tuning System Test")
    print("=" * 80)
    
    test_inputs = [
        "안녕하세요! 오늘 기분 어때요?",  # 일상대화
        "파이썬에서 머신러닝 모델을 어떻게 구현하나요?",  # 기술질문  
        "우주의 나이는 몇 살인가요?",  # QnA
        "SketchUp에서 고급 3D 모델링 기법을 전문적으로 알려주세요",  # 전문가 요청
    ]
    
    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🧪 테스트 {i}: {test_input}")
        result = process_with_finetuned_models(test_input)
        print(f"📂 카테고리: {result['category']}")
        print(f"📊 신뢰도: {result['confidence']:.2f}")
        print(f"🤖 모델: {result['model_used']}")
        print(f"💎 품질: {result['quality_level']}")
        print(f"⏱️ 응답시간: {result['response_time']:.2f}초")
        print(f"💬 응답: {result['response'][:100]}...")
        
    print(f"\n📈 시스템 통계:")
    stats = get_system_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
