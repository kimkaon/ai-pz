#!/usr/bin/env python3
"""
통합형 모델 추론 스크립트
하나의 모델로 모든 대화 유형을 처리
"""

import os
import json
import argparse
import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional

# Transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedModelInference:
    """통합형 모델 추론 클래스"""
    
    def __init__(self, model_path: str = "models/unified_model"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"추론 디바이스: {self.device}")
        
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        
        # 시스템 프롬프트
        self.system_prompt = """당신은 다양한 상황에 적응할 수 있는 멀티태스크 AI 어시스턴트입니다.

역할 가이드:
- QnA/질문답변: 정확하고 구체적인 사실 기반 답변
- 일상대화: 친근하고 감정적인 공감형 대화
- 기술상담: 전문적이고 단계별 설명
- 일반대화: 상황에 맞는 적절한 톤과 스타일

사용자의 입력 유형을 파악하고 그에 맞는 최적의 응답 스타일을 선택하세요."""
    
    def load_model(self) -> bool:
        """통합 모델 로드"""
        if not self.model_path.exists():
            logger.error(f"모델을 찾을 수 없습니다: {self.model_path}")
            return False
        
        try:
            logger.info(f"통합 모델 로딩: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("통합 모델 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            return False
    
    def detect_conversation_type(self, text: str) -> str:
        """입력 텍스트에서 대화 유형 추정 (휴리스틱)"""
        text_lower = text.lower()
        
        # 질문 패턴
        question_patterns = ['?', '무엇', '어떻게', '왜', '언제', '어디서', '누가', '뭐', '어떤']
        if any(pattern in text_lower for pattern in question_patterns):
            return "질문답변"
        
        # 기술 관련 키워드
        tech_keywords = ['프로그래밍', '코딩', '파이썬', 'python', 'cad', '스케치업', '소프트웨어', '버그', '오류']
        if any(keyword in text_lower for keyword in tech_keywords):
            return "기술상담"
        
        # 일상 대화 패턴
        daily_patterns = ['안녕', '반가워', '기분', '날씨', '취미', '좋아해', '싫어해']
        if any(pattern in text_lower for pattern in daily_patterns):
            return "일상대화"
        
        return "일반대화"
    
    def generate_response(self, user_input: str, 
                         conversation_type: Optional[str] = None,
                         max_length: int = 256, 
                         temperature: float = 0.8) -> Dict[str, str]:
        """통합 모델로 응답 생성"""
        if not self.model or not self.tokenizer:
            if not self.load_model():
                return {"error": "모델을 로드할 수 없습니다."}
        
        try:
            # 대화 유형 자동 감지 (지정되지 않은 경우)
            if conversation_type is None:
                conversation_type = self.detect_conversation_type(user_input)
            
            # 입력 포맷팅
            formatted_input = f"시스템: {self.system_prompt}\n\n사용자: {user_input}\n어시스턴트:"
            
            # 토크나이징
            input_ids = self.tokenizer.encode(formatted_input, return_tensors="pt").to(self.device)
            
            # 생성
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩 (입력 부분 제외)
            generated_text = self.tokenizer.decode(
                generated_ids[0][len(input_ids[0]):], 
                skip_special_tokens=True
            ).strip()
            
            return {
                "response": generated_text,
                "detected_type": conversation_type,
                "model_type": "unified"
            }
            
        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            return {"error": f"응답 생성 실패: {e}"}
    
    def interactive_chat(self):
        """대화형 채팅"""
        if not self.load_model():
            print("❌ 모델 로딩에 실패했습니다.")
            return
        
        print("\n" + "="*50)
        print("🤖 통합형 AI 모델 테스트")
        print("="*50)
        print("💡 다양한 유형의 질문을 해보세요!")
        print("   - 질문: '파이썬이 뭐야?'")
        print("   - 일상: '오늘 기분 어때?'") 
        print("   - 기술: '코딩 에러를 어떻게 찾아?'")
        print("종료: 'quit' 입력")
        print()
        
        conversation_count = 0
        
        while True:
            try:
                conversation_count += 1
                user_input = input(f"💬 #{conversation_count} 사용자: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                
                if not user_input:
                    continue
                
                # 응답 생성
                print("🤔 생각 중...")
                result = self.generate_response(user_input)
                
                if "error" in result:
                    print(f"❌ 오류: {result['error']}")
                    continue
                
                print(f"🔍 감지된 대화 유형: {result['detected_type']}")
                print(f"🤖 AI: {result['response']}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"채팅 오류: {e}")
                continue
        
        print("\n👋 대화를 종료합니다.")
    
    def batch_test(self, test_inputs: List[str]):
        """배치 테스트"""
        if not self.load_model():
            print("❌ 모델 로딩에 실패했습니다.")
            return
        
        logger.info("배치 테스트 시작")
        results = []
        
        for i, text in enumerate(test_inputs, 1):
            print(f"\n[{i}/{len(test_inputs)}] 테스트: {text}")
            
            result = self.generate_response(text)
            result['input'] = text
            results.append(result)
            
            if "error" not in result:
                print(f"유형: {result['detected_type']}")
                print(f"응답: {result['response']}")
            else:
                print(f"오류: {result['error']}")
            
            print("-" * 50)
        
        # 결과 저장
        output_path = Path("models/unified_model/batch_test_results.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"배치 테스트 결과 저장: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="통합형 모델 추론 스크립트")
    parser.add_argument("--model_path", default="models/unified_model", help="모델 경로")
    parser.add_argument("--mode", choices=["interactive", "batch"], default="interactive", help="실행 모드")
    parser.add_argument("--test_inputs", nargs="+", help="배치 테스트 입력들")
    
    args = parser.parse_args()
    
    inference = UnifiedModelInference(args.model_path)
    
    if args.mode == "interactive":
        inference.interactive_chat()
    else:
        if not args.test_inputs:
            test_inputs = [
                "파이썬에서 리스트와 튜플의 차이점은 무엇인가요?",
                "오늘 기분이 어떤가요?",
                "CAD 소프트웨어에서 3D 모델링하는 방법을 알려주세요",
                "안녕하세요! 반갑습니다"
            ]
        else:
            test_inputs = args.test_inputs
        
        inference.batch_test(test_inputs)

if __name__ == "__main__":
    main()
