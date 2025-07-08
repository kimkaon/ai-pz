#!/usr/bin/env python3
"""
main.py에 파인튜닝 모델을 통합하는 헬퍼 스크립트
"""

import os
import json
import sys
from pathlib import Path

def check_finetuning_status():
    """파인튜닝 상태 확인"""
    config_path = "finetuning_result.json"
    
    if not os.path.exists(config_path):
        print("❌ 파인튜닝이 완료되지 않았습니다.")
        print("💡 먼저 'python finetuning/integrated_pipeline.py'를 실행하세요.")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print("✅ 파인튜닝 결과 발견")
    print(f"📅 생성 시간: {config['created_at']}")
    print(f"🔧 모델 타입: {config['model_type']}")
    print(f"⚡ GPU 사용: {config['gpu_available']}")
    
    return config

def create_main_integration():
    """main.py 통합 코드 생성"""
    integration_code = '''
# =====================================================
# 파인튜닝 모델 통합 코드 (main.py에 추가)
# =====================================================

# 1. 필요한 라이브러리 임포트 (파일 상단에 추가)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    FINETUNED_AVAILABLE = True
except ImportError:
    FINETUNED_AVAILABLE = False
    print("⚠️ transformers 라이브러리가 없어 파인튜닝 모델을 사용할 수 없습니다.")

# 2. 파인튜닝 모델 로더 클래스
class FinetuningModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self, model_path="./finetuning/models/korean_chat"):
        """파인튜닝된 모델 로드"""
        if not FINETUNED_AVAILABLE:
            return False
        
        try:
            if os.path.exists(model_path):
                print(f"🤖 파인튜닝 모델 로딩: {model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                self.loaded = True
                print("✅ 파인튜닝 모델 로드 완료")
                return True
            else:
                print(f"❌ 모델 경로를 찾을 수 없음: {model_path}")
                return False
        
        except Exception as e:
            print(f"❌ 파인튜닝 모델 로드 실패: {e}")
            return False
    
    def generate_response(self, prompt, max_tokens=100):
        """파인튜닝 모델로 응답 생성"""
        if not self.loaded:
            return None
        
        try:
            # 한국어 대화 형식으로 포맷
            formatted_prompt = f"사용자: {prompt}\\n어시스턴트:"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            if torch.cuda.is_available() and hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 어시스턴트 응답 부분만 추출
            if "어시스턴트:" in full_response:
                response = full_response.split("어시스턴트:")[-1].strip()
                return response
            else:
                return full_response.replace(formatted_prompt, "").strip()
        
        except Exception as e:
            print(f"❌ 파인튜닝 응답 생성 실패: {e}")
            return None

# 3. 전역 인스턴스 생성 (main.py의 적절한 위치에 추가)
finetuned_model = FinetuningModelLoader()

# 4. 기존 generate_response 함수 수정
def generate_response_with_finetuning(prompt, model_type="auto"):
    """
    파인튜닝 모델을 우선 사용하는 응답 생성 함수
    
    Args:
        prompt: 사용자 입력
        model_type: "finetuned", "original", "auto"
    """
    
    # 파인튜닝 모델 시도
    if model_type in ["auto", "finetuned"] and finetuned_model.loaded:
        finetuned_response = finetuned_model.generate_response(prompt)
        if finetuned_response:
            return finetuned_response
    
    # 기존 모델 폴백 (기존 generate_response 함수 호출)
    if model_type in ["auto", "original"]:
        return generate_response(prompt)  # 기존 함수 호출
    
    return "죄송합니다. 응답을 생성할 수 없습니다."

# 5. 초기화 코드 (main() 함수 시작 부분에 추가)
def initialize_finetuning():
    """파인튜닝 모델 초기화"""
    config_path = "finetuning_result.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('finetuning_completed'):
                model_path = config.get('model_path', './finetuning/models/korean_chat')
                
                if finetuned_model.load_model(model_path):
                    print("🎉 파인튜닝 모델이 활성화되었습니다!")
                    return True
                else:
                    print("⚠️ 파인튜닝 모델 로드 실패, 기존 모델 사용")
        
        except Exception as e:
            print(f"⚠️ 파인튜닝 설정 로드 실패: {e}")
    
    return False

# =====================================================
# 사용 방법:
# 1. 위 코드를 main.py에 적절히 추가
# 2. main() 함수 시작 부분에 initialize_finetuning() 호출 추가
# 3. 기존 generate_response 호출을 generate_response_with_finetuning으로 변경
# =====================================================
'''
    
    with open("main_integration.py", 'w', encoding='utf-8') as f:
        f.write(integration_code)
    
    print("✅ main.py 통합 코드 생성: main_integration.py")
    print("📖 이 파일의 내용을 main.py에 복사하여 사용하세요.")

def create_simple_test():
    """간단한 테스트 스크립트 생성"""
    test_code = '''
#!/usr/bin/env python3
"""
파인튜닝 모델 간단 테스트
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# main_integration.py에서 클래스 임포트
try:
    from main_integration import FinetuningModelLoader
    
    def test_finetuned_model():
        """파인튜닝 모델 테스트"""
        print("🧪 파인튜닝 모델 테스트 시작...")
        
        loader = FinetuningModelLoader()
        
        if loader.load_model():
            test_prompts = [
                "안녕하세요",
                "Python을 배우고 싶어요",
                "날씨가 어때요?",
                "AI에 대해 설명해주세요"
            ]
            
            for prompt in test_prompts:
                print(f"\\n사용자: {prompt}")
                response = loader.generate_response(prompt)
                print(f"어시스턴트: {response}")
        
        else:
            print("❌ 모델 로드 실패")

    if __name__ == "__main__":
        test_finetuned_model()

except ImportError:
    print("❌ main_integration.py를 찾을 수 없습니다.")
    print("💡 먼저 'python finetuning/integration_helper.py'를 실행하세요.")
'''
    
    with open("test_finetuned.py", 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print("✅ 테스트 스크립트 생성: test_finetuned.py")

def main():
    """메인 함수"""
    print("🔧 파인튜닝 통합 헬퍼")
    print("=" * 50)
    
    # 파인튜닝 상태 확인
    config = check_finetuning_status()
    if not config:
        return
    
    # 통합 코드 생성
    print("\n📝 통합 코드 생성 중...")
    create_main_integration()
    
    # 테스트 스크립트 생성
    print("\n🧪 테스트 스크립트 생성 중...")
    create_simple_test()
    
    print("\n" + "=" * 50)
    print("✅ 통합 준비 완료!")
    print("\n📋 다음 단계:")
    print("1. main_integration.py의 내용을 main.py에 통합")
    print("2. test_finetuned.py로 테스트 실행")
    print("3. main.py에서 파인튜닝 모델 사용")
    print("\n💡 파인튜닝 모델은 한국어 대화에 최적화되어 있습니다!")

if __name__ == "__main__":
    main()
