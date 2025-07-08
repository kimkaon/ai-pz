#!/usr/bin/env python3
"""
GPU 환경 빠른 검증 스크립트
"""

import torch
import time

def test_gpu_environment():
    """GPU 환경 빠른 테스트"""
    print("🔍 GPU 환경 검증")
    print("=" * 40)
    
    # 기본 정보
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA가 사용 불가능합니다!")
        return False
    
    # GPU 정보
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    # 메모리 정보
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"Total GPU memory: {total_memory / 1e9:.1f} GB")
    
    # GPU 연산 테스트
    print("\n🧪 GPU 연산 테스트")
    device = torch.device("cuda:0")
    
    # 간단한 텐서 연산
    start_time = time.time()
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"✅ 매트릭스 곱셈 (1000x1000) 소요시간: {end_time - start_time:.4f}초")
    
    # 메모리 사용량 확인
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    print(f"GPU 메모리 - 할당: {allocated / 1e6:.1f} MB, 예약: {reserved / 1e6:.1f} MB")
    
    # 메모리 정리
    del x, y, z
    torch.cuda.empty_cache()
    print("✅ GPU 메모리 정리 완료")
    
    return True

def test_transformers_import():
    """Transformers 라이브러리 임포트 테스트"""
    print("\n📚 Transformers 라이브러리 테스트")
    
    try:
        from transformers import AutoTokenizer
        print("✅ AutoTokenizer 임포트 성공")
        
        from transformers import AutoModelForCausalLM
        print("✅ AutoModelForCausalLM 임포트 성공")
        
        from peft import LoraConfig
        print("✅ PEFT (LoRA) 임포트 성공")
        
        return True
    except Exception as e:
        print(f"❌ 임포트 실패: {e}")
        return False

def test_basic_model_loading():
    """간단한 모델 로딩 테스트"""
    print("\n🤖 기본 모델 로딩 테스트")
    
    try:
        # 가장 작은 GPT-2 모델로 테스트
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        model_name = "gpt2"
        print(f"모델 로딩 중: {model_name}")
        
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            print("✅ 모델을 GPU로 이동 완료")
        
        # 간단한 생성 테스트
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        inputs = tokenizer("Hello", return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 생성 테스트 성공: {result}")
        
        # 메모리 정리
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 GPU 파인튜닝 환경 빠른 검증")
    print("=" * 60)
    
    # 1. GPU 환경 테스트
    gpu_ok = test_gpu_environment()
    
    # 2. 라이브러리 임포트 테스트
    import_ok = test_transformers_import()
    
    # 3. 기본 모델 로딩 테스트
    model_ok = test_basic_model_loading()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print(f"GPU 환경: {'✅ 성공' if gpu_ok else '❌ 실패'}")
    print(f"라이브러리 임포트: {'✅ 성공' if import_ok else '❌ 실패'}")
    print(f"모델 로딩: {'✅ 성공' if model_ok else '❌ 실패'}")
    
    if all([gpu_ok, import_ok, model_ok]):
        print("\n🎉 모든 테스트 통과!")
        print("✅ GPU 기반 파인튜닝 환경이 정상적으로 설정되었습니다.")
        print("💡 이제 실제 파인튜닝을 진행할 수 있습니다.")
    else:
        print("\n⚠️ 일부 테스트가 실패했습니다.")
        print("🔧 환경 설정을 다시 확인해주세요.")

if __name__ == "__main__":
    main()
