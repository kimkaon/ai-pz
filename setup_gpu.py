#!/usr/bin/env python3
"""
GPU 및 환경 설정 상태 확인 스크립트
"""

import sys
import os

def check_environment():
    """환경 상태 전체 점검"""
    print("=== AI PZ2 환경 상태 점검 ===\n")
    
    # 1. PyTorch 및 CUDA
    print("1. PyTorch & CUDA 상태:")
    try:
        import torch
        print(f"   ✅ PyTorch 버전: {torch.__version__}")
        
        if hasattr(torch, 'cuda'):
            if torch.cuda.is_available():
                print(f"   ✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
                print(f"   💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("   ❌ CUDA 사용 불가")
        else:
            print("   ❌ PyTorch가 CPU 전용 버전입니다")
            print("   🔧 해결: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    except ImportError:
        print("   ❌ PyTorch가 설치되지 않았습니다")
    
    # 2. Transformers
    print("\n2. Transformers 라이브러리:")
    try:
        import transformers
        print(f"   ✅ Transformers 버전: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers 미설치")
        print("   🔧 해결: pip install transformers")
    
    # 3. 파인튜닝 관련 패키지
    print("\n3. 파인튜닝 패키지:")
    
    packages = {
        'datasets': 'datasets',
        'peft': 'peft', 
        'bitsandbytes': 'bitsandbytes',
        'accelerate': 'accelerate'
    }
    
    missing_packages = []
    for name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n   🔧 해결: pip install {' '.join(missing_packages)}")
    
    # 4. GGUF 모델 지원 (llama-cpp-python)
    print("\n4. GGUF 모델 지원:")
    try:
        import llama_cpp
        print(f"   ✅ llama-cpp-python 설치됨")
    except ImportError:
        print("   ❌ llama-cpp-python 미설치")
        print("   🔧 해결: pip install llama-cpp-python")
    
    # 5. RTX 3070 최적화 모델 확인
    print("\n5. RTX 3070 최적화 모델:")
    rtx_models = ["models/rtx3070_optimized_best", "models/rtx3070_optimized_final"]
    
    for model_path in rtx_models:
        if os.path.exists(model_path):
            adapter_file = os.path.join(model_path, "adapter_model.safetensors")
            config_file = os.path.join(model_path, "adapter_config.json")
            
            if os.path.exists(adapter_file) and os.path.exists(config_file):
                size_mb = os.path.getsize(adapter_file) / 1024**2
                print(f"   ✅ {os.path.basename(model_path)}: LoRA 어댑터 ({size_mb:.1f}MB)")
            else:
                print(f"   ⚠️ {os.path.basename(model_path)}: 불완전한 모델")
        else:
            print(f"   ❌ {os.path.basename(model_path)}: 없음")
    
    # 6. 메인 GGUF 모델 확인
    print("\n6. 기본 GGUF 모델:")
    model_path = "models/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / 1024**3
        print(f"   ✅ {model_path} ({size_gb:.1f}GB)")
    else:
        print(f"   ❌ {model_path} 찾을 수 없음")
    
    # 6. 파인튜닝 데이터 확인
    print("\n6. 파인튜닝 데이터:")
    data_files = [
        "finetuning/datasets/processed_english/unified_train.jsonl",
        "finetuning/datasets/processed_english/unified_validation.jsonl"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = sum(1 for line in f if line.strip() and not line.startswith('#'))
            print(f"   ✅ {os.path.basename(file_path)}: {lines}개 데이터")
        else:
            print(f"   ❌ {file_path} 없음")
    
    print("\n=== RTX 3070 모델 테스트 ===")
    
    # PyTorch CUDA 여부에 따른 권장사항
    try:
        import torch
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            print("🚀 GPU 파인튜닝 권장:")
            print("   - 모델: NousResearch/Nous-Hermes-2-Mistral-7B-DPO")
            print("   - 방법: LoRA + 4bit 양자화")
            print("   - 예상 시간: 30분-1시간")
            print("   - GPU 메모리: 6-8GB 사용")
            
            print("\n🔧 RTX 3070 모델 로딩 테스트:")
            try:
                # 간단한 RTX 3070 모델 로딩 테스트
                from transformers import AutoTokenizer
                
                rtx_path = "./models/rtx3070_optimized_best"
                if os.path.exists(rtx_path):
                    print("   � 모델 폴더 확인됨")
                    
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(rtx_path, local_files_only=True)
                        print("   ✅ 토크나이저 로딩 성공")
                    except Exception as e:
                        print(f"   ⚠️ 토크나이저 로딩 실패: {e}")
                        print("   💡 기본 모델에서 토크나이저를 가져올 수 있습니다")
                    
                    # 어댑터 파일 확인
                    adapter_file = os.path.join(rtx_path, "adapter_model.safetensors")
                    if os.path.exists(adapter_file):
                        print("   ✅ LoRA 어댑터 파일 확인됨")
                    else:
                        print("   ❌ LoRA 어댑터 파일 없음")
                        
                else:
                    print("   ❌ RTX 3070 모델 폴더 없음")
                    
            except ImportError:
                print("   ❌ transformers 라이브러리 없음")
                
        else:
            print("�💻 CPU 파인튜닝:")
            print("   - 모델: microsoft/DialoGPT-small")
            print("   - 방법: 전체 파인튜닝")
            print("   - 예상 시간: 2-4시간")
            print("   - RAM: 8-16GB 사용")
    except:
        pass

def test_rtx3070_model():
    """RTX 3070 모델 로딩 테스트"""
    print("\n=== RTX 3070 모델 로딩 테스트 ===")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA가 사용 불가능합니다")
            return False
            
        print("🔄 RTX 3070 모델 로딩 시도 중...")
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        import gc
        
        gc.collect()
        
        # 모델 경로
        model_path = "./models/rtx3070_optimized_best"
        if not os.path.exists(model_path):
            print(f"❌ 모델 경로 없음: {model_path}")
            return False
        
        # 토크나이저 로드
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            print("✅ 토크나이저 로드 성공")
        except:
            tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-Mistral-7B-DPO")
            print("✅ 기본 토크나이저 로드 성공")
            
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 4bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        
        # 메모리 제한
        max_memory = {0: "7GB"}
        
        try:
            # 기본 모델 로드
            print("🔄 기본 모델 로딩 중...")
            base = AutoModelForCausalLM.from_pretrained(
                "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            print("✅ 기본 모델 로드 성공")
            
            # LoRA 어댑터 로드
            print("🔄 LoRA 어댑터 로딩 중...")
            model = PeftModel.from_pretrained(base, model_path)
            model.eval()
            print("✅ RTX 3070 모델 로드 완전 성공!")
            
            # 간단한 테스트
            test_prompt = "Hello, how are you?"
            inputs = tokenizer.encode(test_prompt, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"🧪 테스트 응답: {response}")
            
            return True
            
        except Exception as e:
            print(f"❌ Transformers 모델 로딩 실패: {e}")
            
            # GGUF 폴백 테스트
            print("🔄 GGUF 모델 폴백 테스트...")
            try:
                from llama_cpp import Llama
                gguf_path = "./models/Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"
                
                if os.path.exists(gguf_path):
                    model = Llama(
                        model_path=gguf_path,
                        n_gpu_layers=-1,
                        n_ctx=1024,
                        verbose=False
                    )
                    
                    response = model("Hello", max_tokens=10)
                    print("✅ GGUF 모델 폴백 성공!")
                    print(f"🧪 테스트 응답: {response['choices'][0]['text']}")
                    return True
                else:
                    print("❌ GGUF 모델 파일 없음")
                    return False
                    
            except Exception as gguf_e:
                print(f"❌ GGUF 모델도 실패: {gguf_e}")
                return False
        
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    check_environment()
    
    # RTX 3070 모델 테스트 옵션
    print("\n" + "="*50)
    test_choice = input("RTX 3070 모델 로딩 테스트를 실행하시겠습니까? (y/N): ").strip().lower()
    
    if test_choice in ['y', 'yes']:
        success = test_rtx3070_model()
        if success:
            print("\n🎉 RTX 3070 모델이 정상적으로 작동합니다!")
        else:
            print("\n⚠️ RTX 3070 모델 로딩에 문제가 있습니다.")
            print("   해결 방안:")
            print("   1. GPU 메모리 확인 (다른 프로그램 종료)")
            print("   2. CUDA 드라이버 업데이트")
            print("   3. 가상환경 패키지 재설치")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_rtx3070_model()
    else:
        main()
