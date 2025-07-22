#!/usr/bin/env python3
"""
RTX 3070 최종 LoRA 어댑터를 GGUF 양자화 파일로 변환
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def convert_final_lora_to_merged():
    """RTX 3070 최종 LoRA 어댑터를 기본 모델과 병합"""
    
    print("🔄 RTX 3070 최종 모델을 병합 모델로 변환 시작...")
    
    # 경로 설정 - 최종 완료 모델 사용
    base_model_path = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    adapter_path = "./models/rtx3070_optimized_final"
    output_path = "./models/rtx3070_final_merged"
    
    # 어댑터 존재 확인
    if not os.path.exists(adapter_path):
        print(f"❌ 최종 모델을 찾을 수 없습니다: {adapter_path}")
        return False
    
    print(f"📁 사용할 최종 모델: {adapter_path}")
    
    try:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 GPU 메모리 정리 완료")
        
        # 1. 기본 모델 로드 (CPU 사용으로 메모리 절약)
        print("📚 기본 모델 로딩 중... (CPU 사용으로 메모리 절약)")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float32,  # CPU에서는 float32 사용
            device_map="cpu",  # CPU로 강제 로드하여 GPU 메모리 절약
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 2. 토크나이저 로드
        print("🔤 토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 3. LoRA 어댑터 적용 (CPU에서)
        print("🔧 RTX 3070 최종 LoRA 어댑터 적용 중...")
        model = PeftModel.from_pretrained(
            base_model, 
            adapter_path,
            torch_dtype=torch.float32
        )
        
        # 4. LoRA를 기본 모델과 병합
        print("🔀 LoRA 어댑터를 기본 모델과 병합 중...")
        merged_model = model.merge_and_unload()
        
        # 5. 병합된 모델 저장
        print("💾 병합된 최종 모델 저장 중...")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        merged_model.save_pretrained(output_path, safe_serialization=True)
        tokenizer.save_pretrained(output_path)
        
        print(f"✅ 병합된 최종 모델이 {output_path}에 저장되었습니다.")
        
        # 파일 크기 확인
        model_files = os.listdir(output_path)
        print(f"📂 저장된 파일들: {model_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 변수들 삭제
        try:
            del base_model, model, merged_model
        except:
            pass

if __name__ == "__main__":
    # 시스템 정보 출력
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {total_memory // 1024**3}GB")
    else:
        print("🖥️ CPU 모드로 실행")
    
    # 변환 실행
    success = convert_final_lora_to_merged()
    
    if success:
        print("\n🎉 RTX 3070 최종 모델 병합 완료!")
        print("\n다음 단계:")
        print("1. llama.cpp를 사용해서 GGUF로 변환:")
        print("   python llama.cpp/convert.py ./models/rtx3070_final_merged --outdir ./models/ --outfile rtx3070_final.gguf")
        print("2. Q5_K_M 양자화:")
        print("   ./llama.cpp/quantize ./models/rtx3070_final.gguf ./models/rtx3070_final.Q5_K_M.gguf Q5_K_M")
    else:
        print("\n❌ 병합에 실패했습니다.")
