#!/usr/bin/env python3
"""
GPU 기반 파인튜닝 테스트 스크립트
Nous-Hermes-2-Mistral 모델을 기반으로 한 간단한 파인튜닝 테스트
"""

import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import json

def check_gpu_setup():
    """GPU 설정 확인"""
    print("=== GPU 설정 확인 ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # CUDA 메모리 정리
        torch.cuda.empty_cache()
        print(f"Available GPU memory: {torch.cuda.memory_reserved(0) / 1e9:.1f} GB")
    else:
        print("⚠️ CUDA not available! CPU will be used.")
    
    print("=" * 50)

def create_sample_dataset():
    """파인튜닝용 샘플 데이터셋 생성"""
    # 한국어 대화 데이터 샘플
    sample_data = [
        {
            "instruction": "안녕하세요",
            "input": "",
            "output": "안녕하세요! 저는 AI 어시스턴트입니다. 무엇을 도와드릴까요?"
        },
        {
            "instruction": "오늘 날씨가 어때요?",
            "input": "",
            "output": "죄송하지만 실시간 날씨 정보에 접근할 수 없습니다. 날씨 앱이나 웹사이트를 확인해 주세요."
        },
        {
            "instruction": "Python에서 리스트를 정렬하는 방법을 알려주세요",
            "input": "",
            "output": "Python에서 리스트를 정렬하는 방법은 여러 가지가 있습니다:\n1. sort() 메서드: my_list.sort() - 원본 리스트를 변경\n2. sorted() 함수: sorted(my_list) - 새로운 정렬된 리스트 반환"
        },
        {
            "instruction": "감사합니다",
            "input": "",
            "output": "천만에요! 언제든지 도움이 필요하시면 말씀해 주세요."
        }
    ]
    
    # 대화 형식으로 변환
    formatted_data = []
    for item in sample_data:
        if item["input"]:
            text = f"### 지시사항:\n{item['instruction']}\n\n### 입력:\n{item['input']}\n\n### 응답:\n{item['output']}"
        else:
            text = f"### 지시사항:\n{item['instruction']}\n\n### 응답:\n{item['output']}"
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def test_model_loading():
    """모델 로딩 테스트"""
    print("=== 모델 로딩 테스트 ===")
    
    # Mistral 기반 모델 (허깅페이스에서 사용 가능한 버전)
    model_name = "mistralai/Mistral-7B-v0.1"
    
    try:
        print(f"토크나이저 로딩: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 패딩 토큰 설정 (Mistral은 기본적으로 패딩 토큰이 없음)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"모델 로딩: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✓ 모델 로딩 성공")
        print(f"✓ 모델 파라미터 수: {model.num_parameters():,}")
        print(f"✓ 모델 디바이스: {next(model.parameters()).device}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        return None, None

def setup_lora_config():
    """LoRA 구성 설정"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Mistral 아키텍처에 맞는 모듈
    )
    return lora_config

def test_minimal_training(model, tokenizer, dataset):
    """최소한의 훈련 테스트"""
    print("=== 최소 훈련 테스트 ===")
    
    try:
        # LoRA 적용
        lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        print(f"✓ LoRA 적용 완료")
        print(f"✓ 훈련 가능한 파라미터: {model.num_parameters()}")
        
        # 토크나이징
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 데이터 콜렉터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # 훈련 인자 설정 (매우 가벼운 설정)
        training_args = TrainingArguments(
            output_dir="./test_training_output",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
            warmup_steps=10,
            logging_steps=1,
            save_strategy="no",
            evaluation_strategy="no",
            fp16=torch.cuda.is_available(),  # GPU가 있으면 fp16 사용
            dataloader_pin_memory=False,
        )
        
        # 트레이너 초기화
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        print("✓ 트레이너 설정 완료")
        print("🚀 테스트 훈련 시작...")
        
        # 훈련 실행
        trainer.train()
        
        print("✅ 테스트 훈련 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 훈련 테스트 실패: {e}")
        return False

def test_inference(model, tokenizer):
    """추론 테스트"""
    print("=== 추론 테스트 ===")
    
    try:
        test_prompt = "### 지시사항:\n안녕하세요\n\n### 응답:\n"
        
        inputs = tokenizer(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 생성된 응답:\n{response}")
        return True
        
    except Exception as e:
        print(f"❌ 추론 테스트 실패: {e}")
        return False

def main():
    """메인 함수"""
    print("🚀 GPU 기반 파인튜닝 환경 테스트 시작")
    print("=" * 60)
    
    # 1. GPU 설정 확인
    check_gpu_setup()
    
    # 2. 샘플 데이터셋 생성
    print("📝 샘플 데이터셋 생성...")
    dataset = create_sample_dataset()
    print(f"✓ 데이터셋 크기: {len(dataset)}")
    
    # 3. 모델 로딩 테스트
    model, tokenizer = test_model_loading()
    if model is None:
        print("❌ 모델 로딩 실패로 테스트 중단")
        return
    
    # 4. 최소 훈련 테스트
    training_success = test_minimal_training(model, tokenizer, dataset)
    
    # 5. 추론 테스트
    if training_success:
        inference_success = test_inference(model, tokenizer)
    
    print("\n" + "=" * 60)
    print("🎉 GPU 기반 파인튜닝 환경 테스트 완료!")
    
    if training_success:
        print("✅ 파인튜닝 환경이 정상적으로 설정되었습니다.")
        print("💡 이제 실제 데이터셋으로 파인튜닝을 진행할 수 있습니다.")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 환경을 다시 확인해주세요.")

if __name__ == "__main__":
    main()
