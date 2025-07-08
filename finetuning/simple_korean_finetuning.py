#!/usr/bin/env python3
"""
실용적인 GPU 기반 파인튜닝 스크립트
Nous-Hermes-2-Mistral과 호환되는 모델로 파인튜닝
"""

import torch
import os
import json
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse

def setup_environment():
    """환경 설정 및 확인"""
    print("🚀 파인튜닝 환경 설정")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        torch.cuda.empty_cache()
    else:
        print("⚠️ GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다.")

def load_model_and_tokenizer(model_name="microsoft/DialoGPT-medium"):
    """
    모델과 토크나이저 로드
    더 가벼운 DialoGPT를 사용하여 테스트 (Mistral 대신)
    """
    print(f"📥 모델 로딩: {model_name}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    print(f"✅ 모델 로딩 완료 - 파라미터: {model.num_parameters():,}")
    return model, tokenizer

def create_korean_dataset():
    """한국어 대화 데이터셋 생성"""
    korean_conversations = [
        {
            "user": "안녕하세요!",
            "assistant": "안녕하세요! 반갑습니다. 무엇을 도와드릴까요?"
        },
        {
            "user": "오늘 날씨가 어때요?",
            "assistant": "죄송하지만 현재 날씨 정보에 접근할 수 없습니다. 날씨 앱이나 웹사이트를 확인해 주세요."
        },
        {
            "user": "Python에서 리스트를 정렬하는 방법을 알려주세요.",
            "assistant": "Python에서 리스트를 정렬하는 방법:\n1. sort() 메서드: my_list.sort() (원본 수정)\n2. sorted() 함수: new_list = sorted(my_list) (새 리스트 반환)"
        },
        {
            "user": "감사합니다!",
            "assistant": "천만에요! 언제든지 궁금한 것이 있으시면 물어보세요."
        },
        {
            "user": "컴퓨터가 느려요. 어떻게 해야 하나요?",
            "assistant": "컴퓨터 속도 개선 방법:\n1. 불필요한 프로그램 종료\n2. 디스크 정리\n3. 백그라운드 앱 확인\n4. 재시작\n5. 하드웨어 업그레이드 고려"
        },
        {
            "user": "좋은 하루 되세요!",
            "assistant": "감사합니다! 좋은 하루 되시고, 도움이 필요하시면 언제든지 말씀해 주세요."
        }
    ]
    
    # 대화 형식으로 변환
    formatted_data = []
    for conv in korean_conversations:
        text = f"사용자: {conv['user']}\n어시스턴트: {conv['assistant']}"
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)

def setup_lora(model):
    """LoRA (Low-Rank Adaptation) 설정"""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # 더 작은 rank로 메모리 절약
        lora_alpha=16,
        lora_dropout=0.1,
        # DialoGPT에 맞는 target modules
        target_modules=["c_attn", "c_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    print(f"🔧 LoRA 적용 완료")
    print(f"📊 훈련 가능한 파라미터: {model.num_parameters():,}")
    
    return model

def train_model(model, tokenizer, dataset, output_dir="./finetuned_model"):
    """모델 훈련"""
    print("🎯 모델 훈련 시작")
    
    # 토크나이징
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,  # 더 짧은 길이로 메모리 절약
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 데이터 콜렉터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # 트레이너
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 훈련 실행
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 훈련 완료! 모델 저장됨: {output_dir}")
    return output_dir

def test_inference(model, tokenizer):
    """추론 테스트"""
    print("🧪 추론 테스트")
    
    test_prompts = [
        "사용자: 안녕하세요!\n어시스턴트:",
        "사용자: Python을 배우고 싶어요.\n어시스턴트:",
    ]
    
    for prompt in test_prompts:
        print(f"\n입력: {prompt.split('어시스턴트:')[0].strip()}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = response.split("어시스턴트:")[-1].strip()
        print(f"응답: {assistant_response}")

def save_config(output_dir, model_name, training_params):
    """훈련 설정 저장"""
    config = {
        "base_model": model_name,
        "training_params": training_params,
        "timestamp": str(torch.cuda.current_device() if torch.cuda.is_available() else "cpu"),
        "gpu_used": torch.cuda.is_available()
    }
    
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📝 설정 저장됨: {config_path}")

def main():
    parser = argparse.ArgumentParser(description="GPU 기반 한국어 대화 모델 파인튜닝")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", help="베이스 모델")
    parser.add_argument("--output", default="./finetuned_korean_chat", help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=3, help="훈련 에포크")
    parser.add_argument("--test-only", action="store_true", help="추론만 테스트")
    
    args = parser.parse_args()
    
    # 환경 설정
    setup_environment()
    
    # 모델 로드
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    if args.test_only:
        # 추론만 테스트
        test_inference(model, tokenizer)
        return
    
    # 데이터셋 생성
    print("📚 데이터셋 준비")
    dataset = create_korean_dataset()
    print(f"데이터 수: {len(dataset)}")
    
    # LoRA 적용
    model = setup_lora(model)
    
    # 훈련
    output_dir = train_model(model, tokenizer, dataset, args.output)
    
    # 설정 저장
    save_config(output_dir, args.model, {"epochs": args.epochs})
    
    # 추론 테스트
    test_inference(model, tokenizer)
    
    print("\n🎉 파인튜닝 완료!")
    print(f"📁 모델 위치: {output_dir}")
    print("💡 다음 단계: GGUF 변환 후 main.py에서 사용")

if __name__ == "__main__":
    main()
