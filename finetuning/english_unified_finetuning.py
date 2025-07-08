#!/usr/bin/env python3
"""
영어 대화 데이터 기반 파인튜닝 스크립트
기존 simple_korean_finetuning.py를 영어 데이터로 수정
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
    print("🚀 영어 대화 파인튜닝 환경 설정")
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
    영어 대화에 특화된 DialoGPT 사용
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
    
    print(f"✅ 모델 로딩 완료 - 파라미터: {model.num_parameters()}")
    return model, tokenizer

def load_english_data():
    """영어 대화 데이터 로드"""
    print("📚 영어 데이터셋 로딩...")
    
    # 영어 대화 데이터 준비
    data_files = [
        "./datasets/processed_english/unified_train.jsonl",
        "./datasets/processed_english/dialogue_qna_train.jsonl",
        "./datasets/processed_english/dialogue_general_train.jsonl"
    ]
    
    conversations = []
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"📂 로딩: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            item = json.loads(line)
                            if 'input_text' in item and 'target_text' in item:
                                # 영어 대화 형식으로 포맷
                                text = f"User: {item['input_text']}<|endoftext|>Assistant: {item['target_text']}<|endoftext|>"
                                conversations.append({"text": text})
                        except json.JSONDecodeError:
                            continue
    
    print(f"✅ 총 {len(conversations)}개 대화 로드됨")
    
    if len(conversations) == 0:
        print("❌ 데이터가 없습니다. 기본 영어 데이터를 생성합니다.")
        conversations = [
            {"text": "User: What is artificial intelligence?<|endoftext|>Assistant: Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.<|endoftext|>"},
            {"text": "User: How does machine learning work?<|endoftext|>Assistant: Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.<|endoftext|>"},
            {"text": "User: Explain deep learning.<|endoftext|>Assistant: Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to model and understand complex patterns in data.<|endoftext|>"},
            {"text": "User: What is the difference between AI and ML?<|endoftext|>Assistant: AI is the broader concept of creating intelligent machines, while ML is a specific approach within AI that focuses on learning from data.<|endoftext|>"},
            {"text": "User: How can I start learning programming?<|endoftext|>Assistant: Start with a beginner-friendly language like Python, practice regularly with small projects, and use online resources like tutorials and coding platforms.<|endoftext|>"}
        ]
    
    return Dataset.from_list(conversations)

def setup_lora_config():
    """LoRA 설정"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # 더 높은 rank로 성능 향상
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # DialoGPT에 맞는 타겟 모듈
    )

def tokenize_function(examples, tokenizer, max_length=512):
    """토큰화 함수"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    print("=" * 60)
    print("🚀 영어 대화 파인튜닝 시작")
    print("=" * 60)
    
    # 환경 설정
    setup_environment()
    
    # 모델과 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer()
    
    # LoRA 설정 적용
    print("🔧 LoRA 설정 적용...")
    peft_config = setup_lora_config()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 데이터 로드 및 토큰화
    dataset = load_english_data()
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="./models/english_unified",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2 if torch.cuda.is_available() else 1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,  # 메모리 문제 방지
        remove_unused_columns=False,
    )
    
    # 데이터 컬렉터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # 파인튜닝 실행
    print("🎯 파인튜닝 시작...")
    try:
        trainer.train()
        print("✅ 파인튜닝 완료!")
        
        # 모델 저장
        print("💾 모델 저장 중...")
        trainer.save_model()
        tokenizer.save_pretrained("./models/english_unified")
        
        # 테스트
        print("🧪 모델 테스트...")
        test_prompts = [
            "What is machine learning?",
            "How do neural networks work?",
            "Explain artificial intelligence."
        ]
        
        model.eval()
        for prompt in test_prompts:
            print(f"\n입력: {prompt}")
            input_text = f"User: {prompt}<|endoftext|>Assistant:"
            inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("Assistant:")[-1].strip()
            print(f"응답: {assistant_response}")
        
        print("\n🎉 영어 대화 파인튜닝 완료!")
        print("📁 모델 저장 위치: ./models/english_unified")
        
    except Exception as e:
        print(f"❌ 파인튜닝 실패: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
