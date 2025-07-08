#!/usr/bin/env python3
"""
영어 대화 데이터 기반 간단한 파인튜닝 스크립트
- PyTorch 기본 DataLoader 사용
- TensorFlow/datasets 라이브러리 의존성 없음
- 영어 대화/QnA 데이터셋으로 통합 모델 생성
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EnglishChatDataset(Dataset):
    """영어 대화 데이터셋 클래스"""
    
    def __init__(self, data_files, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 여러 JSONL 파일에서 데이터 로드
        for file_path in data_files:
            if os.path.exists(file_path):
                print(f"📂 데이터 로딩: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                item = json.loads(line)
                                if 'input_text' in item and 'target_text' in item:
                                    self.data.append(item)
                            except json.JSONDecodeError:
                                continue
        
        print(f"✅ 총 {len(self.data)}개 데이터 로드됨")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Mistral/Nous-Hermes 형식으로 포맷팅
        text = f"<|im_start|>user\n{item['input_text']}<|im_end|>\n<|im_start|>assistant\n{item['target_text']}<|im_end|>"
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def load_english_datasets():
    """영어 데이터셋 파일 경로 반환 - processed_english의 모든 데이터 활용"""
    # 스크립트 위치에 관계없이 절대 경로 사용
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "datasets", "processed_english")
    
    print(f"📁 데이터셋 기본 경로: {base_path}")
    
    # 모든 영어 훈련 데이터 파일 포함
    train_files = [
        os.path.join(base_path, "unified_train.jsonl"),
        os.path.join(base_path, "dialogue_qna_train.jsonl"),
        os.path.join(base_path, "dialogue_general_train.jsonl"),
        os.path.join(base_path, "dialogue_technical_train.jsonl"),
        os.path.join(base_path, "classification_train.jsonl")
    ]
    
    # 모든 영어 검증 데이터 파일 포함
    val_files = [
        os.path.join(base_path, "unified_validation.jsonl"),
        os.path.join(base_path, "dialogue_qna_validation.jsonl"),
        os.path.join(base_path, "dialogue_general_validation.jsonl"),
        os.path.join(base_path, "dialogue_technical_validation.jsonl"),
        os.path.join(base_path, "classification_validation.jsonl")
    ]
    
    return train_files, val_files

def main():
    print("🚀 영어 대화 파인튜닝 시작")
    print("="*50)
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 기본 모델 로드 - Nous-Hermes-2-Mistral-7B-DPO 사용
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"  # 원본 HuggingFace 모델
    print(f"📥 기본 모델 로딩: {model_name}")
    print("⚠️  주의: 이 모델은 크기가 큽니다. 다운로드에 시간이 걸릴 수 있습니다.")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ 모델 로드 완료")
        print(f"📊 모델 파라미터 수: {model.num_parameters():,}")
        
        # LoRA 설정 적용 (메모리 효율성을 위해)
        print("🔧 LoRA 설정 적용 중...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # rank
            lora_alpha=32,  # scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # Mistral 타겟 모듈
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        print("💡 인터넷 연결을 확인하고 HuggingFace Hub 접근이 가능한지 확인하세요.")
        return
    
    # 데이터셋 로드
    train_files, val_files = load_english_datasets()
    
    train_dataset = EnglishChatDataset(train_files, tokenizer)
    val_dataset = EnglishChatDataset(val_files, tokenizer)
    
    if len(train_dataset) == 0:
        print("❌ 훈련 데이터가 없습니다.")
        return
    
    # 데이터로더 생성 - 7B 모델에 맞게 배치 크기 조정
    batch_size = 1 if torch.cuda.is_available() else 1  # 메모리 절약을 위해 작은 배치
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 최적화 설정 - 7B 모델에 맞게 조정
    num_epochs = 2  # 에포크 수 줄임
    learning_rate = 2e-4  # LoRA에 맞는 더 높은 학습률
    gradient_accumulation_steps = 4  # 그래디언트 누적으로 효과적인 배치 크기 증가
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),  # 전체 스텝의 10%를 워밍업으로
        num_training_steps=total_steps
    )
    
    print(f"🎯 훈련 설정:")
    print(f"   배치 크기: {batch_size}")
    print(f"   그래디언트 누적 스텝: {gradient_accumulation_steps}")
    print(f"   효과적 배치 크기: {batch_size * gradient_accumulation_steps}")
    print(f"   에포크: {num_epochs}")
    print(f"   학습률: {learning_rate}")
    print(f"   총 스텝: {total_steps}")
    
    # 파인튜닝 실행
    model.train()
    
    for epoch in range(num_epochs):
        print(f"\n📚 에포크 {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"훈련 중")
        
        for step, batch in enumerate(progress_bar):
            
            # 배치를 디바이스로 이동
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 순전파
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps  # 누적을 위해 나누기
            total_loss += loss.item()
            
            # 역전파
            loss.backward()
            
            # 그래디언트 누적
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        avg_loss = total_loss / len(train_loader)
        print(f"📊 평균 손실: {avg_loss:.4f}")
        
        # 검증
        if len(val_dataset) > 0:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f"🔍 검증 손실: {avg_val_loss:.4f}")
            
            model.train()
    
    # 모델 저장 - LoRA 어댑터와 토크나이저 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "models", "english_unified")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 LoRA 모델 저장: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 설정 파일 저장
    config_info = {
        "base_model": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "training_data": "processed_english datasets",
        "model_type": "LoRA fine-tuned",
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }
    
    with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    
    # 테스트 생성
    print(f"\n🧪 모델 테스트")
    model.eval()
    
    test_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning briefly.",
        "What are the benefits of renewable energy?",
        "How do neural networks learn?"
    ]
    
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\n입력: {prompt}")
            
            # Mistral/Nous-Hermes 프롬프트 형식으로 토큰화
            input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(input_text, return_tensors='pt').to(device)
            
            # 응답 생성
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # 응답 디코딩
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            
            print(f"응답: {assistant_response}")
    
    print(f"\n🎉 영어 대화 파인튜닝 완료!")
    print(f"📁 모델 저장 위치: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()
