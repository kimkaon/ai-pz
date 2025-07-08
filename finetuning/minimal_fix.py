#!/usr/bin/env python3
"""
최소 설정 파인튜닝 - 그래디언트 문제 해결용
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings("ignore")

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class SimpleDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = data[:10]  # 최소 데이터만 사용
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 간단한 형식
        text = f"User: {item['input_text']}\nAssistant: {item['target_text']}"
        
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

def main():
    print("🚀 최소 설정 파인튜닝 시작")
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 디바이스: {device}")
    
    # 가장 작은 모델로 테스트
    model_name = "microsoft/DialoGPT-small"
    print(f"📥 모델 로딩: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 최소 설정
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # float32 사용
            device_map={"": 0}
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ 모델 로드 완료")
        
        # 최소 LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=2,  # 최소 rank
            lora_alpha=4,
            lora_dropout=0.1,
            target_modules=["c_attn"],  # DialoGPT의 어텐션 모듈
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 훈련 모드
        model.train()
        
        print("✅ LoRA 설정 완료")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 가짜 데이터 생성
    fake_data = [
        {"input_text": "What is AI?", "target_text": "AI is artificial intelligence."},
        {"input_text": "How are you?", "target_text": "I am fine, thank you."},
        {"input_text": "What is Python?", "target_text": "Python is a programming language."},
        {"input_text": "Hello", "target_text": "Hello! How can I help you?"},
        {"input_text": "Goodbye", "target_text": "Goodbye! Have a nice day!"}
    ]
    
    # 데이터셋 생성
    dataset = SimpleDataset(fake_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 최소 훈련 설정
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_epochs = 1
    
    print(f"🎯 훈련 시작 (에포크: {num_epochs}, 데이터: {len(dataset)})")
    
    # 훈련
    for epoch in range(num_epochs):
        print(f"\n📚 에포크 {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        for step, batch in enumerate(tqdm(dataloader, desc="훈련 중")):
            
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
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 역전파
            loss.backward()
            
            # 옵티마이저 스텝
            optimizer.step()
            optimizer.zero_grad()
            
            clear_memory()
        
        avg_loss = total_loss / len(dataloader)
        print(f"📊 평균 손실: {avg_loss:.4f}")
    
    # 테스트
    print("\n🧪 테스트")
    model.eval()
    
    test_input = "What is machine learning?"
    inputs = tokenizer(f"User: {test_input}\nAssistant:", return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"입력: {test_input}")
    print(f"응답: {response}")
    
    print("\n🎉 최소 설정 파인튜닝 완료!")

if __name__ == "__main__":
    main()
