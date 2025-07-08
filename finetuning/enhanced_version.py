#!/usr/bin/env python3
"""
성공한 설정 기반 추가 개선 버전
- 더 긴 시퀀스 길이
- 더 다양한 LoRA 모듈
- 개선된 데이터 전처리
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
    get_cosine_schedule_with_warmup,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import warnings
import gc

warnings.filterwarnings("ignore")

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class EnhancedDataset(Dataset):
    """개선된 데이터셋 클래스"""
    
    def __init__(self, data_files, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
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
                                    # 품질 필터링
                                    if len(item['input_text'].strip()) > 5 and len(item['target_text'].strip()) > 10:
                                        self.data.append(item)
                            except json.JSONDecodeError:
                                continue
        
        print(f"✅ 총 {len(self.data)}개 고품질 데이터 로드됨")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = item['input_text'].strip()
        target_text = item['target_text'].strip()
        
        # 더 자연스러운 응답을 위한 전처리
        sentences = target_text.split('. ')
        if len(sentences) > 3:
            # 3-4문장으로 제한하되, 의미가 완성되도록
            target_text = '. '.join(sentences[:3])
            if not target_text.endswith('.'):
                target_text += '.'
        
        # 고품질 대화 형식
        text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 개선된 라벨 마스킹
        labels = encoding['input_ids'].clone()
        
        # assistant 부분만 학습하도록 마스킹
        assistant_start = text.find("<|im_start|>assistant\n")
        if assistant_start != -1:
            user_part = text[:assistant_start + len("<|im_start|>assistant\n")]
            user_tokens = self.tokenizer.encode(user_part, add_special_tokens=False)
            
            if len(user_tokens) < labels.shape[-1]:
                labels[0][:len(user_tokens)] = -100
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def load_datasets():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "datasets", "processed_english")
    
    train_files = [
        os.path.join(base_path, "unified_train.jsonl"),
        os.path.join(base_path, "dialogue_qna_train.jsonl"),
        os.path.join(base_path, "dialogue_general_train.jsonl"),
        os.path.join(base_path, "dialogue_technical_train.jsonl"),
        os.path.join(base_path, "classification_train.jsonl")
    ]
    
    val_files = [
        os.path.join(base_path, "unified_validation.jsonl"),
        os.path.join(base_path, "dialogue_qna_validation.jsonl"),
        os.path.join(base_path, "dialogue_general_validation.jsonl"),
        os.path.join(base_path, "dialogue_technical_validation.jsonl"),
        os.path.join(base_path, "classification_validation.jsonl")
    ]
    
    return train_files, val_files

def main():
    print("🚀 성공 기반 개선된 파인튜닝 버전 2.0")
    print("="*60)
    
    clear_memory()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    print(f"📥 기본 모델 로딩: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # 4bit 양자화 설정 (메모리 최적화)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ 모델 로드 완료")
        
        # 개선된 LoRA 설정 (성공한 설정 기반)
        print("🔧 개선된 LoRA 설정 적용 중...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # 성공한 설정 유지
            lora_alpha=32,  # 성공한 설정 유지
            lora_dropout=0.05,  # 약간 감소
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 모든 주요 모듈
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.train()
        
        print("✅ LoRA 설정 완료")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 데이터셋 로드
    train_files, val_files = load_datasets()
    
    train_dataset = EnhancedDataset(train_files, tokenizer)
    val_dataset = EnhancedDataset(val_files, tokenizer)
    
    if len(train_dataset) == 0:
        print("❌ 훈련 데이터가 없습니다.")
        return
    
    # 데이터로더 생성
    batch_size = 1
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 개선된 훈련 설정
    num_epochs = 6  # 조금 더 많이
    learning_rate = 8e-5  # 조금 낮춤 (안정성)
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    
    print(f"🎯 전체 {len(train_dataset.data)}개 고품질 데이터 사용")
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8
    )
    
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = total_steps // 20
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"🎯 개선된 훈련 설정:")
    print(f"   에포크: {num_epochs}")
    print(f"   학습률: {learning_rate}")
    print(f"   배치 크기: {batch_size}")
    print(f"   그래디언트 누적: {gradient_accumulation_steps}")
    print(f"   총 스텝: {total_steps}")
    print(f"   훈련 데이터: {len(train_dataset)} 샘플")
    print(f"   검증 데이터: {len(val_dataset)} 샘플")
    
    # 훈련 실행
    model.train()
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📚 에포크 {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(train_loader, desc=f"훈련 중 (에포크 {epoch+1})")
        
        for step, batch in enumerate(progress_bar):
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            if loss is None or not torch.isfinite(loss):
                continue
            
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()
            valid_batches += 1
            
            loss.backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                clear_memory()
            
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            if step % 20 == 0:
                clear_memory()
        
        avg_loss = total_loss / max(valid_batches, 1)
        print(f"📊 평균 훈련 손실: {avg_loss:.4f}")
        
        # 검증
        if len(val_dataset) > 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="검증 중"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    if outputs.loss is not None and torch.isfinite(outputs.loss):
                        val_loss += outputs.loss.item()
                        val_steps += 1
            
            avg_val_loss = val_loss / max(val_steps, 1)
            print(f"🔍 검증 손실: {avg_val_loss:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print("✅ 검증 손실 개선! 최고 모델 저장...")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                best_dir = os.path.join(script_dir, "models", "english_unified_v2_best")
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
            
            model.train()
    
    # 최종 모델 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "models", "english_unified_v2")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 최종 모델 저장: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 설정 파일 저장
    config_info = {
        "base_model": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "training_data": "processed_english datasets (quality filtered)",
        "model_type": "Enhanced LoRA fine-tuned v2.0",
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_length": 256,
        "quantization": "4bit",
        "target_modules": "all_major",
        "best_val_loss": best_val_loss,
        "improvements": [
            "Quality filtered data",
            "Enhanced label masking",
            "4bit quantization",
            "More LoRA modules",
            "Improved text preprocessing"
        ]
    }
    
    with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    
    print(f"\n🎉 개선된 파인튜닝 v2.0 완료!")
    print(f"📁 최종 모델: {output_dir}")
    print(f"📁 최고 성능 모델: {os.path.join(script_dir, 'models', 'english_unified_v2_best')}")
    print(f"🏆 최고 검증 손실: {best_val_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
