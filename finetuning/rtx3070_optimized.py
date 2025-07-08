#!/usr/bin/env python3
"""
RTX 3070 8GB 최적화: 작은 모델 5시간 연속 학습
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, BitsAndBytesConfig
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
    def __init__(self, data_files, tokenizer, max_length=512):
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
                                    # 품질 필터링: 너무 짧거나 긴 데이터 제외
                                    input_len = len(item['input_text'].strip())
                                    target_len = len(item['target_text'].strip())
                                    
                                    if 5 <= input_len <= 300 and 10 <= target_len <= 500:
                                        self.data.append(item)
                            except json.JSONDecodeError:
                                continue
        
        print(f"✅ 총 {len(self.data)}개 고품질 데이터 로드됨")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 개선된 응답 전처리
        input_text = item['input_text'].strip()
        target_text = item['target_text'].strip()
        
        # 1. 너무 긴 응답 제한 (품질 향상)
        sentences = target_text.split('. ')
        if len(sentences) > 3:
            # 의미 있는 마무리를 위해 3문장으로 제한
            target_text = '. '.join(sentences[:3])
            if not target_text.endswith('.'):
                target_text += '.'
        
        # 2. 더 자연스러운 대화 형식 (Mistral 최적화)
        text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 3. 개선된 라벨 마스킹 (사용자 입력 부분만 마스킹)
        labels = encoding['input_ids'].clone()
        
        # assistant 부분만 학습하도록 정확한 마스킹
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

def main():
    print("🚀 RTX 3070 8GB 최적화: 5시간 연속 학습")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 디바이스: {device}")
    
    # 성공 검증된 Mistral 7B 모델 사용 + 4bit 양자화
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    print(f"📥 모델 로딩: {model_name}")
    
    try:
        # 4bit 양자화로 RTX 3070에 최적화
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
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
        
        # Mistral 7B 최적화 LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # 높은 rank (품질 우선)
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Mistral 전용 모듈
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "datasets", "processed_english")
    
    train_files = [
        os.path.join(base_path, "unified_train.jsonl"),
        os.path.join(base_path, "dialogue_qna_train.jsonl"),
        os.path.join(base_path, "dialogue_general_train.jsonl"),
        os.path.join(base_path, "dialogue_technical_train.jsonl"),
        os.path.join(base_path, "classification_train.jsonl")
    ]
    
    dataset = SimpleDataset(train_files, tokenizer, max_length=384)  # 최적화된 길이
    
    # RTX 3070 8GB 최적화: batch_size 1 + gradient accumulation
    batch_size = 1
    gradient_accumulation_steps = 4  # 실질적으로 batch_size 4와 동일
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 5시간 최적화 설정 (메모리 효율성 + 안정성)
    num_epochs = 6  # 더 효율적인 에포크 수
    learning_rate = 8e-5  # 안정적인 학습률
    max_grad_norm = 1.0  # 그래디언트 클리핑
    
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=learning_rate, 
        weight_decay=0.01,
        eps=1e-8
    )
    
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    warmup_steps = total_steps // 20
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"🎯 RTX 3070 최적화 설정:")
    print(f"   에포크: {num_epochs}")
    print(f"   실질 배치 크기: {batch_size * gradient_accumulation_steps}")
    print(f"   학습률: {learning_rate}")
    print(f"   총 스텝: {total_steps}")
    print(f"   데이터: {len(dataset)} 샘플")
    print(f"   최대 시퀀스 길이: 384")
    
    # 검증용 데이터 로드 (품질 모니터링)
    val_files = [
        os.path.join(base_path, "unified_validation.jsonl"),
        os.path.join(base_path, "dialogue_qna_validation.jsonl")
    ]
    val_dataset = SimpleDataset(val_files, tokenizer, max_length=384)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False) if len(val_dataset) > 0 else None
    
    best_val_loss = float('inf')
    
    # 고품질 훈련 시작
    model.train()
    for epoch in range(num_epochs):
        print(f"\n📚 에포크 {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(dataloader, desc=f"훈련 중 (에포크 {epoch+1})")
        
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
            
            # Gradient accumulation으로 메모리 최적화
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()
            valid_batches += 1
            
            loss.backward()
            
            # Gradient accumulation 단계마다 업데이트
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                clear_memory()
            
            # 진행률 표시
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            if step % 20 == 0:
                clear_memory()
        
        avg_loss = total_loss / max(valid_batches, 1)
        print(f"📊 평균 훈련 손실: {avg_loss:.4f}")
        
        # 검증 (품질 모니터링)
        if val_dataloader and len(val_dataset) > 0:
            model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="검증 중"):
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
            
            if val_steps > 0:
                avg_val_loss = val_loss / val_steps
                print(f"� 검증 손실: {avg_val_loss:.4f}")
                
                # 최고 성능 모델 저장
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print("✅ 검증 손실 개선! 최고 모델 저장...")
                    output_dir = "models/rtx3070_optimized_best"
                    os.makedirs(output_dir, exist_ok=True)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
            
            model.train()
        
        # 정기 체크포인트 저장
        if (epoch + 1) % 2 == 0:
            output_dir = f"models/rtx3070_checkpoint_epoch_{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            print(f"💾 체크포인트 저장: {output_dir}")
    
    # 최종 모델 저장
    final_dir = "models/rtx3070_optimized_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    # 훈련 결과 정보 저장
    result_info = {
        "model": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "training_type": "RTX 3070 최적화 LoRA 파인튜닝",
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": 384,
        "lora_r": 16,
        "lora_alpha": 32,
        "best_validation_loss": best_val_loss,
        "optimizations": [
            "4bit 양자화",
            "Mistral 전용 LoRA 모듈",
            "Gradient accumulation",
            "개선된 라벨 마스킹",
            "검증 기반 최고 모델 저장"
        ]
    }
    
    with open(f"{final_dir}/training_info.json", "w", encoding="utf-8") as f:
        json.dump(result_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 RTX 3070 최적화 훈련 완료!")
    print(f"📁 최종 모델: {final_dir}")
    print(f"📁 최고 성능 모델: models/rtx3070_optimized_best")
    print(f"🏆 최고 검증 손실: {best_val_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
