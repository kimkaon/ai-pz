#!/usr/bin/env python3
"""
Nous-Hermes-2-Mistral-7B-DPO 영어 파인튜닝 스크립트
- processed_english 데이터셋 기반
- LoRA를 이용한 효율적 파인튜닝
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
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
import warnings
import gc

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def clear_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class EnglishChatDataset(Dataset):
    """영어 대화 데이터셋 클래스"""
    
    def __init__(self, data_files, tokenizer, max_length=128):  # 길이 감소 (메모리 절약)
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
        
        # 고품질 응답을 위한 개선된 전처리
        input_text = item['input_text'].strip()
        target_text = item['target_text'].strip()
        
        # 너무 긴 응답은 3-4문장으로 제한 (품질 유지하면서 간결화)
        sentences = target_text.split('. ')
        if len(sentences) > 4:
            target_text = '. '.join(sentences[:4]) + '.'
        
        # 고품질 Mistral/Nous-Hermes 형식으로 포맷팅
        text = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n{target_text}<|im_end|>"
        
        # 토큰화
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 라벨 마스킹: 사용자 입력 부분은 -100으로 마스킹 (고품질 학습)
        labels = encoding['input_ids'].clone()
        
        # 간단하고 안전한 마스킹 방식
        text_parts = text.split("<|im_start|>assistant\n")
        if len(text_parts) > 1:
            user_part = text_parts[0] + "<|im_start|>assistant\n"
            user_tokens = self.tokenizer.encode(user_part, add_special_tokens=False)
            
            # 안전한 마스킹
            mask_length = min(len(user_tokens), labels.shape[-1])
            labels[0][:mask_length] = -100
        else:
            # 대안: 전체 시퀀스의 절반만 마스킹
            mask_length = labels.shape[-1] // 2
            labels[0][:mask_length] = -100
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

def load_english_datasets():
    """영어 데이터셋 파일 경로 반환"""
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
    print("🚀 Nous-Hermes-2-Mistral-7B-DPO 영어 파인튜닝 시작")
    print("="*60)
    
    # 초기 메모리 정리
    clear_memory()
    
    # GPU 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 사용 디바이스: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"🧹 메모리 최적화 설정 적용됨")
    
    # 원본 Nous-Hermes-2-Mistral-7B-DPO 모델 사용 (최적화 적용)
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    print(f"📥 기본 모델 로딩: {model_name}")
    print("⚡ 메모리 최적화 설정 적용")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        # RTX 3070 8GB 최적화: 4bit 양자화
        from transformers import BitsAndBytesConfig
        
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
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("✅ 모델 로드 완료")
        print(f"📊 모델 파라미터 수: {model.num_parameters():,}")
        
        # RTX 3070 8GB 최적화된 LoRA 설정
        print("🔧 RTX 3070 최적화 LoRA 설정 적용 중...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # 적절한 rank (품질과 메모리 균형)
            lora_alpha=16,  # 2*r (일반적 설정)
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 주요 어텐션 모듈
            bias="none"
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 훈련 모드로 설정
        model.train()
        
        print("✅ LoRA 파라미터 설정 완료")
        print("✅ 4bit 양자화로 메모리 사용량 대폭 감소")
        
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
    
    # 데이터로더 생성 - 안전한 설정
    batch_size = 1  # 배치 크기 감소 (안정성)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=False,  # 메모리 문제 방지
        drop_last=True  # 불완전한 배치 제거
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=False,
        drop_last=False
    )
    
    # 5시간 연속 학습을 위한 최적화된 설정
    num_epochs = 5  # 에포크 증가 (5시간 충분히 활용)
    learning_rate = 1e-4  # 적절한 학습률
    gradient_accumulation_steps = 4  # 메모리 효율적
    max_grad_norm = 1.0  # 그래디언트 클리핑
    
    # 전체 데이터셋 사용 (고품질 학습)
    print(f"🎯 5시간 연속 학습을 위해 전체 {len(train_dataset.data)}개 데이터 사용")
    
    # 웜업과 코사인 스케줄러를 위한 고급 옵티마이저 설정
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=learning_rate,
        weight_decay=0.01,  # 가중치 감쇠 추가
        eps=1e-8
    )
    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = total_steps // 20  # 5% 웜업
    
    # 고급 스케줄러 설정
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"🎯 RTX 3070 8GB 최적화 설정 (5시간 연속):")
    print(f"   배치 크기: {batch_size}")
    print(f"   그래디언트 누적 스텝: {gradient_accumulation_steps}")
    print(f"   효과적 배치 크기: {batch_size * gradient_accumulation_steps}")
    print(f"   에포크: {num_epochs}")
    print(f"   학습률: {learning_rate}")
    print(f"   총 스텝: {total_steps}")
    print(f"   웜업 스텝: {warmup_steps}")
    print(f"   훈련 데이터: {len(train_dataset)} 샘플")
    print(f"   검증 데이터: {len(val_dataset)} 샘플")
    print(f"   💡 4bit 양자화로 메모리 사용량 ~75% 감소")
    
    # 고품질 파인튜닝 실행
    model.train()
    best_val_loss = float('inf')
    patience = 0
    max_patience = 2
    
    for epoch in range(num_epochs):
        print(f"\n📚 에포크 {epoch + 1}/{num_epochs}")
        
        total_loss = 0
        valid_batches = 0
        progress_bar = tqdm(train_loader, desc=f"훈련 중 (에포크 {epoch+1})")
        
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
            
            loss = outputs.loss
            
            # 손실 유효성 검사
            if loss is None or not torch.isfinite(loss):
                print("⚠️ 유효하지 않은 손실값입니다. 스킵합니다.")
                continue
            
            # 그래디언트 누적을 위한 손실 조정
            loss = loss / gradient_accumulation_steps
            total_loss += loss.item()
            valid_batches += 1
            
            # 역전파
            loss.backward()
            
            # 그래디언트 누적 및 업데이트
            if (step + 1) % gradient_accumulation_steps == 0:
                # 그래디언트 클리핑 (안정적 훈련)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                clear_memory()  # 메모리 정리
            
            # 현재 학습률 표시
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # 주기적 메모리 정리
            if step % 20 == 0:
                clear_memory()
        
        avg_loss = total_loss / max(valid_batches, 1)
        print(f"📊 평균 훈련 손실: {avg_loss:.4f}")
        print(f"📈 현재 학습률: {scheduler.get_last_lr()[0]:.2e}")
        
        # 검증 (조기 종료 포함)
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
            
            # 조기 종료 체크
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience = 0
                print("✅ 검증 손실 개선! 모델 저장...")
                # 최고 성능 모델 중간 저장
                script_dir = os.path.dirname(os.path.abspath(__file__))
                temp_dir = os.path.join(script_dir, "models", "english_unified_best")
                os.makedirs(temp_dir, exist_ok=True)
                model.save_pretrained(temp_dir)
            else:
                patience += 1
                print(f"⏳ 검증 손실 미개선 ({patience}/{max_patience})")
                
            if patience >= max_patience:
                print("🛑 조기 종료: 검증 성능이 개선되지 않음")
                break
            
            model.train()
    
    # 모델 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "models", "english_unified")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 LoRA 모델 저장: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 설정 파일 저장
    config_info = {
        "base_model": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        "training_data": "processed_english datasets (full)",
        "model_type": "High-Quality LoRA fine-tuned",
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "lora_r": 16,
        "lora_alpha": 32,
        "max_length": 256,
        "training_time": "~5 hours",
        "scheduler": "cosine_with_warmup",
        "best_val_loss": best_val_loss
    }
    
    with open(f"{output_dir}/training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)
    
    # 고품질 테스트 생성
    print(f"\n🧪 고품질 모델 테스트")
    model.eval()
    
    # 다양한 유형의 테스트 프롬프트
    test_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?", 
        "Explain deep learning in simple terms.",
        "What are the main benefits of renewable energy?",
        "How do neural networks learn from data?",
        "What is the difference between AI and ML?",
        "Describe the concept of natural language processing.",
        "How does computer vision work?"
    ]
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"\n🔸 테스트 {i+1}: {prompt}")
            
            # Mistral/Nous-Hermes 프롬프트 형식으로 토큰화
            input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            inputs = tokenizer(input_text, return_tensors='pt').to(device)
            
            # 고품질 응답 생성
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,  # 충분한 길이
                do_sample=True,
                temperature=0.7,  # 창의적이지만 일관된 응답
                top_p=0.9,  # 다양성과 품질 균형
                repetition_penalty=1.1,  # 반복 방지
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                no_repeat_ngram_size=3  # 3-gram 반복 방지
            )
            
            # 응답 디코딩
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_response = response.split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            
            print(f"💬 응답: {assistant_response}")
            
            # 응답 품질 간단 체크
            if len(assistant_response.split()) < 5:
                print("⚠️ 응답이 너무 짧습니다.")
            elif len(assistant_response) > 500:
                print("⚠️ 응답이 너무 깁니다.")
            else:
                print("✅ 적절한 길이의 응답")
    
    print(f"\n🎉 고품질 Nous-Hermes-2-Mistral 영어 파인튜닝 완료!")
    print(f"📁 최종 모델 저장 위치: {output_dir}")
    print(f"📁 최고 성능 모델 위치: {os.path.join(script_dir, 'models', 'english_unified_best')}")
    print(f"🏆 최고 검증 손실: {best_val_loss:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
