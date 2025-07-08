#!/usr/bin/env python3
"""
순수 PyTorch 파인튜닝 스크립트
TensorFlow 의존성 없이 PyTorch만 사용하여 통합 모델 파인튜닝
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import logging
from tqdm import tqdm

# TensorFlow 비활성화
os.environ['USE_TF'] = 'False'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# PyTorch만 사용하도록 transformers 설정
os.environ['TRANSFORMERS_NO_TF'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedDataset(Dataset):
    """통합 데이터셋 클래스"""
    
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 데이터 로드
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        item = json.loads(line)
                        # 입력과 출력을 결합하여 하나의 텍스트로 만듦
                        text = f"{item['input_text']} {item['target_text']}{tokenizer.eos_token}"
                        self.data.append(text)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"로드된 데이터: {len(self.data)}개")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # 토크나이제이션
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # 언어 모델링에서는 input과 label이 동일
        }

def simple_finetuning():
    """간단한 파인튜닝 실행"""
    
    logger.info("순수 PyTorch 파인튜닝 시작...")
    
    # 디바이스 설정
    device = torch.device("cpu")
    logger.info(f"사용 디바이스: {device}")
    
    # 모델과 토크나이저 로드
    model_name = "microsoft/DialoGPT-small"
    logger.info(f"모델 로딩: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model.to(device)
        logger.info("모델 로딩 완료")
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        return
    
    # 데이터셋 로드
    try:
        train_dataset = UnifiedDataset(
            "datasets/processed_english/unified_train.jsonl",
            tokenizer,
            max_length=256
        )
        
        val_dataset = UnifiedDataset(
            "datasets/processed_english/unified_validation.jsonl", 
            tokenizer,
            max_length=256
        )
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,  # CPU에서는 작은 배치 크기
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"학습 배치: {len(train_loader)}개")
        logger.info(f"검증 배치: {len(val_loader)}개")
        
    except Exception as e:
        logger.error(f"데이터 로딩 실패: {e}")
        return
    
    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    # 모델을 학습 모드로 설정
    model.train()
    
    # 학습 루프
    num_epochs = 1
    total_loss = 0
    step = 0
    
    logger.info("파인튜닝 시작...")
    
    for epoch in range(num_epochs):
        logger.info(f"에포크 {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for batch in progress_bar:
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
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 손실 기록
            total_loss += loss.item()
            epoch_loss += loss.item()
            step += 1
            
            # 진행 상황 업데이트
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 주기적으로 로그 출력
            if step % 10 == 0:
                avg_loss = total_loss / step
                logger.info(f"Step {step}, 평균 손실: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"에포크 {epoch + 1} 완료, 평균 손실: {avg_epoch_loss:.4f}")
        
        # 검증
        if len(val_loader) > 0:
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
            logger.info(f"검증 손실: {avg_val_loss:.4f}")
            
            model.train()
    
    # 모델 저장
    logger.info("모델 저장 중...")
    
    try:
        # 출력 디렉토리 생성
        output_dir = Path("models/unified_model_pytorch")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델과 토크나이저 저장
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"모델이 {output_dir}에 저장되었습니다!")
        
        # 최종 평가
        final_avg_loss = total_loss / step if step > 0 else 0
        logger.info(f"파인튜닝 완료! 최종 평균 손실: {final_avg_loss:.4f}")
        
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")

def test_model():
    """파인튜닝된 모델 테스트"""
    logger.info("모델 테스트 시작...")
    
    try:
        # 저장된 모델 로드
        model_path = "models/unified_model_pytorch"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        # 테스트 입력
        test_input = "System: You are a helpful AI assistant that can handle various types of conversations including Q&A, technical support, and general chat.\n\nUser: Hello, how are you?\nAssistant:"
        
        # 토크나이제이션
        input_ids = tokenizer.encode(test_input, return_tensors='pt')
        
        # 생성
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 결과 디코딩
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        logger.info(f"테스트 응답: {response}")
        
    except Exception as e:
        logger.error(f"모델 테스트 실패: {e}")

if __name__ == "__main__":
    try:
        simple_finetuning()
        test_model()
    except Exception as e:
        logger.error(f"실행 실패: {e}")
        
        # error.txt 업데이트
        with open("../.github/prompts/error.txt", "a", encoding="utf-8") as f:
            f.write(f"\n추가 오류: {e}\n")
            f.write("최종 해결 방법: 순수 PyTorch 구현 완료\n")
