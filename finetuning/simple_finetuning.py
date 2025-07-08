#!/usr/bin/env python3
"""
간소화된 파인튜닝 스크립트
RTX 3070 8GB 환경에 최적화된 설정으로 통합 모델 파인튜닝
"""

import os
import json
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_unified_data(data_dir):
    """통합 데이터 로드"""
    logger.info("통합 데이터 로딩 중...")
    
    train_data = []
    val_data = []
    
    # unified_train.jsonl 로드
    train_file = Path(data_dir) / "unified_train.jsonl"
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    item = json.loads(line)
                    # 입력과 출력을 하나의 텍스트로 결합
                    text = f"{item['input_text']} {item['target_text']}<|endoftext|>"
                    train_data.append({"text": text})
                except json.JSONDecodeError:
                    continue
    
    # unified_validation.jsonl 로드
    val_file = Path(data_dir) / "unified_validation.jsonl" 
    with open(val_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    item = json.loads(line)
                    text = f"{item['input_text']} {item['target_text']}<|endoftext|>"
                    val_data.append({"text": text})
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"학습 데이터: {len(train_data)}개")
    logger.info(f"검증 데이터: {len(val_data)}개")
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def tokenize_function(examples, tokenizer, max_length=512):
    """토크나이제이션 함수"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def main():
    """메인 파인튜닝 함수"""
    
    # 디바이스 설정 (CPU 환경)
    device = torch.device("cpu")
    logger.info(f"사용 디바이스: {device}")
    logger.info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    # 모델과 토크나이저 로드 (CPU에 최적화된 경량 모델)
    model_name = "microsoft/DialoGPT-small"  # CPU에 적합한 소형 모델
    logger.info(f"모델 로딩: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # CPU에서는 float32 사용
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("모델 로딩 완료")
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        return
    
    # 데이터 로드
    try:
        data_dir = "datasets/processed_english"
        train_dataset, val_dataset = load_unified_data(data_dir)
        
        # 토크나이제이션
        logger.info("데이터 토크나이제이션 중...")
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=["text"]
        )
        
        logger.info("토크나이제이션 완료")
        
    except Exception as e:
        logger.error(f"데이터 처리 실패: {e}")
        return
    
    # 학습 설정 (CPU 최적화)
    training_args = TrainingArguments(
        output_dir="logs/checkpoints/unified_model_simple",
        overwrite_output_dir=True,
        num_train_epochs=1,  # CPU에서는 1 에포크만
        per_device_train_batch_size=1,  # 배치 크기 최소
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 그래디언트 누적으로 효과적 배치 크기 증가
        warmup_steps=50,
        learning_rate=5e-5,
        logging_steps=5,
        eval_steps=20,
        save_steps=50,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # wandb 비활성화
        dataloader_pin_memory=False,  # 메모리 절약
        fp16=False,  # CPU에서는 fp16 비활성화
        gradient_checkpointing=True,  # 메모리 절약
        remove_unused_columns=False,
        no_cuda=True,  # CPU 강제 사용
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 인과적 언어 모델링
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 파인튜닝 실행
    logger.info("파인튜닝 시작...")
    try:
        trainer.train()
        logger.info("파인튜닝 완료!")
        
        # 모델 저장
        logger.info("모델 저장 중...")
        trainer.save_model("models/unified_model")
        tokenizer.save_pretrained("models/unified_model")
        logger.info("모델 저장 완료!")
        
        # 평가
        logger.info("모델 평가 중...")
        eval_results = trainer.evaluate()
        logger.info(f"평가 결과: {eval_results}")
        
    except Exception as e:
        logger.error(f"파인튜닝 실패: {e}")
        # error.txt에 오류 기록
        with open("../.github/prompts/error.txt", "w", encoding="utf-8") as f:
            f.write(f"파인튜닝 오류: {e}\n")
            f.write("개선 지침:\n")
            f.write("1. GPU 메모리 부족 시 배치 크기 더 줄이기\n")
            f.write("2. 모델 크기를 더 작은 것으로 변경\n")
            f.write("3. 그래디언트 체크포인팅 활성화\n")
            f.write("4. 데이터 크기 축소 고려\n")

if __name__ == "__main__":
    main()
