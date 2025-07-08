#!/usr/bin/env python3
"""
통합형 단일 모델 파인튜닝 스크립트
하나의 모델로 모든 대화 유형을 처리하도록 파인튜닝
"""

import os
import sys
import json
import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List
import logging
from datetime import datetime

# Transformers 라이브러리
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling,
    TrainerCallback, EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedModelTrainer:
    """통합형 모델 파인튜닝 클래스"""
    
    def __init__(self, config_path: str = "configs/training_config.yaml"):
        """초기화"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"사용 디바이스: {self.device}")
        
        # 출력 디렉토리 생성
        self.output_dir = Path("logs/checkpoints/unified_model")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logging_dir = Path("logs/training_logs/unified_model")
        self.logging_dir.mkdir(parents=True, exist_ok=True)
    
    def create_unified_system_prompt(self) -> str:
        """통합 모델용 시스템 프롬프트 생성"""
        return """당신은 다양한 상황에 적응할 수 있는 멀티태스크 AI 어시스턴트입니다.

역할 가이드:
- QnA/질문답변: 정확하고 구체적인 사실 기반 답변
- 일상대화: 친근하고 감정적인 공감형 대화
- 기술상담: 전문적이고 단계별 설명
- 일반대화: 상황에 맞는 적절한 톤과 스타일

사용자의 입력 유형을 파악하고 그에 맞는 최적의 응답 스타일을 선택하세요."""

    def load_all_datasets(self) -> List[Dict]:
        """모든 데이터셋을 하나로 통합하여 로드"""
        unified_data = []
        system_prompt = self.create_unified_system_prompt()
        
        # 각 카테고리별 데이터 로드
        categories = {
            'qna': '질문답변',
            'daily_chat': '일상대화', 
            'technical': '기술상담',
            'conversation_types': '일반대화'
        }
        
        for category, korean_name in categories.items():
            data_files = [
                f"datasets/processed/dialogue_{category}_train.jsonl",
                f"datasets/processed/classification_train.jsonl" if category == 'conversation_types' else None
            ]
            
            for data_file in data_files:
                if data_file and os.path.exists(data_file):
                    logger.info(f"{category} 데이터 로딩: {data_file}")
                    
                    with open(data_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                item = json.loads(line)
                                
                                # 분류 데이터인 경우
                                if 'label' in item:
                                    text = item.get('text', '')
                                    # 분류를 응답 형태로 변환
                                    response = f"이것은 {korean_name} 유형의 대화입니다."
                                    
                                    unified_item = {
                                        'input_text': text,
                                        'target_text': response,
                                        'category': category,
                                        'type': 'classification'
                                    }
                                
                                # 대화 데이터인 경우
                                elif 'input_text' in item and 'target_text' in item:
                                    unified_item = {
                                        'input_text': item['input_text'],
                                        'target_text': item['target_text'], 
                                        'category': category,
                                        'type': 'dialogue'
                                    }
                                
                                else:
                                    continue
                                
                                # 시스템 프롬프트와 함께 포맷팅
                                formatted_input = f"시스템: {system_prompt}\n\n사용자: {unified_item['input_text']}\n어시스턴트:"
                                
                                unified_data.append({
                                    'input_text': formatted_input,
                                    'target_text': unified_item['target_text'],
                                    'category': category
                                })
        
        logger.info(f"총 {len(unified_data)}개의 통합 데이터 로드 완료")
        return unified_data
    
    def preprocess_unified_data(self, examples, tokenizer):
        """통합 데이터 전처리"""
        inputs = []
        
        for input_text, target_text in zip(examples['input_text'], examples['target_text']):
            # 입력과 타겟을 결합하여 완전한 대화 형태로 만들기
            combined = f"{input_text} {target_text}{tokenizer.eos_token}"
            inputs.append(combined)
        
        model_inputs = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=self.config['training']['max_seq_length']
        )
        
        # 레이블은 input_ids와 동일 (언어 모델링)
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    def train_unified_model(self):
        """통합 모델 파인튜닝"""
        logger.info("통합 모델 파인튜닝 시작")
        
        # 모델 설정 - 일반적인 대화 모델 사용
        base_model = "microsoft/DialoGPT-medium"  # 또는 사용하고 싶은 다른 모델
        
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.config['training']['fp16'] else torch.float32
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id
        
        # LoRA 적용
        if self.config['lora']['use_lora']:
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['lora_dropout'],
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        # 통합 데이터 로드
        unified_data = self.load_all_datasets()
        if not unified_data:
            raise ValueError("통합 데이터가 없습니다. 먼저 데이터를 준비하세요.")
        
        # 데이터 분할 (80% 학습, 10% 검증, 10% 테스트)
        import random
        random.shuffle(unified_data)
        
        total_size = len(unified_data)
        train_size = int(total_size * 0.8)
        val_size = int(total_size * 0.1)
        
        train_data = unified_data[:train_size]
        val_data = unified_data[train_size:train_size + val_size]
        test_data = unified_data[train_size + val_size:]
        
        # Dataset 객체 생성
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # 데이터 전처리
        train_dataset = train_dataset.map(
            lambda x: self.preprocess_unified_data(x, tokenizer),
            batched=True
        )
        val_dataset = val_dataset.map(
            lambda x: self.preprocess_unified_data(x, tokenizer),
            batched=True
        )
        
        # 학습 인자 설정
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_steps=self.config['training']['save_steps'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            fp16=self.config['training']['fp16'],
            dataloader_num_workers=self.config['training']['dataloader_num_workers'],
            save_total_limit=self.config['training']['save_total_limit'],
            logging_dir=self.logging_dir,
            report_to=None
        )
        
        # 트레이너 설정
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # 학습 실행
        logger.info("통합 모델 학습 시작...")
        trainer.train()
        
        # 모델 저장
        model_save_path = Path("models/unified_model")
        model_save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"통합 모델 저장 완료: {model_save_path}")
        
        # 테스트 데이터 저장 (평가용)
        test_file = Path("datasets/processed/unified_test.jsonl")
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"테스트 데이터 저장: {test_file}")

def main():
    parser = argparse.ArgumentParser(description="통합형 모델 파인튜닝 스크립트")
    parser.add_argument("--config", default="configs/training_config.yaml", help="설정 파일 경로")
    
    args = parser.parse_args()
    
    trainer = UnifiedModelTrainer(args.config)
    trainer.train_unified_model()

if __name__ == "__main__":
    main()
