#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
전문가 모델 훈련 스크립트 (하이브리드 시스템용)
Specialist Model Training Script (for Hybrid System)

각 도메인별 전문가 모델을 개별적으로 훈련합니다.
Trains specialist models for each domain individually.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpecialistModelTrainer:
    """전문가 모델 훈련 클래스"""
    
    def __init__(self, base_model_name: str = "microsoft/DialoGPT-medium"):
        """
        Args:
            base_model_name: 기본 모델명 (Microsoft DialoGPT 사용)
        """
        self.base_model_name = base_model_name
        self.tokenizer = None
        self.model = None
        
        # 전문가 도메인 정의
        self.specialist_domains = {
            'qna_specialist': {
                'description': 'Question & Answer specialist',
                'data_prefix': 'dialogue_qna_',
                'output_dir': 'qna_specialist'
            },
            'technical_specialist': {
                'description': 'Technical knowledge specialist',
                'data_prefix': 'dialogue_technical_',
                'output_dir': 'technical_specialist'
            }
        }
    
    def load_base_model(self):
        """기본 모델과 토크나이저 로딩"""
        logger.info(f"기본 모델 로딩 중: {self.base_model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.base_model_name)
            
            # 패딩 토큰 설정
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("✅ 기본 모델 로딩 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            return False
    
    def load_specialist_data(self, domain: str) -> List[Dict]:
        """특정 도메인의 훈련 데이터 로딩"""
        
        data_dir = Path(__file__).parent.parent / "datasets" / "processed_english"
        data_prefix = self.specialist_domains[domain]['data_prefix']
        
        train_file = data_dir / f"{data_prefix}train.jsonl"
        
        if not train_file.exists():
            logger.warning(f"⚠️ 훈련 데이터 없음: {train_file}")
            return []
        
        try:
            data = []
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            
            logger.info(f"📁 {domain} 데이터 로딩: {len(data)}개 샘플")
            return data
            
        except Exception as e:
            logger.error(f"❌ 데이터 로딩 실패 ({domain}): {e}")
            return []
    
    def prepare_training_data(self, data: List[Dict]) -> Dataset:
        """훈련 데이터 전처리"""
        
        # 대화 형태로 변환
        formatted_data = []
        
        for item in data:
            # 한국어 댓글은 참고용으로 유지하되, 실제 훈련은 영어로
            input_text = item.get('input', '')
            output_text = item.get('output', '')
            
            # 대화 형식 구성 
            conversation = f"User: {input_text}\nAssistant: {output_text}"
            formatted_data.append({'text': conversation})
        
        # Hugging Face Dataset으로 변환
        dataset = Dataset.from_list(formatted_data)
        
        # 토크나이징
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        logger.info(f"✅ 훈련 데이터 준비 완료: {len(tokenized_dataset)}개 샘플")
        return tokenized_dataset
    
    def train_specialist_model(self, domain: str, epochs: int = 3, 
                             learning_rate: float = 5e-5) -> bool:
        """특정 도메인의 전문가 모델 훈련"""
        
        logger.info(f"🎯 {domain} 전문가 모델 훈련 시작")
        
        # 데이터 로딩
        data = self.load_specialist_data(domain)
        if not data:
            logger.error(f"❌ {domain} 훈련 데이터 없음")
            return False
        
        # 데이터 전처리
        dataset = self.prepare_training_data(data)
        
        # 출력 디렉토리 설정
        output_dir = Path(__file__).parent.parent / "models" / self.specialist_domains[domain]['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 훈련 인수 설정
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=100,
            logging_steps=50,
            learning_rate=learning_rate,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
        )
        
        # 데이터 컬렉터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 인과적 언어 모델링 사용
        )
        
        # 트레이너 설정
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        try:
            # 훈련 실행
            logger.info(f"🚀 {domain} 모델 훈련 중...")
            trainer.train()
            
            # 모델 저장
            trainer.save_model()
            self.tokenizer.save_pretrained(str(output_dir))
            
            logger.info(f"✅ {domain} 전문가 모델 훈련 완료: {output_dir}")
            
            # 모델 정보 저장
            model_info = {
                'domain': domain,
                'base_model': self.base_model_name,
                'training_samples': len(data),
                'epochs': epochs,
                'learning_rate': learning_rate,
                'description': self.specialist_domains[domain]['description']
            }
            
            with open(output_dir / "model_info.json", 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ {domain} 모델 훈련 실패: {e}")
            return False
    
    def train_all_specialists(self, epochs: int = 3):
        """모든 전문가 모델 훈련"""
        
        logger.info("🎯 모든 전문가 모델 훈련 시작")
        
        if not self.load_base_model():
            logger.error("❌ 기본 모델 로딩 실패로 훈련 중단")
            return
        
        results = {}
        
        for domain in self.specialist_domains:
            logger.info(f"\n{'='*60}")
            logger.info(f"📚 {domain} 훈련 중...")
            
            success = self.train_specialist_model(domain, epochs=epochs)
            results[domain] = success
            
            if success:
                logger.info(f"✅ {domain} 성공")
            else:
                logger.error(f"❌ {domain} 실패")
        
        # 최종 결과 요약
        logger.info(f"\n{'='*60}")
        logger.info("📊 전문가 모델 훈련 결과:")
        
        successful = sum(results.values())
        total = len(results)
        
        for domain, success in results.items():
            status = "✅" if success else "❌"
            logger.info(f"   {status} {domain}")
        
        logger.info(f"\n🎯 성공: {successful}/{total}")
        
        if successful == total:
            logger.info("🎉 모든 전문가 모델 훈련 완료!")
        else:
            logger.warning(f"⚠️ {total - successful}개 모델 훈련 실패")

def main():
    """메인 실행 함수"""
    
    parser = argparse.ArgumentParser(description="전문가 모델 훈련")
    parser.add_argument('--domain', type=str, choices=['qna_specialist', 'daily_chat_specialist', 'technical_specialist', 'all'], 
                       default='all', help='훈련할 도메인 선택')
    parser.add_argument('--epochs', type=int, default=3, help='훈련 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='학습률')
    parser.add_argument('--base_model', type=str, default="microsoft/DialoGPT-medium", help='기본 모델명')
    
    args = parser.parse_args()
    
    trainer = SpecialistModelTrainer(base_model_name=args.base_model)
    
    if args.domain == 'all':
        trainer.train_all_specialists(epochs=args.epochs)
    else:
        if not trainer.load_base_model():
            logger.error("❌ 기본 모델 로딩 실패")
            return
            
        success = trainer.train_specialist_model(args.domain, epochs=args.epochs, learning_rate=args.learning_rate)
        
        if success:
            logger.info(f"✅ {args.domain} 훈련 완료")
        else:
            logger.error(f"❌ {args.domain} 훈련 실패")

if __name__ == "__main__":
    main()
