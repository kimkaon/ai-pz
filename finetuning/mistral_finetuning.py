#!/usr/bin/env python3
"""
메인 프로그램과 호환되는 Nous-Hermes-2-Mistral 파인튜닝 스크립트
GPU 환경에서 실행되도록 최적화
"""

import os
import json
import torch
from pathlib import Path
import logging

# 환경 변수 설정 (TensorFlow 문제 해결)
os.environ['USE_TF'] = 'NO'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        TrainingArguments, 
        Trainer,
        DataCollatorForLanguageModeling,
        BitsAndBytesConfig
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Transformers 라이브러리 문제: {e}")
    TRANSFORMERS_AVAILABLE = False

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_status():
    """GPU 상태 확인 및 설정"""
    logger.info("=== GPU 상태 확인 ===")
    
    # CUDA 확인
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU 사용 가능: {gpu_name}")
        logger.info(f"💾 GPU 메모리: {gpu_memory:.1f}GB")
        
        # 메모리 정리
        torch.cuda.empty_cache()
        return device, True
    else:
        logger.warning("❌ CUDA를 사용할 수 없습니다. CPU 모드로 진행합니다.")
        return torch.device("cpu"), False

def install_requirements():
    """필요한 패키지 설치"""
    logger.info("=== 필요한 패키지 확인 및 설치 ===")
    
    packages_to_install = []
    
    # PyTorch CUDA 버전 확인
    if not hasattr(torch, 'cuda') or not torch.cuda.is_available():
        logger.warning("PyTorch CUDA 버전이 필요합니다.")
        print("\n🔧 PyTorch CUDA 설치 명령어:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    # Transformers 확인
    if not TRANSFORMERS_AVAILABLE:
        packages_to_install.append("transformers")
    
    # 기타 필요 패키지
    try:
        import datasets
    except ImportError:
        packages_to_install.append("datasets")
    
    try:
        import peft
    except ImportError:
        packages_to_install.append("peft")
    
    try:
        import bitsandbytes
    except ImportError:
        packages_to_install.append("bitsandbytes")
    
    if packages_to_install:
        logger.info(f"설치 필요한 패키지: {packages_to_install}")
        install_cmd = f"pip install {' '.join(packages_to_install)}"
        print(f"\n🔧 설치 명령어: {install_cmd}")
        return False
    
    return True

def load_unified_data(data_dir):
    """통합 데이터 로드 (메인 프로그램 호환)"""
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
                    # Nous-Hermes 형식에 맞게 변환
                    conversation = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{item['input_text'].split('User: ')[-1].split('\\nAssistant:')[0]}<|im_end|>\n<|im_start|>assistant\n{item['target_text']}<|im_end|>"
                    train_data.append({"text": conversation})
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
                    conversation = f"<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n{item['input_text'].split('User: ')[-1].split('\\nAssistant:')[0]}<|im_end|>\n<|im_start|>assistant\n{item['target_text']}<|im_end|>"
                    val_data.append({"text": conversation})
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"학습 데이터: {len(train_data)}개")
    logger.info(f"검증 데이터: {len(val_data)}개")
    
    return Dataset.from_list(train_data), Dataset.from_list(val_data)

def main():
    """메인 파인튜닝 함수"""
    
    logger.info("=== Nous-Hermes-2-Mistral 파인튜닝 시작 ===")
    
    # 필수 패키지 확인
    if not install_requirements():
        logger.error("필수 패키지가 설치되지 않았습니다. 위의 명령어로 설치해주세요.")
        return
    
    # GPU 상태 확인
    device, use_gpu = check_gpu_status()
    
    if not TRANSFORMERS_AVAILABLE:
        logger.error("Transformers 라이브러리를 사용할 수 없습니다.")
        return
    
    # 메인 프로그램과 동일한 모델 사용
    model_name = "NousResearch/Nous-Hermes-2-Mistral-7B-DPO"
    logger.info(f"모델 로딩: {model_name}")
    
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 특수 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 양자화 설정 (GPU 메모리 절약)
        if use_gpu:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # 모델 로드 (4bit 양자화)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:
            # CPU 모드
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        logger.info("모델 로딩 완료")
        
        # LoRA 설정 (효율적인 파인튜닝)
        if use_gpu:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
            )
            
            model = get_peft_model(model, lora_config)
            logger.info("LoRA 설정 완료")
        
    except Exception as e:
        logger.error(f"모델 로딩 실패: {e}")
        # error.txt에 기록
        with open("../.github/prompts/error.txt", "a", encoding="utf-8") as f:
            f.write(f"\n모델 로딩 오류: {e}\n")
            f.write("개선 지침:\n")
            f.write("1. HuggingFace 토큰이 필요할 수 있음\n")
            f.write("2. 모델 크기가 클 수 있으니 더 작은 모델 시도\n")
            f.write("3. 양자화 설정 조정\n")
        return
    
    # 데이터 로드
    try:
        data_dir = "datasets/processed_english"
        train_dataset, val_dataset = load_unified_data(data_dir)
        
        # 토크나이제이션
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
        
        logger.info("데이터 토크나이제이션 중...")
        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        logger.info("토크나이제이션 완료")
        
    except Exception as e:
        logger.error(f"데이터 처리 실패: {e}")
        return
    
    # 학습 설정
    if use_gpu:
        training_args = TrainingArguments(
            output_dir="logs/checkpoints/nous_hermes_finetuned",
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            warmup_steps=100,
            learning_rate=2e-4,
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            dataloader_pin_memory=False,
            fp16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )
    else:
        training_args = TrainingArguments(
            output_dir="logs/checkpoints/nous_hermes_finetuned",
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
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
            report_to=None,
            dataloader_pin_memory=False,
            fp16=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            no_cuda=True,
        )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
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
        if use_gpu:
            # LoRA 어댑터만 저장 (메모리 효율적)
            model.save_pretrained("models/nous_hermes_finetuned_lora")
        else:
            trainer.save_model("models/nous_hermes_finetuned")
        
        tokenizer.save_pretrained("models/nous_hermes_finetuned")
        logger.info("모델 저장 완료!")
        
        # 평가
        logger.info("모델 평가 중...")
        eval_results = trainer.evaluate()
        logger.info(f"평가 결과: {eval_results}")
        
        # 메인 프로그램 연동을 위한 정보 저장
        model_info = {
            "model_name": model_name,
            "finetuned_path": "models/nous_hermes_finetuned",
            "use_lora": use_gpu,
            "eval_loss": eval_results.get("eval_loss", 0),
            "training_completed": True
        }
        
        with open("models/finetuning_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ 파인튜닝 완료! 메인 프로그램에서 사용 가능합니다.")
        
    except Exception as e:
        logger.error(f"파인튜닝 실패: {e}")
        with open("../.github/prompts/error.txt", "a", encoding="utf-8") as f:
            f.write(f"\n파인튜닝 실행 오류: {e}\n")
            f.write("개선 지침:\n")
            f.write("1. GPU 메모리 부족 시 배치 크기 줄이기\n")
            f.write("2. LoRA rank 값 줄이기\n")
            f.write("3. 양자화 비트 수 늘리기 (8bit)\n")
            f.write("4. 그래디언트 체크포인팅 활성화\n")

if __name__ == "__main__":
    main()
