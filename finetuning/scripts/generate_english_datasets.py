#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path

# 출력 디렉토리 설정
OUTPUT_DIR = Path("finetuning/datasets/processed_english")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_classification_data():
    """분류 데이터 생성 (daily_chat 제거됨)"""
    print("🏷️  분류 데이터 생성 중...")
    
    # QnA 샘플
    qna_samples = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Explain quantum physics",
        "What causes earthquakes?", 
        "How do vaccines work?",
        "What is DNA?",
        "Explain climate change",
        "What is artificial intelligence?",
        "How does the internet work?",
        "What is blockchain technology?"
    ]
    
    # 기술 상담 샘플
    technical_samples = [
        "How to use SketchUp for 3D modeling?",
        "AutoCAD best practices for beginners",
        "Programming in Python for data analysis",
        "JavaScript frameworks comparison",
        "Database optimization techniques",
        "Cloud computing architecture",
        "Software testing methodologies",
        "Web development with React",
        "Mobile app development guide",
        "API design principles"
    ]
    
    # 일반 대화 샘플 (기존 daily_chat 포함)
    general_samples = [
        "Hello, how are you?",
        "Good morning! Nice weather today",
        "I'm feeling stressed lately",
        "What do you do for fun?", 
        "Have a great day!",
        "Thank you for your help",
        "I had a wonderful weekend",
        "How can I improve my mood?",
        "I'm learning something new",
        "See you later!"
    ]
    
    # 데이터 결합 및 섞기
    all_data = []
    all_data.extend([(text, 0) for text in qna_samples])
    all_data.extend([(text, 1) for text in technical_samples])
    all_data.extend([(text, 2) for text in general_samples])
    
    random.shuffle(all_data)
    
    # 훈련/검증/테스트 분할 (80/10/10)
    total = len(all_data)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = all_data[:train_size]
    val_data = all_data[train_size:train_size + val_size]
    test_data = all_data[train_size + val_size:]
    
    # 파일 저장
    datasets = {
        "classification_train.jsonl": train_data,
        "classification_validation.jsonl": val_data,
        "classification_test.jsonl": test_data
    }
    
    for filename, data in datasets.items():
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 분류 모델 데이터 - 레이블: 0=qna, 1=technical, 2=general (일반 대화는 통합모델에서 처리)\n")
            f.write("# Classification model data - Labels: 0=qna, 1=technical, 2=general (general conversation handled by unified model)\n")
            for text, label in data:
                f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")
    
    print(f"✅ 분류 데이터 생성 완료")
    print(f"   - 훈련: {len(train_data)}개")
    print(f"   - 검증: {len(val_data)}개")
    print(f"   - 테스트: {len(test_data)}개")

def generate_dialogue_data():
    """대화 데이터 생성 (daily_chat 제거됨)"""
    print("💬 대화 데이터 생성 중...")
    
    # QnA 대화 데이터
    qna_dialogues = [
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task."
        },
        {
            "input": "How does photosynthesis work?",
            "output": "Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll."
        },
        {
            "input": "What causes climate change?",
            "output": "Climate change is primarily caused by increased greenhouse gas emissions from human activities, particularly burning fossil fuels and deforestation."
        },
        {
            "input": "Explain DNA replication",
            "output": "DNA replication is the process by which DNA makes a copy of itself during cell division, involving unwinding, base pairing, and synthesis of new strands."
        }
    ]
    
    # 기술 상담 대화 데이터
    technical_dialogues = [
        {
            "input": "How to start 3D modeling in SketchUp?",
            "output": "To start 3D modeling in SketchUp: 1. Learn basic tools (Push/Pull, Move, Rotate), 2. Practice with simple shapes, 3. Use guides and dimensions for precision, 4. Explore plugins for advanced features."
        },
        {
            "input": "Best practices for AutoCAD beginners?",
            "output": "AutoCAD best practices: 1. Use layers properly, 2. Set up templates with standards, 3. Learn keyboard shortcuts, 4. Use object snaps, 5. Save regularly and use backup files."
        },
        {
            "input": "How to optimize Python code performance?",
            "output": "Python optimization tips: 1. Use list comprehensions, 2. Avoid global variables, 3. Use built-in functions, 4. Profile your code, 5. Consider NumPy for numerical operations."
        },
        {
            "input": "What are the advantages of cloud computing?",
            "output": "Cloud computing advantages include: 1) Cost efficiency - pay for what you use, 2) Scalability, 3) Accessibility from anywhere, 4) Automatic backups, 5) Access to latest technology."
        }
    ]
    
    # 일반 대화 데이터 (기존 daily_chat 포함)
    general_dialogues = [
        {
            "input": "Hello, nice to meet you",
            "output": "Hello! Nice to meet you too. I'm glad we have the chance to chat. How can I help you today?"
        },
        {
            "input": "Good morning! How are you today?",
            "output": "Good morning! I'm doing well, thank you for asking. How are you feeling today? I hope you have a wonderful day ahead!"
        },
        {
            "input": "I'm feeling a bit stressed lately",
            "output": "I understand that stress can be overwhelming. Have you tried taking deep breaths or going for a walk? Sometimes small breaks can make a big difference."
        },
        {
            "input": "What do you do for fun?",
            "output": "I enjoy learning about new topics and having interesting conversations with people like you! What about you? What activities bring you joy?"
        },
        {
            "input": "Thank you for your help",
            "output": "You're very welcome! I'm happy I could help. If you have any other questions, feel free to ask anytime."
        },
        {
            "input": "I had a great vacation",
            "output": "That's wonderful! I'm so glad you had a great time. Where did you go? I'd love to hear about your favorite moments from the trip!"
        },
        {
            "input": "What's your favorite season?",
            "output": "I find all seasons fascinating in their own way! Spring brings renewal, summer brings energy, autumn brings beautiful colors, and winter brings cozy moments. What's your favorite and why?"
        },
        {
            "input": "Have a great day!",
            "output": "Thank you! You have a wonderful day too. Take care and I hope to chat with you again soon!"
        }
    ]
    
    # 각 타입별로 파일 생성 (daily_chat 제거됨)
    dialogue_types = {
        "qna": qna_dialogues,
        "technical": technical_dialogues,
        "general": general_dialogues  # daily_chat은 여기에 포함됨
    }
    
    for dialogue_type, dialogues in dialogue_types.items():
        # 훈련/검증/테스트 분할
        random.shuffle(dialogues)
        total = len(dialogues)
        train_size = int(total * 0.7)
        val_size = int(total * 0.15)
        
        train_data = dialogues[:train_size]
        val_data = dialogues[train_size:train_size + val_size]
        test_data = dialogues[train_size + val_size:]
        
        datasets = {
            f"dialogue_{dialogue_type}_train.jsonl": train_data,
            f"dialogue_{dialogue_type}_validation.jsonl": val_data,
            f"dialogue_{dialogue_type}_test.jsonl": test_data
        }
        
        for filename, data in datasets.items():
            filepath = OUTPUT_DIR / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {dialogue_type} 대화 데이터 / {dialogue_type} dialogue data\n")
                for item in data:
                    f.write(json.dumps({"input_text": item["input"], "target_text": item["output"]}, ensure_ascii=False) + "\n")
        
        print(f"   - {dialogue_type}: 훈련 {len(train_data)}, 검증 {len(val_data)}, 테스트 {len(test_data)}")

def generate_unified_data():
    """통합 모델용 데이터 생성"""
    print("🔄 통합 모델 데이터 생성 중...")
    
    system_prompt = "You are a helpful AI assistant that can handle various types of conversations including Q&A, technical support, and general chat."
    
    # 통합 학습 데이터
    unified_samples = [
        # QnA 샘플
        {
            "input_text": f"System: {system_prompt}\n\nUser: What is machine learning?\nAssistant:",
            "target_text": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed for every task.",
            "category": "qna"
        },
        {
            "input_text": f"System: {system_prompt}\n\nUser: How does photosynthesis work?\nAssistant:",
            "target_text": "Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll in their leaves.",
            "category": "qna"
        },
        
        # 일반 대화 샘플 (기존 daily_chat 포함)
        {
            "input_text": f"System: {system_prompt}\n\nUser: Good morning! How are you?\nAssistant:",
            "target_text": "Good morning! I'm doing great, thank you for asking. How are you feeling today? I hope you're having a wonderful start to your day!",
            "category": "general"
        },
        {
            "input_text": f"System: {system_prompt}\n\nUser: I'm feeling stressed lately\nAssistant:",
            "target_text": "I'm sorry to hear you're feeling stressed. That can be really tough. Have you tried any relaxation techniques like deep breathing or taking short walks? Sometimes even small breaks can help a lot.",
            "category": "general"
        },
        {
            "input_text": f"System: {system_prompt}\n\nUser: What do you do for fun?\nAssistant:",
            "target_text": "I enjoy learning about new topics and having interesting conversations with people like you! What about you? What activities bring you joy and relaxation?",
            "category": "general"
        },
        {
            "input_text": f"System: {system_prompt}\n\nUser: I had a great weekend\nAssistant:",
            "target_text": "That's wonderful to hear! I'm so glad you had a great weekend. What made it special? I'd love to hear about the highlights!",
            "category": "general"
        },
        
        # 기술 상담 샘플
        {
            "input_text": f"System: {system_prompt}\n\nUser: How to start 3D modeling in SketchUp?\nAssistant:",
            "target_text": "To start 3D modeling in SketchUp: 1. Learn basic tools (Push/Pull, Move, Rotate), 2. Practice with simple shapes, 3. Use guides and dimensions for precision, 4. Explore plugins for advanced features.",
            "category": "technical"
        },
        {
            "input_text": f"System: {system_prompt}\n\nUser: What are the advantages of cloud computing?\nAssistant:",
            "target_text": "Cloud computing advantages include: 1) Cost efficiency - pay for what you use, 2) Scalability - easily scale up/down, 3) Accessibility - access from anywhere, 4) Automatic backups, 5) Access to latest technology, 6) Reduced maintenance burden.",
            "category": "technical"
        },
        
        # 일반적인 도움 요청
        {
            "input_text": f"System: {system_prompt}\n\nUser: Can you help me with something?\nAssistant:",
            "target_text": "Of course! I'd be happy to help you. Could you tell me what you need assistance with? I can help with questions, provide information, or just have a conversation.",
            "category": "general"
        },
        {
            "input_text": f"System: {system_prompt}\n\nUser: Thank you for your help\nAssistant:",
            "target_text": "You're very welcome! I'm glad I could help. If you have any other questions or need assistance with anything else, please don't hesitate to ask.",
            "category": "general"
        }
    ]
    
    # 추가 샘플 생성 (다양성 증대)
    additional_samples = []
    categories = ["qna", "technical", "general"]
    
    for _ in range(50):  # 50개 추가 샘플
        category = random.choice(categories)
        if category == "qna":
            sample = {
                "input_text": f"System: {system_prompt}\n\nUser: Tell me about science\nAssistant:",
                "target_text": "Science is the systematic study of the natural world through observation, experimentation, and analysis. It helps us understand how things work and discover new knowledge.",
                "category": "qna"
            }
        elif category == "technical":
            sample = {
                "input_text": f"System: {system_prompt}\n\nUser: I need technical help\nAssistant:",
                "target_text": "I'd be happy to provide technical assistance! Could you please tell me more specifically what you need help with? Whether it's software, programming, or technical concepts, I'm here to help.",
                "category": "technical"
            }
        else:  # general
            sample = {
                "input_text": f"System: {system_prompt}\n\nUser: How are you doing today?\nAssistant:",
                "target_text": "I'm doing well, thank you for asking! I'm here and ready to help or chat about whatever you'd like. How has your day been going?",
                "category": "general"
            }
        additional_samples.append(sample)
    
    all_unified_data = unified_samples + additional_samples
    random.shuffle(all_unified_data)
    
    # 훈련/검증/테스트 분할
    total = len(all_unified_data)
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    
    train_data = all_unified_data[:train_size]
    val_data = all_unified_data[train_size:train_size + val_size]
    test_data = all_unified_data[train_size + val_size:]
    
    datasets = {
        "unified_train.jsonl": train_data,
        "unified_validation.jsonl": val_data,
        "unified_test.jsonl": test_data
    }
    
    for filename, data in datasets.items():
        filepath = OUTPUT_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 통합 모델 학습 데이터 (daily_chat은 general로 통합됨)\n")
            f.write("# Unified model training data (daily_chat integrated as general)\n")
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"✅ 통합 데이터 생성 완료")
    print(f"   - 훈련: {len(train_data)}개")
    print(f"   - 검증: {len(val_data)}개")
    print(f"   - 테스트: {len(test_data)}개")

def validate_data():
    """생성된 데이터 검증"""
    print("🔍 데이터 검증 중...")
    
    required_files = [
        "classification_train.jsonl", "classification_validation.jsonl", "classification_test.jsonl",
        "dialogue_qna_train.jsonl", "dialogue_qna_validation.jsonl", "dialogue_qna_test.jsonl", 
        "dialogue_technical_train.jsonl", "dialogue_technical_validation.jsonl", "dialogue_technical_test.jsonl",
        "dialogue_general_train.jsonl", "dialogue_general_validation.jsonl", "dialogue_general_test.jsonl",
        "unified_train.jsonl", "unified_validation.jsonl", "unified_test.jsonl"
    ]
    
    for filename in required_files:
        filepath = OUTPUT_DIR / filename
        if not filepath.exists():
            print(f"❌ 누락된 파일: {filename}")
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                data_lines = [line for line in lines if not line.startswith('#')]
                print(f"✅ {filename}: {len(data_lines)}개 샘플")

def main():
    """메인 함수"""
    print("🚀 영어 기반 파인튜닝 데이터셋 생성 시작")
    print("🔄 하이브리드 시스템용 - daily_chat은 general로 통합됨")
    print("=" * 50)
    
    # 기존 파일 정리
    print("🧹 기존 파일 정리 중...")
    for file in OUTPUT_DIR.glob("*.jsonl"):
        if file.name.startswith("dialogue_daily_chat"):
            print(f"   🗑️ 제거: {file.name} (daily_chat 전문모델 제거됨)")
            file.unlink()
    
    # 데이터 생성
    generate_classification_data()
    generate_dialogue_data()
    generate_unified_data()
    
    # 검증
    validate_data()
    
    print("=" * 50)
    print("✅ 모든 데이터셋 생성 완료!")
    print(f"📁 저장 위치: {OUTPUT_DIR}")
    print("\n📋 하이브리드 시스템 구조:")
    print("   - 통합모델: 기본 처리 (general, 간단한 qna/technical)")
    print("   - QnA 전문모델: 복잡한 질문 처리")
    print("   - 기술 전문모델: 고급 기술 상담")
    print("   - daily_chat: 통합모델에서 general로 처리됨")
    
    print("\n🎯 다음 단계:")
    print("   1. python finetuning/scripts/train_unified_model.py")
    print("   2. python finetuning/scripts/train_specialist.py --type qna")
    print("   3. python finetuning/scripts/train_specialist.py --type technical")

if __name__ == "__main__":
    main()
