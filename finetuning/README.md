# AI PZ2 파인튜닝 환경

## 🎯 목표
대규모 LLM 모델 보완을 위한 파인튜닝 환경 구축
- 대화 유형 분류기 모델 파인튜닝
- 특화 작업별 모델 개발

## 📁 폴더 구조 (하이브리드 시스템)

```
finetuning/
├── README.md                    <- 이 파일
├── datasets/                    <- 데이터셋 저장
│   └── processed_english/       <- 영어 기반 학습 데이터 ⭐
├── models/                      <- 파인튜닝된 모델 저장
│   ├── unified_model/           <- 통합형 모델 (기본) ⭐
│   ├── qna_specialist/          <- QnA 전문 모델 (동적 로딩)
│   └── technical_specialist/    <- 기술 상담 전문 모델 (동적 로딩)
├── scripts/                     <- 학습/추론 스크립트
│   ├── train_unified_model.py   <- 통합형 모델 학습 ⭐
│   ├── unified_inference.py     <- 통합형 모델 추론
│   ├── generate_english_datasets.py <- 영어 데이터 생성 ⭐
│   └── validate_english_data.py <- 데이터 검증 ⭐
├── configs/                     <- 설정 파일들
│   ├── training_config.yaml     <- 학습 설정
│   └── model_config.yaml        <- 모델 설정
└── logs/                        <- 학습 로그 및 체크포인트
    ├── training_logs/
    └── checkpoints/
```

⭐ = 핵심 구성요소

## 🚀 하이브리드 아키텍처 (Hybrid Architecture)

### 🎯 새로운 접근법: 통합형 + 전문모델 연결
**기본은 통합형, 필요시 전문모델 동적 로딩**

```
사용자 입력
    ↓
💡 통합형 모델 (기본 처리)
- 내장된 분류 기능
- 일반적인 QnA, 일상대화, 기술상담 처리
- 빠른 응답, 낮은 리소스
    ↓
🤔 품질 판단
- 응답 신뢰도 낮음?
- 복잡한 전문 질문?
- 사용자 요청?
    ↓
🚀 전문모델 동적 로딩
- QnA 전문모델
- 기술상담 전문모델
- 고품질 응답 생성
```

### 🏗️ 아키텍처 구성

#### 1️⃣ 통합형 모델 (Primary)
- **역할**: 메인 처리기 + 내장 분류기
- **처리**: 90% 일반 질문 빠르게 해결
- **특징**: 
  - 빠른 응답 (< 1초)
  - 낮은 메모리 사용
  - 기본적인 QnA, 일상대화, 기술상담 가능
  - 자체 분류 및 품질 판단 기능

#### 2️⃣ 전문모델 풀 (Specialist Pool)
- **QnA 전문가**: 복잡한 사실 질문 및 연구 지원
- **기술상담 전문가**: 전문 소프트웨어 지원 (SketchUp, CAD 등)
- **일상대화 및 감정상담**: 통합형 모델에서 충분히 처리
- **동적 로딩**: 필요할 때만 전문모델을 메모리에 로드

#### 3️⃣ 스마트 라우터
- **기본**: 통합형 모델이 처리
- **전환 조건**:
  - 통합형 응답 신뢰도 < 70%
  - "더 정확한 답변", "전문가 모드" 요청
  - 복잡도 임계값 초과
- **최적화**: 사용 패턴 학습으로 예측 로딩

### 💡 하이브리드 방식의 장점

| 특성 | 하이브리드 (Unified + Specialists) | 기존 분리형 | 기존 통합형 |
|------|----------------------------------|------------|------------|
| **응답 속도** | ⭐⭐⭐⭐⭐ (기본 빠름) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **메모리 효율** | ⭐⭐⭐⭐ (동적 로딩) | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **전문성** | ⭐⭐⭐⭐⭐ (필요시 전환) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **구현 복잡도** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **확장성** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **리소스 사용** | ⭐⭐⭐⭐ (적응형) | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### 🎮 사용 시나리오

#### 시나리오 1: 일반 질문
```
사용자: "오늘 날씨 어때?"
→ 통합형 모델이 즉시 처리 (< 1초)
→ 빠르고 적절한 답변
```

#### 시나리오 2: 복잡한 기술 질문
```
사용자: "SketchUp에서 복잡한 곡면 모델링하는 고급 기법을 알려줘"
→ 통합형 모델: "기본적인 답변... 더 전문적인 답변이 필요하시나요?"
→ 사용자: "네, 전문가 모드로 해주세요"
→ 기술상담 전문모델 로딩 및 고품질 답변 생성
```

#### 시나리오 3: 일상 감정 대화
```
사용자: "I feel sad today"
→ 통합형 모델이 따뜻하고 적절한 응답 제공
→ 일상 대화는 통합모델로 충분 (전문가 모드 불필요)
→ 응답: "It sounds like you're feeling sad today. Is there anything you'd like to talk about?"
```

#### 시나리오 4: 기술 전문 질문
```
사용자: "How to draw a line in CAD software?"
→ 통합형 모델: "To draw a line in CAD software, you typically select the line tool from the toolbar..."
→ 복잡도 감지 → QnA 전문모델 자동 활성화  
→ 전문가 응답: "Drawing a line in CAD software involves several steps. Here's a detailed guide: 1. Open your CAD software..."
```

#### 시나리오 5: SketchUp 전문가 모드
```
사용자: "SketchUp expert mode advanced modeling"
→ 전문가 모드 키워드 감지 → 기술 전문모델 즉시 로딩
→ 전문가 응답: "SketchUp is a powerful 3D modeling tool. For advanced modeling, consider these expert techniques: 1. Component hierarchies, 2. Advanced inference systems..."
```

## 📊 데이터셋 구조 및 예시

### 분리형 모델용 데이터 형식

#### 1. 대화 분류 데이터 (classification_*.jsonl)
```json
{"text": "What is the difference between lists and tuples in Python?", "label": 0}
{"text": "How to draw a line in CAD software?", "label": 1}
{"text": "Hello, how are you today?", "label": 2}
```
**라벨**: 0=qna, 1=technical, 2=general (일반 대화는 통합모델에서 처리)

#### 2. 대화 생성 데이터 (dialogue_*_*.jsonl)
```json
{"input_text": "What is a Python dictionary?", "target_text": "A dictionary is a built-in data type in Python that stores data in key-value pairs."}
{"input_text": "How to start 3D modeling in AutoCAD?", "target_text": "To start 3D modeling in AutoCAD: 1. Switch to 3D Modeling workspace, 2. Change view using ViewCube..."}
```

### 통합형 모델용 데이터 형식

#### 통합 학습 데이터 (unified_*.jsonl)
```json
{
  "input_text": "System: You are a versatile AI assistant capable of handling various types of conversations...\n\nUser: What is Python?\nAssistant:",
  "target_text": "Python is a simple and readable programming language. It's widely used for web development, data analysis, AI, and more.",
  "category": "technical"
}
```

### 원본 데이터 생성 (자동화)
`generate_english_datasets.py` 스크립트가 다양한 영어 대화 데이터를 자동으로 생성합니다:
```json
{
  "input_text": "What is the difference between lists and tuples in Python?",
  "target_text": "The main differences between lists and tuples are: 1. Mutability: Lists are mutable, tuples are immutable...",
  "category": "technical"
}
```

## 🔧 사용법

### 🚀 빠른 시작 (하이브리드 모드)

#### 1단계: 영어 데이터 생성
```bash
# 영어 기반 파인튜닝 데이터 생성 (필수)
python finetuning/scripts/generate_english_datasets.py

# 데이터 검증 (권장)
python finetuning/scripts/validate_english_data.py
```

#### 2단계: 통합형 모델 학습
```bash
# 통합형 모델 파인튜닝 (기본 모델)
python finetuning/scripts/train_unified_model.py

# 또는 배치 스크립트 사용
run_finetuning.bat
# → 메뉴에서 8번 "통합형 모델 학습" 선택
```

#### 3단계: 전문모델 학습 (선택적)
```bash
# 필요시 특정 전문모델 추가 학습
# QnA 전문가 모델
python finetuning/scripts/train_specialist.py --type qna

# 기술상담 전문가 모델
python finetuning/scripts/train_specialist.py --type technical
```

#### 4단계: 메인 시스템에서 사용
```bash
# AI 어시스턴트 실행
python main.py

# 하이브리드 모드 확인
# → ft: 파인튜닝관리 → 상태 확인
# → 일반 대화는 통합모델로 빠르게 처리
# → 복잡한 질문은 자동으로 전문모델 로딩
```

### 📋 기본 사용법

1. **데이터 준비**: 영어 기반 데이터 자동 생성
2. **통합모델 학습**: 기본 멀티태스크 모델 학습 
3. **전문모델 학습**: 필요시 고품질 전문모델 추가
4. **자동 라우팅**: 시스템이 자동으로 최적 모델 선택
run_finetuning.bat

# 메뉴에서 선택:
# 9번: 통합형 모델 파인튜닝 (간단, 빠름)
# 5,6,7,8번: 분리형 모델들 파인튜닝 (전문성)
```

### 📋 상세 단계

#### 분리형 모델 (Multi-Model) - 하이브리드 시스템
1. **데이터 준비**: `python finetuning/scripts/generate_english_datasets.py`
2. **통합모델 학습**: `python scripts/train_unified_model.py` (기본 모델)
3. **전문모델 학습**: 
   - `python scripts/train_specialist.py --type qna` (QnA 전문가)
   - `python scripts/train_specialist.py --type technical` (기술 전문가)
   - ~~`python scripts/train_specialist.py --type daily_chat`~~ (제거됨 - 통합모델에서 처리)
4. **평가**: `python scripts/model_evaluation.py --model_path models/unified_model --model_type generation`
5. **테스트**: `python main.py` (하이브리드 모드 자동 선택)

#### 통합형 모델 (Single-Model) - 기본 방식
1. **데이터 준비**: `python finetuning/scripts/generate_english_datasets.py`
2. **통합 학습**: `python scripts/train_unified_model.py`
3. **평가**: `python scripts/model_evaluation.py --model_path models/unified_model --model_type generation`
4. **테스트**: `python scripts/unified_inference.py --mode interactive`

### 🌍 영어 데이터셋 생성 / English Dataset Generation
**영어 기반 모델을 위한 데이터셋** (권장 / Recommended)
```bash
# 영어 기반 파인튜닝 데이터 생성 / Generate English-based fine-tuning data
python finetuning/scripts/generate_english_datasets.py
```

**생성되는 파일들 / Generated Files**:
```
finetuning/datasets/processed_english/
├── classification_train.jsonl     # 분류 모델 훈련 데이터
├── classification_validation.jsonl # 분류 모델 검증 데이터  
├── classification_test.jsonl      # 분류 모델 테스트 데이터
├── dialogue_qna_train.jsonl       # QnA 대화 데이터
├── dialogue_technical_train.jsonl  # 기술 상담 데이터
├── dialogue_general_train.jsonl    # 일반 대화 데이터
├── unified_train.jsonl            # 통합 모델 훈련 데이터
├── unified_validation.jsonl       # 통합 모델 검증 데이터
└── unified_test.jsonl             # 통합 모델 테스트 데이터
```

**데이터 검증 기능 / Data Validation Features**:
- ✅ JSON 형식 검증 / JSON format validation
- ✅ 필수 필드 확인 / Required fields check  
- ✅ 텍스트 길이 제한 / Text length limits
- ✅ 레이블 분포 확인 / Label distribution check
- ✅ 한글 주석 포함 / Korean comments included for easy review

### 🎯 추천 학습 순서
1. **처음 시작**: 통합형 모델로 빠른 체험
2. **성능 개선**: 분리형 모델들로 전문성 향상  
3. **비교 분석**: 두 방식의 장단점 직접 비교
