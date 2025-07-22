# AI 어시스턴트 프로젝트 (RTX 3070 최적화) - 완전 모듈화 버전

RTX 3070 GPU에 최적화된 다중 모델 AI 어시스턴트 시스템입니다. **2025-07-14 최종 업데이트: 모듈화 완료 + GGUF Q4_K_M 양자화 적용**

## 🎯 주요 기능

- **완전 모듈화 시스템**: 유지보수와 디버깅을 위한 모듈별 분리 설계
- **GGUF Q4_K_M 양자화**: 13.5GB → 4.1GB로 압축, RTX 3070에서 GPU 35레이어 실행
- **다중 AI 모델 지원**: 기본 모델, RTX 3070 최적화 모델, 영어 특화 모델, 하이브리드 파인튜닝
- **언어 제한 시스템**: RTX 3070 모델은 영어/한국어만 지원 (안전성 강화)
- **실시간 음성 대화**: 스트리밍 STT + TTS로 자연스러운 대화
- **GPU 최적화**: RTX 3070 8GB VRAM에 최적화된 4bit 양자화
- **하이브리드 파인튜닝**: 도메인별 전문 모델 자동 선택

## 🚀 시작하기

### 필수 환경
- NVIDIA RTX 3070 (또는 유사한 8GB GPU)
- Python 3.10+
- CUDA 11.8+

### 설치 및 실행
```bash
# 가상환경 활성화
.\ksmevn\Scripts\Activate.ps1

# 메인 프로그램 실행
python main.py
```

## 📁 새로운 모듈화 구조 (2025-07-14)

### 핵심 모듈들
```
main.py                     # 메인 어플리케이션 (간소화됨)
├── core/                   # 핵심 시스템
│   └── initialization.py  # 시스템 초기화 로직
├── models/                 # 모델 관리
│   └── model_manager.py    # 통합 모델 관리자 (RTX3070GGUFLoader 포함)
├── interfaces/             # 사용자 인터페이스
│   └── menu_system.py      # 메뉴 시스템
├── processing/             # 처리 로직
│   └── response_handler.py # 응답 생성 및 처리
└── utils/                  # 유틸리티
    └── logging_utils.py    # 로깅 시스템
```

### GGUF 모델 파일
```
models/
├── rtx3070_final_merged.gguf        # 원본 GGUF 모델 (13.5GB)
├── rtx3070_final_merged_q4km.gguf   # Q4_K_M 양자화 모델 (4.1GB) ⭐
├── rtx3070_optimized_best/          # LoRA 어댑터
├── rtx3070_optimized_final/         # 백업 LoRA 어댑터
└── rtx3070_checkpoint_epoch_6/      # 체크포인트
```

### 양자화 도구
```
llama.cpp/                  # GGUF 변환 및 양자화 도구
├── build/bin/              # 빌드된 실행파일들
│   └── llama-quantize.exe  # 양자화 도구
quantize_to_q4km.py         # 양자화 스크립트
convert_rtx3070_to_gguf.py  # LoRA → GGUF 변환 스크립트
```

## 📚 상세 파일 설명

### 🔧 핵심 모듈 파일들 (5개)

#### `main.py` - 메인 애플리케이션
**기능**: 프로그램 진입점 및 모듈 오케스트레이션
- 환경 변수 설정 및 경고 메시지 숨기기
- 전역 model_manager 인스턴스 생성 및 관리
- 시스템 초기화, 모델 preload, 메뉴 시스템 실행
- 예외 처리 및 안전한 종료 보장
- **크기**: 약 72줄 (이전 1151줄에서 대폭 간소화)

#### `core/initialization.py` - 시스템 초기화
**기능**: 시스템 환경 설정 및 LLM 초기화
- 환경 변수 설정 (CUDA, HuggingFace 오프라인 모드)
- 로깅 레벨 조정 (faster_whisper, transformers 등)
- 전역 LLM 인스턴스 생성 및 관리
- 하이브리드 파인튜닝 시스템 초기화
- GPU 설정 및 메모리 최적화

#### `models/model_manager.py` - 통합 모델 관리자
**기능**: 모든 AI 모델의 로딩, 관리, 전환
- **EnglishFinetuningModelLoader**: 영어 특화 파인튜닝 모델
- **RTX3070OptimizedLoader**: RTX 3070 LoRA 모델 (언어 제한 포함)
- **RTX3070GGUFLoader**: ⭐ GGUF Q4_K_M 양자화 모델 (GPU 가속)
- **ModelManager**: 모든 모델을 통합 관리하는 메인 클래스
- 동적 GPU 메모리 관리 및 레이어 수 조정
- 모델 preloading 및 실시간 전환 기능

#### `interfaces/menu_system.py` - 사용자 인터페이스
**기능**: 대화형 메뉴 시스템 및 사용자 입력 처리
- 메인 메뉴 (음성/텍스트 대화, 모델 선택, 설정 등)
- 모델 선택 및 전환 인터페이스
- 설정 관리 메뉴 (통합 설정 시스템)
- 실시간 대화 모드 관리
- 사용자 경험 최적화 (팁, 상태 표시 등)

#### `processing/response_handler.py` - 응답 처리
**기능**: AI 모델 응답 생성 및 후처리
- 현재 선택된 모델에 따른 응답 생성 로직
- 하이브리드 파인튜닝 모델 통합
- RTX 3070 특화 모델 (LoRA, GGUF) 처리
- 응답 정제 및 포맷팅
- 오류 처리 및 폴백 메커니즘

#### `utils/logging_utils.py` - 로깅 시스템
**기능**: 통합 로깅 및 설정 관리
- JSON 기반 통합 설정 관리 (`ai_assistant_settings.json`)
- 카테고리별 로깅 (model_loading, conversation, system 등)
- 설정 백업/복원 기능
- 실시간 설정 변경 감지
- 응답 정제 유틸리티 함수들

### 🤖 AI 모델 관련 파일들

#### `nous_hermes2_mistral_loader.py` - 기본 LLM 로더
**기능**: 기본 Nous-Hermes-2-Mistral 모델 로딩
- GGUF 형식 모델 로딩 (Q5_K_M 양자화)
- GPU/CPU 자동 선택 및 최적화
- 컨텍스트 윈도우 및 배치 크기 설정
- 메모리 효율적인 로딩 (mmap 사용)

#### `finetuning_integration.py` - 파인튜닝 통합
**기능**: 파인튜닝된 모델들의 통합 관리
- 도메인별 전문 모델 로딩
- 자동 도메인 분류 및 모델 선택
- 모델 성능 통계 및 모니터링
- 파인튜닝 모델 상태 관리

#### `hybrid_finetuning_integration.py` - 하이브리드 시스템
**기능**: 여러 파인튜닝 모델을 조합한 하이브리드 시스템
- 통합형 모델과 전문형 모델 조합
- 실시간 모델 전환 및 응답 병합
- 한국어 특화 처리
- 성능 최적화 및 캐싱

### 🔧 변환 및 양자화 도구들

#### `convert_rtx3070_to_gguf.py` - LoRA → GGUF 변환
**기능**: LoRA 어댑터를 GGUF 형식으로 변환
- PeftModel 로딩 및 base model과 병합
- HuggingFace 형식으로 병합된 모델 저장
- llama.cpp의 convert_hf_to_gguf.py 호출
- 변환 과정 모니터링 및 오류 처리

#### `quantize_to_q4km.py` - Q4_K_M 양자화 전문 도구
**기능**: GGUF 모델을 Q4_K_M으로 양자화 (RTX 3070 최적화)
- llama-quantize.exe 실행 및 파라미터 설정
- 양자화 진행률 실시간 모니터링
- 파일 크기 비교 및 압축률 계산
- 양자화 품질 검증 및 성능 벤치마크

#### `quantize_model.py` - 범용 양자화 도구
**기능**: 다양한 양자화 옵션 지원하는 통합 양자화 도구
- 지원 형식: Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0
- 사용자 선택형 양자화 인터페이스
- 배치 양자화 지원 (여러 모델 동시 처리)
- 양자화 결과 비교 및 성능 분석

#### `quantize_to_q4km.py` - Q4_K_M 양자화
**기능**: GGUF 모델을 Q4_K_M으로 양자화
- llama-quantize.exe 실행 및 파라미터 설정
- 양자화 진행률 모니터링
- 파일 크기 비교 및 압축률 계산
- 양자화 품질 검증

#### `quantize_model.py` - 범용 양자화 도구
**기능**: 다양한 양자화 옵션 지원
- Q2_K, Q3_K_M, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0 지원
- 사용자 선택형 양자화 인터페이스
- 배치 양자화 지원
- 양자화 결과 비교 및 분석

### 🎤 음성 처리 시스템

#### `voice_utils.py` - 음성 처리 통합 플랫폼
**기능**: STT, 실시간 음성 감지, 오디오 처리 통합
- faster-whisper 기반 STT 엔진 (OpenAI Whisper 최적화 버전)
- 실시간 음성 활동 감지 (VAD) - pyannote.audio 활용
- 마이크 설정 및 오디오 장치 관리 (portaudio 기반)
- 백그라운드 녹음 및 스트리밍 처리
- 노이즈 필터링 및 음성 품질 향상 (noisereduce 라이브러리)
- 한국어/영어 혼합 음성 인식 지원

#### `openvoice_tts.py` - OpenVoice V2 TTS 엔진
**기능**: MyShell-AI OpenVoice V2 기반 고품질 음성 합성
- 다국어 TTS 지원 (영어, 한국어, 중국어, 일본어 등)
- 감정 및 톤 조절 (happy, sad, angry, neutral 등)
- TTS 캐시 시스템 (중복 생성 방지, 빠른 응답)
- 실시간 스트리밍 TTS (청크 단위 생성)
- 음성 품질 최적화 (RTX 3070 GPU 가속)
- 목소리 복제 및 스타일 전환 기능

#### `fast_tts.py` - 경량 TTS 시스템
**기능**: 빠른 응답을 위한 경량 TTS 대안
- edge-tts 기반 빠른 음성 합성
- 낮은 레이턴시 및 메모리 사용량
- 오프라인 작동 가능
- 다양한 음성 스타일 지원
- OpenVoice 백업용 TTS 엔진

#### `realtime_chat.py` - 실시간 음성 대화 시스템
**기능**: 끊김 없는 실시간 음성 대화 구현
- 음성 입력 → STT → AI 응답 → TTS → 음성 출력 파이프라인
- 실시간 응답 스트리밍 (청크 단위 처리)
- 대화 컨텍스트 유지 및 메모리 관리
- 인터럽트 처리 (사용자가 AI 응답 중 말하기 시작)
- 멀티스레딩 기반 비동기 처리

### ⚙️ 설정 및 구성 파일들

#### `config_gpu.py` - GPU 설정 최적화
**기능**: CUDA 및 GPU 메모리 최적화 (RTX 3070 특화)
- RTX 3070 특화 설정 (8GB VRAM 최적화)
- 동적 GPU 메모리 관리 (pytorch cuda cache)
- CUDA 커널 최적화 (cuDNN 벤치마크 활성화)
- 메모리 파편화 방지 (clear_cache 전략)
- GPU 온도 및 성능 모니터링 (nvidia-ml-py)
- 자동 GPU/CPU 폴백 메커니즘

#### `setup_gpu.py` - GPU 환경 초기 설정 도구
**기능**: GPU 드라이버 및 CUDA 환경 검증
- NVIDIA 드라이버 버전 확인 (GeForce Experience 연동)
- CUDA 툴킷 설치 검증 (nvcc 버전 체크)
- cuDNN 라이브러리 확인 (딥러닝 가속)
- GPU 메모리 테스트 (메모리 안정성 검사)
- 환경 문제 진단 및 해결 가이드 제공

#### `settings_manager.py` - 레거시 설정 관리 (구형)
**기능**: 기존 설정 시스템 (현재는 logging_utils.py로 통합됨)
- JSON 기반 설정 저장/로드
- 설정 마이그레이션 지원 (버전별 호환성)
- 백호환성 유지 (기존 설정 파일 읽기)

#### `prompt_templates.py` - 프롬프트 템플릿 관리
**기능**: AI 모델별 프롬프트 최적화
- 모델별 최적화된 프롬프트 형식 (ChatML, Alpaca, Vicuna 등)
- 역할 기반 프롬프트 (assistant, user, system)
- 다국어 프롬프트 지원 (한국어, 영어)
- 컨텍스트 길이 최적화 (모델별 최대 토큰 고려)
- 특수 용도 프롬프트 (코딩, 창작, 분석 등)

### 📊 유틸리티 및 관리 도구들

#### `run_finetuning.bat` - 파인튜닝 관리 스크립트
**기능**: 파인튜닝 작업 자동화 배치 스크립트
- 데이터셋 준비 및 검증 (JSON/CSV 형식 지원)
- 하이퍼파라미터 설정 (learning rate, batch size, epochs)
- 학습 진행률 모니터링 (tensorboard 로그)
- 체크포인트 관리 (자동 저장 및 복원)
- 학습 완료 후 모델 변환 (LoRA → HuggingFace → GGUF)

#### `small_model_options.py` - 경량 모델 옵션
**기능**: 저사양 환경을 위한 경량 모델 설정
- 3GB/4GB VRAM 환경 지원 (GTX 1060/1070 등)
- 모델 크기별 추천 설정 (1B, 3B, 7B 모델)
- 메모리 사용량 프로파일링 (실시간 모니터링)
- 성능 vs 품질 트레이드오프 설정 (양자화 수준 조정)
- CPU 전용 모드 지원 (GPU 없는 환경)
**기능**: AI 모델별 프롬프트 최적화
- 모델별 최적화된 프롬프트 형식
- 역할 기반 프롬프트 (assistant, user, system)
- 다국어 프롬프트 지원
- 컨텍스트 길이 최적화

### 📊 유틸리티 및 관리 도구들

#### `setup_gpu.py` - GPU 환경 초기 설정
**기능**: GPU 드라이버 및 CUDA 환경 검증
- NVIDIA 드라이버 버전 확인
- CUDA 툴킷 설치 검증
- cuDNN 라이브러리 확인
- GPU 메모리 테스트
- 환경 문제 진단 및 해결 가이드

#### `run_finetuning.bat` - 파인튜닝 관리 스크립트
**기능**: 파인튜닝 작업 자동화
- 데이터셋 준비 및 검증
- 하이퍼파라미터 설정
- 학습 진행률 모니터링
- 체크포인트 관리
- 학습 완료 후 모델 변환

#### `small_model_options.py` - 경량 모델 옵션
**기능**: 저사양 환경을 위한 경량 모델 설정
- 3GB/4GB VRAM 환경 지원
- 모델 크기별 추천 설정
- 메모리 사용량 프로파일링
- 성능 vs 품질 트레이드오프 설정

### 📁 디렉토리 및 데이터 관리

#### `models/` - AI 모델 저장소
- **rtx3070_final_merged_q4km.gguf**: ⭐ 메인 Q4_K_M 양자화 모델 (4.1GB)
  - RTX 3070 최적화, GPU 35레이어 처리 가능
  - 추론 속도와 품질의 최적 균형점
- **rtx3070_final_merged.gguf**: 원본 FP16 GGUF 모델 (13.5GB)
  - 최고 품질이지만 큰 메모리 사용량
- **rtx3070_optimized_best/**: 메인 LoRA 어댑터 (HuggingFace 형식)
  - 파인튜닝 최종 결과물, 한국어 특화
- **rtx3070_optimized_final/**: 백업 LoRA 어댑터
- **rtx3070_checkpoint_epoch_6/**: 파인튜닝 체크포인트 (중간 저장)
- **Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf**: 기본 베이스 모델

#### `finetuning/` - 파인튜닝 개발 환경
- **mistral_english_finetuning.py**: 영어 파인튜닝 스크립트
- **korean_dataset.json**: 한국어 학습 데이터셋
- **training_config.yaml**: 하이퍼파라미터 설정
- **logs/**: 학습 로그 및 텐서보드 파일
- **checkpoints/**: 에폭별 체크포인트 저장
- **evaluation/**: 모델 성능 평가 결과

#### `llama.cpp/` - GGUF 변환 및 양자화 도구
- **build/bin/llama-quantize.exe**: GGUF 양자화 실행파일
- **build/bin/llama-server.exe**: GGUF 모델 추론 서버
- **convert_hf_to_gguf.py**: HuggingFace → GGUF 변환 스크립트
- **scripts/**: 각종 유틸리티 스크립트들
- **examples/**: 샘플 코드 및 예제들

#### `archive/` - 백업 및 버전 관리
- **log_settings.py.bak**: 이전 로그 설정 백업
- **mic_settings.py.bak**: 이전 마이크 설정 백업  
- **mic_settings.json.bak**: JSON 설정 백업
- 리팩터링 전 파일들의 안전한 보관

#### `deta/` - 음성 데이터 저장소
- **recorded_*.wav**: 음성 대화 녹음 파일들 (numbered sequentially)
- STT 처리용 임시 오디오 파일
- 음성 품질 테스트 샘플들
- VAD (Voice Activity Detection) 결과 데이터

#### `tts_output/` - TTS 출력 디렉토리
- 생성된 TTS 음성 파일들 (MP3/WAV)
- TTS 캐시 파일들 (.gitignore로 버전 관리 제외)
- 감정별/스타일별 음성 샘플들

#### `tts_cache/` - TTS 캐시 시스템
- 중복 TTS 생성 방지용 캐시
- 텍스트 해시 기반 빠른 검색
- 자동 캐시 정리 및 최적화

#### `OpenVoice/` - OpenVoice TTS 엔진 리소스
- **checkpoints/**: OpenVoice V2 체크포인트 및 설정
- **converter/**: 목소리 변환 모델
- **se_extractor/**: 화자 임베딩 추출기
- **base_speakers/**: 기본 화자 음성 데이터

#### `ksmevn/` - Python 가상환경 (E: 드라이브 전용)
- **Scripts/python.exe**: Python 3.10 실행파일
- **Lib/site-packages/**: 모든 의존성 패키지
- **include/**: C 헤더 파일들
- **CUDA 지원 패키지들**: llama-cpp-python, torch+cu118 등

### 📄 문서 및 설정 파일들

#### `ai_assistant_settings.json` - 통합 설정 파일
**기능**: 모든 시스템 설정을 하나의 JSON에서 관리
```json
{
  "model": {
    "current_model": "rtx3070_gguf",     // 현재 선택된 모델
    "gpu_layers": 35,                    // GPU에 로드할 레이어 수
    "context_length": 4096,              // 컨텍스트 윈도우 크기
    "temperature": 0.7,                  // 생성 온도 (창의성)
    "max_tokens": 512                    // 최대 응답 토큰 수
  },
  "audio": {
    "stt_model": "faster-whisper-base",  // STT 모델 선택
    "tts_engine": "openvoice",           // TTS 엔진 선택
    "auto_play": true,                   // 자동 음성 재생
    "voice_threshold": 0.5               // 음성 감지 임계값
  },
  "system": {
    "logging_level": "INFO",             // 로깅 레벨
    "cache_responses": true,             // 응답 캐싱 활성화
    "gpu_monitoring": true               // GPU 모니터링 활성화
  }
}
```

#### `log_settings.json` - 로깅 설정 파일
**기능**: 세부적인 로깅 카테고리 관리
```json
{
  "categories": {
    "model_loading": "INFO",     // 모델 로딩 관련 로그
    "conversation": "DEBUG",     // 대화 로그 (사용자-AI 교환)
    "system": "WARNING",         // 시스템 관련 로그
    "audio": "INFO",            // 음성 처리 로그
    "gpu": "INFO"               // GPU 관련 로그
  },
  "file_logging": true,         // 파일 로깅 활성화
  "console_logging": true,      // 콘솔 로깅 활성화
  "max_log_size": "100MB"       // 최대 로그 파일 크기
}
```

#### `README.md` - 프로젝트 문서 (현재 파일)
**기능**: 프로젝트 전체 가이드 및 기술 문서
- 프로젝트 개요 및 목적
- 상세한 파일별 설명 (현재 섹션)
- 설치 및 사용 가이드
- 모델 관리 및 최적화 방법
- 트러블슈팅 및 FAQ

#### `star ledd me.txt` - 사용자 가이드
**기능**: 일반 사용자를 위한 간단한 사용법
- 빠른 시작 가이드
- 주요 기능 설명
- 모델 선택 가이드
- 일반적인 문제 해결법

### 🔄 레거시 및 기타 파일들

#### `nous_hermes2_mistral_loader.py` - 기본 모델 로더 (레거시)
**기능**: Nous-Hermes-2-Mistral 기본 모델 로딩 (현재는 model_manager.py에 통합)
- GGUF 형식 모델 로딩 (Q5_K_M 양자화)
- GPU/CPU 자동 선택 및 최적화
- 컨텍스트 윈도우 및 배치 크기 설정
- 메모리 효율적인 로딩 (mmap 사용)

#### `finetuning_integration.py` - 파인튜닝 통합 (레거시)
**기능**: 파인튜닝된 모델들의 통합 관리 (현재는 model_manager.py에 통합)
- 도메인별 전문 모델 로딩
- 자동 도메인 분류 및 모델 선택
- 모델 성능 통계 및 모니터링
- 파인튜닝 모델 상태 관리

#### `hybrid_finetuning_integration.py` - 하이브리드 시스템 (레거시)
**기능**: 여러 파인튜닝 모델을 조합한 하이브리드 시스템 (현재는 model_manager.py에 통합)
- 통합형 모델과 전문형 모델 조합
- 실시간 모델 전환 및 응답 병합
- 한국어 특화 처리
- 성능 최적화 및 캐싱
    "current": "rtx3070_gguf",
    "auto_save": true,
    "available_models": ["original", "hybrid", "rtx3070_gguf"]
  },
  "logging": {
    "verbose_mode": false,
    "show_model_loading": true
  },
  "microphone": {
    "device_index": null,
    "auto_detect": true
  },
  "tts": {
    "cache_enabled": true,
    "cache_max_size": 100
  }
}
```

#### `README.md` - 프로젝트 문서
- 전체 프로젝트 개요 및 사용법
- 모듈화 구조 설명
- 설치 가이드 및 기술 스택
- GGUF 양자화 성과 및 성능 비교

#### `star ledd me.txt` - 개발 이력
- 개발 과정 상세 기록
- 버전별 변경사항
- 기술적 도전과 해결방법
- 성능 최적화 과정

#### `.gitignore` - Git 버전 관리 제외
- 모델 파일들 (용량 문제)
- TTS 출력 파일들
- 가상환경 디렉토리
- 로그 및 캐시 파일들

### 파인튜닝 시스템
```
finetuning_integration.py  # 파인튜닝 모델 통합 시스템
hybrid_finetuning_integration.py  # 하이브리드 모델 관리
run_finetuning.bat         # 파인튜닝 관리 스크립트
finetuning/                # 파인튜닝 환경 및 데이터
```

### 디렉토리 구조
```
models/                    # 학습된 모델 저장소
  ├── rtx3070_optimized_best/   # RTX 3070 최적화 모델 (메인)
  ├── rtx3070_optimized_final/  # RTX 3070 최적화 모델 (백업)
  └── Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf  # 기본 모델

deta/                      # 음성 녹음 파일 저장
tts_output/               # TTS 생성 음성 파일 (gitignore)
OpenVoice/                # OpenVoice V2 TTS 엔진 파일
ksmevn/                   # Python 가상환경
```

## 🤖 지원 모델 상세

### 1. 기본 베이스 모델 (Nous-Hermes-2-Mistral)
- **모델명**: Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf
- **크기**: 약 4.8GB (Q5_K_M 양자화)
- **특징**: 
  - 다국어 지원 범용 대화 모델
  - 높은 품질의 instruction following
  - DPO (Direct Preference Optimization) 적용
- **용도**: 일반적인 대화, 질문답변, 기본 추론
- **VRAM**: 약 5-6GB 사용
- **상태**: ✅ 완전 작동

### 2. ⭐ RTX 3070 GGUF 최적화 모델 (권장)
- **모델명**: rtx3070_final_merged_q4km.gguf
- **크기**: 약 4.1GB (Q4_K_M 양자화)
- **특징**:
  - RTX 3070 8GB VRAM 최적화
  - LoRA 어댑터 완전 병합 (LoRA-free 추론)
  - GPU 35레이어 처리 (CPU hybrid)
  - 한국어 특화 파인튜닝 적용
  - llama-cpp-python 기반 빠른 추론
- **성능**: 토큰/초 15-25 (GPU hybrid)
- **VRAM**: 약 6-7GB 사용
- **상태**: ✅ 완전 작동, 메인 추천 모델

### 3. RTX 3070 LoRA 모델 (레거시)
- **모델명**: rtx3070_optimized_best (HuggingFace LoRA)
- **특징**:
  - 영어/한국어 제한 (안전성 강화)
  - LoRA 어댑터 기반 (실시간 로딩 필요)
  - 4bit 양자화 BitsAndBytes
  - 정규식 기반 언어 감지 및 차단
- **VRAM**: 약 6GB 사용
- **상태**: ✅ 작동 (GGUF 모델 권장으로 레거시화)

### 4. 영어 전문 파인튜닝 모델 (레거시)
- **특징**: 영어 대화 및 QnA 특화
- **용도**: 영어 전용 고품질 대화
- **상태**: ✅ 작동 (GGUF 모델에 통합됨)

### 5. 하이브리드 파인튜닝 모델 (레거시)
- **특징**:
  - 한국어 특화 전문 모델 조합
  - 자동 도메인 분류 및 모델 선택
  - 통합 모델 + 전문 모델 하이브리드
- **상태**: ✅ 작동 (GGUF 모델에 통합됨)

## 🎮 사용법

### 빠른 시작
1. **모델 선택**: 메인 메뉴에서 `md` → **RTX3070 GGUF** 선택 (권장)
2. **음성 대화**: `s` → 마이크에 대고 말하기
3. **텍스트 대화**: `m` → 키보드로 질문 입력

### 메인 메뉴 상세
```
s: 음성대화       - 마이크로 실시간 대화 (스트리밍 TTS)
m: 텍스트대화     - 키보드로 대화 (빠른 응답)
p: 마이크설정     - 입력 장치 설정 및 테스트
md: 모델선택      - AI 모델 변경 (GGUF/LoRA/기본모델)
lg: 로그         - 상세 로그 on/off 토글
ft: 파인튜닝관리  - 하이브리드 모델 관리 (레거시)
ts: TTS통계      - TTS 캐시 효율 및 성능 확인
rt: 실시간통계    - 실시간 처리 성능 모니터링
st: 통합설정     - 모든 설정 통합 관리 ⭐
q: 종료          - 프로그램 안전 종료
```

### 📋 통합 설정 관리 (`st` 메뉴)
**새로운 통합 설정 관리 시스템**으로 모든 설정을 `ai_assistant_settings.json`에서 관리합니다.

#### 설정 관리 기능
- **설정 요약**: 현재 모든 설정값 한눈에 보기
- **설정 백업**: 현재 설정을 백업 파일로 저장
- **설정 복원**: 백업된 설정 파일로 복원
- **부분 초기화**: 특정 영역만 기본값으로 초기화
- **설정 파일 정보**: 설정 파일 위치 및 상태 확인

#### 관리되는 설정 영역
```json
{
  "model": {
    "current": "rtx3070_unfiltered",
    "auto_save": true,
    "last_used": { ... }
  },
  "logging": {
    "verbose_mode": false,
    "show_tts_debug": false,
    "show_model_loading": true
  },
  "microphone": {
    "device_index": null,
    "auto_detect": true,
    "sample_rate": 16000
  },
  "tts": {
    "cache_enabled": true,
    "cache_max_size": 100,
    "default_language": "English"
  },
  "conversation": {
    "history_max_length": 5,
    "context_window": 3
  },
  "realtime": {
    "enabled": true,
    "parallel_processing": true
  },
  "ui": {
    "show_tips": true,
    "compact_mode": false
  }
}
```

#### 설정 파일 통합 
- **이전**: `log_settings.py`, `mic_settings.py`, `mic_settings.json` 등 분산
- **현재**: `ai_assistant_settings.json` 하나로 통합
- **장점**: 
  - 설정 파일 관리 간소화
  - 백업/복원 편의성
  - 자동 저장 기능
  - 설정 변경 감지 및 로깅

### 모델 선택
1. `md` 입력으로 모델 선택 메뉴 진입
2. 원하는 모델 번호 선택
3. **RTX 3070 모델**: 영어/한국어 제한 확인 메시지
4. 모델 자동 로드 및 전환

### 음성 대화 (추천)
1. `s` 입력으로 음성 모드 시작
2. 마이크에 대고 자연스럽게 말하기
3. **실시간 스트리밍**: AI가 토큰 단위로 응답 + 문장 단위 TTS
4. "quit" 또는 "over"로 종료

### 언어 제한 기능
- **RTX 3070 모델**: 영어/한국어 외 언어 자동 차단
- **자동 감지**: 정규식 기반 언어 패턴 분석
- **안전성**: 지원하지 않는 언어 입력시 경고 메시지

## ⚙️ 기술 스택

### AI 프레임워크
- **LLM**: Transformers, PEFT (LoRA), BitsAndBytes
- **파인튜닝**: LoRA (Low-Rank Adaptation)
- **양자화**: 4bit NF4 quantization

### 음성 처리
- **STT**: Whisper (faster-whisper)
- **TTS**: OpenVoice V2
- **오디오**: SoundDevice, SoundFile

### GPU 최적화
- **CUDA**: 11.8+ 지원
- **메모리**: Dynamic loading, 4bit quantization
- **최적화**: RTX 3070 8GB 특화

## 🔧 고급 설정

### GPU 메모리 최적화
```python
# 4bit 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

### 언어 제한 시스템
```python
# 언어 패턴 감지
language_patterns = {
    'ko': re.compile(r'[\uAC00-\uD7A3]'),  # 한글
    'en': re.compile(r'[A-Za-z]'),          # 영어
    'other': re.compile(r'[\u4E00-\u9FFF...]')  # 기타 언어
}
```

### 실시간 스트리밍
```python
# 토큰 단위 출력 + 문장 단위 TTS
for token in chat_stream(llm, prompt):
    print(token, end="", flush=True)
    if sentence_end_detected(token):
        tts_queue.put(sentence)
```

## 🆕 최신 개발 현황 (2025-07-14)

### ✅ 완료된 작업
1. **완전 모듈화 리팩터링**
   - monolithic main.py → 5개 모듈로 분리
   - 유지보수성 및 디버깅 용이성 대폭 향상
   - 코드 재사용성 증대

2. **GGUF Q4_K_M 양자화 성공**
   - 13.5GB FP16 모델 → 4.1GB Q4_K_M 모델
   - RTX 3070에서 GPU 35레이어 실행 확인
   - 성능 손실 최소화하면서 VRAM 사용량 절반 이하로 감소

3. **실제 파인튜닝 모델 통합**
   - 기존 wrapper 모델이 아닌 실제 LoRA 어댑터 사용
   - LoRA → HuggingFace → GGUF 완전 변환 파이프라인 구축
   - llama-cpp-python을 통한 GPU 가속 추론 구현

4. **모델 인스턴스 문제 해결**
   - 전역 model_manager 인스턴스 공유 구현
   - preload와 runtime 모델 동기화 완료
   - 즉시 재로딩 폴백 시스템 구현

### 🔧 기술적 성과
- **GPU 메모리 효율성**: 8GB VRAM에서 4.1GB 모델로 여유 공간 확보
- **추론 속도**: CPU 대비 GPU 35레이어로 대폭 향상
- **모델 품질**: Q4_K_M 양자화로 최소한의 품질 손실
- **시스템 안정성**: 모듈화로 오류 격리 및 디버깅 용이

### 📊 성능 비교
```
원본 FP16 모델:    13.5GB (GPU 로드 불가)
Q4_K_M 양자화:     4.1GB  (GPU 35레이어 실행)
메모리 절약률:     약 70% 감소
추론 모드:         GPU 하이브리드 (35레이어)
```

## 📝 설치 가이드

### 1. 환경 준비
```bash
# CUDA 11.8+ 설치 확인
nvidia-smi

# Python 3.10 가상환경 생성
python -m venv ksmevn
.\ksmevn\Scripts\Activate.ps1
```

### 2. 모델 다운로드
- [Nous-Hermes-2-Mistral-7B-DPO GGUF](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF)에서 Q5_K_M.gguf 다운로드
- `models/` 폴더에 저장

### 3. OpenVoice 설정
- OpenVoice V2 체크포인트 및 리소스를 `OpenVoice/` 폴더에 준비

### 4. 실행
```bash
python main.py
```

## 📝 라이선스

이 프로젝트는 개인 연구 목적으로 개발되었습니다.

## 👨‍💻 개발자

RTX 3070 최적화 및 다중 모델 통합 시스템 개발

---
**마지막 업데이트**: 2025년 7월 8일  
**현재 상태**: RTX 3070 언어 제한 완료, 안정적인 다중 모델 시스템 구축 완료
