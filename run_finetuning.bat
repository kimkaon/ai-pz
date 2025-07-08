@echo off
rem 파인튜닝 환경 설정 및 실행 스크립트 (Windows)

echo ========================================
echo   AI PZ2 파인튜닝 환경 관리
echo ========================================

:menu
echo.
echo 1. 환경 설정 (패키지 설치)
echo 2. 기본 샘플 데이터 생성
echo 3. 풍부한 샘플 데이터 생성 (권장)
echo 4. 데이터 전처리
echo 5. 분류기 모델 파인튜닝
echo 6. QnA 전문모델 파인튜닝
echo 7. 일상대화 전문모델 파인튜닝
echo 8. 기술 전문모델 파인튜닝
echo 9. 통합형 모델 파인튜닝 (NEW!)
echo 10. 모델 평가
echo 11. 대화형 테스트
echo 12. 통합형 모델 테스트 (NEW!)
echo 0. 현재 상태 확인
echo q. 종료
echo.
set /p choice="선택하세요: "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto sample_data
if "%choice%"=="3" goto rich_data
if "%choice%"=="4" goto preprocess
if "%choice%"=="5" goto train_classifier
if "%choice%"=="6" goto train_qna
if "%choice%"=="7" goto train_daily
if "%choice%"=="8" goto train_technical
if "%choice%"=="9" goto train_unified
if "%choice%"=="10" goto evaluate
if "%choice%"=="11" goto test
if "%choice%"=="12" goto test_unified
if "%choice%"=="0" goto status
if "%choice%"=="q" goto exit
if "%choice%"=="Q" goto exit

echo 잘못된 선택입니다.
goto menu

:setup
echo 환경 설정 중...
python setup_finetuning.py --setup
pause
goto menu

:sample_data
echo 기본 샘플 데이터 생성 중...
python setup_finetuning.py --sample-data
pause
goto menu

:rich_data
echo 풍부한 샘플 데이터 생성 중...
echo 분리형 모델과 통합형 모델 모두에 사용할 수 있는 데이터를 생성합니다.
python setup_finetuning.py --rich-data
pause
goto menu

:preprocess
echo 데이터 전처리 중...
cd finetuning
python scripts/data_preprocessing.py --input_dir datasets/raw
cd ..
pause
goto menu

:train_classifier
echo 분류기 모델 파인튜닝 중...
cd finetuning
python scripts/fine_tuning.py --model_type conversation_classifier
cd ..
echo 파인튜닝 완료!
pause
goto menu

:train_qna
echo QnA 전문모델 파인튜닝 중...
cd finetuning
python scripts/fine_tuning.py --model_type qna_specialist
cd ..
echo 파인튜닝 완료!
pause
goto menu

:train_daily
echo 일상대화는 이제 통합모델에서 처리됩니다. 통합모델을 학습해주세요.
echo Daily chat is now handled by the unified model. Please train the unified model instead.
pause
goto menu

:train_technical
echo 기술 전문모델 파인튜닝 중...
cd finetuning
python scripts/fine_tuning.py --model_type technical_specialist
cd ..
echo 파인튜닝 완료!
pause
goto menu

:train_unified
echo 통합형 모델 파인튜닝 중...
echo 하나의 모델로 모든 대화 유형을 처리합니다.
cd finetuning
python scripts/train_unified_model.py
cd ..
echo 통합형 모델 파인튜닝 완료!
pause
goto menu

:evaluate
echo.
echo 평가할 모델을 선택하세요:
echo 1. 분류기 모델
echo 2. QnA 전문모델
echo 3. 일상대화 전문모델
echo 4. 기술 전문모델
echo 5. 통합형 모델
set /p eval_choice="선택: "

cd finetuning

if "%eval_choice%"=="1" (
    python scripts/model_evaluation.py --model_path models/conversation_classifier --model_type classification --test_data datasets/processed/classification_test.jsonl
)
if "%eval_choice%"=="2" (
    python scripts/model_evaluation.py --model_path models/qna_specialist --model_type generation --test_data datasets/processed/dialogue_qna_test.jsonl
)
if "%eval_choice%"=="3" (
    echo 일상대화는 이제 통합모델에서 처리됩니다.
    echo Daily chat evaluation is now included in unified model evaluation.
)
if "%eval_choice%"=="4" (
    python scripts/model_evaluation.py --model_path models/technical_specialist --model_type generation --test_data datasets/processed/dialogue_technical_test.jsonl
)
if "%eval_choice%"=="5" (
    python scripts/model_evaluation.py --model_path models/unified_model --model_type generation --test_data datasets/processed/unified_test.jsonl
)

cd ..
echo 평가 완료!
pause
goto menu

:test
echo 대화형 테스트 시작 (분리형 모델)...
cd finetuning
python scripts/inference.py --mode interactive
cd ..
pause
goto menu

:test_unified
echo 통합형 모델 테스트 시작...
cd finetuning
python scripts/unified_inference.py --mode interactive
cd ..
pause
goto menu

:status
echo 현재 상태 확인 중...
python setup_finetuning.py --status
pause
goto menu

:exit
echo 종료합니다.
pause
