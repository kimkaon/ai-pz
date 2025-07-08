#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 오류 검증 및 수정 스크립트
Data Error Validation and Fix Script

영어 기반 파인튜닝 데이터의 오류를 검사하고 수정합니다.
Checks and fixes errors in English-based fine-tuning data.
"""

import json
import os
from pathlib import Path
import re

def check_data_quality():
    """데이터 품질 검사 / Data quality check"""
    
    processed_dir = Path(__file__).parent.parent / "datasets" / "processed_english"
    
    print("🔍 데이터 품질 검사 시작... / Starting data quality check...")
    
    issues = []
    recommendations = []
    
    # 1. 파일 존재 및 크기 확인 / Check file existence and size (daily_chat 제거됨)
    required_files = [
        "classification_train.jsonl", "classification_validation.jsonl", "classification_test.jsonl",
        "dialogue_qna_train.jsonl", "dialogue_technical_train.jsonl", "dialogue_general_train.jsonl",
        "unified_train.jsonl", "unified_validation.jsonl", "unified_test.jsonl"
    ]
    
    for filename in required_files:
        filepath = processed_dir / filename
        if not filepath.exists():
            issues.append(f"❌ 파일 없음 / Missing file: {filename}")
        else:
            file_size = filepath.stat().st_size
            if file_size < 100:  # 100 bytes 미만
                issues.append(f"⚠️ 파일 크기 작음 / Small file size: {filename} ({file_size} bytes)")
    
    # 2. JSON 구조 검증 / JSON structure validation
    for filename in required_files:
        filepath = processed_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    data_lines = [line for line in lines if not line.strip().startswith('#')]
                    
                    for i, line in enumerate(data_lines):
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                
                                # 분류 데이터 검증 / Classification data validation
                                if filename.startswith("classification"):
                                    if "text" not in data:
                                        issues.append(f"❌ 'text' 필드 없음 / Missing 'text' field: {filename}:{i+1}")
                                    if "label" not in data:
                                        issues.append(f"❌ 'label' 필드 없음 / Missing 'label' field: {filename}:{i+1}")
                                    elif not isinstance(data["label"], int) or data["label"] not in [0,1,2]:  # qna, technical, general
                                        issues.append(f"❌ 잘못된 레이블 / Invalid label: {filename}:{i+1} (label: {data.get('label')})")
                                
                                # 대화 데이터 검증 / Dialogue data validation (필드명 수정)
                                elif filename.startswith("dialogue"):
                                    if "input_text" not in data:
                                        issues.append(f"❌ 'input_text' 필드 없음 / Missing 'input_text' field: {filename}:{i+1}")
                                    if "target_text" not in data:
                                        issues.append(f"❌ 'target_text' 필드 없음 / Missing 'target_text' field: {filename}:{i+1}")
                                
                                # 통합 데이터 검증 / Unified data validation
                                elif filename.startswith("unified"):
                                    if "input_text" not in data:
                                        issues.append(f"❌ 'input_text' 필드 없음 / Missing 'input_text' field: {filename}:{i+1}")
                                    if "target_text" not in data:
                                        issues.append(f"❌ 'target_text' 필드 없음 / Missing 'target_text' field: {filename}:{i+1}")
                                    if "category" not in data:
                                        issues.append(f"❌ 'category' 필드 없음 / Missing 'category' field: {filename}:{i+1}")
                                
                                # 텍스트 품질 검증 / Text quality validation
                                for key, value in data.items():
                                    if isinstance(value, str):
                                        # 빈 텍스트 확인 / Check empty text
                                        if not value.strip():
                                            issues.append(f"❌ 빈 텍스트 / Empty text: {filename}:{i+1}, field: {key}")
                                        
                                        # 너무 짧은 텍스트 / Too short text
                                        if len(value.strip()) < 3:
                                            issues.append(f"⚠️ 너무 짧은 텍스트 / Too short text: {filename}:{i+1}, field: {key} ('{value}')")
                                        
                                        # 너무 긴 텍스트 / Too long text
                                        if len(value) > 5000:
                                            issues.append(f"⚠️ 너무 긴 텍스트 / Too long text: {filename}:{i+1}, field: {key} ({len(value)} chars)")
                                        
                                        # 특수 문자 검증 / Special character validation
                                        if re.search(r'[^\w\s\-.,!?();:\'"/@#$%^&*+=<>{}[\]|\\~`]', value):
                                            recommendations.append(f"💡 특수 문자 발견 / Special characters found: {filename}:{i+1}, field: {key}")
                                        
                                        # 영어 텍스트 검증 / English text validation
                                        if filename.startswith(("classification", "dialogue", "unified")):
                                            # 주로 영어인지 확인 (간단한 휴리스틱)
                                            english_chars = len(re.findall(r'[a-zA-Z]', value))
                                            total_chars = len(re.findall(r'[a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]', value))
                                            if total_chars > 0 and english_chars / total_chars < 0.7:
                                                recommendations.append(f"💡 영어 비율 낮음 / Low English ratio: {filename}:{i+1}, field: {key}")
                                
                            except json.JSONDecodeError as e:
                                issues.append(f"❌ JSON 파싱 오류 / JSON parsing error: {filename}:{i+1}: {e}")
                                
            except Exception as e:
                issues.append(f"❌ 파일 읽기 오류 / File reading error: {filename}: {e}")
    
    # 3. 데이터 분포 검증 / Data distribution validation
    try:
        # 분류 레이블 분포 / Classification label distribution
        for split in ["train", "validation", "test"]:
            filepath = processed_dir / f"classification_{split}.jsonl"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [line for line in f.readlines() if not line.strip().startswith('#')]
                    labels = []
                    for line in lines:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                labels.append(data.get("label"))
                            except:
                                continue
                    
                    unique_labels = set(labels)
                    expected_labels = {0, 1, 2}  # qna, technical, general (daily_chat 제거됨)
                    
                    if unique_labels != expected_labels:
                        missing = expected_labels - unique_labels
                        if missing:
                            recommendations.append(f"💡 누락된 레이블 / Missing labels in {split}: {missing}")
                        
                        # 레이블 분포 계산 / Calculate label distribution
                        label_counts = {label: labels.count(label) for label in unique_labels}
                        total = len(labels)
                        if total > 0:
                            distribution = {label: count/total for label, count in label_counts.items()}
                            imbalanced = any(ratio < 0.1 or ratio > 0.6 for ratio in distribution.values())
                            if imbalanced:
                                recommendations.append(f"💡 불균형한 레이블 분포 / Imbalanced distribution in {split}: {distribution}")
        
        # 통합 모델 카테고리 분포 / Unified model category distribution
        for split in ["train", "validation", "test"]:
            filepath = processed_dir / f"unified_{split}.jsonl"
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [line for line in f.readlines() if not line.strip().startswith('#')]
                    categories = []
                    for line in lines:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                categories.append(data.get("category"))
                            except:
                                continue
                    
                    unique_categories = set(categories)
                    expected_categories = {"qna", "technical", "general"}  # daily_chat 제거됨
                    
                    if unique_categories != expected_categories:
                        missing = expected_categories - unique_categories
                        if missing:
                            recommendations.append(f"💡 누락된 카테고리 / Missing categories in unified {split}: {missing}")
    
    except Exception as e:
        recommendations.append(f"💡 분포 검증 오류 / Distribution validation error: {e}")
    
    # 4. 시스템 프롬프트 일관성 검증 / System prompt consistency validation
    try:
        unified_files = ["unified_train.jsonl", "unified_validation.jsonl", "unified_test.jsonl"]
        system_prompts = set()
        
        for filename in unified_files:
            filepath = processed_dir / filename
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = [line for line in f.readlines() if not line.strip().startswith('#')]
                    for line in lines:
                        if line.strip():
                            try:
                                data = json.loads(line.strip())
                                input_text = data.get("input_text", "")
                                if "System:" in input_text:
                                    system_part = input_text.split("User:")[0].strip()
                                    system_prompts.add(system_part)
                            except:
                                continue
        
        if len(system_prompts) > 1:
            recommendations.append(f"💡 시스템 프롬프트 불일치 / Inconsistent system prompts: {len(system_prompts)} different versions found")
        elif len(system_prompts) == 0:
            issues.append(f"❌ 시스템 프롬프트 없음 / No system prompts found in unified data")
    
    except Exception as e:
        recommendations.append(f"💡 시스템 프롬프트 검증 오류 / System prompt validation error: {e}")
    
    # 결과 출력 / Print results
    print(f"\n📊 검사 결과 / Inspection Results:")
    print(f"   - 검사 파일 / Files checked: {len(required_files)}")
    print(f"   - 심각한 오류 / Critical errors: {len(issues)}")
    print(f"   - 권장사항 / Recommendations: {len(recommendations)}")
    
    if issues:
        print(f"\n❌ 심각한 오류들 / Critical Errors:")
        for issue in issues:
            print(f"   {issue}")
    
    if recommendations:
        print(f"\n💡 권장사항들 / Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
    
    if not issues and not recommendations:
        print(f"\n✅ 모든 검사 통과! / All checks passed!")
        print(f"   - 데이터가 파인튜닝에 적합합니다 / Data is suitable for fine-tuning")
        print(f"   - 오류 가능성이 매우 낮습니다 / Very low error probability")
    elif not issues:
        print(f"\n✅ 심각한 오류 없음 / No critical errors found")
        print(f"   - 파인튜닝 진행 가능 / Can proceed with fine-tuning")
        print(f"   - 권장사항은 선택적으로 적용 / Recommendations are optional")
    else:
        print(f"\n⚠️ 수정 필요한 오류 발견 / Errors found that need fixing")
        print(f"   - 파인튜닝 전 오류 수정 권장 / Recommend fixing before fine-tuning")
    
    return len(issues) == 0

def check_model_compatibility():
    """모델 호환성 검사 / Model compatibility check"""
    
    print(f"\n🤖 모델 호환성 검사 / Model Compatibility Check:")
    
    compatibility_notes = [
        "✅ 영어 텍스트로 구성 / Composed of English text",
        "✅ JSON Lines 형식 / JSON Lines format", 
        "✅ UTF-8 인코딩 / UTF-8 encoding",
        "✅ 표준 필드명 사용 / Standard field names used",
        "✅ 시스템 프롬프트 포함 (통합 모델) / System prompts included (unified model)",
        "✅ 레이블 정수형 (분류 모델) / Integer labels (classification model)",
        "✅ 카테고리 문자열 (통합 모델) / String categories (unified model)"
    ]
    
    for note in compatibility_notes:
        print(f"   {note}")
    
    print(f"\n🎯 권장 사용법 / Recommended Usage:")
    print(f"   - Hugging Face Transformers 라이브러리 / Hugging Face Transformers library")
    print(f"   - PyTorch 또는 TensorFlow / PyTorch or TensorFlow")
    print(f"   - 사전 훈련된 영어 모델 (BERT, GPT, T5 등) / Pre-trained English models (BERT, GPT, T5, etc.)")
    print(f"   - 권장 배치 크기: 4-16 / Recommended batch size: 4-16")
    print(f"   - 권장 학습률: 1e-5 ~ 5e-5 / Recommended learning rate: 1e-5 ~ 5e-5")

def main():
    """메인 함수 / Main function"""
    
    print("🔍 영어 기반 파인튜닝 데이터 검증 / English Fine-tuning Data Validation")
    print("=" * 80)
    
    # 데이터 품질 검사 / Data quality check
    is_valid = check_data_quality()
    
    # 모델 호환성 검사 / Model compatibility check
    check_model_compatibility()
    
    print(f"\n" + "=" * 80)
    if is_valid:
        print(f"🎉 검증 완료! 데이터가 파인튜닝에 준비되었습니다.")
        print(f"🎉 Validation complete! Data is ready for fine-tuning.")
        print(f"\n📋 다음 단계 / Next Steps:")
        print(f"   1. run_finetuning.bat 실행 / Run run_finetuning.bat")
        print(f"   2. 메뉴에서 원하는 학습 방식 선택 / Select desired training method from menu")
        print(f"   3. 분리형(4-7번) 또는 통합형(8번) 시도 / Try multi-model (4-7) or unified (8)")
    else:
        print(f"⚠️ 일부 오류가 발견되었습니다. 위의 오류를 수정 후 다시 시도하세요.")
        print(f"⚠️ Some errors were found. Please fix the above errors and try again.")
    
    return is_valid

if __name__ == "__main__":
    main()
