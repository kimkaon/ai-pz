"""
로그 및 유틸리티 함수들
main.py에서 분리된 로그 관련 기능
"""

import os
import re
from settings_manager import get_settings_manager

# 통합 설정 관리자 초기화
settings_manager = get_settings_manager()

def log_print(message, log_type="general"):
    """로그 출력 (통합 설정 관리자 기반)"""
    if settings_manager.should_show_log(log_type):
        print(message)

def is_verbose_mode():
    """로그 상세 모드 확인"""
    return settings_manager.is_verbose_mode()

def toggle_verbose_mode():
    """로그 모드 토글"""
    return {"verbose_mode": settings_manager.toggle_verbose_mode()}

def get_current_settings():
    """현재 설정 가져오기 (호환성 유지)"""
    return settings_manager.get_logging_settings()

def save_mic_index(index):
    """마이크 인덱스 저장 (호환성 유지)"""
    settings_manager.set_microphone_device(index)

def load_mic_index():
    """마이크 인덱스 로드 (호환성 유지)"""
    return settings_manager.get('microphone.device_index')

def clean_llm_response(response):
    """
    LLM 응답에서 불필요한 부분을 제거하여 TTS에 적합한 텍스트만 추출
    """
    if not response:
        return ""
    
    # 여러 가지 패턴으로 정리
    cleaned = response.strip()
    
    # <|im_start|>, <|im_end|> 태그 제거
    cleaned = re.sub(r'<\|im_start\|>\s*\w*\s*', '', cleaned)
    cleaned = re.sub(r'<\|im_end\|>', '', cleaned)
    
    # 반복되는 인사말이나 불필요한 문구 제거
    cleaned = re.sub(r'(Hi,?\s*)?(nice to meet you\.?\s*)?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Hello there!\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'I\'m glad we can chat today\.\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Let me know if you have any questions.*?\.\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'I look forward to being helpful\.\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Your success is important to me\.\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'🤗\s*', '', cleaned)  # 이모지 제거
    
    # 여러 줄에 걸친 처리를 위해 줄별로 분리
    lines = cleaned.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Type:, Classification:, Category: 패턴 제거
        if re.match(r'^(Type|Classification|Category)\s*:', line, flags=re.IGNORECASE):
            continue
            
        # 대괄호로 둘러싸인 분류만 있는 줄 제거
        if re.match(r'^\[.*?\]$', line):
            continue
            
        # 분류명만 있는 줄 제거
        if line.lower() in ['qna', 'daily chat', 'specific program', 'unknown']:
            continue
        
        # Answer:, QnA:, User:, Assistant: 등의 접두사 제거
        line = re.sub(r'^(answer|qna|daily chat|specific program|unknown|user|assistant)\s*:\s*', '', line, flags=re.IGNORECASE)
        
        # 대괄호로 둘러싸인 분류 정보 제거
        line = re.sub(r'^\[.*?\]\s*', '', line)
        
        if line:  # 빈 줄이 아닌 경우만 추가
            filtered_lines.append(line)
    
    # 줄들을 공백으로 연결
    cleaned = ' '.join(filtered_lines)
    
    # 연속된 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 양쪽 공백 제거
    cleaned = cleaned.strip()
    
    # 빈 문자열이면 기본값 반환
    if not cleaned:
        return "Sorry, I couldn't generate a proper response."
    
    return cleaned
