# log_settings.py
"""
로그 설정 관리 모듈
- 로그 표시 여부 설정 저장/로드
- 로그 메시지 출력 제어
"""
import json
import os

LOG_SETTINGS_FILE = "log_settings.json"

# 기본 로그 설정
DEFAULT_LOG_SETTINGS = {
    "verbose_mode": False,
    "show_tts_debug": False,
    "show_tts_status": False,
    "show_model_loading": False,
    "show_audio_processing": False
}

def load_log_settings():
    """로그 설정을 파일에서 로드"""
    try:
        if os.path.exists(LOG_SETTINGS_FILE):
            with open(LOG_SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                # 기본값과 병합 (새로운 설정이 추가된 경우를 대비)
                merged_settings = DEFAULT_LOG_SETTINGS.copy()
                merged_settings.update(settings)
                return merged_settings
        else:
            return DEFAULT_LOG_SETTINGS.copy()
    except Exception as e:
        print(f"로그 설정 로드 오류: {e}")
        return DEFAULT_LOG_SETTINGS.copy()

def save_log_settings(settings):
    """로그 설정을 파일에 저장"""
    try:
        with open(LOG_SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"로그 설정 저장 오류: {e}")
        return False

def toggle_verbose_mode():
    """상세 로그 모드 토글"""
    settings = load_log_settings()
    settings["verbose_mode"] = not settings["verbose_mode"]
    
    if settings["verbose_mode"]:
        # verbose 모드면 모든 로그 활성화
        settings["show_tts_debug"] = True
        settings["show_tts_status"] = True
        settings["show_model_loading"] = True
        settings["show_audio_processing"] = True
    else:
        # 비활성화면 모든 로그 숨김
        settings["show_tts_debug"] = False
        settings["show_tts_status"] = False
        settings["show_model_loading"] = False
        settings["show_audio_processing"] = False
    
    save_log_settings(settings)
    return settings

def is_verbose_mode():
    """현재 상세 로그 모드인지 확인"""
    settings = load_log_settings()
    return settings.get("verbose_mode", False)

def should_show_tts_debug():
    """TTS 디버그 메시지 표시 여부"""
    settings = load_log_settings()
    return settings.get("show_tts_debug", False)

def should_show_tts_status():
    """TTS 상태 메시지 표시 여부"""
    settings = load_log_settings()
    return settings.get("show_tts_status", False)

def should_show_model_loading():
    """모델 로딩 메시지 표시 여부"""
    settings = load_log_settings()
    return settings.get("show_model_loading", False)

def should_show_audio_processing():
    """오디오 처리 메시지 표시 여부"""
    settings = load_log_settings()
    return settings.get("show_audio_processing", False)

def log_print(message, log_type="general"):
    """
    조건부 로그 출력
    log_type: "tts_debug", "tts_status", "model_loading", "audio_processing", "general"
    """
    settings = load_log_settings()
    
    if not settings.get("verbose_mode", False):
        return  # verbose 모드가 아니면 아무것도 출력하지 않음
    
    should_print = False
    
    if log_type == "tts_debug":
        should_print = settings.get("show_tts_debug", False)
    elif log_type == "tts_status":
        should_print = settings.get("show_tts_status", False)
    elif log_type == "model_loading":
        should_print = settings.get("show_model_loading", False)
    elif log_type == "audio_processing":
        should_print = settings.get("show_audio_processing", False)
    else:
        should_print = True  # general 메시지는 verbose 모드에서 항상 출력
    
    if should_print:
        print(message)

def get_current_settings():
    """현재 로그 설정 반환"""
    return load_log_settings()
