# settings_manager.py
"""
통합 설정 관리 시스템
- 모든 설정을 하나의 JSON 파일로 관리
- 모델 선택, 로그 설정, 마이크 설정, TTS 설정 등
"""
import json
import os
from typing import Dict, Any, Optional

SETTINGS_FILE = "ai_assistant_settings.json"

# 기본 설정값
DEFAULT_SETTINGS = {
    "version": "1.0",
    "last_updated": None,
    
    # 모델 설정
    "model": {
        "current": "original",
        "auto_save": True,
        "last_used": {
            "original": True,
            "hybrid": False,
            "english_unified": False,
            "rtx3070_unfiltered": False,
            "rtx3070_language_limited": False
        }
    },
    
    # 로그 설정
    "logging": {
        "verbose_mode": False,
        "show_tts_debug": False,
        "show_tts_status": True,
        "show_model_loading": True,
        "show_audio_processing": True,
        "show_realtime_chat": False,
        "show_general": True
    },
    
    # 마이크 설정
    "microphone": {
        "device_index": None,
        "auto_detect": True,
        "sample_rate": 16000,
        "channels": 1
    },
    
    # TTS 설정
    "tts": {
        "cache_enabled": True,
        "cache_max_size": 100,
        "default_language": "English",
        "voice": "en-US-AriaNeural",
        "speed": 1.0,
        "volume": 0.9
    },
    
    # 대화 설정
    "conversation": {
        "history_max_length": 5,
        "auto_save_history": False,
        "context_window": 3
    },
    
    # 실시간 처리 설정
    "realtime": {
        "enabled": True,
        "parallel_processing": True,
        "sentence_streaming": True,
        "audio_buffering": True,
        "predictive_caching": True
    },
    
    # UI 설정
    "ui": {
        "show_tips": True,
        "show_model_info": True,
        "compact_mode": False,
        "color_theme": "default"
    }
}

class SettingsManager:
    """통합 설정 관리자"""
    
    def __init__(self):
        self._observers = []  # 설정 변경 감지자들
        self.settings = self.load_settings()
    
    def load_settings(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                
                # 기본값과 병합 (새 설정이 추가된 경우 대비)
                merged_settings = self._merge_settings(DEFAULT_SETTINGS, loaded_settings)
                
                # 버전 확인 및 마이그레이션
                if loaded_settings.get('version') != DEFAULT_SETTINGS['version']:
                    print(f"⬆️ 설정 파일 버전 업데이트: {loaded_settings.get('version', '0.0')} -> {DEFAULT_SETTINGS['version']}")
                    merged_settings['version'] = DEFAULT_SETTINGS['version']
                    self.save_settings(merged_settings)
                
                return merged_settings
            else:
                print("📝 기본 설정으로 설정 파일을 생성합니다.")
                self.save_settings(DEFAULT_SETTINGS)
                return DEFAULT_SETTINGS.copy()
                
        except Exception as e:
            print(f"❌ 설정 로드 실패: {e}")
            print("📝 기본 설정을 사용합니다.")
            return DEFAULT_SETTINGS.copy()
    
    def _merge_settings(self, default: Dict, loaded: Dict) -> Dict:
        """기본 설정과 로드된 설정을 병합"""
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged:
                if isinstance(value, dict) and isinstance(merged[key], dict):
                    merged[key] = self._merge_settings(merged[key], value)
                else:
                    merged[key] = value
            else:
                merged[key] = value
        
        return merged
    
    def save_settings(self, settings: Optional[Dict] = None):
        """설정 파일 저장"""
        if settings is None:
            settings = self.settings
        
        try:
            import time
            settings['last_updated'] = time.time()
            
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            
            # 옵저버들에게 알림
            for observer in self._observers:
                try:
                    observer(settings)
                except:
                    pass
                    
        except Exception as e:
            print(f"❌ 설정 저장 실패: {e}")
    
    def get(self, path: str, default=None):
        """설정값 가져오기 (점으로 구분된 경로 지원)"""
        keys = path.split('.')
        value = self.settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any, auto_save: bool = True):
        """설정값 설정하기 (점으로 구분된 경로 지원)"""
        keys = path.split('.')
        current = self.settings
        
        # 마지막 키를 제외하고 경로 생성
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        
        # 값 설정
        current[keys[-1]] = value
        
        if auto_save:
            self.save_settings()
    
    def add_observer(self, callback):
        """설정 변경 감지자 추가"""
        self._observers.append(callback)
    
    def remove_observer(self, callback):
        """설정 변경 감지자 제거"""
        if callback in self._observers:
            self._observers.remove(callback)
    
    # 편의 메서드들
    def get_model_settings(self) -> Dict:
        """모델 설정 가져오기"""
        return self.get('model', {})
    
    def set_current_model(self, model_name: str):
        """현재 모델 설정"""
        self.set('model.current', model_name)
        if self.get('model.auto_save', True):
            # 사용한 모델 기록
            self.set(f'model.last_used.{model_name}', True, auto_save=False)
            self.save_settings()
    
    def get_logging_settings(self) -> Dict:
        """로그 설정 가져오기"""
        return self.get('logging', {})
    
    def is_verbose_mode(self) -> bool:
        """상세 로그 모드 여부"""
        return self.get('logging.verbose_mode', False)
    
    def toggle_verbose_mode(self) -> bool:
        """상세 로그 모드 토글"""
        current = self.is_verbose_mode()
        self.set('logging.verbose_mode', not current)
        return not current
    
    def should_show_log(self, log_type: str) -> bool:
        """특정 로그 타입 표시 여부"""
        return self.get(f'logging.show_{log_type}', True)
    
    def get_microphone_settings(self) -> Dict:
        """마이크 설정 가져오기"""
        return self.get('microphone', {})
    
    def set_microphone_device(self, device_index: int):
        """마이크 장치 설정"""
        self.set('microphone.device_index', device_index)
    
    def get_tts_settings(self) -> Dict:
        """TTS 설정 가져오기"""
        return self.get('tts', {})
    
    def get_conversation_settings(self) -> Dict:
        """대화 설정 가져오기"""
        return self.get('conversation', {})
    
    def get_realtime_settings(self) -> Dict:
        """실시간 처리 설정 가져오기"""
        return self.get('realtime', {})
    
    def get_ui_settings(self) -> Dict:
        """UI 설정 가져오기"""
        return self.get('ui', {})
    
    def export_settings(self, file_path: str = None) -> str:
        """설정 내보내기"""
        if file_path is None:
            import time
            timestamp = int(time.time())
            file_path = f"ai_assistant_settings_backup_{timestamp}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            return file_path
        except Exception as e:
            raise Exception(f"설정 내보내기 실패: {e}")
    
    def import_settings(self, file_path: str):
        """설정 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_settings = json.load(f)
            
            # 병합 후 저장
            self.settings = self._merge_settings(DEFAULT_SETTINGS, imported_settings)
            self.save_settings()
            
        except Exception as e:
            raise Exception(f"설정 가져오기 실패: {e}")
    
    def reset_to_defaults(self, section: str = None):
        """설정 초기화"""
        if section is None:
            # 전체 초기화
            self.settings = DEFAULT_SETTINGS.copy()
        else:
            # 특정 섹션만 초기화
            if section in DEFAULT_SETTINGS:
                self.settings[section] = DEFAULT_SETTINGS[section].copy()
        
        self.save_settings()
    
    def get_settings_summary(self) -> str:
        """설정 요약 정보"""
        model = self.get('model.current', 'unknown')
        verbose = "ON" if self.is_verbose_mode() else "OFF"
        mic_device = self.get('microphone.device_index')
        mic_status = f"장치 {mic_device}" if mic_device is not None else "자동 감지"
        
        summary = f"""
📋 AI 어시스턴트 설정 요약
{'='*40}
🤖 활성 모델: {model}
📝 상세 로그: {verbose}
🎤 마이크: {mic_status}
🔊 TTS 캐시: {'ON' if self.get('tts.cache_enabled') else 'OFF'}
⚡ 실시간 처리: {'ON' if self.get('realtime.enabled') else 'OFF'}
💡 UI 팁: {'ON' if self.get('ui.show_tips') else 'OFF'}
"""
        return summary

# 전역 설정 관리자 인스턴스
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """설정 관리자 싱글톤 반환"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

# 편의 함수들 (기존 코드와의 호환성)
def load_settings():
    """설정 로드 (호환성 유지)"""
    return get_settings_manager().settings

def save_settings(settings=None):
    """설정 저장 (호환성 유지)"""
    if settings is not None:
        get_settings_manager().settings = settings
    get_settings_manager().save_settings()

def is_verbose_mode():
    """상세 로그 모드 여부 (호환성 유지)"""
    return get_settings_manager().is_verbose_mode()

def toggle_verbose_mode():
    """상세 로그 모드 토글 (호환성 유지)"""
    manager = get_settings_manager()
    new_state = manager.toggle_verbose_mode()
    return {"verbose_mode": new_state}

if __name__ == "__main__":
    # 테스트 코드
    manager = SettingsManager()
    print("🧪 설정 관리자 테스트")
    print(manager.get_settings_summary())
    
    # 설정 변경 테스트
    manager.set_current_model("rtx3070_unfiltered")
    manager.toggle_verbose_mode()
    print("\n📝 설정 변경 후:")
    print(manager.get_settings_summary())
