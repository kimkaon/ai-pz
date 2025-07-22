# fast_tts.py
"""
초고속 TTS 시스템
- 캐싱 기능으로 반복 문구 빠른 재생
- edge-tts로 더 빠른 음성 생성
- 메모리 기반 캐시와 디스크 캐시 병용
"""
import os
import sys
import hashlib
import asyncio
import threading
import time
import json
from typing import Dict, Optional, Tuple
from pathlib import Path
import tempfile

# 통합 설정 관리자 임포트
try:
    from settings_manager import get_settings_manager
    settings_manager = get_settings_manager()
    
    def log_print(message, log_type="general"):
        """로그 출력 (호환성 유지)"""
        if settings_manager.should_show_log(log_type):
            print(message)
            
except ImportError:
    def log_print(message, log_type="general"):
        print(message)

# 선택적 라이브러리 import
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    log_print("edge-tts 없음 - gTTS 사용", "tts_status")

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    log_print("pyttsx3 없음", "tts_status")

try:
    from gtts import gTTS
    from pydub import AudioSegment
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    log_print("gTTS 없음", "tts_status")

class FastTTSCache:
    """초고속 TTS 캐시 시스템"""
    
    def __init__(self, cache_dir="tts_cache", max_memory_cache=50):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}  # 메모리 캐시
        self.max_memory_cache = max_memory_cache
        self.cache_stats = {"hits": 0, "misses": 0, "created": 0}
        
        # 캐시 메타데이터 로드
        self.meta_file = self.cache_dir / "cache_meta.json"
        self.load_cache_meta()
    
    def load_cache_meta(self):
        """캐시 메타데이터 로드"""
        if self.meta_file.exists():
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache_stats = data.get('stats', {"hits": 0, "misses": 0, "created": 0})
            except:
                pass
    
    def save_cache_meta(self):
        """캐시 메타데이터 저장"""
        try:
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'stats': self.cache_stats,
                    'last_update': time.time()
                }, f, indent=2)
        except:
            pass
    
    def get_cache_key(self, text: str, voice: str, language: str) -> str:
        """캐시 키 생성"""
        cache_input = f"{text}|{voice}|{language}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def get_cached_audio(self, text: str, voice: str, language: str) -> Optional[str]:
        """캐시에서 오디오 파일 경로 반환"""
        cache_key = self.get_cache_key(text, voice, language)
        
        # 1. 메모리 캐시 확인
        if cache_key in self.memory_cache:
            file_path = self.memory_cache[cache_key]
            if os.path.exists(file_path):
                self.cache_stats["hits"] += 1
                log_print(f"🎯 메모리 캐시 히트: {text[:20]}...", "tts_cache")
                return file_path
            else:
                del self.memory_cache[cache_key]
        
        # 2. 디스크 캐시 확인
        cache_file = self.cache_dir / f"{cache_key}.wav"
        if cache_file.exists():
            file_path = str(cache_file)
            # 메모리 캐시에 추가 (용량 제한)
            if len(self.memory_cache) < self.max_memory_cache:
                self.memory_cache[cache_key] = file_path
            
            self.cache_stats["hits"] += 1
            log_print(f"🎯 디스크 캐시 히트: {text[:20]}...", "tts_cache")
            return file_path
        
        self.cache_stats["misses"] += 1
        return None
    
    def store_cached_audio(self, text: str, voice: str, language: str, audio_path: str):
        """오디오 파일을 캐시에 저장"""
        cache_key = self.get_cache_key(text, voice, language)
        cache_file = self.cache_dir / f"{cache_key}.wav"
        
        try:
            # 파일 복사
            import shutil
            shutil.copy2(audio_path, cache_file)
            
            # 메모리 캐시에 추가
            if len(self.memory_cache) < self.max_memory_cache:
                self.memory_cache[cache_key] = str(cache_file)
            
            self.cache_stats["created"] += 1
            log_print(f"💾 캐시 저장: {text[:20]}...", "tts_cache")
            
        except Exception as e:
            log_print(f"캐시 저장 오류: {e}", "tts_cache")
    
    def get_stats(self) -> Dict:
        """캐시 통계 반환"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate": f"{hit_rate:.1f}%",
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_size": len(list(self.cache_dir.glob("*.wav")))
        }

class UltraFastTTS:
    """초고속 TTS 엔진"""
    
    def __init__(self, cache_dir="tts_cache"):
        self.cache = FastTTSCache(cache_dir)
        self.default_voice = "en-US-AriaNeural"  # Edge TTS 기본 음성
        self.pyttsx3_engine = None
        
        # pyttsx3 엔진 초기화 (로컬 TTS)
        if PYTTSX3_AVAILABLE:
            try:
                self.pyttsx3_engine = pyttsx3.init()
                self.pyttsx3_engine.setProperty('rate', 180)  # 속도 향상
                self.pyttsx3_engine.setProperty('volume', 0.9)
                log_print("✅ pyttsx3 엔진 초기화 완료", "tts_status")
            except:
                self.pyttsx3_engine = None
                log_print("❌ pyttsx3 엔진 초기화 실패", "tts_status")
    
    async def synthesize_with_edge_tts(self, text: str, output_path: str, voice: str = None) -> bool:
        """Edge TTS로 음성 합성 (가장 빠름)"""
        if not EDGE_TTS_AVAILABLE:
            return False
        
        try:
            voice = voice or self.default_voice
            communicate = edge_tts.Communicate(text, voice)
            
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name
            
            await communicate.save(temp_path)
            
            # MP3를 WAV로 변환 (필요시)
            if output_path.endswith('.wav'):
                if GTTS_AVAILABLE:
                    audio = AudioSegment.from_mp3(temp_path)
                    audio = audio.set_frame_rate(22050).set_channels(1)
                    audio.export(output_path, format="wav")
                else:
                    # 단순 복사 (확장자만 변경)
                    import shutil
                    shutil.copy2(temp_path, output_path)
            else:
                import shutil
                shutil.copy2(temp_path, output_path)
            
            # 임시 파일 정리
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return True
            
        except Exception as e:
            log_print(f"Edge TTS 오류: {e}", "tts_debug")
            return False
    
    def synthesize_with_pyttsx3(self, text: str, output_path: str) -> bool:
        """pyttsx3로 음성 합성 (로컬, 매우 빠름)"""
        if not self.pyttsx3_engine:
            return False
        
        try:
            # pyttsx3은 WAV 직접 저장 지원
            self.pyttsx3_engine.save_to_file(text, output_path)
            self.pyttsx3_engine.runAndWait()
            
            return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
        except Exception as e:
            log_print(f"pyttsx3 오류: {e}", "tts_debug")
            return False
    
    def synthesize_with_gtts(self, text: str, output_path: str, language: str = "en") -> bool:
        """gTTS로 음성 합성 (폴백)"""
        if not GTTS_AVAILABLE:
            return False
        
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            
            # 임시 mp3 파일
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name
                tts.save(temp_mp3_path)
            
            # mp3를 wav로 변환
            audio = AudioSegment.from_mp3(temp_mp3_path)
            audio = audio.set_frame_rate(22050).set_channels(1)
            audio.export(output_path, format="wav")
            
            # 임시 파일 정리
            if os.path.exists(temp_mp3_path):
                os.remove(temp_mp3_path)
            
            return True
            
        except Exception as e:
            log_print(f"gTTS 오류: {e}", "tts_debug")
            return False
    
    async def synthesize_ultra_fast(self, text: str, output_path: str, 
                                   voice: str = None, language: str = "English") -> str:
        """초고속 TTS 합성 (캐시 + 최적 엔진 선택)"""
        if not text.strip():
            return ""
        
        # 1. 캐시 확인
        voice = voice or self.default_voice
        lang_code = "en" if language == "English" else "ko"
        
        cached_path = self.cache.get_cached_audio(text, voice, language)
        if cached_path:
            # 캐시된 파일을 출력 경로로 복사
            import shutil
            shutil.copy2(cached_path, output_path)
            return output_path
        
        # 2. 엔진 우선순위: Edge TTS → pyttsx3 → gTTS
        success = False
        
        # Edge TTS 시도 (가장 빠름, 고품질)
        if EDGE_TTS_AVAILABLE:
            log_print("🚀 Edge TTS로 음성 생성 중...", "tts_status")
            success = await self.synthesize_with_edge_tts(text, output_path, voice)
            if success:
                log_print("✅ Edge TTS 완료", "tts_status")
        
        # pyttsx3 시도 (로컬, 매우 빠름)
        if not success and PYTTSX3_AVAILABLE:
            log_print("🏠 pyttsx3로 음성 생성 중...", "tts_status")
            success = self.synthesize_with_pyttsx3(text, output_path)
            if success:
                log_print("✅ pyttsx3 완료", "tts_status")
        
        # gTTS 시도 (폴백)
        if not success and GTTS_AVAILABLE:
            log_print("🌐 gTTS로 음성 생성 중...", "tts_status")
            success = self.synthesize_with_gtts(text, output_path, lang_code)
            if success:
                log_print("✅ gTTS 완료", "tts_status")
        
        if success:
            # 3. 캐시에 저장
            self.cache.store_cached_audio(text, voice, language, output_path)
            return output_path
        else:
            raise Exception("모든 TTS 엔진 실패")
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 반환"""
        return self.cache.get_stats()

# 전역 초고속 TTS 인스턴스
_ultra_fast_tts = None

def get_ultra_fast_tts() -> UltraFastTTS:
    """초고속 TTS 인스턴스 반환 (싱글톤)"""
    global _ultra_fast_tts
    if _ultra_fast_tts is None:
        _ultra_fast_tts = UltraFastTTS()
    return _ultra_fast_tts

def synthesize_ultra_fast(text: str, output_path: str, 
                         voice: str = None, language: str = "English") -> str:
    """초고속 TTS 합성 (동기 래퍼)"""
    tts = get_ultra_fast_tts()
    
    try:
        # 현재 스레드가 메인 스레드인지 확인
        import threading
        current_thread = threading.current_thread()
        
        # 워커 스레드에서는 새로운 이벤트 루프 생성
        if current_thread.name != "MainThread":
            # 새로운 이벤트 루프 생성
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    tts.synthesize_ultra_fast(text, output_path, voice, language)
                )
                return result
            finally:
                loop.close()
        else:
            # 메인 스레드에서는 기존 방식 사용
            try:
                # 기존 이벤트 루프가 있는지 확인
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(tts.synthesize_ultra_fast(text, output_path, voice, language))
                        )
                        return future.result()
                else:
                    # 루프가 실행 중이 아니면 직접 실행
                    return asyncio.run(tts.synthesize_ultra_fast(text, output_path, voice, language))
            except RuntimeError:
                # 이벤트 루프 없으면 새로 생성
                return asyncio.run(tts.synthesize_ultra_fast(text, output_path, voice, language))
    
    except Exception as e:
        log_print(f"초고속 TTS 오류: {e}", "tts_debug")
        
        # 오류 시 fallback으로 동기 TTS 사용
        try:
            return _fallback_sync_tts(text, output_path, language)
        except Exception as fallback_error:
            log_print(f"Fallback TTS 오류: {fallback_error}", "tts_debug")
            raise

def _fallback_sync_tts(text: str, output_path: str, language: str = "English") -> str:
    """동기 TTS 대체 함수 (비동기 실패시 사용)"""
    try:
        # pyttsx3 사용 (완전 동기)
        if PYTTSX3_AVAILABLE:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.setProperty('volume', 0.9)
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return output_path
        
        # gTTS 사용 (동기 처리)
        if GTTS_AVAILABLE:
            from gtts import gTTS
            from pydub import AudioSegment
            import tempfile
            
            lang_code = "en" if language == "English" else "ko"
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # 임시 mp3 파일
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
                temp_mp3_path = temp_mp3.name
                tts.save(temp_mp3_path)
            
            try:
                # mp3를 wav로 변환
                audio = AudioSegment.from_mp3(temp_mp3_path)
                audio = audio.set_frame_rate(22050).set_channels(1)
                audio.export(output_path, format="wav")
                
                return output_path
                
            finally:
                # 임시 파일 정리
                if os.path.exists(temp_mp3_path):
                    os.remove(temp_mp3_path)
        
        raise Exception("사용 가능한 TTS 엔진이 없습니다")
        
    except Exception as e:
        log_print(f"Fallback TTS 실패: {e}", "tts_debug")
        raise

def get_tts_stats() -> Dict:
    """TTS 캐시 통계 반환"""
    tts = get_ultra_fast_tts()
    return tts.get_cache_stats()

# 필요한 패키지 설치 명령어 출력
if __name__ == "__main__":
    print("초고속 TTS 시스템 - 필요한 패키지:")
    print("pip install edge-tts pyttsx3 gtts pydub")
    
    # 테스트
    tts = get_ultra_fast_tts()
    print(f"캐시 통계: {tts.get_cache_stats()}")
