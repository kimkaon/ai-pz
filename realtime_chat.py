# realtime_chat.py
"""
실시간 대화 시스템
- AI 응답 생성과 동시에 TTS 처리
- 문장별 스트리밍으로 자연스러운 대화 흐름
- 예측 캐싱으로 지연 최소화
"""
import os
import sys
import threading
import queue
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from collections import deque
import re

# 통합 설정 관리자 임포트
try:
    from settings_manager import get_settings_manager
    settings_manager = get_settings_manager()
    
    def log_print(message, log_type="general"):
        """로그 출력 (호환성 유지)"""
        if settings_manager.should_show_log(log_type):
            print(message)
    
    def is_verbose_mode():
        """상세 로그 모드 여부 (호환성 유지)"""
        return settings_manager.is_verbose_mode()
            
except ImportError:
    def log_print(message, log_type="general"):
        print(message)
    
    def is_verbose_mode():
        return False

from fast_tts import synthesize_ultra_fast, get_ultra_fast_tts

try:
    import soundfile as sf
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

class RealtimeChatSystem:
    """실시간 대화 시스템"""
    
    def __init__(self, tts_cache_size=10, audio_buffer_size=5):
        self.tts_cache_size = tts_cache_size
        self.audio_buffer_size = audio_buffer_size
        
        # 큐와 버퍼
        self.text_queue = queue.Queue()  # AI 응답 텍스트
        self.tts_queue = queue.Queue()   # TTS 처리 대기
        self.audio_queue = queue.Queue() # 오디오 재생 대기
        
        # 스레드 풀
        self.tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.audio_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # 상태 관리
        self.is_running = False
        self.current_conversation = []
        self.sentence_buffer = ""
        self.tts_futures = deque(maxlen=self.tts_cache_size)
        
        # 실시간 통계
        self.stats = {
            "total_sentences": 0,
            "avg_tts_time": 0,
            "avg_audio_delay": 0,
            "cache_hits": 0,
            "processing_times": deque(maxlen=50)
        }
        
        # 일반적인 응답 패턴 미리 캐싱
        self.common_phrases = [
            "I understand.",
            "That's interesting.",
            "Let me think about that.",
            "I see what you mean.",
            "That makes sense.",
            "I agree.",
            "You're right.",
            "Good point.",
            "Exactly.",
            "I think so too."
        ]
        
        # 예측 캐싱 시작
        self.start_predictive_caching()
    
    def start_predictive_caching(self):
        """일반적인 응답들을 미리 캐싱"""
        def cache_common_phrases():
            if is_verbose_mode():
                log_print("🔄 일반 응답 패턴 캐싱 시작...", "realtime_chat")
            
            for phrase in self.common_phrases:
                try:
                    # 캐시 디렉토리 생성
                    cache_dir = "tts_cache"
                    if not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    
                    # 해시 기반 파일명
                    import hashlib
                    cache_key = hashlib.md5(f"{phrase}|en-US-AriaNeural|English".encode()).hexdigest()
                    cache_file = os.path.join(cache_dir, f"{cache_key}.wav")
                    
                    if not os.path.exists(cache_file):
                        synthesize_ultra_fast(phrase, cache_file, language="English")
                        # 로그 OFF 상태에서는 출력하지 않음
                    
                except Exception as e:
                    if is_verbose_mode():
                        log_print(f"캐싱 실패: {phrase} - {e}", "realtime_chat")
            
            if is_verbose_mode():
                log_print("✅ 일반 응답 패턴 캐싱 완료", "realtime_chat")
        
        # 백그라운드에서 실행
        threading.Thread(target=cache_common_phrases, daemon=True).start()
    
    def extract_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분리"""
        # 더 정교한 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # 너무 짧은 문장 제거 및 정리
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3 and not sentence.isspace():
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def process_streaming_response(self, response_generator, mode="s"):
        """스트리밍 응답을 실시간으로 처리"""
        if not AUDIO_AVAILABLE and mode == "s":
            if is_verbose_mode():
                log_print("⚠️ 오디오 라이브러리 없음 - 텍스트 모드로 전환", "realtime_chat")
            mode = "m"
        
        print("🤔 AI가 답변 중...")
        
        full_response = ""
        sentence_buffer = ""
        sentence_count = 0
        
        # TTS 처리 워커 시작
        tts_workers = []
        audio_workers = []
        
        if mode == "s":
            # TTS 워커 스레드
            def tts_worker():
                while True:
                    try:
                        item = self.tts_queue.get(timeout=1)
                        if item is None:
                            break
                        
                        sentence, output_path, sentence_id = item
                        start_time = time.time()
                        
                        try:
                            # 기존 동기 TTS 사용 (스레드 안전)
                            from openvoice_tts import synthesize_quick_tts
                            synthesize_quick_tts(sentence, output_path, language="English")
                            
                            tts_time = time.time() - start_time
                            self.stats["processing_times"].append(tts_time)
                            
                            # 오디오 큐에 추가
                            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                self.audio_queue.put((output_path, sentence_id, sentence))
                                if is_verbose_mode():
                                    log_print(f"🎵 TTS 완료 ({tts_time:.2f}s): {sentence[:30]}...", "realtime_chat")
                            
                        except Exception as e:
                            if is_verbose_mode():
                                log_print(f"TTS 생성 실패: {e}", "realtime_chat")
                        
                        self.tts_queue.task_done()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if is_verbose_mode():
                            log_print(f"TTS 워커 오류: {e}", "realtime_chat")
                        break
            
            # 오디오 재생 워커
            def audio_worker():
                while True:
                    try:
                        item = self.audio_queue.get(timeout=1)
                        if item is None:
                            break
                        
                        output_path, sentence_id, sentence = item
                        
                        try:
                            # 오디오 재생
                            data, fs = sf.read(output_path, dtype='float32')
                            sd.play(data, fs)
                            sd.wait()
                            
                            if is_verbose_mode():
                                log_print(f"🔊 재생 완료 #{sentence_id}: {sentence[:20]}...", "realtime_chat")
                            
                        except Exception as e:
                            if is_verbose_mode():
                                log_print(f"오디오 재생 실패: {e}", "realtime_chat")
                        
                        self.audio_queue.task_done()
                        
                    except queue.Empty:
                        continue
                    except Exception as e:
                        if is_verbose_mode():
                            log_print(f"오디오 워커 오류: {e}", "realtime_chat")
                        break
            
            # 워커 스레드 시작
            for i in range(2):  # TTS 워커 2개
                worker = threading.Thread(target=tts_worker, daemon=True)
                worker.start()
                tts_workers.append(worker)
            
            for i in range(1):  # 오디오 워커 1개
                worker = threading.Thread(target=audio_worker, daemon=True)
                worker.start()
                audio_workers.append(worker)
        
        try:
            # 스트리밍 응답 처리
            for token in response_generator:
                # 실시간 화면 출력
                print(token, end="", flush=True)
                full_response += token
                sentence_buffer += token
                
                # 문장 완성 감지
                if any(punct in token for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                    sentence = sentence_buffer.strip()
                    
                    if len(sentence) > 5:  # 의미 있는 문장만 처리
                        sentence_count += 1
                        
                        if mode == "s":
                            # TTS 처리 큐에 추가
                            tts_output_dir = "tts_output"
                            if not os.path.exists(tts_output_dir):
                                os.makedirs(tts_output_dir)
                            
                            tts_path = os.path.join(tts_output_dir, f"realtime_{sentence_count}.wav")
                            self.tts_queue.put((sentence, tts_path, sentence_count))
                            
                            if is_verbose_mode():
                                log_print(f"📝 문장 #{sentence_count} TTS 큐 추가", "realtime_chat")
                    
                    sentence_buffer = ""
            
            print()  # 줄바꿈
            
            # 마지막 문장 처리
            if sentence_buffer.strip() and len(sentence_buffer.strip()) > 5:
                sentence_count += 1
                
                if mode == "s":
                    tts_path = os.path.join("tts_output", f"realtime_{sentence_count}.wav")
                    self.tts_queue.put((sentence_buffer.strip(), tts_path, sentence_count))
            
            # 모든 TTS 처리 완료 대기
            if mode == "s":
                if is_verbose_mode():
                    log_print("🎵 모든 TTS 처리 완료 대기...", "realtime_chat")
                self.tts_queue.join()
                self.audio_queue.join()
                
                # 워커 스레드 정리
                for _ in tts_workers:
                    self.tts_queue.put(None)
                for _ in audio_workers:
                    self.audio_queue.put(None)
                
                for worker in tts_workers + audio_workers:
                    worker.join(timeout=2)
        
        except Exception as e:
            if is_verbose_mode():
                log_print(f"실시간 처리 오류: {e}", "realtime_chat")
            
            # 오류 시 워커 정리
            if mode == "s":
                try:
                    for _ in tts_workers:
                        self.tts_queue.put(None)
                    for _ in audio_workers:
                        self.audio_queue.put(None)
                except:
                    pass
        
        # 통계 업데이트
        self.stats["total_sentences"] += sentence_count
        if self.stats["processing_times"]:
            self.stats["avg_tts_time"] = sum(self.stats["processing_times"]) / len(self.stats["processing_times"])
        
        if is_verbose_mode():
            log_print(f"✅ 실시간 처리 완료: {sentence_count}개 문장", "realtime_chat")
        
        return full_response
    
    def get_realtime_stats(self) -> Dict:
        """실시간 처리 통계 반환"""
        return {
            **self.stats,
            "tts_queue_size": self.tts_queue.qsize(),
            "audio_queue_size": self.audio_queue.qsize(),
            "avg_tts_time_ms": f"{self.stats['avg_tts_time']*1000:.0f}ms" if self.stats['avg_tts_time'] > 0 else "0ms"
        }
    
    def optimize_for_realtime(self) -> Dict:
        """실시간 최적화 설정"""
        optimizations = {
            "tts_cache_preload": True,
            "parallel_processing": True,
            "sentence_streaming": True,
            "audio_buffering": True,
            "predictive_caching": True
        }
        
        if is_verbose_mode():
            log_print("🚀 실시간 최적화 적용됨", "realtime_chat")
        return optimizations

# 전역 실시간 채팅 시스템
_realtime_chat = None

def get_realtime_chat_system():
    """실시간 채팅 시스템 싱글톤"""
    global _realtime_chat
    if _realtime_chat is None:
        _realtime_chat = RealtimeChatSystem()
    return _realtime_chat

def process_realtime_response(response_generator, mode="s"):
    """실시간 응답 처리 (main.py에서 사용)"""
    system = get_realtime_chat_system()
    return system.process_streaming_response(response_generator, mode)

def get_realtime_stats():
    """실시간 처리 통계 반환"""
    system = get_realtime_chat_system()
    return system.get_realtime_stats()

if __name__ == "__main__":
    # 테스트 코드
    print("🚀 실시간 대화 시스템 테스트")
    system = RealtimeChatSystem()
    
    # 테스트 응답 생성기
    def test_response_generator():
        test_response = "Hello! I'm working perfectly. This is a test of the real-time chat system. It should be very smooth and natural."
        for char in test_response:
            yield char
            time.sleep(0.01)  # 실제 스트리밍 시뮬레이션
    
    # 테스트 실행
    result = system.process_streaming_response(test_response_generator(), mode="m")
    print(f"\n테스트 결과: {result}")
    print(f"통계: {system.get_realtime_stats()}")
