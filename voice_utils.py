import sounddevice as sd
import scipy.io.wavfile as wav
import os
from faster_whisper import WhisperModel
import numpy as np
import time

# 통합 설정 관리자 임포트
try:
    from settings_manager import get_settings_manager
    settings_manager = get_settings_manager()
    
    def save_mic_index(device_index):
        """마이크 인덱스 저장 (호환성 유지)"""
        settings_manager.set_microphone_device(device_index)
    
    def load_mic_index():
        """마이크 인덱스 로드 (호환성 유지)"""
        return settings_manager.get('microphone.device_index')
    
    def log_print(message, log_type="general"):
        """로그 출력 (호환성 유지)"""
        if settings_manager.should_show_log(log_type):
            print(message)
            
except ImportError:
    # 폴백: 기본 함수들
    def save_mic_index(device_index):
        print(f"마이크 인덱스 저장: {device_index}")
    
    def load_mic_index():
        return None
    
    def log_print(message, log_type="general"):
        print(message)

def select_input_device():
    import sounddevice as sd
    devices = sd.query_devices()
    # Windows의 마이크 설정 창과 유사하게, hostapi가 WASAPI/DirectSound/Windows MME 등인 입력 장치만 표시
    hostapi_names = [sd.query_hostapis()[i]['name'] for i in range(len(sd.query_hostapis()))]
    input_devices = []
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0 and d['name'].strip() and d['default_samplerate'] > 0:
            hostapi = hostapi_names[d['hostapi']] if d['hostapi'] < len(hostapi_names) else 'Unknown'
            input_devices.append((i, d, hostapi))
    print("[사용 가능한 마이크 목록]")
    for idx, (i, d, hostapi) in enumerate(input_devices, 1):
        print(f"{idx}. {d['name']} (id={i}, {hostapi})")
    if not input_devices:
        print("❗ 입력 가능한 마이크가 없습니다.")
        return None
    while True:
        try:
            sel = int(input(f"사용할 마이크 번호를 선택하세요 (1~{len(input_devices)}): "))
            if 1 <= sel <= len(input_devices):
                device_index = input_devices[sel-1][0]
                save_mic_index(device_index)
                return device_index
        except Exception:
            pass
        print("잘못된 입력입니다. 다시 선택하세요.")

def record_voice(duration=5, fs=44100, device_name="erpon", device_index=None):
    log_print("🎤 말하세요...", "audio_processing")
    if device_index is None:
        device_index = None
        for idx, dev in enumerate(sd.query_devices()):
            if device_name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                device_index = idx
                break
    if device_index is None:
        print(f"❗ '{device_name}' 장치를 찾을 수 없습니다. 기본 입력 장치를 사용합니다.")
        device_index = None
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device_index)
    sd.wait()
    log_print("🛑 녹음이 종료되었습니다.", "audio_processing")
    return audio, fs

def record_voice_until_silence(fs=44100, device_name="erpon", silence_sec=2, silence_threshold=0.015, max_duration=30, device_index=None):
    """
    실시간 대화를 위한 개선된 음성 녹음
    - 더 짧은 침묵 감지 시간 (3초 -> 2초)
    - 더 민감한 threshold
    - 더 짧은 최대 녹음 시간 (60초 -> 30초)
    """
    if device_index is None:
        device_index = None
        for idx, dev in enumerate(sd.query_devices()):
            if device_name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                device_index = idx
                break
    
    print(f"🎤 녹음 중... ({silence_sec}초 침묵 시 자동 종료)")
    if device_index is None:
        log_print(f"❗ '{device_name}' 장치를 찾을 수 없습니다. 기본 입력 장치를 사용합니다.", "audio_processing")
        device_index = None

    buffer = []
    last_sound = time.time()
    start_time = time.time()
    sound_detected = False

    def callback(indata, frames, time_info, status):
        nonlocal last_sound, sound_detected
        buffer.append(indata.copy())
        if np.abs(indata).max() > silence_threshold:
            last_sound = time.time()
            sound_detected = True

    with sd.InputStream(samplerate=fs, channels=1, device=device_index, callback=callback):
        while True:
            sd.sleep(100)  # 더 빠른 체크
            current_time = time.time()
            
            # 소리가 감지된 후 침묵 시간 체크
            if sound_detected and (current_time - last_sound > silence_sec):
                log_print(f"🛑 {silence_sec}초 침묵으로 녹음 종료", "audio_processing")
                break
                
            # 최대 녹음 시간 체크
            if current_time - start_time > max_duration:
                log_print("⏰ 최대 녹음 시간 초과로 종료", "audio_processing")
                break
                
            # 소리 감지 없이 5초가 지나면 종료
            if not sound_detected and (current_time - start_time > 5):
                log_print("❗ 소리가 감지되지 않아 녹음 종료", "audio_processing")
                break

    audio = np.concatenate(buffer, axis=0)
    
    # 녹음 품질 체크
    if len(audio) < fs * 0.5:  # 0.5초 미만이면 너무 짧음
        log_print("❗ 녹음이 너무 짧습니다", "audio_processing")
        return audio, fs
    
    log_print(f"✅ 녹음 완료 ({len(audio)/fs:.1f}초)", "audio_processing")
    return audio, fs

def save_temp_wav(audio, fs, silence_threshold=0):
    save_dir = os.path.join(os.getcwd(), "deta")
    os.makedirs(save_dir, exist_ok=True)
    file_count = len([f for f in os.listdir(save_dir) if f.endswith('.wav')])
    file_path = os.path.join(save_dir, f"recorded_{file_count+1}.wav")
    if abs(audio).max() < silence_threshold:
        print("❗ 녹음된 오디오가 너무 작아서 저장하지 않습니다.")
        return None
    wav.write(file_path, fs, audio)
    return file_path

def transcribe(audio_path):
    # GPU 메모리 절약을 위해 smaller 모델 사용 (medium → base)
    # LLM이 이미 GPU 메모리를 많이 사용하고 있으므로 CPU 모델 우선 시도
    try:
        # CPU 모델로 먼저 시도 (GPU 메모리 절약)
        model = WhisperModel("base", device="cpu", compute_type="int8")
        log_print("[Whisper] CPU 모델 사용 (GPU 메모리 절약)", "model_loading")
    except Exception as e:
        log_print(f"[Whisper] CPU 모델 로드 실패: {e}", "model_loading")
        # 최후 수단으로 GPU 시도
        try:
            model = WhisperModel("base", device="cuda", compute_type="float16")
            log_print("[Whisper] GPU 모델 사용", "model_loading")
        except Exception as e2:
            log_print(f"[Whisper] GPU 모델도 실패: {e2}", "model_loading")
            raise Exception("Whisper 모델 로드 실패")

    segments, _ = model.transcribe(audio_path, language="en")
    return " ".join([seg.text for seg in segments])


def wait_for_voice(fs=44100, device_index=None, threshold=0.015, max_wait=300):
    """
    마이크에서 소리가 감지될 때까지 대기, 감지되면 True 반환
    실시간 대화를 위해 더 민감하고 빠른 감지
    """
    import time
    import numpy as np
    log_print("🎤 대기 중... (말씀하세요)", "audio_processing")
    start_time = time.time()
    detected = False
    
    def callback(indata, frames, time_info, status):
        nonlocal detected
        # 더 민감한 감지를 위해 threshold 조정
        if np.abs(indata).max() > threshold:
            detected = True
    
    with sd.InputStream(samplerate=fs, channels=1, device=device_index, callback=callback):
        while not detected:
            sd.sleep(50)  # 더 빠른 체크 (100ms -> 50ms)
            if time.time() - start_time > max_wait:
                print("⏰ 대기 시간 초과. 메뉴로 돌아갑니다.")
                return False
    
    log_print("🔊 음성 감지! 녹음 시작", "audio_processing")
    return True
