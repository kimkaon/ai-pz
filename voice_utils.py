import sounddevice as sd
import scipy.io.wavfile as wav
import os
from faster_whisper import WhisperModel
import numpy as np
import time
from mic_settings import save_mic_index, load_mic_index

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
    print("🎤 말하세요...")
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
    print("🛑 녹음이 종료되었습니다.")
    return audio, fs

def record_voice_until_silence(fs=44100, device_name="erpon", silence_sec=3, silence_threshold=0.01, max_duration=60, device_index=None):
    if device_index is None:
        device_index = None
        for idx, dev in enumerate(sd.query_devices()):
            if device_name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                device_index = idx
                break
    print(f"🎤 말하세요... ({silence_sec}초 이상 조용하면 자동 종료)")
    if device_index is None:
        print(f"❗ '{device_name}' 장치를 찾을 수 없습니다. 기본 입력 장치를 사용합니다.")
        device_index = None

    buffer = []
    last_sound = time.time()
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        nonlocal last_sound
        buffer.append(indata.copy())
        if np.abs(indata).max() > silence_threshold:
            last_sound = time.time()

    with sd.InputStream(samplerate=fs, channels=1, device=device_index, callback=callback):
        while True:
            sd.sleep(200)
            if time.time() - last_sound > silence_sec:
                print(f"🛑 {silence_sec}초 이상 조용하여 녹음을 종료합니다.")
                break
            if time.time() - start_time > max_duration:
                print("⏰ 최대 녹음 시간 초과로 종료합니다.")
                break

    audio = np.concatenate(buffer, axis=0)
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
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language="en")
    return " ".join([seg.text for seg in segments])

def wait_for_voice(fs=44100, device_index=None, threshold=0.01, max_wait=60):
    """
    마이크에서 소리가 감지될 때까지 대기, 감지되면 True 반환
    """
    import time
    import numpy as np
    print("🎤 녹음 대기 중... (마이크에 소리가 감지되면 자동 녹음 시작)")
    start_time = time.time()
    detected = False
    def callback(indata, frames, time_info, status):
        nonlocal detected
        if np.abs(indata).max() > threshold:
            detected = True
    with sd.InputStream(samplerate=fs, channels=1, device=device_index, callback=callback):
        while not detected:
            sd.sleep(100)
            if time.time() - start_time > max_wait:
                print("⏰ 대기 시간 초과. 다시 시도하세요.")
                return False
    print("🔊 소리 감지! 녹음을 시작합니다.")
    return True
