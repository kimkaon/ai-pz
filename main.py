import torch
import numpy as np
import warnings
import os

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 로그 숨기기
os.environ['PYTORCH_WARNINGS'] = '0'  # PyTorch 경고 숨기기

print("main.py에서 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

from config_gpu import setup_gpu
setup_gpu(cuda_device=0)
from voice_utils import record_voice, save_temp_wav, transcribe, select_input_device, record_voice_until_silence
from nous_hermes2_mistral_loader import load_nous_hermes2_mistral, chat_nous_hermes2_mistral
from mic_settings import save_mic_index, load_mic_index
from prompt_templates import make_role_prompt
from openvoice_tts import synthesize_with_openvoice
from log_settings import toggle_verbose_mode, is_verbose_mode, log_print, get_current_settings
import sounddevice as sd
import soundfile as sf
import re

def clean_llm_response(response):
    """
    LLM 응답에서 불필요한 부분을 제거하여 TTS에 적합한 텍스트만 추출
    """
    if not response:
        return ""
    
    # 여러 가지 패턴으로 정리
    cleaned = response.strip()
    
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
        
        # Answer:, QnA: 등의 접두사 제거
        line = re.sub(r'^(answer|qna|daily chat|specific program|unknown)\s*:\s*', '', line, flags=re.IGNORECASE)
        
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

def main():
    llm = load_nous_hermes2_mistral()

    # 마이크 설정 불러오기 또는 재설정
    device_index = load_mic_index()

    # 입력 방식 선택 (잘못된 입력 시 반복)
    while True:
        current_verbose = "ON" if is_verbose_mode() else "OFF"
        mode = input(f"실행 모드를 선택하세요 (s: 음성, m: 메시지, p: 마이크 재설정, lg: 로그 모드 [{current_verbose}]): ").strip().lower()
        if mode in ["s", "m", "p", "lg"]:
            break
        print("잘못된 입력입니다. 다시 입력해 주세요. (s: 음성, m: 메시지, p: 마이크 재설정, lg: 로그 모드)")

    if mode == "lg":
        settings = toggle_verbose_mode()
        new_verbose = "ON" if settings["verbose_mode"] else "OFF"
        print(f"✅ 로그 모드가 [{new_verbose}]로 변경되었습니다.")
        log_print(f"로그 설정: {settings}", "general")
        # 로그 모드 변경 후 다시 입력 방식 선택
        while True:
            mode = input("실행 모드를 선택하세요 (s: 음성, m: 메시지, p: 마이크 재설정): ").strip().lower()
            if mode in ["s", "m", "p"]:
                break
            print("잘못된 입력입니다. 다시 입력해 주세요. (s: 음성, m: 메시지, p: 마이크 재설정)")

    if mode == "p":
        device_index = select_input_device()
        print(f"선택된 마이크 index: {device_index}")
        # 마이크 재설정 후 다시 입력 방식 선택
        while True:
            mode = input("실행 모드를 선택하세요 (s: 음성, m: 메시지): ").strip().lower()
            if mode in ["s", "m"]:
                break
            print("잘못된 입력입니다. 다시 입력해 주세요. (s: 음성, m: 메시지)")

    while True:
        if mode == "s":
            from voice_utils import wait_for_voice
            # 녹음 대기 모드: 소리 감지될 때까지 대기
            if not wait_for_voice(device_index=device_index):
                continue
            audio, fs = record_voice_until_silence(device_index=device_index)
            # 음성 입력이 너무 작으면 재녹음(대기모드로 복귀, 파일 저장 X)
            if np.abs(audio).max() < 0.01:
                print("❗ 음성 인식이 되지 않았습니다. 다시 녹음 대기모드로 전환합니다.")
                continue
            # 음성 신호가 충분할 때만 파일 저장
            wav_path = save_temp_wav(audio, fs, silence_threshold=0.01)
            if wav_path is None:
                print("❗ 음성 인식이 되지 않았습니다. 다시 녹음 대기모드로 전환합니다.")
                continue
            print(f"저장된 오디오 파일 경로: {wav_path}")
            prompt = transcribe(wav_path)
            print(f"🗣 인식된 말: {prompt}")
            if not prompt.strip():
                print("❗ 음성 인식이 되지 않았습니다. 다시 녹음 대기모드로 전환합니다.")
                continue
        else:
            # 메시지 입력 모드
            prompt = input("메시지를 입력하세요: ").strip()
            if not prompt:
                print("❗ 아무 메시지도 입력되지 않았어요.")
                continue

        if not prompt:
            print("❗ 아무 말도 인식되지 않았어요.")
            continue

        # 종료 명령어 처리
        if any(x in prompt.lower() for x in ["over", "it's over"]):
            print("🛑 'over'가 감지되어 프로그램을 종료합니다.")
            break

        # Nous Hermes 2 - Mistral 응답 생성
        prompt_for_llm = make_role_prompt(prompt)
        raw_response = chat_nous_hermes2_mistral(llm, prompt_for_llm)
        
        # 응답 정리 (TTS용)
        response = clean_llm_response(raw_response)
        
        # 로그 모드에 따른 출력 분기
        if is_verbose_mode():
            print(f"🤖 Nous Hermes 2 - Mistral의 원본 응답:\n{raw_response}\n")
            log_print(f"🎙️ TTS용 정리된 응답: {response}", "tts_debug")
        else:
            print(f"🤖 {response}")  # 로그 OFF일 때는 정리된 응답만 표시

        # OpenVoice TTS로 응답을 음성(wav)으로 합성 및 재생
        tts_output_dir = "tts_output"
        if not os.path.exists(tts_output_dir):
            os.makedirs(tts_output_dir)
        
        tts_wav_path = os.path.join(tts_output_dir, "response.wav")
        try:
            # 경고 메시지 숨기기
            import warnings
            warnings.filterwarnings("ignore")
            
            synthesize_with_openvoice(response, tts_wav_path, language="English", verbose=is_verbose_mode())
            
            # 생성된 파일이 있는지 확인
            if os.path.exists(tts_wav_path) and os.path.getsize(tts_wav_path) > 0:
                data, fs = sf.read(tts_wav_path, dtype='float32')
                log_print("🔊 TTS 음성 재생 중...", "tts_status")
                sd.play(data, fs)
                sd.wait()
                log_print("✅ TTS 음성 재생 완료", "tts_status")
            else:
                print("❌ TTS 파일이 생성되지 않았습니다.")
        except NotImplementedError:
            print("[TTS] OpenVoice TTS 합성 함수가 아직 구현되지 않았습니다.")
        except Exception as e:
            print(f"[TTS] 오류 발생: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
# This code is a simple voice interaction system that records audio, transcribes it using Whisper, and generates a response using Nous Hermes 2 - Mistral GGUF.
# It uses sounddevice for audio recording, scipy for saving audio files, and faster_whisper for transcription.
# The Llama model is loaded using llama-cpp-python, and the chat function generates responses based on the GGUF model.