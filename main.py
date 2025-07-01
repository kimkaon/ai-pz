import torch
import numpy as np
print("main.py에서 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

from config_gpu import setup_gpu
setup_gpu(cuda_device=0)
from voice_utils import record_voice, save_temp_wav, transcribe, select_input_device, record_voice_until_silence
from nous_hermes2_mistral_loader import load_nous_hermes2_mistral, chat_nous_hermes2_mistral
from mic_settings import save_mic_index, load_mic_index
from prompt_templates import make_role_prompt
from openvoice_tts import synthesize_with_openvoice
import sounddevice as sd
import soundfile as sf

def main():
    llm = load_nous_hermes2_mistral()

    # 마이크 설정 불러오기 또는 재설정
    device_index = load_mic_index()

    # 입력 방식 선택 (잘못된 입력 시 반복)
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
        response = chat_nous_hermes2_mistral(llm, prompt_for_llm)
        print(f"🤖 Nous Hermes 2 - Mistral의 응답:\n{response}\n")

        # OpenVoice TTS로 응답을 음성(wav)으로 합성 및 재생
        tts_wav_path = "output_tts.wav"
        try:
            synthesize_with_openvoice(response, tts_wav_path, language="en")
            data, fs = sf.read(tts_wav_path, dtype='float32')
            print("🔊 TTS 음성 재생 중...")
            sd.play(data, fs)
            sd.wait()
        except NotImplementedError:
            print("[TTS] OpenVoice TTS 합성 함수가 아직 구현되지 않았습니다.")
        except Exception as e:
            print(f"[TTS] 오류 발생: {e}")

if __name__ == "__main__":
    main()
# This code is a simple voice interaction system that records audio, transcribes it using Whisper, and generates a response using Nous Hermes 2 - Mistral GGUF.
# It uses sounddevice for audio recording, scipy for saving audio files, and faster_whisper for transcription.
# The Llama model is loaded using llama-cpp-python, and the chat function generates responses based on the GGUF model.