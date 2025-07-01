# openvoice_tts.py
"""
OpenVoice 기반 TTS(음성합성) 함수 모듈
- synthesize_with_openvoice: 텍스트를 음성(wav)으로 변환
- main.py 등에서 import하여 사용
"""
import os
import sys
import torch
import numpy as np
import soundfile as sf
sys.path.append(os.path.join(os.path.dirname(__file__), "OpenVoice"))
from openvoice.api import BaseSpeakerTTS

# OpenVoice V2 기준 (checkpoints_v2 폴더 필요)
# 최초 1회만 모델 로드 (전역 캐시)
_ov_model = None

def synthesize_with_openvoice(text, output_wav_path, speaker_wav=None, language="en"):
    """
    OpenVoice V2 모델을 사용해 텍스트를 음성(wav)으로 합성합니다.
    Args:
        text (str): 합성할 텍스트
        output_wav_path (str): 저장할 wav 파일 경로
        speaker_wav (str, optional): 화자 음성 참조 wav 파일 경로(없으면 기본)
        language (str): 언어 코드(기본: 영어)
    Returns:
        output_wav_path (str): 생성된 wav 파일 경로
    """
    global _ov_model
    if _ov_model is None:
        # OpenVoice V2 모델 로드 (config 경로 필요)
        config_path = os.path.join(os.path.dirname(__file__), "OpenVoice", "checkpoints_v2", "config.json")
        _ov_model = BaseSpeakerTTS(config_path)
    # 화자 프롬프트 설정 (없으면 기본)
    if speaker_wav is not None:
        prompt = speaker_wav
    else:
        prompt = os.path.join(os.path.dirname(__file__), "OpenVoice", "resources", "demo_speaker0.mp3")
        if not os.path.exists(prompt):
            raise FileNotFoundError(f"기본 화자 프롬프트 파일이 없습니다: {prompt}")
    # speaker 이름 지정 (config.json에 정의된 speaker 중 하나)
    speaker = "default"  # 필요시 config.json에서 실제 이름 확인
    # 합성
    _ov_model.tts(
        text,
        output_wav_path,
        speaker=speaker,
        language=language,  # .capitalize() 제거!
        # speed=1.0 등 추가 옵션 필요시 전달
    )
    return output_wav_path
