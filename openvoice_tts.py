# openvoice_tts.py
"""
OpenVoice V2 기반 TTS(음성합성) 함수 모듈
- synthesize_with_openvoice: 텍스트를 음성(wav)으로 변환
- gTTS + OpenVoice ToneColorConverter 사용
"""
import os
import sys
import torch
import numpy as np
import soundfile as sf
import tempfile
import contextlib
sys.path.append(os.path.join(os.path.dirname(__file__), "OpenVoice"))
from openvoice.api import ToneColorConverter
from openvoice import se_extractor
from log_settings import log_print

# OpenVoice V2 모델 캐시
_tone_color_converter = None

@contextlib.contextmanager
def suppress_stdout_stderr():
    """표준 출력과 표준 에러를 일시적으로 숨김"""
    try:
        # Windows에서는 'nul', Unix/Linux에서는 '/dev/null'
        devnull = 'nul' if os.name == 'nt' else '/dev/null'
        with open(devnull, "w") as devnull_file:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull_file
                sys.stderr = devnull_file
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    except:
        # fallback: 출력을 완전히 무시
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            yield
        finally:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def synthesize_with_openvoice(text, output_wav_path, speaker_wav=None, language="English", verbose=False):
    """
    OpenVoice V2를 사용해 텍스트를 음성으로 합성합니다.
    1. gTTS로 기본 음성 생성
    2. OpenVoice ToneColorConverter로 톤 컬러 변환
    
    Args:
        text (str): 합성할 텍스트
        output_wav_path (str): 저장할 wav 파일 경로
        speaker_wav (str, optional): 참조할 화자 음성 파일 경로
        language (str): 언어 ("English", "Korean", "Chinese", "Japanese", "Spanish", "French")
        verbose (bool): 상세 로그 출력 여부
    Returns:
        output_wav_path (str): 생성된 wav 파일 경로
    """
    global _tone_color_converter
    
    device = 'cpu'  # GPU 메모리 절약을 위해 CPU 사용
    
    try:
        # gTTS import
        from gtts import gTTS
        from pydub import AudioSegment
        from pydub.utils import which
    except ImportError as e:
        print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
        print("다음 명령으로 설치하세요: pip install gtts pydub")
        raise
    
    # 1. ToneColorConverter 초기화 (최초 1회)
    if _tone_color_converter is None:
        try:
            ckpt_converter = os.path.join(os.path.dirname(__file__), "OpenVoice", "checkpoints_v2", "converter")
            converter_config = os.path.join(ckpt_converter, "config.json")
            converter_checkpoint = os.path.join(ckpt_converter, "checkpoint.pth")
            
            if not os.path.exists(converter_config):
                raise FileNotFoundError(f"Converter config not found: {converter_config}")
            if not os.path.exists(converter_checkpoint):
                raise FileNotFoundError(f"Converter checkpoint not found: {converter_checkpoint}")
            
            # 출력 메시지를 숨기고 ToneColorConverter 로드
            with suppress_stdout_stderr():
                _tone_color_converter = ToneColorConverter(converter_config, device=device)
                _tone_color_converter.load_ckpt(converter_checkpoint)
            
            log_print(f"✅ OpenVoice ToneColorConverter 로드됨", "model_loading")
            
        except Exception as e:
            log_print(f"ToneColorConverter 초기화 오류: {e}", "model_loading")
            raise
    
    # 2. gTTS로 기본 음성 생성
    try:
        # 언어 매핑
        gtts_lang_map = {
            "English": "en",
            "Korean": "ko", 
            "Chinese": "zh",
            "Japanese": "ja",
            "Spanish": "es",
            "French": "fr"
        }
        
        gtts_lang = gtts_lang_map.get(language, "en")
        
        log_print(f"🎵 gTTS로 {language} 음성 생성 중...", "tts_status")
        
        # gTTS 객체 생성
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # 임시 mp3 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            temp_mp3_path = temp_mp3.name
            tts.save(temp_mp3_path)
        
        # mp3를 wav로 변환
        temp_wav_path = output_wav_path.replace('.wav', '_temp_base.wav')
        
        # pydub로 변환 (ffmpeg 필요 없이)
        try:
            audio = AudioSegment.from_mp3(temp_mp3_path)
            # OpenVoice에 맞는 샘플레이트로 변환 (22050Hz)
            audio = audio.set_frame_rate(22050).set_channels(1)
            audio.export(temp_wav_path, format="wav")
            log_print(f"✅ gTTS 기본 음성 생성 완료: {temp_wav_path}", "tts_status")
        except Exception as e:
            log_print(f"MP3->WAV 변환 오류: {e}", "tts_debug")
            # pydub 변환이 실패하면 scipy로 시도
            import scipy.io.wavfile as wavfile
            from pydub.utils import which
            
            # 단순히 파일을 복사하고 확장자만 변경
            import shutil
            shutil.copy2(temp_mp3_path, temp_wav_path.replace('.wav', '.mp3'))
            raise Exception("MP3 to WAV 변환이 필요합니다. ffmpeg를 설치하거나 다른 방법을 시도해주세요.")
        
        # 임시 mp3 파일 정리
        if os.path.exists(temp_mp3_path):
            os.remove(temp_mp3_path)
        
        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            raise Exception("gTTS 베이스 음성 생성에 실패했습니다.")
            
    except Exception as e:
        log_print(f"gTTS 음성 생성 오류: {e}", "tts_debug")
        raise
    
    # 3. OpenVoice 톤 컬러 변환
    try:
        log_print("🎨 OpenVoice 톤 컬러 변환 시작...", "tts_status")
        
        # 베이스 화자 embedding 로드
        ses_path = os.path.join(os.path.dirname(__file__), "OpenVoice", "checkpoints_v2", "base_speakers", "ses")
        
        # 언어별 베이스 화자 선택
        lang_to_speaker = {
            "English": "zh",  # 중국어 화자가 안정적
            "Korean": "kr", 
            "Chinese": "zh",
            "Japanese": "jp",
            "Spanish": "es",
            "French": "fr"
        }
        
        speaker_file = lang_to_speaker.get(language, "zh")
        source_se_path = os.path.join(ses_path, f"{speaker_file}.pth")
        
        if not os.path.exists(source_se_path):
            # 중국어 기본값으로 fallback
            source_se_path = os.path.join(ses_path, "zh.pth")
        
        if os.path.exists(source_se_path):
            source_se = torch.load(source_se_path, map_location=device)
            log_print(f"✅ 베이스 화자 embedding 로드: {speaker_file}", "model_loading")
        else:
            raise FileNotFoundError(f"베이스 화자 embedding을 찾을 수 없음: {ses_path}")
        
        # 타겟 화자 설정
        if speaker_wav and os.path.exists(speaker_wav):
            # 참조 음성이 있으면 해당 음성의 톤 컬러 추출
            log_print(f"🎯 참조 음성에서 톤 컬러 추출: {speaker_wav}", "tts_debug")
            with suppress_stdout_stderr():
                target_se, _ = se_extractor.get_se(speaker_wav, _tone_color_converter, vad=True)
        else:
            # 참조 음성이 없으면 베이스 화자와 동일하게 설정
            target_se = source_se
        
        # 톤 컬러 변환 실행
        with suppress_stdout_stderr():
            _tone_color_converter.convert(
                audio_src_path=temp_wav_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_wav_path,
                message="@OpenVoice"
            )
        
        # 임시 파일 정리
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        
        # 결과 확인
        if os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0:
            file_size = os.path.getsize(output_wav_path)
            log_print(f"✅ OpenVoice TTS 완료: {output_wav_path} ({file_size} bytes)", "tts_status")
        else:
            raise Exception("최종 음성 파일이 생성되지 않았습니다.")
            
    except Exception as e:
        log_print(f"OpenVoice 톤 컬러 변환 오류: {e}", "tts_debug")
        import traceback
        log_print(f"Traceback: {traceback.format_exc()}", "tts_debug")
        raise Exception(f"OpenVoice TTS 실행 실패: {e}")
    
    return output_wav_path
