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
from nous_hermes2_mistral_loader import load_nous_hermes2_mistral, chat_nous_hermes2_mistral, chat_nous_hermes2_mistral_stream
from mic_settings import save_mic_index, load_mic_index
from prompt_templates import make_role_prompt
from openvoice_tts import synthesize_with_openvoice, synthesize_quick_tts
from log_settings import toggle_verbose_mode, is_verbose_mode, log_print, get_current_settings

# 하이브리드 파인튜닝 통합 기능 추가
try:
    from finetuning_integration import (
        process_with_finetuned_models_main, 
        request_specialist_mode,
        get_finetuning_stats,
        integrate_finetuned_response  # 호환성 유지
    )
    FINETUNING_ENABLED = True
    log_print("하이브리드 파인튜닝 시스템 로드 완료", "model_loading")
except ImportError as e:
    log_print(f"하이브리드 파인튜닝 시스템 로드 실패: {e}", "model_loading")
    FINETUNING_ENABLED = False

# 영어 파인튜닝 모델 통합 기능 추가
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    ENGLISH_FINETUNED_AVAILABLE = True
    log_print("영어 파인튜닝 모델 라이브러리 로드 완료", "model_loading")
except ImportError as e:
    ENGLISH_FINETUNED_AVAILABLE = False
    log_print(f"영어 파인튜닝 모델 라이브러리 로드 실패: {e}", "model_loading")

import sounddevice as sd
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("⚠️ soundfile 모듈이 없습니다. 음성 기능이 제한될 수 있습니다.")
import re
import threading
import queue
import time

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

def try_finetuned_response(prompt):
    """
    선택된 모델에 따라 파인튜닝 응답 생성 시도
    Returns: (response, used_finetuned, model_info) 튜플
    """
    current_model = MODEL_SELECTION["current"]
    
    # RTX 3070 무제한 모델 사용
    if current_model == "rtx3070_unfiltered":
        if not rtx3070_unfiltered_model.loaded:
            if not rtx3070_unfiltered_model.load_model():
                log_print("RTX 3070 무제한 모델 로드 실패, 기본 모델로 폴백", "model_loading")
                return None, False, {}
        
        response = rtx3070_unfiltered_model.generate_response(prompt)
        if response:
            model_info = {
                'model_used': 'rtx3070_unfiltered',
                'category': 'unfiltered_conversation',
                'confidence': 0.95,
                'quality_level': 'high'
            }
            log_print(f"RTX 3070 무제한 모델 사용: {response[:50]}...", "model_loading")
            return response, True, model_info
        
        log_print("RTX 3070 무제한 모델 응답 생성 실패", "model_loading")
        return None, False, {}
    
    # RTX 3070 언어 제한 모델 사용
    elif current_model == "rtx3070_language_limited":
        if not rtx3070_language_limited_model.loaded:
            if not rtx3070_language_limited_model.load_model():
                log_print("RTX 3070 언어 제한 모델 로드 실패, 기본 모델로 폴백", "model_loading")
                return None, False, {}
        
        response = rtx3070_language_limited_model.generate_response(prompt)
        if response:
            model_info = {
                'model_used': 'rtx3070_language_limited',
                'category': 'language_limited_conversation',
                'confidence': 0.95,
                'quality_level': 'high'
            }
            log_print(f"RTX 3070 언어 제한 모델 사용: {response[:50]}...", "model_loading")
            return response, True, model_info
        
        log_print("RTX 3070 언어 제한 모델 응답 생성 실패", "model_loading")
        return None, False, {}
    
    # 영어 통합 파인튜닝 모델 사용
    elif current_model == "english_unified":
        if english_finetuned_model.loaded:
            response = english_finetuned_model.generate_response(prompt)
            if response:
                model_info = {
                    'model_used': 'english_unified',
                    'category': 'english_conversation',
                    'confidence': 0.9,
                    'quality_level': 'high'
                }
                log_print(f"영어 통합 모델 사용: {response[:50]}...", "model_loading")
                return response, True, model_info
        
        log_print("영어 통합 모델 사용 실패, 기본 모델로 폴백", "model_loading")
        return None, False, {}
    
    # 하이브리드 파인튜닝 모델 사용
    elif current_model == "hybrid" and FINETUNING_ENABLED:
        try:
            # 하이브리드 시스템으로 처리
            result = process_with_finetuned_models_main(prompt)
            
            if result and result.get('response'):
                model_info = {
                    'category': result.get('category', 'unknown'),
                    'confidence': result.get('confidence', 0.0),
                    'model_used': result.get('model_used', 'unknown'),
                    'quality_level': result.get('quality_level', 'unknown'),
                    'can_upgrade': result.get('can_upgrade', False)
                }
                
                log_print(f"하이브리드 모델 사용: {model_info['model_used']} ({model_info['category']}, 신뢰도: {model_info['confidence']:.3f})", "model_loading")
                
                return result['response'], True, model_info
            else:
                log_print("하이브리드 모델 응답 생성 실패", "model_loading")
                return None, False, {}
                
        except Exception as e:
            log_print(f"하이브리드 모델 오류: {e}", "model_loading")
            # Fallback to basic integrated response
            try:
                fallback_response = integrate_finetuned_response(prompt)
                return fallback_response, True, {'model_used': 'fallback', 'category': 'unknown'}
            except:
                return None, False, {}
    
    # 기본 모델 사용 (파인튜닝 없음)
    return None, False, {}

def stream_chat_with_tts(llm, prompt, conversation_history, mode, device_index=None):
    """
    실시간 스트리밍 채팅 + TTS
    - LLM에서 토큰 생성과 동시에 화면 출력
    - 문장 단위로 TTS 생성 및 재생
    """
    from nous_hermes2_mistral_loader import chat_nous_hermes2_mistral_stream
    from openvoice_tts import synthesize_quick_tts
    
    # 컨텍스트 포함 프롬프트 생성
    context_prompt = build_conversation_context(conversation_history, prompt)
    
    print("🤔 AI가 답변 중...")
    print("🤖 AI: ", end="", flush=True)
    
    # 실시간 텍스트 수집
    full_response = ""
    sentence_buffer = ""
    sentence_count = 0
    
    # TTS 큐와 스레드
    tts_queue = queue.Queue()
    tts_thread = None
    
    def tts_worker():
        """TTS 처리 워커 스레드"""
        while True:
            try:
                item = tts_queue.get(timeout=1)
                if item is None:  # 종료 신호
                    break
                    
                sentence, file_path = item
                try:
                    synthesize_quick_tts(sentence, file_path, language="English")
                    
                    # 오디오 재생
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        if SOUNDFILE_AVAILABLE:
                            data, fs = sf.read(file_path, dtype='float32')
                            sd.play(data, fs)
                            sd.wait()
                            log_print(f"🔊 문장 재생 완료: {sentence[:30]}...", "tts_status")
                        else:
                            log_print("⚠️ soundfile 없음 - 오디오 재생 건너뜀", "tts_status")
                    
                except Exception as e:
                    log_print(f"TTS 처리 오류: {e}", "tts_debug")
                finally:
                    tts_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                log_print(f"TTS 워커 오류: {e}", "tts_debug")
                break
    
    # TTS 워커 스레드 시작
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    
    try:
        # 스트리밍으로 토큰 받기
        for token in chat_nous_hermes2_mistral_stream(llm, context_prompt):
            # 화면에 실시간 출력
            print(token, end="", flush=True)
            full_response += token
            sentence_buffer += token
            
            # 문장 끝 감지 (.!?로 끝나고 공백이 있는 경우)
            if any(punct in token for punct in ['. ', '! ', '? ', '.\n', '!\n', '?\n']):
                sentence = sentence_buffer.strip()
                if len(sentence) > 5:  # 너무 짧은 문장 제외
                    sentence_count += 1
                    
                    # TTS 파일 경로
                    tts_output_dir = "tts_output"
                    if not os.path.exists(tts_output_dir):
                        os.makedirs(tts_output_dir)
                    
                    tts_wav_path = os.path.join(tts_output_dir, f"stream_{sentence_count}.wav")
                    
                    # TTS 큐에 추가
                    tts_queue.put((sentence, tts_wav_path))
                    log_print(f"🎵 TTS 큐에 추가: {sentence[:30]}...", "tts_status")
                
                sentence_buffer = ""
        
        print()  # 줄바꿈
        
        # 마지막 문장 처리
        if sentence_buffer.strip() and len(sentence_buffer.strip()) > 5:
            sentence_count += 1
            tts_wav_path = os.path.join("tts_output", f"stream_{sentence_count}.wav")
            tts_queue.put((sentence_buffer.strip(), tts_wav_path))
        
        # TTS 큐 완료 대기
        log_print("🎵 모든 TTS 처리 대기 중...", "tts_status")
        tts_queue.join()
        
        # 워커 스레드 종료
        tts_queue.put(None)
        if tts_thread:
            tts_thread.join(timeout=5)
        
        log_print("✅ 스트리밍 응답 완료", "tts_status")
        
    except Exception as e:
        print(f"\n❌ 스트리밍 오류: {e}")
        log_print(f"스트리밍 상세 오류: {e}", "tts_debug")
        
        # 오류 발생시 워커 스레드 정리
        try:
            tts_queue.put(None)
            if tts_thread:
                tts_thread.join(timeout=2)
        except:
            pass
    
    # 응답 정리
    clean_response = clean_llm_response(full_response)
    return clean_response

def main():
    print("🤖 AI 어시스턴트를 시작합니다...")
    llm = load_nous_hermes2_mistral()
    print("✅ 모델 로딩 완료!")
    
    # 하이브리드 파인튜닝 시스템에 LLM 인스턴스 전달
    if FINETUNING_ENABLED:
        try:
            from hybrid_finetuning_integration import initialize_hybrid_system
            initialize_hybrid_system(llm)
        except ImportError:
            pass

    # 마이크 설정 불러기기 또는 재설정
    device_index = load_mic_index()
    
    # 사용 가능한 모델 감지
    detect_available_models()
    
    # 대화 히스토리 저장용
    conversation_history = []

    # 초기 설정 메뉴
    while True:
        current_verbose = "ON" if is_verbose_mode() else "OFF"
        
        # 현재 모델 정보 가져오기
        model_info = get_model_info()
        current_model_name = model_info["current_name"]
        
        # 하이브리드 파인튜닝 상태 확인
        finetuning_status = "OFF"
        if FINETUNING_ENABLED:
            try:
                stats = get_finetuning_stats()
                if stats.get('status') == 'active':
                    finetuning_status = "ON (Hybrid)"
                    loaded_specialists = stats.get('loaded_specialists', [])
                    if loaded_specialists:
                        finetuning_status += f" + {len(loaded_specialists)} specialists"
                elif stats.get('status') == 'unavailable':
                    finetuning_status = "UNAVAILABLE"
                else:
                    finetuning_status = "ERROR"
            except:
                finetuning_status = "ERROR"
        
        print(f"\n{'='*60}")
        print("🎯 AI 어시스턴트 설정 (다중 모델 모드)")
        print(f"{'='*60}")
        print(f"� 현재 활성 모델: {current_model_name}")
        print(f"�🧠 파인튜닝 시스템: {finetuning_status}")
        if FINETUNING_ENABLED and finetuning_status.startswith("ON"):
            try:
                stats = get_finetuning_stats()
                print(f"   💾 통합모델: {'로드됨' if stats.get('unified_model_loaded') else '미로드'}")
                print(f"   📈 통계: 통합응답 {stats.get('unified_responses', 0)}회, 전문응답 {stats.get('specialist_responses', 0)}회")
            except:
                pass
        print(f"📝 상세 로그: {current_verbose}")
        print(f"{'='*60}")
        
        mode = input(f"선택 (s: 음성대화, m: 텍스트대화, p: 마이크설정, md: 모델선택, lg: 로그[{current_verbose}], ft: 파인튜닝관리, q: 종료): ").strip().lower()
        
        if mode == "q":
            print("👋 프로그램을 종료합니다.")
            # 하이브리드 파인튜닝 시스템 정리
            if FINETUNING_ENABLED:
                try:
                    from finetuning_integration import cleanup_finetuning_system
                    cleanup_finetuning_system()
                except:
                    pass
            return
        elif mode == "md":
            # 모델 선택 메뉴
            print(f"\n{'='*50}")
            print("🤖 모델 선택")
            print(f"{'='*50}")
            
            model_info = get_model_info()
            available_models = model_info["available"]
            current_model = model_info["current"]
            
            # 모델 설명
            model_descriptions = {
                "original": "기본 Nous-Hermes-2-Mistral (다국어 지원)",
                "hybrid": "하이브리드 파인튜닝 모델 (한국어 특화)",
                "english_unified": "영어 통합 파인튜닝 모델 (영어 대화/QnA 특화)",
                "rtx3070_unfiltered": "RTX 3070 최적화 모델 (영어/한국어 제한)",
                "rtx3070_language_limited": "RTX 3070 최적화 모델 (영어/한국어 제한)"
            }
            
            print("사용 가능한 모델:")
            for i, model_type in enumerate(available_models, 1):
                marker = "★" if model_type == current_model else " "
                description = model_descriptions.get(model_type, model_type)
                print(f"{marker} {i}. {description}")
                
                # RTX 3070 최적화 모델에 대한 추가 설명
                if model_type == "rtx3070_unfiltered":
                    print("    🚀 최고 성능의 RTX 3070 최적화 모델")
                    print("    🌐 언어 제한: 영어/한국어만 지원 (다른 언어 차단)")
                    rtx_path = "./models/rtx3070_optimized_best"
                    if os.path.exists(rtx_path):
                        print("         ✅ 모델 파일 확인됨")
                    else:
                        print("         ❌ 모델 파일이 없습니다")
                
                # RTX 3070 언어 제한 모델에 대한 추가 설명
                elif model_type == "rtx3070_language_limited":
                    print("    🚀 최고 성능의 RTX 3070 최적화 모델")
                    print("    🌐 언어 제한: 영어/한국어만 지원 (다른 언어 차단)")
                    rtx_path = "./models/rtx3070_optimized_best"
                    if os.path.exists(rtx_path):
                        print("         ✅ 모델 파일 확인됨")
                    else:
                        print("         ❌ 모델 파일이 없습니다")
                
                # 영어 모델에 대한 추가 설명
                elif model_type == "english_unified":
                    print("    ⚠️  주의: 이 모델은 영어 대화에 최적화되어 있습니다.")
                    print("         한국어 응답 품질이 떨어질 수 있습니다.")
                    if os.path.exists("./finetuning/models/unified_model"):
                        print("         ✅ 모델 파일 확인됨")
                    else:
                        print("         ❌ 모델 파일이 없습니다")
            
            print(f"\n현재 활성 모델: {model_descriptions.get(current_model, current_model)}")
            print("\n0. 돌아가기")
            
            try:
                choice = input("\n선택할 모델 번호: ").strip()
                
                if choice == "0":
                    continue
                elif choice.isdigit():
                    model_index = int(choice) - 1
                    if 0 <= model_index < len(available_models):
                        selected_model = available_models[model_index]
                        
                        if selected_model == current_model:
                            print(f"이미 '{model_descriptions[selected_model]}' 모델이 활성화되어 있습니다.")
                        else:
                            # RTX 3070 최적화 모델 선택시 확인
                            if selected_model == "rtx3070_unfiltered":
                                print("\n💡 RTX 3070 최적화 모델을 선택하셨습니다.")
                                print("   이 모델은 영어와 한국어만 지원합니다.")
                                print("   다른 언어 입력은 자동으로 차단됩니다.")
                                confirm = input("   이 모델을 사용하시겠습니까? (Y/n): ").strip().lower()
                                if confirm in ['n', 'no']:
                                    print("모델 변경이 취소되었습니다.")
                                    continue
                            
                            # RTX 3070 언어 제한 모델 선택시 확인
                            elif selected_model == "rtx3070_language_limited":
                                print("\n💡 RTX 3070 최적화 언어 제한 모델을 선택하셨습니다.")
                                print("   이 모델은 영어와 한국어만 지원합니다.")
                                print("   다른 언어 입력은 자동으로 차단됩니다.")
                                confirm = input("   이 모델을 사용하시겠습니까? (Y/n): ").strip().lower()
                                if confirm in ['n', 'no']:
                                    print("모델 변경이 취소되었습니다.")
                                    continue
                            
                            # 영어 모델 선택시 확인
                            elif selected_model == "english_unified":
                                print("\n⚠️  영어 통합 파인튜닝 모델을 선택하셨습니다.")
                                print("   이 모델은 영어 대화와 QnA에 특화되어 있습니다.")
                                print("   한국어 응답 품질이 현저히 떨어질 수 있습니다.")
                                confirm = input("   정말로 이 모델을 사용하시겠습니까? (y/N): ").strip().lower()
                                if confirm not in ['y', 'yes']:
                                    print("모델 변경이 취소되었습니다.")
                                    continue
                            
                            print(f"🔄 모델을 '{model_descriptions[selected_model]}'로 변경 중...")
                            success, message = switch_model(selected_model)
                            
                            if success:
                                print(f"✅ {message}")
                                log_print(f"모델 변경: {current_model} -> {selected_model}", "model_loading")
                            else:
                                print(f"❌ 모델 변경 실패: {message}")
                    else:
                        print("❌ 잘못된 선택입니다.")
                else:
                    print("❌ 잘못된 입력입니다.")
                    
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")
            
            input("\n계속하려면 Enter를 누르세요...")
            continue
        elif mode == "lg":
            settings = toggle_verbose_mode()
            new_verbose = "ON" if settings["verbose_mode"] else "OFF"
            print(f"✅ 로그 모드가 [{new_verbose}]로 변경되었습니다.")
            log_print(f"로그 설정: {settings}", "general")
            continue
        elif mode == "ft":
            # 하이브리드 파인튜닝 관리 메뉴
            if not FINETUNING_ENABLED:
                print("❌ 하이브리드 파인튜닝 기능이 비활성화되어 있습니다.")
                print("   hybrid_finetuning_integration.py를 확인하세요.")
                continue
            
            print(f"\n{'='*40}")
            print("🔧 하이브리드 파인튜닝 관리")
            print(f"{'='*40}")
            
            try:
                stats = get_finetuning_stats()
                if stats.get('status') == 'active':
                    print("✅ 하이브리드 시스템 상태:")
                    print(f"   🧠 통합모델: {'활성' if stats.get('unified_model_loaded') else '미활성'}")
                    print(f"   🎯 로드된 전문모델: {stats.get('loaded_specialists', [])}")
                    print(f"   📊 응답 통계: 통합 {stats.get('unified_responses', 0)}회, 전문 {stats.get('specialist_responses', 0)}회")
                    print(f"   🔄 모델 전환: {stats.get('model_switches', 0)}회")
                    print(f"   ⏱️ 평균 응답시간: {stats.get('avg_response_time', 0):.2f}초")
                
                    print(f"\n하이브리드 파인튜닝 관리 옵션:")
                    print("1. 일반 모드로 테스트 응답")
                    print("2. 전문가 모드로 테스트 응답")
                    print("3. 전문모델 언로드 (메모리 절약)")
                    print("4. 파인튜닝 환경 열기")
                    print("0. 돌아가기")
                    
                    ft_choice = input("선택: ").strip()
                    
                    if ft_choice == "1":
                        test_prompt = input("테스트 입력: ").strip()
                        if test_prompt:
                            finetuned_response, used, model_info = try_finetuned_response(test_prompt)
                            if used and finetuned_response:
                                print(f"🤖 [{model_info.get('model_used', 'unknown')}] {finetuned_response}")
                                print(f"📊 정보: {model_info}")
                            else:
                                print("❌ 하이브리드 응답을 생성할 수 없습니다.")
                    
                    elif ft_choice == "2":
                        test_prompt = input("전문가 모드 테스트 입력: ").strip()
                        if test_prompt:
                            try:
                                specialist_result = request_specialist_mode(test_prompt)
                                print(f"🎓 [{specialist_result.get('model_used', 'unknown')}] {specialist_result.get('response', '응답 없음')}")
                                print(f"📊 정보: 카테고리 {specialist_result.get('category')}, 품질 {specialist_result.get('quality_level')}")
                            except Exception as e:
                                print(f"❌ 전문가 모드 오류: {e}")
                    
                    elif ft_choice == "3":
                        print("🗑️ 메모리 정리는 시스템이 자동으로 관리합니다.")
                        print("   필요시 프로그램을 재시작하세요.")
                    
                    elif ft_choice == "4":
                        print("📁 하이브리드 파인튜닝 관리 스크립트:")
                        print("   python finetuning/scripts/generate_english_datasets.py")
                        print("   python finetuning/scripts/validate_english_data.py") 
                        print("   또는 run_finetuning.bat 실행")
                else:
                    print(f"❌ 하이브리드 시스템 오류: {stats.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ 하이브리드 시스템 상태 확인 실패: {e}")
                
            else:
                print("❌ 사용 가능한 파인튜닝 모델이 없습니다.")
                print("📁 파인튜닝을 시작하려면 run_finetuning.bat를 실행하세요.")
            
            input("\n계속하려면 Enter를 누르세요...")
            continue
        elif mode == "p":
            device_index = select_input_device()
            print(f"✅ 선택된 마이크 index: {device_index}")
            continue
        elif mode in ["s", "m"]:
            break
        else:
            print("❌ 잘못된 입력입니다. 다시 입력해 주세요.")

    # 실시간 대화 모드 시작
    model_info = get_model_info()
    current_model_name = model_info["current_name"]
    
    print(f"\n{'='*60}")
    if mode == "s":
        print("🎤 음성 대화 모드 시작 (종료: 'over' 또는 'quit' 말하기)")
    else:
        print("💬 텍스트 대화 모드 시작 (종료: 'over', 'quit', 또는 빈 입력)")
    print(f"🤖 활성 모델: {current_model_name}")
    
    # 영어 모델 사용시 추가 안내
    if model_info["current"] == "english_unified":
        print("⚠️  영어 대화에 최적화된 모델입니다. 한국어 응답 품질이 제한적일 수 있습니다.")
    
    print(f"{'='*60}")
    print("💡 팁: 자연스럽게 대화하세요. 이전 대화 내용을 기억합니다!")
    print("💡 모델 변경은 메인 메뉴에서 'md'를 선택하세요.")
    print()

    # 실시간 대화 루프
    conversation_count = 0
    while True:
        conversation_count += 1
        
        # 입력 받기
        if mode == "s":
            # 음성 입력
            from voice_utils import wait_for_voice
            print(f"🎤 대화 #{conversation_count} - 말씀하세요...")
            
            if not wait_for_voice(device_index=device_index):
                continue
                
            audio, fs = record_voice_until_silence(device_index=device_index)
            
            if np.abs(audio).max() < 0.01:
                log_print("❗ 음성이 감지되지 않았습니다. 다시 시도하세요.", "audio_processing")
                continue
                
            wav_path = save_temp_wav(audio, fs, silence_threshold=0.01)
            if wav_path is None:
                log_print("❗ 음성 저장에 실패했습니다. 다시 시도하세요.", "audio_processing")
                continue
                
            log_print(f"📁 음성 파일 저장: {wav_path}", "audio_processing")
            prompt = transcribe(wav_path)
            print(f"🗣️ 인식된 내용: \"{prompt}\"")
            
            if not prompt.strip():
                print("❗ 음성 인식에 실패했습니다. 다시 시도하세요.")
                continue
        else:
            # 텍스트 입력
            prompt = input(f"💬 대화 #{conversation_count} - 메시지 입력: ").strip()
            if not prompt:
                print("👋 대화를 종료합니다.")
                break

        # 종료 명령어 체크
        if any(keyword in prompt.lower() for keyword in ["over", "quit", "exit", "bye", "goodbye"]):
            print("👋 대화를 종료합니다.")
            break

        # 1. 먼저 하이브리드 파인튜닝 모델로 응답 시도
        finetuned_response, used_finetuned, model_info = try_finetuned_response(prompt)
        
        if used_finetuned and finetuned_response:
            # 모델 정보 표시
            model_used = model_info.get('model_used', 'unknown')
            category = model_info.get('category', 'unknown')
            quality_level = model_info.get('quality_level', 'standard')
            can_upgrade = model_info.get('can_upgrade', False)
            
            print(f"🤖 AI ({model_used}/{quality_level}): {finetuned_response}")
            
            # 업그레이드 제안 (전문모델로 전환 가능한 경우)
            if can_upgrade:
                print(f"💡 더 전문적인 답변이 필요하시면 '전문가 모드'라고 말씀해 주세요!")
            
            # 파인튜닝 응답을 TTS로 출력
            if mode == "s":  # 음성 모드일 때만 TTS
                try:
                    tts_output_dir = "tts_output"
                    if not os.path.exists(tts_output_dir):
                        os.makedirs(tts_output_dir)
                    
                    tts_wav_path = os.path.join(tts_output_dir, f"{model_used}_response.wav")
                    log_print(f"🎵 {model_used} 응답 음성 합성 중...", "tts_status")
                    synthesize_with_openvoice(finetuned_response, tts_wav_path, language="English", verbose=is_verbose_mode())
                    
                    if os.path.exists(tts_wav_path) and os.path.getsize(tts_wav_path) > 0:
                        if SOUNDFILE_AVAILABLE:
                            data, fs = sf.read(tts_wav_path, dtype='float32')
                            log_print(f"🔊 {model_used} 응답 재생 중...", "tts_status")
                            sd.play(data, fs)
                            sd.wait()
                            log_print(f"✅ {model_used} 응답 재생 완료", "tts_status")
                        else:
                            log_print(f"⚠️ soundfile 없음 - {model_used} 응답 재생 건너뜀", "tts_status")
                    else:
                        print(f"❌ {model_used} TTS 파일 생성에 실패했습니다.")
                except Exception as tts_e:
                    print(f"❌ {model_used} TTS 오류: {tts_e}")
            
            response = finetuned_response
        else:
            # 2. 파인튜닝 응답이 없으면 기존 스트리밍 방식 사용
            log_print("기존 LLM 모델 사용", "model_loading")
            
            # 대화 컨텍스트를 포함한 프롬프트 생성
            context_prompt = build_conversation_context(conversation_history, prompt)
            
            # 실시간 스트리밍 응답 생성
            try:
                response = stream_chat_with_tts(llm, prompt, conversation_history, mode, device_index)
            except Exception as e:
                print(f"❌ 스트리밍 실패, 일반 모드로 전환: {e}")
                # 스트리밍 실패시 기존 방식으로 fallback
                print("🤔 답변 생성 중...")
                raw_response = chat_nous_hermes2_mistral(llm, context_prompt)
                response = clean_llm_response(raw_response)
            
            if is_verbose_mode():
                print(f"\n🤖 원본 응답:\n{raw_response}\n")
            
            print(f"🤖 AI: {response}")
            
            # 기존 TTS 방식으로 처리
            tts_output_dir = "tts_output"
            if not os.path.exists(tts_output_dir):
                os.makedirs(tts_output_dir)
            
            tts_wav_path = os.path.join(tts_output_dir, "response.wav")
            try:
                import warnings
                warnings.filterwarnings("ignore")
                
                log_print("🎵 음성 합성 중...", "tts_status")
                synthesize_with_openvoice(response, tts_wav_path, language="English", verbose=is_verbose_mode())
                
                if os.path.exists(tts_wav_path) and os.path.getsize(tts_wav_path) > 0:
                    if SOUNDFILE_AVAILABLE:
                        data, fs = sf.read(tts_wav_path, dtype='float32')
                        log_print("🔊 음성 재생 중...", "tts_status")
                        sd.play(data, fs)
                        sd.wait()
                        log_print("✅ 음성 재생 완료", "tts_status")
                    else:
                        log_print("⚠️ soundfile 없음 - 음성 재생 건너뜀", "tts_status")
                else:
                    print("❌ TTS 파일 생성에 실패했습니다.")
            except Exception as tts_e:
                print(f"❌ TTS 오류: {tts_e}")
                log_print(f"TTS 상세 오류: {tts_e}", "tts_debug")
        
        # 대화 히스토리에 추가
        conversation_history.append({"user": prompt, "assistant": response})
        
        # 대화 히스토리가 너무 길면 최근 5개만 유지
        if len(conversation_history) > 5:
            conversation_history = conversation_history[-5:]
        
        print(f"\n{'-'*50}")

def build_conversation_context(history, current_prompt):
    """
    대화 히스토리를 포함한 컨텍스트 프롬프트 생성
    """
    if not history:
        return make_role_prompt(current_prompt)
    
    # 최근 대화 컨텍스트 구성
    context = "You are a helpful AI assistant. Here's the recent conversation context:\n\n"
    
    for i, conv in enumerate(history[-3:], 1):  # 최근 3개 대화만 포함
        context += f"User: {conv['user']}\n"
        context += f"Assistant: {conv['assistant']}\n\n"
    
    context += f"Current User Input: {current_prompt}\n\n"
    context += "Please respond naturally, considering the conversation context. Keep your response conversational and helpful."
    
    return context

# 영어 파인튜닝 모델 로더 클래스
class EnglishFinetuningModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.model_type = "none"  # "unified", "classification", "none"
    
    def load_unified_model(self, model_path="./finetuning/models/unified_model"):
        """통합형 영어 파인튜닝 모델 로드"""
        if not ENGLISH_FINETUNED_AVAILABLE:
            log_print("영어 파인튜닝 라이브러리 비활성화", "model_loading")
            return False
        
        try:
            if os.path.exists(model_path):
                log_print(f"영어 통합 모델 로딩: {model_path}", "model_loading")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                self.loaded = True
                self.model_type = "unified"
                log_print("영어 통합 파인튜닝 모델 로드 완료", "model_loading")
                return True
            else:
                log_print(f"모델 경로를 찾을 수 없음: {model_path}", "model_loading")
                return False
        
        except Exception as e:
            log_print(f"영어 파인튜닝 모델 로드 실패: {e}", "model_loading")
            return False
    
    def generate_response(self, prompt, max_tokens=150):
        """영어 파인튜닝 모델로 응답 생성"""
        if not self.loaded:
            return None
        
        try:
            # 영어 대화 형식으로 포맷
            if self.model_type == "unified":
                formatted_prompt = f"System: You are a helpful AI assistant that can handle various types of conversations including Q&A, technical support, and general chat.\n\nUser: {prompt}\nAssistant:"
            else:
                formatted_prompt = f"User: {prompt}\nAssistant:"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            
            if torch.cuda.is_available() and hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Assistant 응답 부분만 추출
            if "Assistant:" in full_response:
                response = full_response.split("Assistant:")[-1].strip()
                return response
            else:
                return full_response.replace(formatted_prompt, "").strip()
        
        except Exception as e:
            log_print(f"영어 파인튜닝 응답 생성 실패: {e}", "model_loading")
            return None

# 전역 영어 파인튜닝 모델 인스턴스
english_finetuned_model = EnglishFinetuningModelLoader()

# RTX 3070 최적화 모델 클래스 (언어 제한 옵션 포함)
class RTX3070OptimizedLoader:
    def __init__(self, language_restriction=False):
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.language_restriction = language_restriction
        
        # 언어 제한 설정
        if language_restriction:
            import re
            self.language_patterns = {
                'ko': re.compile(r'[\uAC00-\uD7A3]'),  # 한글
                'en': re.compile(r'[A-Za-z]'),          # 영어
                'other': re.compile(r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u0400-\u04FF\u0600-\u06FF]')  # 기타 언어
            }
    
    def load_model(self):
        """RTX 3070 최적화 모델 로드"""
        try:
            from transformers import BitsAndBytesConfig
            from peft import PeftModel
            
            # 모델 경로 찾기
            model_paths = ["./models/rtx3070_optimized_best", "./models/rtx3070_optimized_final"]
            model_path = None
            
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                mode_text = "언어 제한" if self.language_restriction else "무제한"
                log_print(f"RTX 3070 최적화 모델을 찾을 수 없음 ({mode_text})", "model_loading")
                return False
            
            mode_text = "언어 제한" if self.language_restriction else "무제한"
            log_print(f"RTX 3070 {mode_text} 모델 로딩: {model_path}", "model_loading")
            
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 4bit 양자화 설정
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # 기본 모델 로드
            base = AutoModelForCausalLM.from_pretrained(
                "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA 어댑터 로드
            self.model = PeftModel.from_pretrained(base, model_path)
            self.model.eval()
            
            self.loaded = True
            log_print(f"RTX 3070 {mode_text} 모델 로드 완료", "model_loading")
            return True
            
        except Exception as e:
            mode_text = "언어 제한" if self.language_restriction else "무제한"
            log_print(f"RTX 3070 {mode_text} 모델 로드 실패: {e}", "model_loading")
            return False
    
    def detect_language(self, text):
        """언어 감지 (언어 제한 모드에서만 사용)"""
        if not self.language_restriction:
            return 'unlimited'
        
        korean_chars = len(self.language_patterns['ko'].findall(text))
        english_chars = len(self.language_patterns['en'].findall(text))
        other_chars = len(self.language_patterns['other'].findall(text))
        
        total_chars = korean_chars + english_chars + other_chars
        
        if total_chars == 0:
            return 'en'
        
        if other_chars / total_chars > 0.2:
            return 'other'
        
        if korean_chars > english_chars:
            return 'ko'
        else:
            return 'en'
    
    def validate_language(self, user_input):
        """언어 유효성 검사 (언어 제한 모드에서만 사용)"""
        if not self.language_restriction:
            return True, 'unlimited'
        
        detected_lang = self.detect_language(user_input)
        
        if detected_lang == 'other':
            return False, "❌ 지원하지 않는 언어입니다. 영어 또는 한국어만 사용해주세요."
        
        return True, detected_lang
    
    def generate_response(self, prompt, max_tokens=200):
        """응답 생성 (언어 제한 옵션 포함)"""
        if not self.loaded:
            return None
        
        # 언어 제한 모드에서 언어 유효성 검사
        if self.language_restriction:
            is_valid, lang_result = self.validate_language(prompt)
            if not is_valid:
                return lang_result
            
            detected_lang = lang_result
            log_print(f"언어 감지됨: {detected_lang}", "model_loading")
        
        try:
            # 프롬프트 형식화
            if self.language_restriction:
                # 언어 제한 모드에서는 더 엄격한 프롬프트
                formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                # 무제한 모드에서는 단순한 프롬프트
                formatted_prompt = f"User: {prompt}\nAssistant:"
            
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.9,
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 응답에서 프롬프트 부분 제거
            if self.language_restriction:
                if "<|im_start|>assistant\n" in response:
                    response = response.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0]
            else:
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1]
            
            return response.strip()
        
        except Exception as e:
            mode_text = "언어 제한" if self.language_restriction else "무제한"
            log_print(f"RTX 3070 {mode_text} 응답 생성 실패: {e}", "model_loading")
            return None

# 전역 RTX 3070 모델 인스턴스들 (모두 영어/한국어 제한)
rtx3070_unfiltered_model = RTX3070OptimizedLoader(language_restriction=True)
rtx3070_language_limited_model = RTX3070OptimizedLoader(language_restriction=True)

# 모델 선택 상태 저장
MODEL_SELECTION = {
    "current": "original",  # "original", "hybrid", "english_unified"
    "available_models": []
}

def detect_available_models():
    """사용 가능한 모델들 감지"""
    available = ["original"]  # 기본 모델은 항상 사용 가능
    
    # 하이브리드 파인튜닝 모델 확인
    if FINETUNING_ENABLED:
        try:
            stats = get_finetuning_stats()
            if stats.get('status') == 'active':
                available.append("hybrid")
        except:
            pass
    
    # 영어 파인튜닝 모델 확인
    if ENGLISH_FINETUNED_AVAILABLE:
        unified_path = "./finetuning/models/unified_model"
        if os.path.exists(unified_path):
            available.append("english_unified")
    
    # RTX 3070 최적화 모델들 확인
    rtx_best_path = "./models/rtx3070_optimized_best"
    rtx_final_path = "./models/rtx3070_optimized_final"
    
    if os.path.exists(rtx_best_path):
        available.append("rtx3070_unfiltered")
        available.append("rtx3070_language_limited")
    elif os.path.exists(rtx_final_path):
        available.append("rtx3070_unfiltered")
        available.append("rtx3070_language_limited")
    
    MODEL_SELECTION["available_models"] = available
    log_print(f"사용 가능한 모델: {available}", "model_loading")
    return available

def switch_model(model_type):
    """모델 전환"""
    if model_type not in MODEL_SELECTION["available_models"]:
        return False, f"모델 '{model_type}'을 사용할 수 없습니다."
    
    # RTX 3070 무제한 모델 로드
    if model_type == "rtx3070_unfiltered":
        if rtx3070_unfiltered_model.load_model():
            MODEL_SELECTION["current"] = model_type
            return True, "RTX 3070 최적화 (무제한) 모델로 전환되었습니다."
        else:
            return False, "RTX 3070 무제한 모델 로드에 실패했습니다."
    
    # RTX 3070 언어 제한 모델 로드
    elif model_type == "rtx3070_language_limited":
        if rtx3070_language_limited_model.load_model():
            MODEL_SELECTION["current"] = model_type
            return True, "RTX 3070 최적화 (언어 제한) 모델로 전환되었습니다."
        else:
            return False, "RTX 3070 언어 제한 모델 로드에 실패했습니다."
    
    # 영어 통합 모델 로드
    elif model_type == "english_unified":
        if english_finetuned_model.load_unified_model():
            MODEL_SELECTION["current"] = model_type
            return True, "영어 통합 파인튜닝 모델로 전환되었습니다."
        else:
            return False, "영어 통합 모델 로드에 실패했습니다."
    
    # 다른 모델들
    MODEL_SELECTION["current"] = model_type
    return True, f"모델이 '{model_type}'로 전환되었습니다."

def get_model_info():
    """현재 모델 정보 반환"""
    current = MODEL_SELECTION["current"]
    available = MODEL_SELECTION["available_models"]
    
    model_names = {
        "original": "기본 Nous-Hermes-2-Mistral",
        "hybrid": "하이브리드 파인튜닝 (한국어)",
        "english_unified": "영어 통합 파인튜닝",
        "rtx3070_unfiltered": "RTX 3070 최적화 (영어/한국어)",
        "rtx3070_language_limited": "RTX 3070 최적화 (영어/한국어)"
    }
    
    return {
        "current": current,
        "current_name": model_names.get(current, current),
        "available": available,
        "available_names": [model_names.get(m, m) for m in available]
    }

# 메인 실행부
if __name__ == "__main__":
    main()