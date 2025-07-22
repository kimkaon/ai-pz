"""
메뉴 시스템 모듈
main.py에서 분리된 모든 메뉴 관련 기능
"""

import os
import time
from utils.logging_utils import log_print, toggle_verbose_mode, is_verbose_mode, settings_manager

# 파인튜닝 기능 사용 가능 여부 확인
try:
    from finetuning_integration import get_finetuning_stats
    FINETUNING_ENABLED = True
except ImportError:
    FINETUNING_ENABLED = False

class MenuSystem:
    """메뉴 시스템 클래스"""
    
    def __init__(self, model_manager, response_handler):
        self.model_manager = model_manager
        self.response_handler = response_handler
        self.conversation_history = []
    
    def run_main_menu(self):
        """메인 메뉴 실행"""
        # 마이크 설정 불러오기
        from utils.logging_utils import load_mic_index
        device_index = load_mic_index()
        
        print("🎯 모든 준비 완료! 어시스턴트 시작합니다.")
        
        while True:
            self.show_main_menu()
            
            mode = input(f"메뉴 (s:음성, m:텍스트, md:모델, st:설정, q:종료): ").strip().lower()
            
            if mode == "q":
                print("👋 프로그램을 종료합니다.")
                return
            elif mode == "st":
                self.handle_settings_menu()
            elif mode == "md":
                self.handle_model_selection()
            elif mode == "s":
                self.handle_voice_conversation()
            elif mode == "m":
                self.handle_text_conversation()
            else:
                print("❌ 잘못된 선택입니다.")
                input("계속하려면 Enter를 누르세요...")
    
    def handle_voice_conversation(self):
        """음성 대화 처리"""
        print("\n🎤 음성 대화 모드입니다. 'q'를 입력하면 메인 메뉴로 돌아갑니다.")
        
        from voice_utils import record_voice, transcribe
        
        while True:
            print("\n음성 입력을 시작하세요...")
            
            try:
                # 음성 녹음
                audio_data, sample_rate = record_voice()
                if audio_data is None:
                    print("❌ 음성 녹음에 실패했습니다.")
                    continue
                
                # 음성 인식
                print("🔄 음성을 텍스트로 변환 중...")
                user_input = transcribe(audio_data, sample_rate)
                
                if not user_input:
                    print("❌ 음성 인식에 실패했습니다.")
                    continue
                
                print(f"👤 입력: {user_input}")
                
                if user_input.lower() == 'q':
                    break
                
                # 응답 생성 및 출력
                self._process_user_input(user_input, use_voice=True)
                
            except KeyboardInterrupt:
                print("\n음성 대화를 종료합니다.")
                break
            except Exception as e:
                log_print(f"음성 대화 오류: {e}", "error")
                print(f"❌ 오류가 발생했습니다: {e}")
    
    def handle_text_conversation(self):
        """텍스트 대화 처리"""
        print("\n💬 텍스트 대화 모드입니다. 'q'를 입력하면 메인 메뉴로 돌아갑니다.")
        
        while True:
            user_input = input("\n👤 입력: ").strip()
            
            if user_input.lower() == 'q':
                break
            
            if not user_input:
                continue
            
            # 응답 생성 및 출력
            self._process_user_input(user_input, use_voice=False)
    
    def _process_user_input(self, user_input, use_voice=False):
        """사용자 입력 처리 및 응답 생성"""
        try:
            # 응답 생성
            response = self.response_handler.generate_response(user_input, self.conversation_history)
            
            if response:
                print(f"🤖 응답: {response}")
                
                # 대화 히스토리에 추가
                self.conversation_history.append({"user": user_input, "assistant": response})
                
                # 음성 출력 (음성 모드인 경우)
                if use_voice:
                    try:
                        from openvoice_tts import synthesize_with_openvoice
                        synthesize_with_openvoice(response)
                    except Exception as e:
                        log_print(f"TTS 오류: {e}", "error")
                        print("❌ 음성 출력에 실패했습니다.")
            else:
                print("❌ 응답 생성에 실패했습니다.")
                
        except Exception as e:
            log_print(f"입력 처리 오류: {e}", "error")
            print(f"❌ 오류가 발생했습니다: {e}")
    
    def show_main_menu(self):
        """메인 메뉴 표시"""
        current_verbose = "ON" if is_verbose_mode() else "OFF"
        model_info = self.model_manager.get_model_info()
        current_model_name = model_info["current_name"]
        
        print(f"\n{'='*50}")
        print("🎯 AI 어시스턴트")
        print(f"{'='*50}")
        print(f"🤖 현재 모델: {current_model_name}")
        print(f"📝 로그: {current_verbose}")
        
        if settings_manager.get('ui.show_tips', True):
            print("\n💡 빠른 시작:")
            print("   • 's': 음성 대화    • 'm': 텍스트 대화")
            print("   • 'md': 모델 변경   • 'st': 모든 설정")
        
        print(f"{'='*50}")

    def handle_settings_menu(self):
        """설정 메뉴 처리"""
        while True:
            print(f"\n{'='*50}")
            print("⚙️ 설정 및 고급 메뉴")
            print(f"{'='*50}")
            
            # 현재 상태 표시
            model_info = self.model_manager.get_model_info()
            current_model_name = model_info["current_name"]
            current_verbose = "ON" if is_verbose_mode() else "OFF"
            
            # 하이브리드 파인튜닝 상태 확인
            finetuning_status = "OFF"
            if FINETUNING_ENABLED:
                try:
                    from finetuning_integration import get_finetuning_stats
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
            
            print(f"🤖 현재 모델: {current_model_name}")
            print(f"🧠 파인튜닝: {finetuning_status}")
            if FINETUNING_ENABLED and finetuning_status.startswith("ON"):
                try:
                    from finetuning_integration import get_finetuning_stats
                    stats = get_finetuning_stats()
                    print(f"   💾 통합모델: {'로드됨' if stats.get('unified_model_loaded') else '미로드'}")
                    print(f"   📈 통계: 통합응답 {stats.get('unified_responses', 0)}회, 전문응답 {stats.get('specialist_responses', 0)}회")
                except:
                    pass
            print(f"📝 상세 로그: {current_verbose}")
            
            print("\n설정 메뉴:")
            print("1. 마이크 설정 (p)")
            print("2. 로그 설정 (lg)")
            print("3. 파인튜닝 관리 (ft)")
            print("4. TTS 통계 (ts)")
            print("5. 실시간 통계 (rt)")
            print("6. 통합 설정 관리")
            print("0. 메인 메뉴로 돌아가기")
            
            st_choice = input("\n선택: ").strip().lower()
            
            if st_choice == "0" or st_choice == "":
                break
            elif st_choice == "1" or st_choice == "p":
                self.handle_microphone_settings()
            elif st_choice == "2" or st_choice == "lg":
                self.handle_log_settings()
            elif st_choice == "3" or st_choice == "ft":
                self.handle_finetuning_menu()
            elif st_choice == "4" or st_choice == "ts":
                self.handle_tts_stats()
            elif st_choice == "5" or st_choice == "rt":
                self.handle_realtime_stats()
            elif st_choice == "6":
                self.handle_advanced_settings()
            else:
                print("❌ 잘못된 선택입니다.")
                input("\n계속하려면 Enter를 누르세요...")

    def handle_microphone_settings(self):
        """마이크 설정 처리"""
        from voice_utils import select_input_device
        
        device_index = select_input_device()
        if device_index is not None:
            settings_manager.set_microphone_device(device_index)
            print(f"✅ 선택된 마이크 index: {device_index} (설정에 저장됨)")
        else:
            print("❌ 마이크 선택이 취소되었습니다.")
        input("\n계속하려면 Enter를 누르세요...")

    def handle_log_settings(self):
        """로그 설정 처리"""
        settings = toggle_verbose_mode()
        new_verbose = "ON" if settings["verbose_mode"] else "OFF"
        print(f"✅ 로그 모드가 [{new_verbose}]로 변경되었습니다.")
        log_print(f"로그 설정 변경: {new_verbose}", "general")
        input("\n계속하려면 Enter를 누르세요...")

    def handle_model_selection(self):
        """모델 선택 메뉴 처리"""
        while True:
            print(f"\n{'='*50}")
            print("🤖 모델 선택 메뉴")
            print(f"{'='*50}")
            
            # 현재 모델 정보 표시
            model_info = self.model_manager.get_model_info()
            current_model = model_info["current"]
            current_model_name = model_info["current_name"]
            available_models = model_info["available"]
            available_names = model_info["available_names"]
            
            print(f"현재 모델: {current_model_name}")
            print("\n사용 가능한 모델들:")
            
            # 모델 옵션 표시
            model_options = {}
            for i, (model_id, model_name) in enumerate(zip(available_models, available_names), 1):
                status = " (현재)" if model_id == current_model else ""
                print(f"{i}. {model_name}{status}")
                model_options[str(i)] = model_id
            
            print("\n모델 설명:")
            descriptions = {
                "original": "기본 Nous-Hermes-2-Mistral 7B 모델 (허깅페이스)",
                "hybrid": "하이브리드 파인튜닝 시스템 (한국어 특화)",
                "english_unified": "영어 통합 파인튜닝 모델",
                "rtx3070_unfiltered": "RTX 3070용 파인튜닝 모델 (무제한, LoRA)",
                "rtx3070_language_limited": "RTX 3070용 파인튜닝 모델 (언어 제한, LoRA)",
                "rtx3070_gguf": "RTX 3070용 파인튜닝 모델 (GGUF, 고속)"
            }
            
            for model_id, model_name in zip(available_models, available_names):
                desc = descriptions.get(model_id, "설명 없음")
                print(f"   • {model_name}: {desc}")
            
            print(f"\n0. 뒤로 가기")
            print(f"{'='*50}")
            
            choice = input("모델을 선택하세요 (번호 입력): ").strip()
            
            if choice == "0":
                break
            elif choice in model_options:
                selected_model = model_options[choice]
                if selected_model == current_model:
                    print(f"✅ 이미 '{model_info['available_names'][available_models.index(selected_model)]}'를 사용 중입니다.")
                else:
                    print(f"🔄 '{model_info['available_names'][available_models.index(selected_model)]}'로 전환 중...")
                    success, message = self.model_manager.switch_model(selected_model)
                    if success:
                        print(f"✅ {message}")
                    else:
                        print(f"❌ {message}")
            else:
                print("❌ 잘못된 선택입니다.")
            
            input("\n계속하려면 Enter를 누르세요...")

    def handle_finetuning_menu(self):
        """파인튜닝 관리 메뉴"""
        if not FINETUNING_ENABLED:
            print("❌ 하이브리드 파인튜닝 기능이 비활성화되어 있습니다.")
            print("   hybrid_finetuning_integration.py를 확인하세요.")
            input("\n계속하려면 Enter를 누르세요...")
            return
        
        print(f"\n{'='*40}")
        print("🔧 하이브리드 파인튜닝 관리")
        print(f"{'='*40}")
        
        try:
            from finetuning_integration import get_finetuning_stats, request_specialist_mode
            from processing.response_handler import try_finetuned_response
            
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
            print("❌ 사용 가능한 파인튜닝 모델이 없습니다.")
            print("📁 파인튜닝을 시작하려면 run_finetuning.bat를 실행하세요.")
        
        input("\n계속하려면 Enter를 누르세요...")

    def handle_tts_stats(self):
        """TTS 통계 처리"""
        print(f"\n{'='*40}")
        print("📊 TTS 통계 및 캐시 관리")
        print(f"{'='*40}")
        
        try:
            from fast_tts import get_tts_stats
            stats = get_tts_stats()
            print(f"📈 캐시 통계:")
            print(f"   • 총 요청: {stats['total_requests']}")
            print(f"   • 캐시 히트: {stats['hits']} ({stats['hit_rate']})")
            print(f"   • 캐시 미스: {stats['misses']}")
            print(f"   • 새로 생성: {stats['created']}")
            print(f"   • 메모리 캐시: {stats['memory_cache_size']}개")
            print(f"   • 디스크 캐시: {stats['disk_cache_size']}개")
            
            if stats['total_requests'] > 0:
                print(f"🚀 TTS 캐시 효율: {stats['hit_rate']}")
                if float(stats['hit_rate'].replace('%', '')) > 50:
                    print("✅ 캐시가 효율적으로 작동하고 있습니다!")
                else:
                    print("💡 더 많은 대화를 나누면 캐시 효율이 향상됩니다.")
            
        except Exception as e:
            print(f"❌ TTS 통계 조회 오류: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")

    def handle_realtime_stats(self):
        """실시간 통계 처리"""
        print(f"\n{'='*40}")
        print("⚡ 실시간 대화 통계")
        print(f"{'='*40}")
        
        try:
            from realtime_chat import get_realtime_stats
            from fast_tts import get_tts_stats
            
            rt_stats = get_realtime_stats()
            print(f"📊 실시간 처리 통계:")
            print(f"   • 처리된 문장: {rt_stats['total_sentences']}개")
            print(f"   • 평균 TTS 시간: {rt_stats['avg_tts_time_ms']}")
            print(f"   • TTS 큐 대기: {rt_stats['tts_queue_size']}개")
            print(f"   • 오디오 큐 대기: {rt_stats['audio_queue_size']}개")
            
            if rt_stats['total_sentences'] > 0:
                print(f"🚀 실시간 처리 효율:")
                if rt_stats['avg_tts_time'] < 1.0:
                    print("   ✅ 매우 빠름 (실시간 대화 가능)")
                elif rt_stats['avg_tts_time'] < 2.0:
                    print("   ✅ 빠름 (자연스러운 대화)")
                else:
                    print("   ⚠️ 보통 (최적화 필요)")
            else:
                print("   📝 아직 실시간 처리 기록이 없습니다.")
            
            # TTS 통계도 함께 표시
            tts_stats = get_tts_stats()
            print(f"\n🎵 TTS 캐시 통계:")
            print(f"   • 캐시 효율: {tts_stats['hit_rate']}")
            print(f"   • 캐시된 항목: {tts_stats['disk_cache_size']}개")
            
        except Exception as e:
            print(f"❌ 실시간 통계 조회 오류: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")

    def handle_advanced_settings(self):
        """고급 설정 관리"""
        print(f"\n{'='*50}")
        print("⚙️ 통합 설정 관리")
        print(f"{'='*50}")
        
        print(settings_manager.get_settings_summary())
        
        print("\n설정 관리 옵션:")
        print("1. 설정 요약 보기")
        print("2. 설정 내보내기 (백업)")
        print("3. 설정 가져오기 (복원)")
        print("4. 기본 설정으로 초기화")
        print("5. 모델 설정 초기화")
        print("6. 로그 설정 초기화")
        print("7. 설정 파일 위치 보기")
        print("0. 돌아가기")
        
        try:
            st_choice = input("\n선택: ").strip()
            
            if st_choice == "0":
                return
            elif st_choice == "1":
                print("\n📋 현재 설정 상세:")
                print(settings_manager.get_settings_summary())
                
                print("\n🔧 상세 설정:")
                print(f"   • 모델 자동 저장: {'ON' if settings_manager.get('model.auto_save') else 'OFF'}")
                print(f"   • 대화 기록 길이: {settings_manager.get('conversation.history_max_length')}개")
                print(f"   • TTS 캐시 크기: {settings_manager.get('tts.cache_max_size')}개")
                print(f"   • 실시간 처리: {'ON' if settings_manager.get('realtime.enabled') else 'OFF'}")
                
            elif st_choice == "2":
                try:
                    backup_path = settings_manager.export_settings()
                    print(f"✅ 설정이 백업되었습니다: {backup_path}")
                except Exception as e:
                    print(f"❌ 백업 실패: {e}")
                    
            elif st_choice == "3":
                backup_file = input("백업 파일 경로를 입력하세요: ").strip()
                if backup_file and os.path.exists(backup_file):
                    try:
                        settings_manager.import_settings(backup_file)
                        print("✅ 설정이 복원되었습니다.")
                    except Exception as e:
                        print(f"❌ 복원 실패: {e}")
                else:
                    print("❌ 파일을 찾을 수 없습니다.")
                    
            elif st_choice == "4":
                confirm = input("❓ 모든 설정을 기본값으로 초기화하시겠습니까? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    settings_manager.reset_to_defaults()
                    print("✅ 모든 설정이 기본값으로 초기화되었습니다.")
                else:
                    print("초기화가 취소되었습니다.")
                    
            elif st_choice == "5":
                confirm = input("❓ 모델 설정만 초기화하시겠습니까? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    settings_manager.reset_to_defaults('model')
                    print("✅ 모델 설정이 기본값으로 초기화되었습니다.")
                else:
                    print("초기화가 취소되었습니다.")
                    
            elif st_choice == "6":
                confirm = input("❓ 로그 설정만 초기화하시겠습니까? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    settings_manager.reset_to_defaults('logging')
                    print("✅ 로그 설정이 기본값으로 초기화되었습니다.")
                else:
                    print("초기화가 취소되었습니다.")
                    
            elif st_choice == "7":
                settings_file = os.path.abspath("ai_assistant_settings.json")
                print(f"📁 설정 파일 위치: {settings_file}")
                if os.path.exists(settings_file):
                    print("✅ 설정 파일이 존재합니다.")
                    file_size = os.path.getsize(settings_file)
                    print(f"📏 파일 크기: {file_size} bytes")
                    mtime = os.path.getmtime(settings_file)
                    print(f"📅 마지막 수정: {time.ctime(mtime)}")
                else:
                    print("❌ 설정 파일을 찾을 수 없습니다.")
                    
            else:
                print("❌ 잘못된 선택입니다.")
                
        except Exception as e:
            print(f"❌ 설정 관리 오류: {e}")
        
        input("\n계속하려면 Enter를 누르세요...")
