#!/usr/bin/env python3
"""
RTX 3070 GGUF 모델을 Q4_K_M으로 양자화하는 스크립트
13.5GB → 약 4-5GB로 크기 축소
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🔥 RTX 3070 GGUF 모델 Q4_K_M 양자화 시작")
    print("=" * 60)
    
    # 경로 설정
    project_root = Path("e:/Work/ai pz2")
    original_model = project_root / "models" / "rtx3070_final_merged.gguf"
    quantized_model = project_root / "models" / "rtx3070_final_merged_q4km.gguf"
    llama_cpp_dir = project_root / "llama.cpp"
    
    print(f"📂 원본 모델: {original_model}")
    print(f"📂 양자화 모델: {quantized_model}")
    print(f"📂 llama.cpp 디렉토리: {llama_cpp_dir}")
    
    # 원본 모델 존재 확인
    if not original_model.exists():
        print(f"❌ 원본 GGUF 모델을 찾을 수 없습니다: {original_model}")
        return False
    
    # 원본 모델 크기 확인
    original_size_gb = original_model.stat().st_size / (1024**3)
    print(f"📊 원본 모델 크기: {original_size_gb:.1f}GB")
    
    # llama.cpp 디렉토리 확인
    if not llama_cpp_dir.exists():
        print(f"❌ llama.cpp 디렉토리를 찾을 수 없습니다: {llama_cpp_dir}")
        return False
    
    # 양자화 방법들 시도
    print("\n🔧 양자화 방법 시도 중...")
    
    # 방법 1: llama.cpp의 quantize 실행파일 사용
    quantize_exe = None
    possible_paths = [
        llama_cpp_dir / "build" / "bin" / "quantize.exe",
        llama_cpp_dir / "build" / "Release" / "quantize.exe",
        llama_cpp_dir / "build" / "Debug" / "quantize.exe",
        llama_cpp_dir / "quantize.exe",
        llama_cpp_dir / "build" / "quantize.exe"
    ]
    
    for path in possible_paths:
        if path.exists():
            quantize_exe = path
            print(f"✅ quantize 실행파일 발견: {quantize_exe}")
            break
    
    if quantize_exe:
        try:
            print(f"⚡ quantize.exe로 Q4_K_M 양자화 시작...")
            cmd = [
                str(quantize_exe),
                str(original_model),
                str(quantized_model),
                "Q4_K_M"
            ]
            
            print(f"🔄 실행 명령: {' '.join(cmd)}")
            
            # 양자화 실행
            result = subprocess.run(
                cmd,
                cwd=str(llama_cpp_dir),
                capture_output=True,
                text=True,
                timeout=1800  # 30분 타임아웃
            )
            
            if result.returncode == 0:
                print("✅ quantize.exe로 양자화 성공!")
                print("📤 출력:", result.stdout)
                
                # 결과 확인
                if quantized_model.exists():
                    quantized_size_gb = quantized_model.stat().st_size / (1024**3)
                    compression_ratio = (1 - quantized_size_gb / original_size_gb) * 100
                    
                    print(f"🎉 양자화 완료!")
                    print(f"📊 양자화 모델 크기: {quantized_size_gb:.1f}GB")
                    print(f"📉 압축률: {compression_ratio:.1f}% 절약")
                    print(f"💾 저장 위치: {quantized_model}")
                    return True
                else:
                    print("❌ 양자화 파일이 생성되지 않았습니다.")
                    return False
            else:
                print(f"❌ quantize.exe 실행 실패: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print("❌ 양자화 시간 초과 (30분)")
            return False
        except Exception as e:
            print(f"❌ quantize.exe 실행 중 오류: {e}")
    
    # 방법 2: Python llama-cpp-python으로 양자화 시도
    print("\n🐍 Python llama-cpp-python으로 시도...")
    
    try:
        from llama_cpp import Llama
        
        print("⚡ llama-cpp-python으로 모델 로드 및 양자화...")
        
        # 원본 모델 로드
        llm = Llama(
            model_path=str(original_model),
            n_gpu_layers=0,  # CPU로 로드
            n_ctx=512,       # 메모리 절약
            n_threads=8,
            verbose=True
        )
        
        print("✅ 모델 로드 완료")
        print("❌ llama-cpp-python은 직접 양자화를 지원하지 않습니다")
        print("💡 대안: quantize.exe를 직접 빌드하거나 다른 도구 사용 필요")
        
    except ImportError:
        print("❌ llama-cpp-python이 설치되지 않았습니다")
    except Exception as e:
        print(f"❌ Python 양자화 실패: {e}")
    
    # 방법 3: ggml-python 사용 시도
    print("\n🔄 대안: llama.cpp 빌드 및 quantize.exe 생성...")
    
    try:
        # CMake로 빌드 시도
        print("⚡ CMake로 llama.cpp 빌드 시도...")
        
        build_dir = llama_cpp_dir / "build"
        if not build_dir.exists():
            build_dir.mkdir()
        
        # CMake 설정
        cmake_cmd = [
            "cmake",
            "..",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_SERVER=OFF"
        ]
        
        print(f"🔧 CMake 설정: {' '.join(cmake_cmd)}")
        result = subprocess.run(
            cmake_cmd,
            cwd=str(build_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print("✅ CMake 설정 성공")
            
            # 빌드 실행
            build_cmd = ["cmake", "--build", ".", "--config", "Release"]
            print(f"🔨 빌드 실행: {' '.join(build_cmd)}")
            
            result = subprocess.run(
                build_cmd,
                cwd=str(build_dir),
                capture_output=True,
                text=True,
                timeout=1200  # 20분
            )
            
            if result.returncode == 0:
                print("✅ llama.cpp 빌드 성공!")
                
                # quantize.exe 재확인
                for path in possible_paths:
                    if path.exists():
                        print(f"🎯 quantize.exe 발견: {path}")
                        # 다시 양자화 시도
                        return main()  # 재귀 호출
                
                print("❌ 빌드 후에도 quantize.exe를 찾을 수 없습니다")
            else:
                print(f"❌ 빌드 실패: {result.stderr}")
        else:
            print(f"❌ CMake 설정 실패: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ 빌드 시간 초과")
    except Exception as e:
        print(f"❌ 빌드 중 오류: {e}")
    
    print("\n💡 수동 해결 방법:")
    print("1. Visual Studio Community 2022 설치")
    print("2. CMake 3.12+ 설치")
    print("3. llama.cpp 폴더에서:")
    print("   mkdir build && cd build")
    print("   cmake ..")
    print("   cmake --build . --config Release")
    print("4. 생성된 quantize.exe로 수동 양자화:")
    print(f"   quantize.exe {original_model} {quantized_model} Q4_K_M")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 양자화 완료! RTX 3070에서 GPU 가속으로 실행 가능합니다.")
    else:
        print("\n😅 자동 양자화 실패. 수동 빌드가 필요합니다.")
    
    input("\n아무 키나 눌러 종료...")
