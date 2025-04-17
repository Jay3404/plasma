import streamlit as st
import requests # API 호출용
import pandas as pd # 파장 이름 로딩용
from pathlib import Path # pathlib 임포트
import numpy as np
import os

# --- 설정 ---
# FastAPI 서버 주소 (uvicorn 실행 시 터미널에 나오는 주소)
API_BASE_URL = "http://127.0.0.1:8000"

# --- 스크립트 기준 경로 설정 ---
# 현재 파일(streamlit_app.py)가 있는 디렉토리
SCRIPT_DIR = Path(__file__).resolve().parent
# 한 칸 위로 옮기기.
PROJECT_ROOT = SCRIPT_DIR.parent.parent
# 프로젝트 루트를 기준으로 데이터 디렉토리 및 파일 경로 조합
DATA_DIR_NAME = "ML_Data" # 실제 디렉토리 이름과 대소문자 일치 확인!
FILENAME = 'ML_data_for_learning_N1000.txt' # 실제 파일 이름과 대소문자 일치 확인!
DATA_FILE_PATH = PROJECT_ROOT / DATA_DIR_NAME / FILENAME

# --- 파장 이름 로딩 함수 (FastAPI와 동일 또는 유사하게) ---
@st.cache_data
def load_wavelengths(filename):
    try:
        df = pd.read_csv(filename, delim_whitespace=True, header=None, nrows=1)
        wavelengths_str = df.iloc[0, 1:].values
        try:
            wavelengths = [float(wl) for wl in wavelengths_str]
        except ValueError:
            wavelengths = list(wavelengths_str)
        # 가정: 항상 11개 파장 사용
        if len(wavelengths) == 11:
            return wavelengths
        else:
            st.error("파장 데이터 로딩 오류: 11개 파장이 아닙니다.")
            return None
    except FileNotFoundError:
        st.warning(f"'{filename}' 파일 찾기 실패. 기본 레이블 사용.")
        return [f"파장 {i+1}" for i in range(11)]
    except Exception as e:
        st.warning(f"파장 로딩 오류: {e}. 기본 레이블 사용.")
        return [f"파장 {i+1}" for i in range(11)]

# --- 저장된 모델 목록 가져오기 함수 ---
def get_available_models():
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status() # 오류 발생 시 예외 발생
        models = response.json() # [{"name": "model1.joblib"}, ...]
        model_names = ["latest"] + [m["name"] for m in models] # 최신 옵션 추가
        return model_names
    except requests.exceptions.RequestException as e:
        st.error(f"API 서버에서 모델 목록을 가져오는 데 실패했습니다: {e}")
        return ["latest"] # 실패 시 기본값

# --- 앱 인터페이스 ---
st.set_page_config(page_title="Te 예측기", layout="wide")
st.title("플라즈마 전자 온도 (Te) 예측")

# 사용 가능한 모델 목록 가져오기
available_models = get_available_models()

# 모델 선택
selected_model = st.selectbox("사용할 모델 선택:", available_models)

st.divider()

# 파장 이름 로드
wavelength_labels = load_wavelengths(DATA_FILE_PATH)

if wavelength_labels:
    st.subheader("각 파장에서의 빛의 세기(Intensity) 입력:")

    # 입력 필드 (11개)
    input_intensities = []
    cols = st.columns(4) # 4열로 배치
    for i in range(11):
        label = f"{wavelength_labels[i]}"
        # number_input 사용, key 필수
        intensity = cols[i % 4].number_input(label=label, value=0.0, format="%.2f", key=f"intensity_{i}")
        input_intensities.append(intensity)

    # 예측 버튼
    if st.button("전자 온도 예측하기"):
        # 입력 데이터 준비
        predict_payload = {
            "model_name": selected_model,
            "intensities": input_intensities
        }

        # FastAPI 예측 API 호출
        try:
            st.info("API 서버에 예측 요청 중...")
            response = requests.post(f"{API_BASE_URL}/predict", json=predict_payload)
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생

            result = response.json() # {"predicted_te": 3.5284, "model_used": "..."}

            st.success(f"예측된 전자 온도 (Te): {result['predicted_te']:.4f} (eV?)")
            st.caption(f"사용된 모델: {result['model_used']}")

        except requests.exceptions.RequestException as e:
            st.error(f"API 호출 실패: {e}")
            try:
                # FastAPI에서 보낸 상세 오류 메시지 표시 시도
                error_detail = response.json().get('detail', '상세 정보 없음')
                st.error(f"서버 응답: {error_detail}")
            except:
                pass # JSON 파싱 실패 등
        except Exception as e:
            st.error(f"예측 처리 중 예상치 못한 오류 발생: {e}")

else:
    st.error("파장 정보를 로드할 수 없어 입력을 진행할 수 없습니다.")

st.sidebar.header("API 서버 정보")
st.sidebar.write(f"백엔드 API 주소: {API_BASE_URL}")
st.sidebar.info("FastAPI 서버가 먼저 실행 중이어야 합니다 (`uvicorn main:app --reload`).")