# 1. 리더 보드
평가 산식 : Brier Skill Score
    본 대회는 각 투구의 control_success = 1일 확률을 예측하는 확률 예측 과제입니다. 추론 확률이 실제 정답에 가까울수록 높은 점수를 받습니다.
    Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))

    Brier Score = mean((p_i - y_i)^2)
    r = mean(y_i)
    평균 제구율 Brier Score = r × (1 - r)

    Public Score : 전체 테스트 데이터 100%
    Private Score : 대회 종료 시점의 Public Score

# 2. 평가 방식

LG Aimers 수료 조건
Phase1을 이수하고 Phase2의 Public Score (LB: 549.51) 이상
기준 점수는 운영진이 제공한 베이스라인 추론 코드를 운영진 평가 환경에서 실행했을 때의 점수를 기준으로 측정
1차 평가 : 리더보드 Private Score 100%
동점자의 경우, 기존 리더보드 순위 산정 방식을 따름 [링크]의 '리더보드 점수' 부분을 참고
2차 평가 : 오프라인 해커톤(Phase3) 진출을 희망하는 팀은 코드 제출 후 코드 검증
Private 리더보드 상위팀(약 100명)은 코드 및 PPT 필수 제출 대상
코드 및 PPT 제출과 검증를 모두 통과한 Private 리더보드 상위팀(약 100명)이 오프라인 해커톤(Phase3) 진출

# 3. 코드 제출 대회 가이드
본 대회는 submit.zip 파일을 제출하는 방식의 '코드 제출 대회'로 진행됩니다. (기본 가이드 문서)

참가자는 아래와 같은 구조로 submit.zip을 구성하여 제출해야 합니다.

아래의 구조와 동일하고 디렉토리 명과 파일 명을 모두 일치 시켜야합니다.

📁 제출 파일 구조 (submit.zip)
submit.zip
├── model/        # 모델 가중치 파일을 저장하는 디렉토리
│      └── (예: model.pt 등)
├── script.py       # 실제 추론이 수행되는 실행 코드
└── requirements.txt   # 필요한 패키지 및 버전 명시

script.py는 submit.zip을 제출 시 평가 서버에서 자동으로 실행됩니다.
requirements.txt는 pip install -r requirements.txt 명령어로 설치 가능한 형태여야 하며, 추론 시 필요한 모든 패키지를 포함해야 합니다.
submit.zip 내 구조는 반드시 일치해야하며, 추가 최상위 폴더가 zip 구조 내 존재하는 경우 등 구조가 불일치하는 경우 설치 오류가 발생합니다.

- 평가 데이터 구성
샘플 평가 데이터 (참가자에게 제공)
폴더 구조와 파일 형식이 실제 평가 데이터와 동일
소량의 더미 평가 샘플 데이터 포함
로컬에서 추론 코드 개발 및 테스트 용도
실제 평가 데이터 (평가 시 자동 적용)
동일한 폴더 구조와 파일 형식
실제 평가에 사용될 전체 데이터셋
⚙️ 평가 서버에서 추가되는 항목
제출 시, 평가 서버에서 참가자가 제출한 submit.zip 파일에는 아래 항목이 자동으로 추가됩니다.

submit.zip
├── model/        # 참가자 구성
├── script.py       # 참가자 구성
├── requirements.txt   # 참가자 구성
├── data/         # 평가에 사용될 테스트 데이터 (디렉토리 자동 생성)
└── output/submission.csv        # 참가자 추론 결과가 저장되는 경로 (디렉토리 자동 생성)
data/ 디렉토리는 실제 평가 데이터를 포함한 경진대회 데이터가 포함되며, 읽기전용으로 쓰기 및 수정이 불가능한 디렉토리입니다.
output/ 디렉토리는 참가자의 script.py 실행 결과로 생성된 예측 결과 파일이 저장되는 디렉토리이며, 해당 디렉토리 내에 반드시 submission.csv으로 생성될 수 있어야합니다.

⏱️ 실행 시간 제한
패키지 설치 시간: 최대 10분 이내 (시간 초과 시 설치 오류)
추론 코드 실행 시간: 최대 10분 이내 (시간 초과 시 제출 오류)

⚙️ 평가 서버 사양
OS : Ubuntu 22.04.5 LTS
GPU : NVIDIA L4 (VRAM 22.4GiB)
CPU: 6 vCPU
CPU RAM: 28GB
Python : 3.11.15
인터넷 접속: ❌ 비활성화 (패키지 설치 외 외부 서버 연결 및 다운로드 불가)
CUDA : 12.8


💾 평가 서버 기본 설치 패키지(라이브러리) 목록

아래의 패키지(라이브러리)는 평가 서버에 기본적으로 설치되어 있으며, 버전이 명시된 아래의 패키지(라이브러리)에 한해서는 다른 버전을 사용할 때 설치 에러가 발생할 수 있으므로 가급적 평가 서버에 기본 설치된 패키지(라이브러리)를 활용하고 제출하는 requirements.txt에는 포함하지 않는 것을 권장드립니다. 
라이브러리 설치 에러가 발생하면 설치 오류에 해당하며, 일일 제출 횟수에는 반영되지 않습니다.

1) 주요 설치 패키지(라이브러리)
torch==2.7.1+cu128
pandas==2.0.3
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.8.0
joblib==1.5.3
threadpoolctl==3.6.0
narwhals==2.21.2
transformers==4.46.3
accelerate==1.9.0
sentencepiece==0.1.99
regex==2023.12.25
tqdm==4.66.4
loguru==0.7.2
pyyaml==6.0.1
rich==13.7.1

2) 주요 설치 시스템 패키지﻿

git
build-essential
python3.11
python3.11-dev
python3.11-venv
python3-pip
libffi-dev
libblas3
liblapack3
libomp-dev
tzdata
unzip
p7zip-full
gfortran
libatlas-base-dev
default-jre-headless
cmake
pkg-config
ninja-build
libgl1
libglib2.0-0

📌 유의사항
제출 시 발생하는 오류의 종류는 두 가지로 정의되며, 일일 제출 횟수 반영에 대한 기준이 다르므로 반드시 숙지하여 진행해야 합니다.
1) 설치 오류 : 제출하는 submit.zip 내부 구조가 불일치한 경우, 패키지 설치 오류 -> 일일 제출 횟수 반영되지 않음
2) 제출 오류 : script.py 코드 실행 후 발생하는 모든 오류 -> 일일 제출 횟수 반영됨
script.py 내에서 open/ 디렉토리의 데이터를 로드하고, output/ 디렉토리에 예측 결과를 반드시 submission.csv의 파일명으로 저장되어야 합니다.
평가 서버 환경은 인터넷 접속이 불가능하므로, 패키지 설치 이후 외부 다운로드가 필요한 코드나 모델은 작동하지 않습니다.

# 각 구성요소 설명

model/ 디렉터리
로컬에서 훈련한 모델의 가중치 파일 저장
파일명은 자유롭게 설정 가능
여러 파일 저장 가능 (예: model.pt, tokenizer.json 등)
script.py 파일
평가 서버에서 자동으로 실행되는 추론 전용 코드
반드시 이 파일명을 사용해야 함
학습 과정은 포함하지 않고, 추론만 수행
데이터 로드 → 모델 로드 → 예측 → 결과 저장의 순서로 구성
requirements.txt 파일
추론에 필요한 추가 패키지(라이브러리) 명시
pip install -r requirements.txt 형식으로 작성
이미 평가 서버에 설치되어 있는 패키지(라이브러리)는 버전 호환이 이미 맞춰져 있으므로 가급적 포함하지 않을 것을 권장

# 평가 서버 동작 과정

자동 실행 과정
평가 서버에서는 다음과 같은 순서로 자동 실행됩니다:

1
환경 구성
submit.zip (제출된 파일)
├── model/
├── script.py
├── requirements.txt
├── data/               # ← 서버에서 자동 추가
└── output/             # ← 서버에서 자동 추가
2
패키지(라이브러리) 설치
pip install -r requirements.txt
제한시간: 각 대회 페이지 확인
실패 시: 설치 오류 (제출 횟수 차감 없음)
3
추론 실행
python script.py
제한시간: 각 대회 페이지 확인
실패 시: 제출 오류 (제출 횟수 차감됨)
4
결과 확인
output/submission.csv 파일 생성 여부 확인
파일 형식 및 내용 검증

# 추론 코드 작성 가이드 

권장 구조
다음은 script.py 작성을 위한 권장 구조입니다:

import os
import pandas as pd
# 기타 필요한 패키지(라이브러리)

def load_model():
    """모델 로드 함수"""
    # model/ 디렉터리에서 모델 가중치 로드
    model_path = os.path.join('model', 'your_model.pt')
    # 모델 로드 코드
    return model

def load_data():
    """평가 데이터 로드 함수"""
    # data/ 디렉터리에서 평가 데이터 로드 (예시 파일명)
    data_path = os.path.join('data', 'test.csv')
    # 실제 데이터 로드 코드
    return data

def predict(model, data):
    """추론 수행 함수"""
    # 실제 추론 로직
    predictions = model.predict(data)
    return predictions

def save_results(predictions):
    """결과 저장 함수"""
    # output/submission.csv로 저장 (필수)
    os.makedirs('output', exist_ok=True)
    submission = pd.DataFrame({'prediction': predictions})
    submission.to_csv('output/submission.csv', index=False)

if __name__ == "__main__":
    # 메인 실행 코드
    model = load_model()
    data = load_data()
    predictions = predict(model, data)
    save_results(predictions)
    print("추론 완료!")