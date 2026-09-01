실험하기 전에 나한테 먼저 계획을 설명시키고 그 다음에 실험시작하자

# 제출 zip 패키징 가이드

`EVALUATION.md`에 명시된 규칙: **평가 서버에서 학습은 절대 안 한다.** 로컬에서 모델을
전부 학습시켜 가중치 파일로 저장하고, 제출 zip에는 그 저장된 모델을 불러와 추론만
하는 코드를 넣는다. 서버가 자동으로 `pip install -r requirements.txt` → `python script.py`
를 실행하고, `output/submission.csv`가 생성되는지만 확인한다.

## 필수 구조 (반드시 이 형태로, 최상위 폴더 추가 금지)

```
submit.zip
├── script.py            # 반드시 이 파일명, 추론 전용 (학습 코드 없음)
├── requirements.txt      # 추론에 필요한 추가 패키지만 (서버 기본설치 패키지는 빼기)
└── model/
    └── model_artifacts_vXX.pkl   # 로컬에서 학습 완료한 모델 (joblib.dump)
```

이 3개만 zip 루트에 있어야 한다. `data/`, `output/`는 서버가 자동으로 붙여준다 —
직접 넣지 말 것. zip 안에 상위 폴더(예: `submit/script.py`처럼 감싸는 폴더)가 있으면
**구조 불일치로 설치 오류**(제출 횟수 안 깎이지만 시간 낭비).

## 실제 패키징 순서 (매번 이대로)

1. 로컬에서 `train_final_vXX.py` 돌려서 `model/model_artifacts_vXX.pkl` 생성
2. `script.py`의 `ARTIFACT_PATH`를 새 버전 파일명으로 수정
3. **스모크 테스트 필수**: `data/`, `output/` 폴더를 `submit/` 안에 임시로 만들고
   (`data/test.csv`, `data/sample_submission.csv` 복사) `python script.py` 직접 실행해서
   `output/submission.csv`가 정상 생성되는지 확인
4. 스모크 테스트 통과하면 **임시로 만든 `data/`, `output/`, `__pycache__` 삭제**
   (이걸 안 지우고 zip 만들면 안에 불필요한 파일이 껴서 구조가 지저분해짐)
5. `zipfile` 모듈로 압축 — **shell `zip` 명령 쓰지 말 것**(환경에 따라 없을 수 있음).
   `zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED)`로 script.py/requirements.txt/
   model/*.pkl 세 개를 **정확한 상대경로**(`model/model_artifacts_vXX.pkl`, 앞에
   `submit/` 안 붙임)로 `write(src, arcname)` 해야 함
6. 만든 zip을 다시 열어서(`zipfile.ZipFile(path).namelist()`) 3개 파일만 정확한
   경로로 들어있는지 최종 확인 — 여기서 걸러야 제출 거절을 안 당함

## 자주 하는 실수 (5번 거절당한 원인 후보)

- **최상위 폴더가 딸려 들어감**: `zipfile.write(src)`만 쓰면 원래 경로가 그대로 들어가서
  `submit/script.py`처럼 폴더가 씌워짐. 반드시 `write(src, arcname)`로 arcname을
  `script.py`(폴더 없이)로 지정해야 함
- **`data/`, `output/` 폴더를 실수로 같이 압축**: 스모크 테스트용으로 만들어둔 걸
  안 지우고 그대로 zip 뜨면 구조가 깨짐. 압축 직전에 항상 `rm -rf submit/data
  submit/output submit/__pycache__`
- **model 경로 오타**: `model/`이어야 하는데 `models/`나 `model_artifacts_vXX.pkl`을
  루트에 바로 넣는 등 — `script.py`의 `MODEL_DIR = os.path.join(BASE_DIR, "model")`
  경로와 정확히 일치해야 함
- **requirements.txt에 서버 기본설치 패키지까지 명시**: 버전 충돌로 설치 오류 남.
  scikit-learn/pandas/numpy 등은 서버에 이미 있으므로, 우리가 추가로 필요한
  라이브러리(예: catboost)만 적기
- **script.py 안에서 상대경로를 잘못 잡음**: 항상
  `BASE_DIR = os.path.dirname(os.path.abspath(__file__))` 기준으로
  `model/`, `data/`, `output/` 경로를 잡아야 함 (실행 위치에 따라 깨지는 상대경로 금지)
- **아티팩트 로드 후 필요한 키가 빠짐**: 새 모델 구조를 추가했으면(예: Hurdle,
  판정축) `script.py`의 로딩부와 추론부를 둘 다 갱신했는지 확인. 구버전 아티팩트도
  `artifacts.get(...)`로 방어적으로 읽어서 하위호환 유지

## 실전에서 반복된 최다 실패: FileNotFoundError (model_artifacts_vXX.pkl)

실제로 받은 거절 사유 4건이 전부 이 패턴이었다:

```
FileNotFoundError: [Errno 2] No such file or directory: '/app/model/model_artifacts_v18.pkl'
FileNotFoundError: [Errno 2] No such file or directory: '/app/model/model_artifacts_v16.pkl'
FileNotFoundError: [Errno 2] No such file or directory: './model/model_artifacts_v16.pkl'
FileNotFoundError: [Errno 2] No such file or directory: './model/model_artifacts_v5.pkl'
```

**원인은 매번 같다**: `script.py`의 `ARTIFACT_PATH`에 하드코딩한 파일명(`model_artifacts_v18.pkl` 등)과
실제로 zip `model/` 안에 넣은 pkl 파일명이 다르다. 버전을 올릴 때마다
① 학습 스크립트가 저장하는 파일명 ② `script.py`가 읽는 파일명 ③ zip에 실제로 담는 파일명,
이 세 곳을 손으로 각각 맞춰야 하는데 그중 하나를 빼먹으면 이 에러가 난다.
에러 위치(line 754/617/616/276)가 매번 다르다는 건 **매번 스크립트를 처음부터 다시 짜면서
버전 동기화를 놓치고 있다는 뜻** — 새로 만들 때마다 이 실수가 반복될 구조다.

**근본 해결책 — 버전 번호를 하드코딩하지 말고 `model/` 폴더 안의 파일을 자동으로 찾게 만들어라.**
이러면 애초에 이름이 어긋날 수가 없다:

```python
import glob

MODEL_DIR = os.path.join(BASE_DIR, "model")
candidates = sorted(glob.glob(os.path.join(MODEL_DIR, "model_artifacts_v*.pkl")))
if not candidates:
    raise FileNotFoundError(f"model/ 안에 model_artifacts_v*.pkl 이 없음: {MODEL_DIR} 내용물={os.listdir(MODEL_DIR)}")
ARTIFACT_PATH = candidates[-1]  # 버전 문자열 정렬이 걱정되면 zip에 pkl을 하나만 넣어서 무조건 그 하나를 쓰게 한다
```

이렇게 하면 "몇 버전을 넣었는지"를 신경 쓸 필요가 없어진다 — `model/` 폴더에 pkl이 몇 개 있든
(이상적으로는 **항상 1개만** 넣어라) 코드가 알아서 찾는다.

**그래도 제출 전 반드시 최종 확인 3줄을 실행해라** (자동탐색을 써도 파일 자체가 안 들어갔으면 소용없음):

```python
z = zipfile.ZipFile(zip_path)
pkl_names = [n for n in z.namelist() if n.startswith("model/") and n.endswith(".pkl")]
assert len(pkl_names) >= 1, f"model/*.pkl이 zip에 없음! namelist={z.namelist()}"
print("zip 안 pkl:", pkl_names)
```

그리고 **zip을 실제로 로컬에서 풀어서 그 안의 `script.py`를 그대로 실행**하는 스모크 테스트를
반드시 거쳐라(위 3번 항목). "코드는 맞는 것 같다"는 느낌만으로 제출하지 말고, 정확히 서버가
하는 것과 같은 순서(압축 해제 → data/output 폴더 옆에 두고 → `python script.py` → 결과 확인)를
로컬에서 그대로 재현해야 이런 경로 불일치를 제출 전에 100% 잡아낼 수 있다.
