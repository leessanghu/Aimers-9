# Codex 작업 요청: 오늘 발견한 "헤드 정리" 결과 이중검증 + v94 재학습 설계 안전성 검토

## 배경

DACON/LG Aimers 9 대회(투구 제구 성공 확률 `control_success` 예측, Brier Skill Score
평가). 현재 최고 실측 제출은 v88 = 1102.83점 (100등 컷 약 1121, 190등 근처). 대회가
얼마 안 남았고 하루 제출 5개 제한이라 신중하게 검증 후 제출해야 한다.

**핵심 규칙(Rule 4): 각 행의 예측은 그 행 시점까지 알 수 있는 정보만 써야 한다
(leakage 금지).** 그리고 지금까지 두 번 "검증 방법 자체가 낙관적으로 틀린" 사고를
쳤다:
1. 프로덕션 pkl(`model_artifacts_v88.pkl`, train 2019-2024 전체로 학습됨)을 그대로
   로드해서 2024 데이터를 예측 → in-sample 오염(수치가 실제보다 훨씬 좋게 나옴).
2. LightGBM 같은 가벼운 대리모델로 얻은 결론이 실제 프로덕션 앙상블(HGB 3변종+
   CatBoost 3변종)로 전이 안 됨(정반대 부호로 나옴).

그래서 지금부터는 **모든 결론을 "정직한 out-of-time 캐시"로만 내고, 반드시 서로
독립인 두 개 이상의 시간분할(fold)에서 같은 방향인지 확인**하는 게 원칙이다.

## 정직한 검증 셋업 (반드시 이 방식만 사용할 것)

- 피처/라벨: `dev/featcache_X.parquet` (162개 피처, 1,475,092행), `dev/featcache_meta.parquet`
  (`season`, `control_success` 컬럼 포함)
- `season` 컬럼으로 폴드를 나눈다:
  - **fold A**: train = season≤2023, val = season==2024
  - **fold C**: train = season≤2021, val = season==2022
  - (fold B: train≤2022, val=2023 은 일부 헤드 캐시가 없어서 이번엔 스킵해도 됨)
- 각 fold의 val 예측값은 out-of-time으로 미리 캐시돼 있다 (아래 "헤드별 캐시 경로" 참고).
  **절대 프로덕션 pkl로 val 연도를 직접 predict하지 말 것** — 그건 in-sample이다.
- BSS 계산: `BSS = 1e5 * (1 - Brier / 0.249807)` (baseline은 대회 공식 상수, 두 fold
  다 이 고정값을 분모로 써서 서로 비교 가능하게 함 — fold 자체 ybar 기반 baseline이
  아님에 주의)

## 헤드별 캐시 경로 (fold prefix `A` 또는 `C`로 교체)

```python
import numpy as np, pandas as pd, joblib
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')  # 가중치/mc5_succ 등 상수용, predict는 안 씀
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

def load_heads(p):  # p = 'A' or 'C'
    H = dict(
        base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6','d8','sub')]),
        hurdle=np.mean([(1-np.load(f'dev/phase90_cache/{p}_core_{n}.npy'))
                         * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6','d8')], axis=0),
        multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42,7)]),
        ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42,7)]),
        midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42,7)]),
        condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42,7)]),
        countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42,7)]),
        future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42,7)]),
    )
    P = np.load(f'dev/idea75_cache/{p}_proba11.npy')  # 11-class 확률, mc5_succ(11개)와 내적
    H['mc5'] = np.clip(P @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
    ing = np.load(f'dev/idea80_cache/{p}_ingame_heads_s42.npy')
    H['ingame'] = np.clip(ing[:,0] if ing.ndim>1 else ing, 0, 1)
    return H

W = dict(base=.1426368, hurdle=.1901824, multires=.0475456, ordinal=.0950912,
         midother=.118864, condball=.06256, countresid=.06256, future50=.06256,
         mc5=.138, ingame=.08)  # v88의 실제 가중치 (합=1.0)
```

## Task 1 (최우선): 헤드별 leave-one-out을 fold C에서도 재현해서 fold A와 일치하는지 확인

fold A에서 이미 다음을 얻었다 (재현 코드는 위 유틸 + 아래 패턴):

```python
def sc(q, yv, unc=0.249807): return 1e5*(1-np.mean((np.clip(q,0,1)-yv)**2)/unc)

def loo_table(p, val_season):
    yv = y[season==val_season]
    H = load_heads(p)
    full = sum(W[k]*H[k] for k in W)
    base_sc = sc(full, yv)
    rows = []
    for k in W:
        W2 = {a:b for a,b in W.items() if a!=k}; t=sum(W2.values())
        q = sum(v/t*H[a] for a,v in W2.items())
        rows.append((k, sc(q,yv)-base_sc))
    return base_sc, sorted(rows, key=lambda t:t[1])
```

**fold A 결과 (val 2024)**: 전체 블렌드 921.7. LOO(제거시 델타):
```
hurdle     -15.88   (제거하면 크게 나빠짐 = 가치 있음)
ordinal     -6.52
base        -4.92
mc5         -1.15
multires    +0.76   (제거하면 좋아짐 = 해로움)
midother    +1.46
condball    +2.25
countresid  +2.36
future50    +2.63
ingame      +2.69
```
해로운 6개(multires, midother, condball, countresid, future50, ingame)를 순차 제거하면
921.7 → 931.6 (+9.9).

**할 일**: 이 정확한 계산을 `p='C', val_season=2022`로 그대로 재현해서
(1) 순위가 fold A와 비슷하게 나오는지 (hurdle/ordinal/base/mc5가 상위, 나머지 6개가
해로운 쪽), (2) 6개 제거 시 fold C에서도 개선되는지, (3) 개선 폭이 fold A(+9.9)와
비슷한 자릿수인지 확인해라. **fold A에서만 좋고 fold C에서 안 맞으면 이 결과는
채택하지 않는다** — 이게 오늘 세션에서 이미 한 번 겪은 실패 패턴(대리모델 결론이
프로덕션에서 뒤집힘)과 같은 종류의 함정이다.

## Task 2: v94 설계에서 Rule 4 위반 위험 지점 감사

v94는 "해로운 6개 헤드를 빼고 나머지 4개(base/hurdle/ordinal/mc5)로만 재구성"하는
계획이다(재학습은 필요 없고 `submit/script.py`에서 해당 6개 헤드의 weight만 0으로
만들고 나머지 4개 weight를 재정규화하면 됨 — 실제로 재학습이 필요한 건 이번엔 없다,
가중치 재분배만).

다만 최종 파이프라인에는 아래 두 보정이 남아있는데, 이게 base/hurdle/ordinal/mc5
비중이 바뀐 새 분포에도 여전히 안전한지 검토해달라:

- `risk` 보정 (`submit/script.py`의 `risk_idx/risk_thr/risk_alpha/risk_center`):
  `risk_vec = proba5[:, risk_idx].sum(axis=1)` (mc5의 성공률 0인 클래스 확률 합),
  `preds -= risk_alpha * (max(0, risk_vec-risk_thr) - risk_center)`.
  현재 `risk_thr=0.25, risk_alpha=0.045, risk_center=0.144112`(train 2024 기준 상수)는
  mc5 weight=0.138일 때 튜닝된 값이다. 헤드 비중이 바뀌어도 `risk_vec` 자체는 mc5
  모델 출력이라 안 바뀌지만, **전체 preds 분포가 달라지므로 risk_alpha/risk_center가
  여전히 최적에 가까운지, 아니면 재추정이 필요한지** 판단해달라. (참고: v88/v91/v92
  세 실측점으로 이 보정의 이차함수 계수는 C=175.1, V=4979.2, 최적alpha*=0.0352로
  이미 역산된 바 있다 — 이 값이 헤드 비중 변경 후에도 유효할지가 질문.)
- `level_shift = -0.00127`: train 통계로 추정된 상수. 헤드 비중이 바뀌면 preds의
  평균 레벨도 미세하게 바뀔 텐데, 이 상수를 그대로 쓸지 재계산해야 할지 판단해달라.
  **주의**: 재계산한다면 반드시 train(2024) 통계만 써야 한다 — val/test의 실제
  평균을 알고 있다는 가정(오라클)으로 레벨을 맞추면 Rule 4 위반이고, 이건 정확히
  v86이 -36점 낸 원인(비중심 보정)과 같은 실수다.

## Task 3: mc5 재학습 여부 판단 자료 정리

mc5(v74, 5-class softmax 디코더)는 fold A LOO에서 제거시 -1.15로 "가치는 있지만
미미"하게 나왔다. mc5는 `recency_weight(half_life=2.0)`으로 학습된 구모델이고,
이번에 base/hurdle/ordinal은 그대로 두되(재학습 안 함, v88 원본 그대로 재사용) mc5도
그대로 재사용할 계획이다. **이게 타당한지, 아니면 -1.15라는 크기가 재학습을 정당화할
만큼 큰지** 판단 근거를 정리해달라 (재학습하려면 시간이 걸리므로 정말 필요한
경우에만).

## 출력 형식

각 Task에 대해 (1) 실행한 코드/명령, (2) 정확한 수치 결과, (3) 결론(채택/기각/보류와
그 이유)을 명확히 구분해서 보고해줘. 특히 Task 1은 fold A와 fold C 숫자를 나란히
표로 보여줘.
