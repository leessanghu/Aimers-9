# 배경

LG Aimers / DACON 대회, `control_success`(투구 커맨드 성공 여부) 예측. 평가식은 Brier Skill Score:

```
Score = max(0, 100000 * (1 - BrierScore/BSref))
BSref = r(1-r), r = 실제 성공률
```

train.csv는 2019~2024(147.5만행), 실제 test는 2025(비공개, 공개샘플 5행뿐). 규칙(`Rule.md`, `data_description.md`)상 **행 내부 공식 컬럼 + (엔티티, season-1) 형태의 train파생 정적 lookup만 허용**, test 다른 행 참조나 rolling/expanding 집계는 금지.

현재 실측 최고점: **1051.40** (`train_final_v35.py`). 신호대잡음이 극도로 낮음 — rho≈0.1025, 즉 결과 변동의 1.05%만 설명 가능.

# 지금 모델 구조 (v35)

```
p = 0.55 * HGB3변종평균 + 0.55 * Hurdle2변종평균  (정확히는 0.55/0.45 blend, 세부는 train_final_v35_fast.py 참고)
```

- **HGB 3변종 + CatBoost 3변종** (depth/subsample만 다르게, 앙상블 다양화) — 실측 검증됨 (+10.21)
- **Hurdle 인수분해**: `P(success) = (1-P(core_fail)) * P(success|no core_fail)`, core_fail = reverse 또는 middle 라벨(누적차분으로 복원 가능, 99.95% 커버리지). 직행모델과 상관 0.87로 낮아서 앙상블 다양성 기여. 실측 검증됨 (+14.98)

# 확립된 방법론 (반드시 따라줘)

1. **최소 3폴드 검증**: fold A(train≤2023→2024), fold C(train≤2021→2022), fold B(train≤2022→2023, **F리그 regime단절을 가로지름** — 기준선이 음수로 깨질 수 있는 스트레스 폴드). 단일 폴드만 보고 채택했다가 실측에서 부호가 반전된 전례 있음(CatBoost early-stop→refit 아이디어: fold A +61.70인데 fold B -456.04, 실측 -13.97로 확인사살).
2. **편향 없는 비모수 스크리닝**: 선형 partial_gain은 함정이 있음(p의 캘리브레이션 곡선 휨을 잔차가 못 지워서 아무 피처나 가짜 신호로 보임). `phase93_splithalf_screen.py`의 분할반(split-half) 방식 사용 — 구간평균을 절반에서 학습해 나머지 절반에서 평가, 참조피처(이미 모델이 쓰는 것) 여러 개를 같이 넣어 0 근처인지 확인.
3. **magnitude(SHAP) ≠ increment(스크리너)**: SHAP로 많이 쓰인다고 새 정보는 아님. `inseason_cmd_index`가 SHAP 상시 5위인데 증분은 항상 0에 가까움(기존 두 피처의 재표현일 뿐).

# 실험 로그 위치 (꼭 읽어줘)

- `dev/phase75_verify_v28.py`, `phase80~91_*.py`, `phase92~94_*.py` — 각 스크립트 상단 docstring에 가설/방법/결과 요약 있음. 특히:
  - `phase93_splithalf_screen.py` 결과: 지금까지 기각된 피처 전부(트릭맨 tagged/auto 불일치 포함) 편향 없는 기준으로도 정보 없음 확인됨
  - `phase94_v35_deepdive.py` 결과: v35 SHAP 상위 25개, 실제 split threshold, dependence 곡선 저장돼 있음 (`phase94_shap_magnitude.csv`, `phase94_split_counts.csv`, `phase94_dependence_summary.csv`)
  - `idea1_composite.py` (가장 최근, 실행완료): 결과 `idea1_results.csv`
- `dev/train_final_v25.py` ~ `v35_fast.py`: 버전별 변경사항 + 배경 docstring (실측 점수 포함)
- `data_description.md`, `Domain.md`, `Rule.md`, `EVALUATION.md`: 규칙/도메인 지식

# 지금까지 실측/3폴드로 확정된 것

**성공** (실측 검증됨): 모델 다양화(+10.21), Hurdle 인수분해(+14.98)
**실패** (3폴드 또는 실측으로 기각): 캘리브레이션, 구간별 라우팅, 등급 타겟, 의도축(실패모드 구성비), 피처 프루닝, objective(RMSE vs Logloss), 스태킹, 투수×타자 임베딩, 트릭맨 tagged≠auto 실행실패, LightGBM/XGBoost 추가, CatBoost early-stop→refit, 시대보정(시즌단위), F리그 regime 보정, 판정축 혼합분해(Hurdle과 상관 과다), R/F 완전분리 전문가(F데이터 13만행으로 부족), **주축 합성피처 완성**(x_ability_here에 타자축 합치기 — 방금 3폴드로 기각, 이유: 축을 합치면 트리가 정보 뭉개진 지름길을 택함)

# 지금 진행 중

**아이디어2** (돌리는 중): 타자축을 신뢰도(표본수 기반)로 **분리**(trackman×low-n 성공 패턴 재현). 아이디어1 교훈("합치지 말고 분리하라")을 반영.

# 요청

위 phase 로그와 실패/성공 목록을 직접 읽고, **아직 안 건드린 각도**로 새 아이디어를 내줘. 특히:
- 정보 자체가 고갈됐다는 게 `phase93`으로 확인됐으니, **순수 피처 추가보다 구조/아키텍처 축**(모델 다양화, 타겟 인수분해 계열)을 우선 고려
- 만약 피처를 제안한다면 반드시 "합치는 연산"이 아니라 "분리하는 연산"(상호작용, 조건부 라우팅) 형태로
- 각 아이디어에 검증 방법(3폴드 기준, 어떤 스크립트를 어떻게 확장할지)도 같이 제시
