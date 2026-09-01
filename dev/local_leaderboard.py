"""로컬 리더보드 (상설 도구) — 새 아이디어를 최고성적 모델과 같은 자로 비교한다.

## 반드시 지킬 측정 규약 (틀리면 캘리브레이션이 깨진다)
  1) 검증셋 = fold A 전체 2024  (3-7월 구간으로 자르면 상관이 음수로 뒤집힘. 실측 확인)
  2) 블렌딩 = (1-w) x 기존블렌드 + w x 신규축   <- 모든 멤버 비례축소
     base에서만 w를 빼면 안 됨. 로컬 base3는 <=2023 학습이라 프로덕션 대비
     상대품질이 달라, 그 왜곡이 델타에 그대로 실린다.
  3) 기준 블렌드 = v47local (0.30 base + 0.40 hurdle + 0.10 multires + 0.20 ordinal)
  4) w = 0.10 고정

## 캘리브레이션 (실측 5건으로 적합)
     실측Δ = 6.29 + 5.02 x 로컬Δ      Pearson +0.818, 잔차SD 1.61
     손익분기 로컬Δ = -1.25
  적합범위는 로컬Δ -1.28~-0.18. 그 밖은 외삽이므로 '*' 표시.

## 범주별 적용 가능 여부 (중요)
  aux_head    : 공유트리 auxiliary head를 w=0.10로 블렌드   -> **캘리브 유효**
  feature_add : 162피처에 피처를 추가해 재학습
                단순 비교는 시드노이즈 15~50이라 판정 불가.
                그러나 **같은 시드끼리 페어링**하면 페어SD가 3.3~4.0으로 떨어진다
                (idea45 실증). |Δ|>4 이고 3시드 부호 일치면 판정 가능.
                그 미만이면 실측 직행.
  procedure   : 학습절차 변경(refit-closure, ES방식, 재가중) -> 실측 0/6 전패. 시도 금지.
  member      : 이미 프로덕션에 들어있는 멤버               -> 델타가 무의미.

## 주의: 로컬Δ 하나만 보지 말 것
  idea44(트랙맨 실행축)는 aux_head 표에서 -0.90~-1.01로 '경계'로 뜨지만
  fold C에서 -8.98~-16.05로 크게 깨졌다. **fold C 안정성도 반드시 함께 확인.**

## 외삽 금지구역 (v61 실패로 확정)
  캘리브는 "미활용 축 1개를 w=0.10로" 라는 좁은 조건에서만 적합됐다.
  로컬Δ>0 이거나 w>0.10 이면 **외삽구간이고 캘리브 무효**.
  실증: mega(6head, w=0.20) 로컬Δ=+1.45로 midother(+1.25)보다 우세하다고 나왔으나
        실측은 -3.97로 역전. 예측오차 4.97 = 적합구간 잔차SD(1.61)의 3.1배.
  원인: fold A(5시즌)는 aux 타겟이 노이즈라 정규화가 이득으로 보이지만,
        프로덕션(6시즌)은 aux 타겟이 정확해져 head0 희석이 순비용으로 드러난다.
        fold A는 이 비용을 구조적으로 볼 수 없다.

## head 선정 규칙 (v61 실패로 확정)
  **미활용 축만 head로 넣을 것.** 이미 잘 쓰이는 축(SHAP 상위/splits 다수)을
  head로 추가하면 새 정보는 0인데 head0(y)의 그래디언트 비중만 깎여 순손실.
  검증: middle(SHAP122위/2splits)+other(대응피처 없음) 2축만 넣은 midother가 최적.
        여기에 ball(7위/41splits)+투수LOO 2종(이미 multires 보유)을 더한 mega는 -3.97.

사용법:
  python local_leaderboard.py            # 전체 스캔 + local_scores.csv 갱신
  새 후보는 fold A 예측을 `<이름>_cache/A_<태그>_s<시드>.npy` 로 저장만 하면 자동 수집됨.
  범주는 CATEGORY 딕셔너리에 등록(미등록은 'unknown'으로 표시되고 캘리브 미적용).
"""
import glob
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()
OUT_CSV = "local_scores.csv"
CAL_A, CAL_B = 6.29, 5.02          # 실측Δ = CAL_A + CAL_B * 로컬Δ
CAL_LO, CAL_HI = -1.28, -0.18      # 적합범위
BREAKEVEN = -CAL_A / CAL_B         # = -1.253

# 실측된 aux head 축 (캘리브레이션 적합에 사용)
KNOWN_LB = {
    "idea31/midaxis": 7.72,
    "idea33/other": 3.25,
    "idea32/ball": 1.83,
    "idea32/strike": 0.20,
    "idea12/head0": 6.99,          # v51 unified5
}
# 범주 등록 (미등록은 unknown)
CATEGORY = {
    "idea31/midaxis": "aux_head", "idea33/other": "aux_head", "idea33/severe": "aux_head",
    "idea32/ball": "aux_head", "idea32/strike": "aux_head", "idea32/reverse": "aux_head",
    "idea12/head0": "aux_head", "idea30/formcast": "aux_head", "idea34/midloo": "aux_head",
    "idea44/exec3": "aux_head", "idea44/ptype_only": "aux_head", "idea44/exec_only": "aux_head",
    "idea13/multires": "member", "idea13/ordinal": "member",
    "idea15/multires_kps50.0": "member", "idea15/multires_kps30.0": "member",
    "idea20/timesplit": "procedure", "idea21/mono": "procedure",
    "idea8/form_role": "feature_add", "idea8/inseason": "feature_add",
    "idea8/trackman": "feature_add", "idea8/lastyear_vol": "feature_add",
    "idea26/d6plus161": "feature_add", "idea28/d6plusTMspin": "feature_add",
    "idea40/streak": "feature_add", "idea40/workload": "feature_add",
    "idea40/hot_only": "feature_add", "idea40/all": "feature_add",
    "idea42/marcel_full": "feature_add", "idea42/marcel_dev_only": "feature_add",
    "idea42/marcel_multi": "feature_add",
    "idea45/base162": "feature_add", "idea45/tmdirect173": "feature_add",
}


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
va = seasons == 2024
yv = y[va]
N = int(va.sum())
_r = yv.mean(); _BS = _r * (1 - _r)


def sc(p):
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / _BS)


def avg(paths):
    return np.mean([np.load(p) for p in paths], axis=0)


log("기준 블렌드 구성...")
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = avg([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = avg([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
V47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
B47 = sc(V47)
mid = avg([f"idea31_cache/A_midaxis_s{k}.npy" for k in [42, 7]])
oth = avg([f"idea33_cache/A_other_s{k}.npy" for k in [42, 7]])
V58 = 0.10 * base + 0.40 * hur + 0.10 * mr + 0.20 * od + 0.10 * mid + 0.10 * oth
log(f"  v47local={B47:.2f} (실측 1066.18)   v50local={sc(0.9*V47+0.1*mid):.2f} (실측 1073.90)")
log(f"  현재 최고 v58 실측=1077.15  (v58 로컬재현은 base비중이 달라 직접비교 금지)")

log("fold A 후보 자동 수집...")
cands = {}
for d in sorted(glob.glob("*_cache")):
    if not os.path.isdir(d):
        continue
    groups = {}
    for f in os.listdir(d):
        m = re.match(r"A_(.+)_s\d+\.npy$", f)
        if m:
            groups.setdefault(m.group(1), []).append(os.path.join(d, f))
    for g, paths in groups.items():
        try:
            a = avg(sorted(paths))
        except Exception:
            continue
        if a.ndim == 1 and len(a) == N:
            cands[f"{d[:-6]}/{g}"] = (a, len(paths))
log(f"  {len(cands)}개 수집")

rows = []
for nm, (p, nseed) in cands.items():
    solo = sc(p)
    d = sc(0.90 * V47 + 0.10 * p) - B47
    cat = CATEGORY.get(nm, "unknown")
    lb = KNOWN_LB.get(nm)
    pred = CAL_A + CAL_B * d if cat == "aux_head" else np.nan
    rows.append(dict(name=nm, category=cat, n_seed=nseed, solo=solo, local_delta=d,
                     pred_lb_delta=pred, actual_lb_delta=lb,
                     extrapolated=(d < CAL_LO or d > CAL_HI)))
df = pd.DataFrame(rows).sort_values("local_delta", ascending=False)
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
log(f"저장 {OUT_CSV} ({len(df)}행)")

print()
print("=" * 100)
print(f"기준 v47local={B47:.2f} | 캘리브: 실측Δ = {CAL_A} + {CAL_B} x 로컬Δ | 손익분기 로컬Δ={BREAKEVEN:.2f}")
print("=" * 100)
aux = df[df.category == "aux_head"]
print(f"\n[aux_head] 캘리브레이션 유효 — 이 표만 제출판단에 사용")
print(f"{'후보':<26}{'시드':>5}{'단독':>9}{'로컬Δ':>9}{'예상실측Δ':>11}{'실측Δ':>9}  판정")
for _, r in aux.iterrows():
    a = f"{r.actual_lb_delta:+.2f}" if pd.notna(r.actual_lb_delta) else "-"
    w = "*" if r.extrapolated else " "
    if pd.notna(r.actual_lb_delta):
        v = "검증완료"
    elif r.pred_lb_delta > 2:
        v = "제출후보"
    elif r.pred_lb_delta > 0:
        v = "경계"
    else:
        v = "기각"
    print(f"{r['name']:<26}{r.n_seed:5d}{r.solo:9.1f}{r.local_delta:+9.2f}{r.pred_lb_delta:+11.2f}{w}{a:>9}  {v}")

for cat, note in [("feature_add", "로컬 판정불가(시드노이즈 15~50). 실측 직행만이 방법"),
                  ("procedure", "실측 0/6 전패 계열. 로컬 양수여도 시도 금지"),
                  ("member", "이미 프로덕션 포함. 델타 무의미"),
                  ("unknown", "범주 미등록 — CATEGORY에 추가 필요")]:
    sub = df[df.category == cat]
    if len(sub) == 0:
        continue
    print(f"\n[{cat}] {note}")
    for _, r in sub.head(8).iterrows():
        print(f"  {r['name']:<26}{r.solo:9.1f}{r.local_delta:+9.2f}")

print()
print("새 후보 등록법: fold A 예측을 <이름>_cache/A_<태그>_s<시드>.npy 로 저장 후 재실행.")
print("               CATEGORY 딕셔너리에 범주 등록 필수.")
log(f"총 {time.time()-t0:.0f}s")
