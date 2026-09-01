"""codex v20 패키지의 fold A(2024) 예측을 우리 v95 블렌드와 비교.

[중요] 이 예측은 오염돼 있다:
  - metadata: "Uses 2024 validation-fitted bounded correction coefficients"
  - base/raw_id 모델도 제출용이라 2024 포함 학습 가능성 높음
따라서 아래 수치는 전부 '낙관적 상한'이다. 오염은 잔차상관을 양수쪽으로 부풀린다
(p_codex가 2024 y에 맞춰졌으므로 d=p_codex-p_ours가 resid=y-p_ours를 닮게 됨).

=> 단방향 검정: 오염된 상한에서조차 잔차상관이 0근처/음수면 확정 기각.
   크게 양수면 판정불가(오염 탓일 수 있음) -> 정직 OOF 필요.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


va = season == 2024
yv = y_all[va]
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
t = sum(W.values()); W = {k: v / t for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
resid = yv - blend

# codex 예측 로드 (row_id 순서 정렬 필수)
sub = pd.read_csv('C:/Users/이상후/AppData/Local/Temp/claude/'
                  'c--Users-----OneDrive-------Aimers-9/3064dbc5-47a2-47b1-b613-f1f60c5848ac/'
                  'scratchpad/codex_v20/output/submission.csv', encoding='utf-8')
order = pd.read_csv('dev/codex_foldA_input/test.csv', encoding='utf-8', usecols=['row_id'])
p_cod = order[['row_id']].merge(sub, on='row_id', how='left')['control_success'].to_numpy(np.float64)
assert np.isfinite(p_cod).all() and len(p_cod) == len(yv), (len(p_cod), len(yv))

sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
print(f'=== fold A(2024) — 전부 오염된 낙관적 상한 ===')
print(f'  우리 v95 블렌드 단독 BSS = {sc(blend):8.2f}')
print(f'  codex v20 단독 BSS       = {sc(p_cod):8.2f}   <- 2024 적합 포함, 부풀려진 값')

d = p_cod - blend
cp = np.corrcoef(p_cod, blend)[0, 1]
cr = np.corrcoef(d, resid)[0, 1]
C = float(np.mean((blend - yv) * d)); V = float(np.mean(d * d))
s_opt = -C / V
gain = K * C ** 2 / V
print(f'\n  예측상관(vs 우리블렌드) = {cp:+.4f}')
print(f'  잔차상관(d vs resid)    = {cr:+.4f}   <- 오염으로 양수쪽 부풀려진 상태')
print(f'  최적가중치 s*           = {s_opt:+.4f}')
print(f'  최대이득(오염상한)      = {gain:+.2f}점')

print(f'\n  [비교] 오늘 측정한 후보들의 잔차상관(정직):')
print(f'     xgb_rawid  -0.0076 | xgb_ctx  -0.0073 | lgbm  -0.0072 | base(우리헤드) +0.0036')

print(f'\n[판정 기준] 위 잔차상관이 0근처/음수 -> 오염상한에서도 실패이므로 확정 기각.')
print(f'            크게 양수 -> 판정불가, codex에게 fold A/C 정직 OOF 요청 필요.')

# 실제 블렌드 시뮬레이션 (오염 상한)
print(f'\n=== 소량가중치 블렌드 시뮬레이션 (오염 상한) ===')
for w in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30):
    p = (1 - w) * blend + w * p_cod
    print(f'  w={w:.2f}: BSS={sc(p):8.2f}  (기준 {sc(blend):.2f} 대비 {sc(p)-sc(blend):+.2f})')
