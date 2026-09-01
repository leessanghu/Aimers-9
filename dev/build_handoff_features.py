"""codex 전달용 신규피처 2종. Rule4 준수(train 룩업만 조회, 행간 참조 없음).
1) li_resid: li - E[li | inning,outs,base_state,score_diff_home] (게임상황으로 설명 안되는 잔차 = 팀전력차 프록시)
2) team_matchup_te: (pitcher_team_id, batter_team_id) EB축소 target encoding

둘 다 row_id 키로 parquet 저장. 각 fold(A/C)에서 train<=upto로만 테이블을 만들어
Rule4 안전성을 명시적으로 보장(merge key에 시즌누수 없음 재확인)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
n = len(df)
print(f'전체 {n:,}행')


def build_li_resid(train_mask):
    """train_mask 구간으로만 룩업 테이블을 만들어 전체 df에 적용."""
    tr = df[train_mask].copy()
    tr['inning_c'] = tr['inning'].clip(upper=10)
    tr['sd_c'] = tr['score_diff_home'].clip(-6, 6)
    key_cols = ['inning_c', 'outs_before', 'base_state', 'sd_c']
    tab = tr.groupby(key_cols)['li'].agg(['mean', 'count']).reset_index()
    global_li = float(tr['li'].mean())
    K = 30.0
    tab['li_expect'] = (tab['count'] * tab['mean'] + K * global_li) / (tab['count'] + K)

    full = df.copy()
    full['inning_c'] = full['inning'].clip(upper=10)
    full['sd_c'] = full['score_diff_home'].clip(-6, 6)
    merged = full.merge(tab[key_cols + ['li_expect']], on=key_cols, how='left')
    merged['li_expect'] = merged['li_expect'].fillna(global_li)
    li_resid = (full['li'].to_numpy(np.float64) - merged['li_expect'].to_numpy(np.float64))
    return li_resid, merged['li_expect'].to_numpy(np.float64)


def build_team_matchup_te(train_mask):
    tr = df[train_mask]
    tab = tr.groupby(['pitcher_team_id', 'batter_team_id'])['control_success'].agg(['mean', 'count']).reset_index()
    global_y = float(tr['control_success'].mean())
    K = 500.0
    tab['te'] = (tab['count'] * tab['mean'] + K * global_y) / (tab['count'] + K)
    merged = df.merge(tab[['pitcher_team_id', 'batter_team_id', 'te']],
                       on=['pitcher_team_id', 'batter_team_id'], how='left')
    merged['te'] = merged['te'].fillna(global_y)
    return merged['te'].to_numpy(np.float64) - global_y  # 중심화된 편차로 제공


# fold A(train<=2023), fold C(train<=2021), 그리고 프로덕션용(train<=2024, 즉 전체)
season = df['season'].to_numpy()

print('\n=== fold A(train<=2023) 룩업 테이블로 생성 (검증용) ===')
li_resid_A, li_exp_A = build_li_resid(season <= 2023)
matchup_A = build_team_matchup_te(season <= 2023)
print(f'  li_resid std={li_resid_A.std():.4f}  matchup_te std={matchup_A.std():.5f}')

print('\n=== 프로덕션용 (train<=2024, 전체 train으로 테이블 생성) ===')
li_resid_full, li_exp_full = build_li_resid(np.ones(n, bool))
matchup_full = build_team_matchup_te(np.ones(n, bool))

out = pd.DataFrame({
    'row_id': df['row_id'],
    'li_resid': li_resid_full,
    'li_expect': li_exp_full,
    'team_matchup_dev': matchup_full,
})
out.to_parquet('dev/handoff_features_production.parquet')
print(f'\n저장: dev/handoff_features_production.parquet  {out.shape}')
print(out.describe())

# 검증용(fold A 버전)도 별도 저장
outA = pd.DataFrame({'row_id': df['row_id'], 'li_resid': li_resid_A, 'team_matchup_dev': matchup_A})
outA.to_parquet('dev/handoff_features_foldA.parquet')
print('\n저장: dev/handoff_features_foldA.parquet (fold A 룩업 버전, 검증전용)')
