"""새 피처군: 경기 내 시퀀스(ingame sequence).
현재 162피처에 완전히 없는 정보축 - 이번 경기에서 몇 구째인지, 타순 몇 바퀴째인지,
직전 투구가 볼/스트라이크였는지.

Rule 4 안전성: 각 행은 '같은 경기에서 자기 앞에 온 행들'만 참조한다. 미래 행은 절대
안 본다. 행 순서(row_num)가 실제 투구 순서이므로 이는 asof_ 컬럼과 동일한 성격이다.
"""
import numpy as np, pandas as pd, sys, time
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()


def build_gameseq(df):
    """df는 row_num 오름차순 정렬 + pitcher_id/asof_pitcher_n/inning/batter_id/
    balls_before/strikes_before/asof_pitcher_prev1_game_success_rate 필요."""
    o = df.sort_values(['pitcher_id', 'row_num']).copy()
    pid = o['pitcher_id'].to_numpy()
    p1 = o['asof_pitcher_prev1_game_success_rate'].to_numpy()
    inn = o['inning'].to_numpy()
    bid = o['batter_id'].to_numpy()
    b = o['balls_before'].to_numpy()
    s = o['strikes_before'].to_numpy()

    same_p = np.r_[False, pid[1:] == pid[:-1]]
    # 경기 경계: 투수 바뀜 OR prev1_game 값 바뀜 OR 이닝이 감소(새 경기)
    p1_chg = np.r_[True, ~np.isclose(p1[1:], p1[:-1], equal_nan=True)]
    inn_drop = np.r_[True, inn[1:] < inn[:-1]]
    newgame = (~same_p) | p1_chg | inn_drop
    game_idx = np.cumsum(newgame)

    out = pd.DataFrame(index=o.index)
    # 1) 경기 내 투구 번호 (0-based)
    pos = np.arange(len(o)) - pd.Series(np.arange(len(o))).groupby(game_idx).transform('min').to_numpy()
    out['g_pitch_idx'] = np.log1p(pos)
    out['g_is_game_start'] = (pos == 0).astype(np.float64)

    # 2) 경기 내 상대한 타자 수 (직전까지, 자기 자신 제외 안 함 -> 현재 타자 포함 순번)
    new_bat = np.r_[True, (bid[1:] != bid[:-1])] | newgame
    bat_seq = np.cumsum(new_bat)
    bf = bat_seq - pd.Series(bat_seq).groupby(game_idx).transform('min').to_numpy()
    out['g_batters_faced'] = np.log1p(bf)
    # 3) 타순 몇 바퀴째 (9타자 = 1바퀴)
    out['g_times_thru'] = bf / 9.0

    # 4) 이 등판에서 몇 이닝 경과했나
    inn_first = pd.Series(inn).groupby(game_idx).transform('min').to_numpy()
    out['g_inning_span'] = inn - inn_first

    # 5) 현재 타석에서 몇 구째 (볼+스트라이크, 파울 누적 반영 위해 PA내 행 순번도)
    pa_pos = np.arange(len(o)) - pd.Series(np.arange(len(o))).groupby(bat_seq).transform('min').to_numpy()
    out['pa_pitch_idx'] = pa_pos
    out['pa_extra_fouls'] = pa_pos - (b + s)   # 파울로 늘어난 투구수

    # 6) 직전 투구 결과 (같은 타석 안에서 카운트 전이로 복원)
    same_pa = ~new_bat
    db = np.r_[0, np.diff(b)]
    ds = np.r_[0, np.diff(s)]
    out['prev_was_ball'] = np.where(same_pa & (db > 0), 1.0, 0.0)
    out['prev_was_strike'] = np.where(same_pa & (ds > 0), 1.0, 0.0)
    out['prev_was_foul2s'] = np.where(same_pa & (db == 0) & (ds == 0), 1.0, 0.0)

    return out.sort_index()


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                     usecols=['row_id', 'season', 'inning', 'pitcher_id', 'batter_id',
                              'balls_before', 'strikes_before', 'asof_pitcher_n',
                              'asof_pitcher_prev1_game_success_rate', 'control_success'])
    df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
    df = df.sort_values('row_num').reset_index(drop=True)
    print(f'로드 {len(df):,} ({time.time()-t0:.0f}s)')

    G = build_gameseq(df)
    print(f'생성 {G.shape[1]}개 피처 ({time.time()-t0:.0f}s)')
    print(G.describe().T[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())

    G['season'] = df['season'].to_numpy()
    G['control_success'] = df['control_success'].to_numpy()
    G.to_parquet('dev/gameseq_feats.parquet')
    print('saved dev/gameseq_feats.parquet')
