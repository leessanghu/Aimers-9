# v43 Feature Factory Candidate Spec

This file is a candidate feature specification, not implementation code.

Sources checked and translated into this spec:
- MLB Player Digital Engagement 3rd place repo: long offline feature generation notebooks, meta-model, LightGBM, NN, ensemble, and robustness/gap-CV workflow.
- NVIDIA / Chris Deotte feature engineering writeup: large-scale groupby aggregations, quantiles/histograms, categorical combinations, and target-aware nested/OOF handling.
- TabPrep: groupby aggregates, arithmetic expansion, categorical interactions, OOF target encoding.
- Data-Centric Tabular Evaluation: dataset-specific feature engineering, temporal tabular structure, and test-time feature construction matter more than pure model swaps.
- NFL Big Data Bowl solution pattern: convert raw object attributes into relative matchup, projection, team/opponent, and symmetry-style features.

Known local constraints reflected here:
- Avoid repeating already weak fine personal splits as primary ideas: pitcher x count_state, batter x count_state, pitcher x batter_hand, pitcher x inning as direct Bayes/TE blocks.
- Prefer whole-sample-preserving context views: league/team/count/inning/hand/game-state strata, rank/bin/dispersion, and pitcher/batter relative deltas.
- Any target-aware item must be expanding or OOF-expanding only.

## Count Summary

- Target-free candidate families: 19
- Expanded target-free atomic candidates: 190
- OOF/expanding target-encoding candidate families: 17
- Expanded OOF/expanding atomic candidates: 170
- Total expanded atomic candidates: 360
- High cell-size concern candidate families explicitly marked high: 3
- Expanded high-concern atomic candidates explicitly marked high: 30
- Medium families that may become high after support screening: 6

## Expansion Rules

Each family below expands into 10 atomic IDs unless stated otherwise. For example, `TF001-TF010` is 10 concrete candidates formed by applying the listed `agg_set` or `source_cols` in order. The implementation should keep the expanded atomic feature names explicit, but this document keeps families compact enough to audit.

Cell-size concern means:
- low: group cells usually broad and stable.
- medium: cells may fragment but should still cover a meaningful share.
- high: useful only if smoothed, clipped, or screened by support/count.

## Target-Free Candidates

- id: TF001-TF010
  category: target-free
  group_keys: [season, game_type, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: li
  window: previous seasons only for validation/test; current train rows use expanding-before-row if implemented rowwise
  rationale: broad count/game-pressure context without personal fragmentation; converts count state into empirical pressure distribution.
  cell_size_concern: low

- id: TF011-TF020
  category: target-free
  group_keys: [season, game_type, inning]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: li
  window: previous seasons only or expanding-before-row
  rationale: inning/game-type pressure curve; likely captures nonlinearity trees may miss with raw inning/li alone.
  cell_size_concern: low

- id: TF021-TF030
  category: target-free
  group_keys: [game_type, top_bottom, base_state, outs_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: home_win_expectancy
  window: previous seasons only or expanding-before-row
  rationale: broad base/out/inning-side state; approximates run expectancy context without target.
  cell_size_concern: low

- id: TF031-TF040
  category: target-free
  group_keys: [game_type, base_state, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: score_diff_pitcher_team
  window: previous seasons only or expanding-before-row
  rationale: combines count and base state with score leverage; whole-population view, not individual split.
  cell_size_concern: low

- id: TF041-TF050
  category: target-free
  group_keys: [pitcher_team_id, season, game_month]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_success_rate
  window: previous rows in same season/month only; no future month rows
  rationale: team pitching environment and roster context; less fragmented than pitcher-level direct split.
  cell_size_concern: medium

- id: TF051-TF060
  category: target-free
  group_keys: [batter_team_id, season, game_month]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_batter_success_rate
  window: previous rows in same season/month only
  rationale: opposing lineup/team contact-control environment; broad team context.
  cell_size_concern: medium

- id: TF061-TF070
  category: target-free
  group_keys: [pitcher_hand, batter_hand, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_ball_rate
  window: previous seasons only or expanding-before-row
  rationale: hand-count zone tendency population view; preserves sample while adding interaction.
  cell_size_concern: low

- id: TF071-TF080
  category: target-free
  group_keys: [pitcher_hand, batter_hand, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_strike_rate
  window: previous seasons only or expanding-before-row
  rationale: complements ball-rate context; useful for calibration of count pressure.
  cell_size_concern: low

- id: TF081-TF090
  category: target-free
  group_keys: [season, game_month, pitcher_hand]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_fastball_rate
  window: expanding-before-row within season
  rationale: season/month arsenal environment by hand; captures league drift in pitch mix without target.
  cell_size_concern: low

- id: TF091-TF100
  category: target-free
  group_keys: [season, game_month, pitcher_hand]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_breaking_rate
  window: expanding-before-row within season
  rationale: breaking-ball environment drift; pairs with fastball/offspeed composition.
  cell_size_concern: low

- id: TF101-TF110
  category: target-free
  group_keys: [season, game_month, pitcher_hand]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_offspeed_rate
  window: expanding-before-row within season
  rationale: offspeed environment drift; useful if 2024/hidden pitch mix regime differs.
  cell_size_concern: low

- id: TF111-TF120
  category: target-free
  group_keys: [pitcher_team_id, batter_hand, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_fastball_rate
  window: previous rows only; require support screen
  rationale: team-level pitch-mix plan versus batter side and count, broader than pitcher-specific arsenal split.
  cell_size_concern: medium

- id: TF121-TF130
  category: target-free
  group_keys: [pitcher_team_id, batter_hand, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_breaking_rate
  window: previous rows only; require support screen
  rationale: team-level breaking usage by matchup count.
  cell_size_concern: medium

- id: TF131-TF140
  category: target-free
  group_keys: [pitcher_team_id, batter_hand, balls_before, strikes_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_offspeed_rate
  window: previous rows only; require support screen
  rationale: team-level offspeed usage by matchup count.
  cell_size_concern: medium

- id: TF141-TF150
  category: target-free
  group_keys: [game_type, base_state, num_runners_on]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_reverse_rate
  window: previous seasons only or expanding-before-row
  rationale: population reverse tendency under runner pressure; avoids individual split.
  cell_size_concern: low

- id: TF151-TF160
  category: target-free
  group_keys: [game_type, base_state, num_runners_on]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_middle_rate
  window: previous seasons only or expanding-before-row
  rationale: broad middle/contact-context distribution by runner state.
  cell_size_concern: low

- id: TF161-TF170
  category: target-free
  group_keys: [pitcher_hand, batter_hand, base_state, outs_before]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_batter_middle_rate
  window: previous seasons only or expanding-before-row
  rationale: opponent contact profile distribution under base/out state and hand matchup.
  cell_size_concern: low

- id: TF171-TF180
  category: target-free
  group_keys: [season, game_month, game_type]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_prev5_game_success_rate
  window: expanding-before-row within season
  rationale: recent-form league/month drift; stabilizes raw prev-game rates with population percentile context.
  cell_size_concern: low

- id: TF181-TF190
  category: target-free
  group_keys: [season, game_month, game_type]
  agg: mean | std | median | q10 | q25 | q75 | q90 | min | max | count
  source_col: asof_pitcher_prev5_game_middle_rate
  window: expanding-before-row within season
  rationale: recent-middle/contact trend by calendar and game type.
  cell_size_concern: low

## Target-Free Nonlinear Single-Row Transforms

These are not counted in the 190 groupby candidates above. They are cheap add-ons if implementation time permits.

- arsenal_entropy = entropy([fastball_rate, breaking_rate, offspeed_rate])
- arsenal_max_share = max(fastball_rate, breaking_rate, offspeed_rate)
- arsenal_min_share = min(fastball_rate, breaking_rate, offspeed_rate)
- arsenal_gap_top2 = largest_share - second_largest_share
- arsenal_fast_vs_nonfast = fastball_rate - (breaking_rate + offspeed_rate)
- prev_success_accel_1_3_5 = prev1_success - 2 * prev3_success + prev5_success
- prev_middle_accel_1_3_5 = prev1_middle - 2 * prev3_middle + prev5_middle
- pitcher_batter_experience_ratio = log1p(asof_pitcher_n) / (log1p(asof_batter_n) + eps)
- pitcher_pitchmix_experience_ratio = log1p(asof_pitcher_pitchmix_n) / (log1p(asof_pitcher_n) + eps)
- leverage_x_count_pressure = li * (balls_before - strikes_before)

## OOF / Expanding Target-Encoding Candidates

- id: OOF001-OOF010
  category: oof-expanding
  group_keys: [season, game_type, balls_before, strikes_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous seasons plus current season rows strictly before row_num
  rationale: broad count/game-type target prior; high support and likely stable.
  cell_size_concern: low

- id: OOF011-OOF020
  category: oof-expanding
  group_keys: [season, game_type, inning]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous seasons plus current season rows strictly before row_num
  rationale: inning/game-type success prior; complements existing inning split with broader regularization.
  cell_size_concern: low

- id: OOF021-OOF030
  category: oof-expanding
  group_keys: [game_type, base_state, outs_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only
  rationale: base/out target prior without player IDs; should be stable and interpretable.
  cell_size_concern: low

- id: OOF031-OOF040
  category: oof-expanding
  group_keys: [pitcher_hand, batter_hand, balls_before, strikes_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only
  rationale: population hand-count success prior; avoids pitcher-level fragmentation.
  cell_size_concern: low

- id: OOF041-OOF050
  category: oof-expanding
  group_keys: [pitcher_hand, batter_hand, base_state, outs_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only
  rationale: broad matchup + base/out prior, analogous NFL relative context instead of raw player IDs.
  cell_size_concern: low

- id: OOF051-OOF060
  category: oof-expanding
  group_keys: [pitcher_team_id, game_type, balls_before, strikes_before]
  agg: mean | count | smooth_k50 | smooth_k100 | logit_smooth_k50 | logit_smooth_k100 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only; fallback to league count prior
  rationale: team pitching plan/quality by count; broader than pitcher ID.
  cell_size_concern: medium

- id: OOF061-OOF070
  category: oof-expanding
  group_keys: [batter_team_id, game_type, balls_before, strikes_before]
  agg: mean | count | smooth_k50 | smooth_k100 | logit_smooth_k50 | logit_smooth_k100 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only; fallback to league count prior
  rationale: lineup/team opponent success prior by count.
  cell_size_concern: medium

- id: OOF071-OOF080
  category: oof-expanding
  group_keys: [pitcher_team_id, batter_hand, balls_before, strikes_before]
  agg: mean | count | smooth_k50 | smooth_k100 | logit_smooth_k50 | logit_smooth_k100 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only; require support screen
  rationale: team-level strategy versus batter side and count; likely more stable than pitcher x hand.
  cell_size_concern: medium

- id: OOF081-OOF090
  category: oof-expanding
  group_keys: [batter_team_id, pitcher_hand, balls_before, strikes_before]
  agg: mean | count | smooth_k50 | smooth_k100 | logit_smooth_k50 | logit_smooth_k100 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only; require support screen
  rationale: batting-team approach against pitcher side and count.
  cell_size_concern: medium

- id: OOF091-OOF100
  category: oof-expanding
  group_keys: [season, game_month, game_type]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: current season rows strictly before row_num plus previous season fallback
  rationale: calendar drift prior; directly targets temporal regime mismatch.
  cell_size_concern: low

- id: OOF101-OOF110
  category: oof-expanding
  group_keys: [season, game_month, pitcher_hand, batter_hand]
  agg: mean | count | smooth_k50 | smooth_k100 | logit_smooth_k50 | logit_smooth_k100 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: current season rows strictly before row_num plus previous season fallback
  rationale: calendar drift by handedness matchup.
  cell_size_concern: medium

- id: OOF111-OOF120
  category: oof-expanding
  group_keys: [base_state, num_runners_on, balls_before, strikes_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only
  rationale: run-state/count target prior; broad, non-player cell.
  cell_size_concern: low

- id: OOF121-OOF130
  category: oof-expanding
  group_keys: [game_type, li_bin, balls_before, strikes_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only
  rationale: leverage-binned count prior; creates nonlinear pressure strata while retaining support.
  cell_size_concern: medium

- id: OOF131-OOF140
  category: oof-expanding
  group_keys: [game_type, score_diff_bin, balls_before, strikes_before]
  agg: mean | count | smooth_k20 | smooth_k50 | logit_smooth_k20 | logit_smooth_k50 | prior_delta | support_log1p | leave_day_out_mean | recency_halflife2_mean
  source_col: control_success
  window: previous rows only
  rationale: score-difference count prior; pressure substitute less noisy than raw score_diff.
  cell_size_concern: medium

- id: OOF141-OOF150
  category: oof-expanding
  group_keys: [pitcher_id, season]
  agg: mean | count | smooth_empirical_bayes | logit_smooth_empirical_bayes | prior_delta | support_log1p | recency_halflife2_mean | recency_halflife4_mean | clipped_residual_to_team | reliability_weight
  source_col: control_success
  window: same pitcher previous rows in current season only; prior is pitcher_team_id x season or league
  rationale: keep a pitcher-season target view, but only as heavily shrunk reliability feature, not direct multires head.
  cell_size_concern: high

- id: OOF151-OOF160
  category: oof-expanding
  group_keys: [pitcher_id, season, batter_hand]
  agg: mean | count | smooth_empirical_bayes | logit_smooth_empirical_bayes | prior_delta | support_log1p | recency_halflife2_mean | recency_halflife4_mean | clipped_residual_to_pitcher | reliability_weight
  source_col: control_success
  window: same pitcher previous rows in current season only; prior is pitcher_id x season
  rationale: if reintroducing pitcher x hand, make it posterior/residual/reliability only; designed to avoid v40 fold-C failure.
  cell_size_concern: high

- id: OOF161-OOF170
  category: oof-expanding
  group_keys: [batter_id, season, pitcher_hand]
  agg: mean | count | smooth_empirical_bayes | logit_smooth_empirical_bayes | prior_delta | support_log1p | recency_halflife2_mean | recency_halflife4_mean | clipped_residual_to_batter | reliability_weight
  source_col: control_success
  window: same batter previous rows in current season only; prior is batter_team_id x season or league
  rationale: batter-side analogue, marked high risk because earlier batter fine splits were weak.
  cell_size_concern: high

## High-Concern Notes

High-concern families explicitly marked high are OOF141-OOF170. TF111-TF140 and OOF071-OOF090 are medium families that may become high after support screening. They exist because the prompt asked for broad candidate coverage, but screening should be harsh:
- require support/count features to be included with every smoothed rate;
- clip residual features to conservative ranges;
- compare A/C minimum gain against seed spread;
- drop a family if its gain comes mostly from fold B or from a single seed.

## Suggested Screening Order

1. Run target-free low-concern families TF001-TF040, TF061-TF110, TF141-TF190 first.
2. Add OOF low-concern families OOF001-OOF050, OOF091-OOF120 next.
3. Try medium team-level families TF041-TF060, TF111-TF140, OOF051-OOF140 as blocks.
4. Only then test high-concern posterior/residual personal families OOF141-OOF170.

## Acceptance Metric

For every block:
- clean folds: A and C only for primary decision;
- reject if `min_gain(A,C) <= max_seed_spread(A,C)`;
- reject if fold B contributes most of the average gain;
- reject if C improves only by probability shrinkage while prediction standard deviation collapses versus v35.
