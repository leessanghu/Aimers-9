# v43 Trackman Feature Candidate Spec

This file is a Trackman-only candidate feature specification, not implementation code.

Prompt reflected:
- Do not re-propose known failed/simple Trackman ideas:
  - tagged vs auto raw disagreement alone;
  - rel_speed - zone_speed raw mean/std alone;
  - existing `pitch_type_group` inferred pitch-type table;
  - existing within-pitch-type pitcher-season mean/SD profile already in `trackman_profile.py`.
- Focus on new angles:
  - A. batter Trackman profile via uncertain batter mapping;
  - B. cross-pitch-type signal, distance, and arsenal geometry;
  - C. pitch-of-PA sequence/transition physical change;
  - D. league/population Trackman distributions by count/game context.

Local files checked:
- `data/trackman_history.csv`: 30 columns including pitcher/batter Trackman IDs, pitch-of-PA, tagged/auto pitch type, pitch group, and 8 physical columns.
- `dev/pitcher_map.csv`: `pitcher_id`, `tm_id`, `sim`; pitcher mapping exists, batter mapping does not.
- `dev/trackman_profile.py`: existing pitcher-season physical profile, release SD, movement SD, velo decay, pressure release SD.
- `dev/phase82_tm_intent.py` and `dev/phase93_splithalf_screen.py`: raw intent/disagreement-style features were screened and should not be repeated as standalone ideas.
- `dev/pitchshape.py`: pitch-shape clustering already exists in a target-aware matched setting; this spec avoids direct reuse of that exact target-control table.

## Count Summary

- A. Batter mapping/profile candidates: 42, gated by mapping confidence.
- B. Cross-pitch-type candidates: 84.
- C. Pitch-of-PA sequence candidates: 72.
- D. Population/context candidates: 96.
- Total candidate atoms: 294.
- Directly implementable without batter mapping: 252.
- High cell-size concern atoms: 34.

## Global Implementation Rules

- Trackman has no 2025 rows. Final artifacts must store precomputed profile/stat tables; inference must not require raw Trackman.
- Row lookup should use `season - 1` cumulative profiles unless a feature explicitly says same-game previous pitch or same-PA previous pitch in Trackman history.
- For validation rows, never use Trackman rows from the validation season if the row-level train/test analogue would not have them at prediction time.
- Prefer cumulative pitcher profiles through previous season, or league/population tables through previous season. Same-game/same-PA sequence features are only for Trackman-internal profile construction, not direct train-row matching unless a leakage-safe alignment is proven.
- Every rate/distance/SD feature with a personal key needs support count and shrinkage/reliability companion features.

## A. Batter Mapping And Batter Trackman Profiles

Mapping is uncertain because no `batter_id <-> batter_trackman_id` table exists. Treat these as a separate experiment. Do not mix with B/C/D until mapping coverage and false-match rate are measured.

### A0. Proposed Batter Mapping Methods

- id: MAP_A01
  category: mapping
  method: co-occurrence bipartite matching
  source_keys_train: [season, game_month, game_dayofweek, inning, top_bottom, balls_before, strikes_before, outs_before, pitcher_id]
  source_keys_trackman: [season, game_month, game_dayofweek, inning, top_bottom, balls_before, strikes_before, outs_before, pitcher_trackman_id]
  candidate_link: batter_id -> batter_trackman_id
  rationale: for rows with mapped pitcher, same game state and pitcher should narrow batter candidates; aggregate repeated co-occurrences.
  expected_precision: medium-high if uniqueness rate is high
  risk: false positives when multiple pitches share same state and pitcher in same game

- id: MAP_A02
  category: mapping
  method: frequency-signature matching
  source_keys_train: [season, pitcher_id, batter_id]
  source_keys_trackman: [season, pitcher_trackman_id, batter_trackman_id]
  candidate_link: batter_id -> batter_trackman_id
  rationale: compare season-level pitcher-batter exposure frequency vectors after applying pitcher_map.
  expected_precision: medium
  risk: common batters with similar schedules can collide

- id: MAP_A03
  category: mapping
  method: team-season roster constraint
  source_keys_train: [season, batter_team_id, batter_id]
  source_keys_trackman: [season, batter_team, batter_trackman_id]
  candidate_link: batter_id -> batter_trackman_id
  rationale: if team IDs can be aligned to Trackman team strings, constrain candidate set before co-occurrence matching.
  expected_precision: high after team mapping
  risk: team-name mapping and transfers

- id: MAP_A04
  category: mapping
  method: consensus mapping
  source: MAP_A01 + MAP_A02 + MAP_A03
  accept_rule: accept only if top candidate margin is large and at least two methods agree
  rationale: precision matters more than coverage; bad batter mapping can poison every downstream feature.
  expected_precision: high
  risk: lower coverage

### A1. Batter Exposure-To-Stuff Profiles

- id: TM_A001-TM_A006
  category: target-free
  group_keys: [batter_trackman_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1 after accepted batter mapping
  rationale: how much velocity a batter typically sees; may interact with pitcher velocity/stuff gap.
  cell_size_concern: high

- id: TM_A007-TM_A012
  category: target-free
  group_keys: [batter_trackman_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: spin_rate
  window: cumulative through season-1
  rationale: batter exposure to spin environment.
  cell_size_concern: high

- id: TM_A013-TM_A018
  category: target-free
  group_keys: [batter_trackman_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: sqrt(induced_vert_break^2 + horz_break^2)
  window: cumulative through season-1
  rationale: batter exposure to movement magnitude, independent of current pitcher.
  cell_size_concern: high

- id: TM_A019-TM_A024
  category: target-free
  group_keys: [batter_trackman_id, season, pitcher_hand]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1
  rationale: batter velocity exposure by pitcher side.
  cell_size_concern: high

- id: TM_A025-TM_A030
  category: target-free
  group_keys: [batter_trackman_id, season, pitch_type_group]
  agg: share | entropy_component | mean_speed | mean_break_mag | count | reliability
  source_col: pitch_type_group + physical columns
  window: cumulative through season-1
  rationale: what pitch families the batter actually sees; not batter outcome, only exposure distribution.
  cell_size_concern: high

- id: TM_A031-TM_A036
  category: target-free
  group_keys: [mapped_batter_id, season]
  agg: mean_gap | std_gap | q25_gap | q75_gap | count | reliability
  source_col: current_pitcher_profile_minus_batter_exposure_profile
  window: pitcher and batter cumulative through season-1
  rationale: relative matchup: current pitcher stuff versus batter's historical exposure.
  cell_size_concern: high

- id: TM_A037-TM_A042
  category: target-free
  group_keys: [mapped_batter_id, season]
  agg: percentile | zscore | clipped_zscore | above_q75_flag | below_q25_flag | reliability
  source_col: pitcher velocity/break/spin relative to batter exposure distribution
  window: cumulative through season-1
  rationale: flags unfamiliar physical looks; this is the actual reason to risk batter mapping.
  cell_size_concern: high

## B. Cross-Pitch-Type Signal

These features use mapped pitcher profiles only. They avoid the existing within-type mean/SD profile by measuring geometry between pitch types and arsenal separation.

- id: TM_B001-TM_B007
  category: target-free
  group_keys: [pitcher_id, season]
  agg: weighted_mean | min | max | q25 | q75 | count | reliability
  source_col: pairwise Euclidean distance between pitch_type_group centroids in standardized physical space
  window: cumulative through season-1
  rationale: how distinguishable a pitcher's pitch families are physically; command may improve when shapes separate cleanly.
  cell_size_concern: medium

- id: TM_B008-TM_B014
  category: target-free
  group_keys: [pitcher_id, season]
  agg: weighted_mean | min | max | q25 | q75 | count | reliability
  source_col: release-point distance between pitch_type_group centroids
  window: cumulative through season-1
  rationale: pitch families that require different release points may indicate tunneling difficulty or intent reveal.
  cell_size_concern: medium

- id: TM_B015-TM_B021
  category: target-free
  group_keys: [pitcher_id, season]
  agg: weighted_mean | min | max | q25 | q75 | count | reliability
  source_col: movement-vector distance between pitch_type_group centroids
  window: cumulative through season-1
  rationale: movement separation across fastball/breaking/offspeed, not within-type noisiness.
  cell_size_concern: medium

- id: TM_B022-TM_B028
  category: target-free
  group_keys: [pitcher_id, season]
  agg: weighted_mean | min | max | q25 | q75 | count | reliability
  source_col: speed distance between pitch_type_group centroids
  window: cumulative through season-1
  rationale: speed ladder quality across arsenal.
  cell_size_concern: medium

- id: TM_B029-TM_B035
  category: target-free
  group_keys: [pitcher_id, season]
  agg: angle_mean | angle_std | angle_min | angle_max | count | reliability | entropy_weighted_angle
  source_col: angle between movement vectors of pitch_type_group centroids
  window: cumulative through season-1
  rationale: captures direction diversity, not just magnitude.
  cell_size_concern: medium

- id: TM_B036-TM_B042
  category: target-free
  group_keys: [pitcher_id, season]
  agg: primary_vs_secondary_gap | primary_vs_rest_mean_gap | primary_vs_best_gap | primary_share | count | reliability | clipped_gap
  source_col: primary pitch_type_group centroid versus other groups
  window: cumulative through season-1
  rationale: whether the main pitch is physically distinct from secondary offerings.
  cell_size_concern: medium

- id: TM_B043-TM_B049
  category: target-free
  group_keys: [pitcher_id, season]
  agg: covariance_trace | covariance_det_log | first_pc_share | second_pc_share | condition_number | count | reliability
  source_col: pitch_type_group centroid cloud in physical space
  window: cumulative through season-1
  rationale: arsenal geometry dimensionality; one-dimensional arsenals may be easier to anticipate.
  cell_size_concern: medium

- id: TM_B050-TM_B056
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | min | max | count | reliability | clipped
  source_col: within-group physical dispersion / between-group centroid distance ratio
  window: cumulative through season-1
  rationale: separates "wide arsenal" from noisy command; ratio can be more meaningful than raw SD.
  cell_size_concern: medium

- id: TM_B057-TM_B063
  category: target-free
  group_keys: [pitcher_id, season, pitcher_hand]
  agg: percentile | zscore | clipped_zscore | rank | count | reliability | league_delta
  source_col: pairwise pitch-type centroid distance relative to same-hand league distribution
  window: cumulative through season-1; league table through season-1
  rationale: relative arsenal separation; reduces season/hand scaling drift.
  cell_size_concern: medium

- id: TM_B064-TM_B070
  category: target-free
  group_keys: [pitcher_id, season]
  agg: entropy_weighted_speed_gap | entropy_weighted_movement_gap | entropy_weighted_release_gap | top2_gap | bottom2_gap | count | reliability
  source_col: pitch mix share times cross-type physical distances
  window: cumulative through season-1
  rationale: large gap matters only if both pitch families are actually used.
  cell_size_concern: medium

- id: TM_B071-TM_B077
  category: target-free
  group_keys: [pitcher_id, season]
  agg: fb_breaking_gap | fb_offspeed_gap | breaking_offspeed_gap | fb_breaking_release_gap | fb_offspeed_release_gap | count | reliability
  source_col: named pair centroid distances
  window: cumulative through season-1
  rationale: explicit pair features are easier for GBDT to use than unordered pair summaries.
  cell_size_concern: medium

- id: TM_B078-TM_B084
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | min | max | count | reliability | clipped_delta
  source_col: auto_pitch_type centroid distance minus tagged_pitch_type centroid distance
  window: cumulative through season-1
  rationale: not raw disagreement; asks whether intended and physically realized pitch spaces are geometrically separated.
  cell_size_concern: medium

## C. Pitch-Of-PA Sequence And Physical Transition Profiles

These are pitcher profile features computed from Trackman sequences, then looked up through season-1. They should not require matching the current train row to a Trackman row.

- id: TM_C001-TM_C006
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: abs(delta rel_speed from previous pitch within same PA)
  window: cumulative through season-1
  rationale: how aggressively a pitcher changes speed pitch-to-pitch within plate appearances.
  cell_size_concern: medium

- id: TM_C007-TM_C012
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: Euclidean delta of [rel_height, rel_side] from previous pitch within same PA
  window: cumulative through season-1
  rationale: release-point movement between sequential pitches; may proxy mechanical adjustment or deception.
  cell_size_concern: medium

- id: TM_C013-TM_C018
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: Euclidean delta of [induced_vert_break, horz_break] from previous pitch within same PA
  window: cumulative through season-1
  rationale: movement sequencing profile.
  cell_size_concern: medium

- id: TM_C019-TM_C024
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: pitch_type_group switch flag between consecutive pitches within same PA
  window: cumulative through season-1
  rationale: sequence diversity independent of target.
  cell_size_concern: medium

- id: TM_C025-TM_C030
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: physical transition distance conditional on pitch_type_group switch
  window: cumulative through season-1
  rationale: not just whether pitcher switches, but how large the physical jump is when switching.
  cell_size_concern: medium

- id: TM_C031-TM_C036
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: physical transition distance conditional on same pitch_type_group repeat
  window: cumulative through season-1
  rationale: same-family execution variability across repeated offerings.
  cell_size_concern: medium

- id: TM_C037-TM_C042
  category: target-free
  group_keys: [pitcher_id, season, pitch_of_pa_bin]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1
  rationale: whether speed changes as PA deepens, using pitch_of_pa directly.
  cell_size_concern: medium

- id: TM_C043-TM_C048
  category: target-free
  group_keys: [pitcher_id, season, pitch_of_pa_bin]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: release distance sqrt(rel_height^2 + rel_side^2)
  window: cumulative through season-1
  rationale: mechanical drift within plate appearances.
  cell_size_concern: medium

- id: TM_C049-TM_C054
  category: target-free
  group_keys: [pitcher_id, season, balls_before, strikes_before]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: previous-to-current physical transition distance
  window: cumulative through season-1
  rationale: count-conditioned sequencing physical change; higher risk but still pitcher-profile level.
  cell_size_concern: high

- id: TM_C055-TM_C060
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: signed speed delta when previous count was favorable versus unfavorable
  window: cumulative through season-1
  rationale: speed adjustment after pressure changes.
  cell_size_concern: medium

- id: TM_C061-TM_C066
  category: target-free
  group_keys: [pitcher_id, season]
  agg: mean | std | q25 | q75 | count | reliability
  source_col: signed release-height delta after ball versus after strike
  window: cumulative through season-1
  rationale: mechanical response to previous pitch result proxy without target labels.
  cell_size_concern: medium

- id: TM_C067-TM_C072
  category: target-free
  group_keys: [pitcher_id, season]
  agg: first_to_late_mean_delta | first_to_late_std_delta | late_pa_speed_slope | late_pa_release_slope | count | reliability
  source_col: pitch_of_pa physical trajectories
  window: cumulative through season-1
  rationale: compact profile of within-PA physical evolution.
  cell_size_concern: medium

## D. Population / Count / Game Context Trackman Distributions

These avoid individual fragmentation. Most should be low-risk because they use broad Trackman context and can be applied as population priors or z-score baselines.

- id: TM_D001-TM_D008
  category: target-free
  group_keys: [season, game_type_proxy, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1, or expanding-before-date if date-safe
  rationale: league velocity distribution by count context.
  cell_size_concern: low

- id: TM_D009-TM_D016
  category: target-free
  group_keys: [season, game_type_proxy, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: spin_rate
  window: cumulative through season-1
  rationale: count-dependent league spin environment.
  cell_size_concern: low

- id: TM_D017-TM_D024
  category: target-free
  group_keys: [season, game_type_proxy, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: induced_vert_break
  window: cumulative through season-1
  rationale: count-dependent vertical movement distribution.
  cell_size_concern: low

- id: TM_D025-TM_D032
  category: target-free
  group_keys: [season, game_type_proxy, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: horz_break
  window: cumulative through season-1
  rationale: count-dependent horizontal movement distribution.
  cell_size_concern: low

- id: TM_D033-TM_D040
  category: target-free
  group_keys: [season, inning, top_bottom]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1
  rationale: league velocity by inning and side; fatigue/context prior.
  cell_size_concern: low

- id: TM_D041-TM_D048
  category: target-free
  group_keys: [season, inning, top_bottom]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: release distance sqrt(rel_height^2 + rel_side^2)
  window: cumulative through season-1
  rationale: league release dispersion by inning/top-bottom context.
  cell_size_concern: low

- id: TM_D049-TM_D056
  category: target-free
  group_keys: [season, pitcher_hand, batter_hand, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1
  rationale: population physical baseline for handedness and count.
  cell_size_concern: low

- id: TM_D057-TM_D064
  category: target-free
  group_keys: [season, pitcher_hand, batter_hand, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: break magnitude sqrt(ivb^2 + hb^2)
  window: cumulative through season-1
  rationale: population movement baseline for handedness and count.
  cell_size_concern: low

- id: TM_D065-TM_D072
  category: target-free
  group_keys: [season, pitch_type_group, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1
  rationale: pitch-family count baseline; can support pitcher-vs-league residuals.
  cell_size_concern: low

- id: TM_D073-TM_D080
  category: target-free
  group_keys: [season, pitch_type_group, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: break magnitude sqrt(ivb^2 + hb^2)
  window: cumulative through season-1
  rationale: movement count baseline by pitch family.
  cell_size_concern: low

- id: TM_D081-TM_D088
  category: target-free
  group_keys: [season, pitch_of_pa_bin, balls_before, strikes_before]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: rel_speed
  window: cumulative through season-1
  rationale: population speed progression in the PA by count.
  cell_size_concern: medium

- id: TM_D089-TM_D096
  category: target-free
  group_keys: [season, game_month, pitcher_hand]
  agg: mean | std | q10 | q25 | q75 | q90 | count | reliability
  source_col: physical PC1 or standardized stuff index from rel_speed/spin/break
  window: expanding-before-date within season or previous-season fallback
  rationale: league/month physical environment drift; should be tested as a broad drift feature.
  cell_size_concern: low

## Recommended Screening Order

1. D low-risk population baselines first. These are broad, target-free, and least likely to reproduce v40-style fold-C instability.
2. B cross-pitch-type geometry next. These are the most promising "new information" from Trackman because existing code mostly used within-type profile quality.
3. C sequence profiles third. They are plausible but more implementation-heavy and require careful same-PA ordering.
4. A batter mapping only after mapping precision audit. Do not include A features in a final candidate unless mapping confidence is high and coverage is reported.

## Mapping Audit For A

Report before using A features:
- mapped batter coverage overall and by season;
- one-to-one conflict rate;
- top-1 versus top-2 score margin distribution;
- consistency across MAP_A01/MAP_A02/MAP_A03;
- manual sanity sample of high-frequency batters;
- downstream split-half screen with A features isolated.

## Rejection Rules

- Reject if improvement is mostly fold B or a single seed.
- Reject if A/C minimum gain is smaller than seed spread.
- Reject if feature only shrinks probability variance without improving residual separation.
- Reject if a personal Trackman feature has no support/reliability companion.
- Reject A entirely if batter mapping precision is not defensible.
