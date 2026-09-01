# Domain Notes for Control Success Prediction

This file summarizes baseball domain knowledge that can help reframe the LG Aimers control_success task. The goal is not to add generic baseball trivia, but to translate domain mechanisms into legal, row-local feature ideas for this competition.

## 0. Competition Framing

The target is `control_success`, a binary event for whether a pitch was controlled successfully. In baseball terms this is closer to "control/command execution probability" than generic pitcher quality.

For this competition, the useful mental model is:

```text
P(control_success)
  = f(
      pitcher baseline command,
      current season state,
      count-dependent intent,
      pitch-type intent and pitch-type command difficulty,
      release/stuff consistency,
      workload/fatigue state,
      batter/hand/context pressure,
      league/game-type environment
    )
```

The most important lesson from v15 and later failures:

```text
Good domain idea != useful feature.

Useful feature must expose information v15 does not already absorb,
and it must be computable from each test row + stored train/trackman artifacts only.
```

## 1. Control vs Command vs Stuff

Baseball separates three related but different skills:

- Control: ability to throw strikes or avoid bad misses.
- Command: ability to locate a pitch where intended, including edges and chase spots.
- Stuff: raw pitch quality from velocity, spin, movement, and deception, independent of location.

FanGraphs/PitchingBot-style public models make a similar distinction. Stuff+ isolates raw pitch quality without location, while Location+/Command-like models care about where and when a pitch is thrown. FanGraphs notes that Stuff+ stabilizes quickly, while Location+ requires a much larger sample, roughly hundreds of pitches, to reach similar stability. PitchingBot describes CommandBot as focusing on whether pitches are thrown to appropriate locations for pitch type and count.

Competition translation:

```text
Trackman physical averages alone are not control.
They are mostly stuff.

Command signal is more likely in:
  asof_success / reverse / middle / ball
  pitch-type conditional control rates
  count-specific command behavior
  residual: command given stuff
```

Candidate feature family:

```text
stuff_score_p = g(rel_speed, spin_rate, IVB, HB, extension, pitch_type)
command_over_stuff_p = asof_pitcher_success_rate - E(success | stuff_score_p)
reverse_over_stuff_p = asof_pitcher_reverse_rate - E(reverse | stuff_score_p)
```

Risk:

We already tested simple Trackman physical averages and release SDs, and they were weak. Therefore this axis should not be re-tested as raw means. It must be expressed as a residual or interaction with command states.

Sources:

- FanGraphs Sabermetrics Library, "Stuff+, Location+, and Pitching+ Primer": https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/
- FanGraphs Community, "PitchingBot: Using Machine Learning To Understand What Makes a Good Pitch": https://community.fangraphs.com/pitchingbot-using-machine-learning-to-understand-what-makes-a-good-pitch/
- FanGraphs, "Do Better Pitchers Actually Have Better Command?": https://blogs.fangraphs.com/do-betters-pitchers-actually-have-better-command/

## 2. Count Changes Pitcher Intent

Count is not merely a situation label. It changes how pitchers throw.

A large MLB Statcast study found that release speed and spin rate decrease as ball count increases and increase as strike count increases. The paper interprets this through a speed-accuracy tradeoff: with more balls, pitchers may prioritize accuracy to avoid walks; with more strikes, they can afford more aggressive or chase-oriented execution.

The same study reports strong count dependence in strike probability. In its four-seam example, strike probability generally rises with ball count and falls with strike count.

Competition translation:

```text
count_state = 4 * balls_before + strikes_before
pressure = balls_before - strikes_before
```

But v15 already has `count_state`, `balls_before`, `strikes_before`, `x_ability_x_count`, and `x_ability_x_pressure`. So the new domain translation must be more specific:

```text
pitcher_count_command:
  P(success | pitcher, count, season<=t-1)
  P(reverse | pitcher, count, season<=t-1)
  P(middle | pitcher, count, season<=t-1)
  P(ball | pitcher, count, season<=t-1)

count_intent_shift:
  P(type | pitcher, count) - P(type | pitcher)

pressure_response:
  P(success | pitcher, pressure_bucket) - pitcher_prior
```

Formula:

```text
rate_smooth(p, c, k)
  = (S_{p,c} + k * prior_c) / (N_{p,c} + k)

dev(p, c)
  = rate_smooth(p, c, k) - prior_pitcher(p)
```

Important warning:

Count table features will likely overlap with v15 pitchtype/platoon/crosses. Only use them if they explain v15 residuals across rolling-year folds.

Sources:

- Hashimoto and Nakata, "Performance-environment mutual flow model using big data on baseball pitchers": https://pmc.ncbi.nlm.nih.gov/articles/PMC9715958/
- MLB Glossary, Strike Zone: https://www.mlb.com/glossary/rules/strike-zone

## 3. Release Parameters Determine Pitch Location

A pitching accuracy study focused on release parameters found:

- vertical pitch location is strongly influenced by elevation angle, speed, and spin axis;
- horizontal pitch location is influenced by azimuth angle, spin axis, and horizontal release point;
- the relation varies by pitcher.

Our Trackman file does not include direct release angles or actual pitch location for the current pitch. It includes proxies:

```text
rel_speed
spin_rate
induced_vert_break
horz_break
extension
rel_height
rel_side
zone_speed
pitch_type_group
```

Competition translation:

Do not use raw physical averages as "control". Instead derive pitch-type command difficulty proxies:

```text
vertical_command_proxy(type)
  = h(rel_speed, induced_vert_break, rel_height, extension)

horizontal_command_proxy(type)
  = h(horz_break, rel_side, pitcher_hand)
```

Better feature idea:

```text
by pitcher, pitch_type:
  expected_success_from_shape
  expected_reverse_from_shape
  expected_middle_from_shape

shape_command_gap:
  actual_asof_command - expected_command_from_shape
```

This reframes Trackman as "what command should be expected given pitch physics", not as a direct target predictor.

Sources:

- "Influence of Release Parameters on Pitch Location in Skilled Baseball Pitching": https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00036/full
- "Control of Accuracy during Movements of High Speed: Implications from Baseball Pitching": https://pubmed.ncbi.nlm.nih.gov/34376126/

## 4. Reproducibility Matters, but Not as Simple SD

Research comparing MLB and MiLB pitchers found that MLB pitchers had smaller ball release point variability, especially horizontal variability, and that release point variability related to pitching performance. Other work on high-speed accuracy suggests skilled pitchers rely on reproducibility of individual release parameters more than compensating covariation.

However, our prior simple release-point SD feature was weak. That suggests "lower SD is better" is too crude for this competition.

Better domain representation:

```text
release_consistency_by_type:
  std(rel_side | pitcher, pitch_type)
  std(rel_height | pitcher, pitch_type)
  std(extension | pitcher, pitch_type)

release_separation_between_types:
  distance(mean_release_fastball, mean_release_breaking)
  distance(mean_release_fastball, mean_release_offspeed)

deception_index:
  low release separation + high pitch movement/velocity separation

same_slot_multi_pitch:
  entropy(pitch_type mix) / release_spread
```

Possible formula:

```text
release_dist(t1,t2)
  = sqrt(
      (mean_rel_side_t1 - mean_rel_side_t2)^2
    + (mean_rel_height_t1 - mean_rel_height_t2)^2
    + (mean_extension_t1 - mean_extension_t2)^2
    )

deception_proxy
  = late_movement_spread / (release_dist + eps)
```

Risk:

We do not have full trajectory or current-pitch location. Trackman is historical and not 1:1 joined to test. Use only stored pitcher-season summaries.

Sources:

- "Relationship between ball release point variability and pitching performance in major league baseball": https://pmc.ncbi.nlm.nih.gov/articles/PMC11608975/
- "Control of Accuracy during Movements of High Speed": https://pubmed.ncbi.nlm.nih.gov/34376126/
- Driveline, "The Interaction of Biomechanics and Command": https://www.drivelinebaseball.com/2026/02/the-interaction-of-biomechanics-and-command/

## 5. Speed-Accuracy Tradeoff and Launch Window

The speed-accuracy tradeoff is relevant to throwing. A baseball throwing study found that higher effort throws can reduce the "launch window", especially vertically, leading to more accuracy error. Another pitch-specific study similarly emphasizes release-parameter precision.

Competition translation:

Aggressive high-stuff pitches may have lower command reliability:

```text
power_command_tradeoff:
  rel_speed_z_by_pitcher_type
  spin_z_by_pitcher_type
  movement_z_by_pitcher_type
  interacted with count pressure
```

But because simple velocity/spin trend failed earlier, the usable representation is probably:

```text
count_adjusted_stuff:
  stuff in hitter counts vs pitcher counts

intent_sacrifice:
  E(stuff | 3-ball counts) - E(stuff | neutral counts)
```

Hypothesis:

Pitchers who can maintain stuff in high-ball counts may be better; pitchers who sharply reduce stuff may be prioritizing "just get it over", which may increase middle-zone risk.

Sources:

- "The Launch Window Hypothesis and the Speed-Accuracy Trade-Off in Baseball Throwing": https://journals.sagepub.com/doi/10.2466/25.30.PMS.121c13x4
- "Influence of Release Parameters on Pitch Location": https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00036/full

## 6. Fatigue, Workload, and Role

A systematic review of pitcher fatigue links fatigue to kinematic changes, performance decline, and injury risk. It notes that performance can decrease with elevated pitch counts and innings thrown, and other reviews find mixed evidence at college/pro levels for exact pitch-count thresholds.

This matters for the competition because `prev1/3/5` game rates likely encode recent workload and form, but their denominators are hidden. v15 uses the rates, but not reliable workload sizes.

Competition translation:

```text
within_game_fatigue:
  inning
  starter_vs_reliever proxy
  recent game workload

season_fatigue:
  inseason_n
  current season share of career pitches
  workload acceleration

role:
  median pitches per appearance
  prev1 hidden workload if reliably recoverable
```

Critical warning from v19 failure:

The hidden denominator approach using "minimum q" was noisy because many rows had multiple candidate denominators. Do not reuse ambiguous denominator reconstruction.

Safer direction:

```text
Use only workload estimates that are uniquely recoverable
or directly validated against train row transitions.
```

Potential structural search:

```text
For consecutive rows by pitcher:
  asof_pitcher_n increments exactly by 1.

Can previous-game windows be validated by detecting game boundaries
from row_id/order + pitcher asof_n changes + prev1/3/5 rate changes?
```

Sources:

- "Manifestations of muscle fatigue in baseball pitchers: a systematic review": https://pmc.ncbi.nlm.nih.gov/articles/PMC6673423/
- "Current Workload Recommendations in Baseball Pitchers: A Systematic Review": https://journals.sagepub.com/doi/10.1177/0363546519831010
- "Alterations in pitching biomechanics and performance with an increasing number of pitches": https://pubmed.ncbi.nlm.nih.gov/37574914/

## 7. Attack Zones, Shadow Zone, and Framing

The real strike zone is not just binary inside/outside. Modern public baseball analysis often uses zones:

```text
Heart  = middle / hittable zone
Shadow = edge / borderline zone
Chase  = near-ball area that can induce chase
Waste  = obvious ball
```

MLB Statcast catcher framing focuses heavily on the Shadow Zone, where borderline pitches can become called strikes or balls depending on receiving, location, and context. TruMedia's framing model uses pitch location, batter handedness, strike-zone bounds, and count, calibrated by season/level.

Competition translation:

Our target's original labels include categories such as success, reverse, middle, ball, strike rates. That likely maps roughly to:

```text
success = acceptable control outcome
middle  = too hittable / heart risk
reverse = wrong-side miss
ball    = non-zone miss
strike  = strike outcome
```

Useful representation:

```text
command_profile:
  success_rate
  reverse_rate
  middle_rate
  ball_rate
  strike_rate

risk_profile:
  bad_miss = reverse + ball
  heart_risk = middle
  attack = strike or success depending definition
```

Feature candidates:

```text
pitcher_command_entropy
  = -sum_c p_c log(p_c)

command_badness
  = w_reverse * reverse + w_ball * ball + w_middle * middle

edge_command_proxy
  = success - middle - reverse
```

But v15 already uses smoothed rates for pitcher success/reverse/middle/ball/strike. Therefore the promising angle is not raw profile, but conditional profile:

```text
command_profile_by_pitch_type
command_profile_by_count
command_profile_by_game_type
command_profile_by_batter_hand
```

Sources:

- MLB Glossary, Catcher Framing: https://www.mlb.com/glossary/statcast/catcher-framing
- TruMedia Catcher Framing Model: https://baseball.help.trumedianetworks.com/baseball/catcher-framing-model
- Deshpande and Wyner, "A hierarchical Bayesian model of pitch framing": https://www.degruyterbrill.com/document/doi/10.1515/jqas-2017-0027/html
- 1.02 Glossary, Heart/Shadow/Chase/Waste: https://1point02.jp/op/gnav/glossary/gls_index_detail.aspx?gid=10285

## 8. Count Value Is Nonlinear

Called strikes have different run value depending on count. A framing model paper defines count-specific value:

```text
rho(count)
  = E[runs | count, taken, ball] - E[runs | count, taken, strike]
```

The value of a strike is much higher in counts such as 3-2 than 0-0. MLB's framing glossary also notes large downstream differences between 1-0 and 0-1 counts.

Competition translation:

The same control outcome has different strategic value by count. Model features should not treat all counts linearly.

Candidate:

```text
count_value_weighted_command
  = count_value(count) * pitcher_command_dev

count_value_weighted_bad_miss
  = count_value(count) * (reverse + ball + middle risk)
```

However, BSS target is probability of control_success, not run value. So this is useful only if pitchers alter behavior in high-value counts.

Sources:

- Deshpande and Wyner, "A hierarchical Bayesian model of pitch framing": https://www.degruyterbrill.com/document/doi/10.1515/jqas-2017-0027/html
- MLB Glossary, Catcher Framing: https://www.mlb.com/glossary/statcast/catcher-framing

## 9. Pitch Tunneling, Arsenal Entropy, and Deception

Pitch tunneling describes different pitch types looking similar early, then diverging late. Baseball Prospectus defines a tunnel point near batter decision time and measures differences between pitches at release, tunnel point, and plate. Pitcher Arsenal Analysis defines:

```text
Tunnel Spread
Plate Spread
Late Divergence = Plate Spread - Tunnel Spread
Velo Spread
Arsenal Entropy
```

We cannot reconstruct full 3D trajectories exactly from the competition Trackman fields. But we can derive partial proxies:

```text
arsenal_entropy
  = -sum_t P(type=t) log P(type=t)

velo_spread
  = usage-weighted pairwise abs(mean_speed_t1 - mean_speed_t2)

movement_spread
  = usage-weighted pairwise distance((IVB, HB)_t1, (IVB, HB)_t2)

release_spread
  = usage-weighted pairwise distance((rel_side, rel_height, extension)_t1, ...)

late_deception_proxy
  = movement_spread / (release_spread + eps)
```

Why this may matter for control_success:

Tunneling itself is more about batter deception than command. But pitchers with multiple pitches from similar slots may be able to attack edges without telegraphing pitch type, and their pitch-type mix can change count-dependent success probability.

Risk:

This can easily become another weak Trackman feature. Only pursue if it is combined with command outcomes:

```text
deception_proxy * command_profile_by_type
```

Sources:

- Baseball Prospectus, "Introducing Pitch Tunnels": https://legacy.baseballprospectus.com/article_legacy.php?articleid=31030
- Pitcher Arsenal Analysis Glossary: https://www.pitcherarsenal.com/

## 10. Level, League, and Environment Matter

The game_type F/R analysis showed a real environment gap in our data. Same pitchers were much more successful in F than R. But direct league-separated features did not improve because v15's `cat_game_type` already captures much of this split.

Domain translation:

League/level changes affect:

```text
opponent quality
strike zone / umpiring environment
pitcher role
pitch mix
aggressiveness
```

Feature candidates must therefore avoid duplicating `cat_game_type`. Better versions:

```text
within_pitcher_league_transfer:
  R_rate - F_rate for same pitcher, shrinked

league_adjusted_lastyear:
  lastyear_rate - league_rate(game_type, season)

league_adjusted_inseason:
  inseason_rate - league_rate(game_type, season)
```

But prior tests showed direct F/R separation was negative. Treat this axis as a calibration diagnostic, not first-priority feature.

## 11. Batter Effects

Control_success is pitcher-centered, but batter matters:

- hitter hand affects pitch shape and location difficulty;
- disciplined hitters may make pitchers throw closer to the zone;
- count changes batter swing/take behavior;
- batters differ in ability to force balls, avoid chase, or punish middle misses.

v15 already uses:

```text
asof_batter_success_rate_smooth
asof_batter_middle_rate_smooth
batter_id_count
batter_team features
same_hand
hand_matchup
platoon_diff
```

Potential missing representation:

```text
batter_pressure_profile:
  batter takes/patience proxy from pitcher success against that batter type

batter_middle_punish_proxy:
  if batter middle_rate faced is low/high
```

But batter in-season failed before. So batter features are lower priority unless tied to a structural recovered state.

## 12. Highest-Priority Feature Directions After v15

Updated priority after adding role/count/sequence domain knowledge:

```text
1. Command Profile Mixture by Pitch Type
2. Role and Usage State
3. Count Intent Matrix
4. Count-Specific Command Profile
5. Unique Workload Recovery
6. Release-Deception Residual
7. Sequencing Traits
```

Reason:

v15 already proved that pitcher state + pitchtype mixture can survive LB. The best next ideas should extend a surviving axis, not restart from a failed axis.

### Direction A: Command Profile Mixture by Pitch Type

v15 pitchtype uses a success-focused mixture:

```text
pt_pred = sum_t P(type=t | pitcher, count) * P(success | pitcher, type=t)
pt_dev  = pt_pred - pitcher_prior
```

Domain suggests extending this to full command profile:

```text
pt_success_pred = sum_t P(t | p,count,hand,gtype) * P(success | p,t)
pt_reverse_pred = sum_t P(t | p,count,hand,gtype) * P(reverse | p,t)
pt_middle_pred  = sum_t P(t | p,count,hand,gtype) * P(middle | p,t)
pt_ball_pred    = sum_t P(t | p,count,hand,gtype) * P(ball | p,t)
pt_strike_pred  = sum_t P(t | p,count,hand,gtype) * P(strike | p,t)
```

Why promising:

- v15 pitchtype survived LB.
- reverse/middle/ball are directly related to command quality.
- It is still a table-factory feature, not a model novelty.

Validation requirement:

Must beat v15 residual in rolling-year folds. If only target correlation is positive but residual correlation is zero, reject.

### Direction B: Count-Specific Command Profile

Construct:

```text
P(success/reverse/middle/ball/strike | pitcher, count)
```

Use heavy shrinkage to count-level global priors.

Why:

Count changes intent and pitch execution. v15 uses count but not fully count-specific pitcher command profiles.

Risk:

Potentially absorbed by CatBoost through asof rates + count crosses.

### Direction C: Release-Deception Residual

Construct pitcher-season summaries:

```text
release_spread_by_type
movement_spread_by_type
velo_spread_by_type
arsenal_entropy
deception_proxy = movement_spread / (release_spread + eps)
```

Then combine with command:

```text
deception_adjusted_command = deception_proxy * pt_success_dev
deception_bad_miss_risk = deception_proxy * pt_reverse_pred
```

Why:

Raw Trackman failed, but Trackman as a modifier of command intent may still have value.

### Direction D: Unique Workload Recovery

Revisit prev1/3/5 only if denominator/window can be validated uniquely.

Reject:

```text
minimum denominator q when multiple q fit
```

Accept only:

```text
unique q rate >= 95%
or directly validated from train transitions/game boundaries
```

### Direction E: Role and Usage State

Pitcher role changes almost every other mechanism:

```text
starter:
  longer outings
  wider pitch mix
  more pacing
  more exposure to fatigue and batter learning

reliever:
  shorter outings
  higher effort
  narrower pitch mix
  often used in specific inning/leverage pockets

closer/high-leverage reliever:
  aggressive zone attack may change
  pitch mix may concentrate on best weapons
```

The public sabermetric idea of the Times Through the Order Penalty says pitchers tend to perform worse as batters see them again in the same game. A Bayesian paper by Brill, Deshpande, and Wyner argues the exact "third time through" discontinuity may be overstated after adjustment, but still treats within-game pitcher performance evolution as real enough to model. For this competition, the practical point is not the exact TTOP cutoff. It is that starter-like and reliever-like usage are different states.

Possible row-local proxies:

```text
role_proxy_pitcher:
  median appearance length
  mean appearance length
  share of appearances starting in inning <= 2
  share of appearances starting in inning >= 7
  share of appearances with <= 10 pitches
  share of appearances with >= 50 pitches

role_context:
  role_proxy * inning
  role_proxy * li
  role_proxy * inseason_n
  role_proxy * game_type
```

How to build legally:

Use train history only. Derive pitcher-season or pitcher-career appearance profiles from row order and `asof_pitcher_n` transitions if game boundaries can be detected safely. For test rows, lookup stored role profiles for that pitcher up to season-1 or use career history from train.

Why this may be better than generic fatigue:

Prior velocity/spin fatigue features failed. Role is a higher-level state that changes pitch mix, effort, and command strategy at once. It may also explain why `inning` alone is weak but `inning_diff` helps.

Sources:

- FanGraphs Library, "The Beginner's Guide To Pulling A Starting Pitcher": https://library.fangraphs.com/the-beginners-guide-to-pulling-a-starting-pitcher/
- Brill, Deshpande, Wyner, "A Bayesian analysis of the time through the order penalty in baseball": https://doi.org/10.48550/arXiv.2210.06724
- FanGraphs, "Are Starters Improving Relative to Relievers?": https://blogs.fangraphs.com/are-starters-improving-relative-to-relievers/

### Direction F: Count Intent Matrix

Counts have different intent regimes:

```text
0-0:
  establish strike, but not necessarily middle

0-2:
  pitcher advantage
  chase/waste behavior becomes common
  ball cost is relatively small

3-0:
  must avoid walk
  zone rate and middle risk can rise

3-2:
  high consequence
  walk/strikeout boundary
```

FanGraphs analysis of 0-2 counts notes that pitchers throw in the zone much less often in 0-2 than in other counts. It cites 2019-2020 Baseball Savant data where pitches in the zone were about 49% in non-0-2 counts, but about 36% in 0-2. The article's point is that "control success" in 0-2 may not mean the same thing as in 3-0. A pitch outside the zone can be intentional and strategically sound.

Competition implication:

The target definition may classify some chase/waste outcomes as failure or success depending on the competition's hidden labeling. Therefore raw success by count can mix "bad command" with "intentional miss."

Feature candidate:

```text
count_intent_profile(count):
  attack_count
  chase_count
  protect_count
  get_me_over_count

pitcher_intent_deviation:
  P(ball/reverse/middle | pitcher,count) - P(ball/reverse/middle | count)
```

Useful hand-made buckets:

```text
ahead_count: strikes_before > balls_before
behind_count: balls_before > strikes_before
even_count: strikes_before == balls_before
two_strike_count: strikes_before == 2
three_ball_count: balls_before == 3
putaway_count: balls_before == 0 and strikes_before == 2
must_strike_count: balls_before == 3 and strikes_before <= 1
full_count: balls_before == 3 and strikes_before == 2
```

Why this matters:

v15 has `count_state`, but trees must infer baseball intent from ordinal numbers. A compact intent basis may let the model learn more stable splits across seasons.

Source:

- FanGraphs, "How Should Pitchers Approach 0-2 Counts?": https://blogs.fangraphs.com/how-should-pitchers-approach-0-2-counts/

### Direction G: Failure Direction Is Not Symmetric

The competition exposes several asof rates:

```text
success
reverse
middle
ball
strike
```

Domain-wise, these are not interchangeable:

```text
middle:
  often a miss toward the heart of the zone
  may still be a strike-like outcome but dangerous

ball:
  non-zone miss
  walk-count pressure

reverse:
  wrong-side miss
  can imply glove-side/arm-side command issue

strike:
  zone or strike outcome
  not always the same as command quality
```

Baseball's official strike definition includes swings, foul balls, and zone passage. Therefore `strike_rate` and `control_success` need not be identical. A pitcher can throw many strikes while missing intended spots into the middle.

Feature candidate:

```text
bad_miss_rate = ball_rate + reverse_rate
heart_miss_rate = middle_rate
attack_without_middle = strike_rate - middle_rate
command_quality = success_rate - a*reverse_rate - b*middle_rate - c*ball_rate
```

But v15 already uses raw smoothed components. The useful version is conditional:

```text
bad_miss_by_count
bad_miss_by_pitch_type
bad_miss_by_hand_matchup
heart_miss_by_game_type
```

Source:

- Baseball-Reference Bullpen, "Strike": https://www.baseball-reference.com/bullpen/Strike

### Direction H: Pitch Sequencing Without Test Leakage

Pitch sequencing matters because pitch intent is not independent from previous pitches:

```text
fastball after breaking ball
breaking ball after fastball
same pitch repeated
putaway pitch after two strikes
get-me-over pitch after falling behind
```

But the rule forbids using other test rows. Therefore current-test previous pitch reconstruction is unsafe unless the information is present in the row itself, which it is not.

Legal version:

Build historical sequencing tendency from train/Trackman and use it as a stored pitcher trait:

```text
pitcher_sequence_entropy
P(type_t | pitcher, count)
P(type_t | pitcher, count, previous_type)  # aggregated into trait, not current-row previous_type
repeat_tendency_by_pitcher
putaway_pitch_concentration
behind_count_get_me_over_tendency
```

If game/order reconstruction is safe in train only, possible summary traits:

```text
two_strike_breaking_share
three_ball_fastball_share
first_pitch_fastball_share
repeat_pitch_share
```

Why this may help:

v15 pitchtype uses `P(type | pitcher,count)` indirectly from Trackman matching. Sequencing traits can explain whether a pitcher has predictable or flexible intent, but must not depend on the actual previous test pitch.

Source:

- Pitcher Arsenal Analysis glossary, sequence/deception-related arsenal summaries: https://www.pitcherarsenal.com/
- Baseball Prospectus, "Introducing Pitch Tunnels": https://legacy.baseballprospectus.com/article_legacy.php?articleid=31030

### Direction I: Season Phase and Preparation State

Month matters in v15. Domain reasons:

```text
early season:
  command ramp-up
  pitch mix experimentation
  lower accumulated workload

mid season:
  stable role and scouting reports
  heat/fatigue effects

late season:
  roster changes
  playoff/ranking pressure
  expanded or altered usage
```

Possible features:

```text
season_phase:
  early = month <= 4
  mid = 5 <= month <= 8
  late = month >= 9

phase_workload:
  season_phase * inseason_n
  season_phase * role_proxy
  season_phase * game_type
  season_phase * pitcher_experience
```

Risk:

`game_month` is already in v15 and had nontrivial importance. Additional season-phase features must be compact and tested against v15 residuals. Do not create a 100-feature month interaction set.

### Direction J: Leverage and Risk Preference

`li`, base state, score, inning, and runners describe pressure. Pitchers may change risk preference:

```text
with base open:
  avoid middle, accept ball

runner on third / fewer than two outs:
  avoid wild miss, avoid middle depending batter

high leverage:
  use best pitch more often
  narrower command target

large lead:
  attack zone, avoid walks
```

v15 already has `li`, score, runners, and base_state, but it may not have pitcher-specific pressure response.

Candidate:

```text
pitcher_pressure_response:
  P(success | pitcher, li_bucket) - pitcher_prior
  P(ball/reverse/middle | pitcher, li_bucket) - pitcher_prior_component

context_risk_profile:
  base_open
  runner_scoring_position
  tying_run_on_base_proxy
  blowout_proxy
```

Risk:

Sparse and likely absorbed by existing context columns. Use only if shrinkage is strong and residual gain is stable.

## 13. What Not To Do

Avoid these because prior experiments already failed:

```text
more model seeds without new information
generic XGBoost/LightGBM blending
raw Trackman physical averages
simple release-point SD
ambiguous hidden denominator minimum-q
F/R split as direct feature
H2H sparse table
generic career volatility
label-conditioned features not robust to v15
```

## 14. Practical Screening Rule

Every new domain feature should answer four questions:

```text
1. Is it row-local at inference?
2. Is it a new state, not just a v15 re-expression?
3. Does it explain e = y - p_v15?
4. Is the sign stable in 2022, 2023, and 2024 rolling-year validation?
```

Minimal numerical gate:

```text
local +1~5 BSS: ignore
local +5~10 BSS: diagnostic only
local +15+ BSS and stable residual gain: candidate
structure opened like in-season: priority
```

## 15. Short Source List

- Release parameter and pitch location: https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2020.00036/full
- Accuracy and high-speed reproducibility: https://pubmed.ncbi.nlm.nih.gov/34376126/
- Count affects pitching performance: https://pmc.ncbi.nlm.nih.gov/articles/PMC9715958/
- 0-2 count pitcher approach: https://blogs.fangraphs.com/how-should-pitchers-approach-0-2-counts/
- Release point variability and performance: https://pmc.ncbi.nlm.nih.gov/articles/PMC11608975/
- Fatigue systematic review: https://pmc.ncbi.nlm.nih.gov/articles/PMC6673423/
- Workload recommendations systematic review: https://journals.sagepub.com/doi/10.1177/0363546519831010
- Pitch-count/biomechanics narrative review: https://pubmed.ncbi.nlm.nih.gov/37574914/
- Times through the order guide: https://library.fangraphs.com/the-beginners-guide-to-pulling-a-starting-pitcher/
- Bayesian TTOP paper: https://doi.org/10.48550/arXiv.2210.06724
- Starter/reliever role analysis: https://blogs.fangraphs.com/are-starters-improving-relative-to-relievers/
- MLB Catcher Framing glossary: https://www.mlb.com/glossary/statcast/catcher-framing
- TruMedia framing model: https://baseball.help.trumedianetworks.com/baseball/catcher-framing-model
- Hierarchical Bayesian pitch framing: https://www.degruyterbrill.com/document/doi/10.1515/jqas-2017-0027/html
- Strike definition: https://www.baseball-reference.com/bullpen/Strike
- Stuff+/Location+ primer: https://library.fangraphs.com/pitching/stuff-location-and-pitching-primer/
- PitchingBot overview: https://community.fangraphs.com/pitchingbot-using-machine-learning-to-understand-what-makes-a-good-pitch/
- Pitch tunnels: https://legacy.baseballprospectus.com/article_legacy.php?articleid=31030
- Pitcher Arsenal Analysis glossary: https://www.pitcherarsenal.com/
