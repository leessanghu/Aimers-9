# Run Plan

## CPU First

1. Finish Phase 2 rolling-fold baselines.
2. Fix Optuna parameter reconstruction before trusting tuned results.
3. Add XGBoost as a CPU candidate after LightGBM baselines are stable.
4. Keep CatBoost as a tuned-but-expensive candidate, not the first tuning target.
5. Use 2024 validation-size timing with 6 CPU threads before packaging.

## Colab GPU Only

1. Train embedding MLP with ID dropout and unknown embeddings.
2. Train TabM if the MLP produces useful OOF diversity.
3. Export fold predictions:
   - `fold_2022_pred_<model>.csv`
   - `fold_2023_pred_<model>.csv`
   - `fold_2024_pred_<model>.csv`
4. Bring prediction files back to local and blend with NNLS.

## Do Not Prioritize Yet

- TabNet: lower expected value than embedding MLP or TabM.
- TabPFN, TabICL, TabDPT: interesting but pretrained/foundation-model rule risk.
- Full AutoGluon in final submit: useful for scouting, too heavy as a default packaging path.

