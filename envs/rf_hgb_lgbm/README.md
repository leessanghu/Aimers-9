# RF/HGB/LGBM rolling experiment

This environment rebuilds the reliable tree ensemble on leakage-safe rolling folds:

- train through 2021, validate 2022
- train through 2022, validate 2023
- train through 2023, validate 2024

The already validated RF/HGB/classifier OOF files are reused from `dev/phase2_preds`.
The tuned LGBM L2 model is retrained: early stopping only chooses the number of
iterations, then the fold model is refit on all available fold rows. The final
trainer refits every model on all 2019-2024 rows.

Run from this directory:

```powershell
python run_oof_refit.py
python optimize_blend.py
python train_final.py
```

Outputs are kept inside this directory and do not overwrite `submit/`.
