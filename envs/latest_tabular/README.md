# Latest Tabular Experiments

This folder separates local CPU experiments from Colab GPU experiments.

Default rule:

- Run core validation, LightGBM, XGBoost, CatBoost, Optuna, and final packaging on local CPU.
- Use Colab GPU only for neural tabular models where GPU actually changes iteration speed.
- Do not overwrite the current `submit/` safe artifact from this folder.

## Local CPU Track

Use this for models that should be deployable under the competition server CPU budget:

- LightGBM classifier
- LightGBM L2 regressor
- XGBoost classifier/regressor
- CatBoost, only if tuned and fast enough
- AutoGluon scouting with caution, not as a final packaged dependency by default

Setup:

```powershell
cd "C:\Users\이상후\OneDrive\바탕 화면\Aimers 9"
.\envs\latest_tabular\setup_cpu.ps1
```

The local default Python shown on this machine is 3.14, which is too new for many ML packages.
Use Python 3.11 for these environments because the evaluation server is Python 3.11.

## Colab GPU Track

Use this only for neural candidates:

- embedding MLP
- TabM
- optionally FT-Transformer or TabICL/TabPFN risk checks

Colab should export only prediction files or native model artifacts back into the repo.
Final submission still needs CPU-thread-limited timing unless the final script explicitly uses GPU safely.

Setup in Colab:

```bash
bash envs/latest_tabular/setup_colab.sh
```

## Pretrained/Foundation Model Caution

TabPFN, TabICL, TabDPT, and similar models may rely on pretrained checkpoints.
Because the competition restricts data sources to the official train/test/trackman files,
do not use pretrained checkpoints in a final submission without organizer confirmation.

They are allowed here only as research/scouting experiments.

