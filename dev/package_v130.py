import os
import shutil
import zipfile
import glob as _glob

BASE = os.path.dirname(os.path.abspath(__file__))
SUBMIT = os.path.abspath(os.path.join(BASE, "..", "submit"))
DATA_DIR = os.path.join(SUBMIT, "data")
OUT_DIR = os.path.join(SUBMIT, "output")
MODEL_DIR = os.path.join(SUBMIT, "model")
ZIP_PATH = os.path.abspath(os.path.join(BASE, "..", "submit_v130_bundle.zip"))

CODEX_LEFTOVERS = [
    "advanced_features.py", "downside_features.py", "orthogonal_features.py",
    "persona_features.py", "raw_id_features.py", "v7_features.py",
    "phase_v12_submission.pkl", "v20_905_submission.pkl",
]

pkl_path = os.path.join(MODEL_DIR, "model_artifacts_v130.pkl")
assert os.path.exists(pkl_path), f"missing model: {pkl_path}"
print(f"model ok: {pkl_path} ({os.path.getsize(pkl_path)/1e6:.1f}MB)")

to_move = [f for f in os.listdir(MODEL_DIR)
           if (f.endswith(".pkl") and f != "model_artifacts_v130.pkl") or f in CODEX_LEFTOVERS]
moved = []
try:
    for f in to_move:
        src = os.path.join(MODEL_DIR, f)
        dst = src + ".bak"
        shutil.move(src, dst)
        moved.append((src, dst))
    active = sorted(os.path.basename(p) for p in _glob.glob(
        os.path.join(MODEL_DIR, "model_artifacts_v*.pkl")))
    assert active == ["model_artifacts_v130.pkl"], f"isolation failed: {active}"
    for f in CODEX_LEFTOVERS:
        assert not os.path.exists(os.path.join(MODEL_DIR, f)), f"codex leftover: {f}"

    os.makedirs(DATA_DIR, exist_ok=True)
    real_data = os.path.abspath(os.path.join(BASE, "..", "data"))
    shutil.copy(os.path.join(real_data, "test.csv"), os.path.join(DATA_DIR, "test.csv"))
    shutil.copy(os.path.join(real_data, "sample_submission.csv"),
                os.path.join(DATA_DIR, "sample_submission.csv"))
    os.makedirs(OUT_DIR, exist_ok=True)

    print("smoke test...")
    ret = os.system(f'cd "{SUBMIT}" && python script.py')
    sub_csv = os.path.join(OUT_DIR, "submission.csv")
    assert ret == 0, f"script.py failed (exit={ret})"
    assert os.path.exists(sub_csv), "submission.csv not created"
    with open(sub_csv, encoding="utf-8") as f:
        lines = f.readlines()
    print(f"smoke ok: {len(lines)} lines")
    print("".join(lines[:3]))

    shutil.rmtree(DATA_DIR, ignore_errors=True)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    shutil.rmtree(os.path.join(SUBMIT, "__pycache__"), ignore_errors=True)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(SUBMIT, "script.py"), "script.py")
        z.write(os.path.join(SUBMIT, "requirements.txt"), "requirements.txt")
        z.write(pkl_path, "model/model_artifacts_v130.pkl")
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = sorted(z.namelist())
    print(f"packaged: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)/1e6:.1f}MB) files={names}")
finally:
    for src, dst in moved:
        try:
            shutil.move(dst, src)
        except Exception as e:
            print(f"restore failed: {dst} -> {src}: {e}")
