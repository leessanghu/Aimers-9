import argparse
import importlib
import platform
import sys


CPU_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
]

GPU_PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "tabm",
    "optuna",
]


def probe(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:
        return False, str(exc)
    version = getattr(mod, "__version__", "unknown")
    return True, version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["cpu", "colab-gpu"], required=True)
    args = parser.parse_args()

    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")

    packages = CPU_PACKAGES if args.track == "cpu" else GPU_PACKAGES
    ok = True
    for pkg in packages:
        found, info = probe(pkg)
        status = "OK" if found else "MISSING"
        print(f"{pkg:12s} {status:8s} {info}")
        ok = ok and found

    if args.track == "colab-gpu":
        found, _ = probe("torch")
        if found:
            import torch

            print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"cuda device: {torch.cuda.get_device_name(0)}")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

