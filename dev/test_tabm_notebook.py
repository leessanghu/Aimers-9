"""Static integrity checks for the self-contained Colab TabM notebook."""

import base64
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_gpu_tabm.ipynb"


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"<tabm-notebook-cell-{i}>", "exec")

    core_cell = "".join(next(
        cell["source"] for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and any("_MLP_CORE_B64" in line for line in cell["source"])
    ))
    values = {}
    for name in ("_MLP_CORE_B64", "_TABM_CORE_B64"):
        line = next(x for x in core_cell.splitlines() if x.startswith(f"{name} ="))
        values[name] = line.split("'", 2)[1]

    assert gzip.decompress(base64.b64decode(values["_MLP_CORE_B64"])) == (
        ROOT / "dev" / "phase3_embedding_mlp.py"
    ).read_bytes()
    assert gzip.decompress(base64.b64decode(values["_TABM_CORE_B64"])) == (
        ROOT / "dev" / "phase3_tabm.py"
    ).read_bytes()
    print("TabM notebook integrity: OK")


if __name__ == "__main__":
    main()

