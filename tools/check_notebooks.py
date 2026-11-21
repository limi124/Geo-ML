"""Check the public notebook files without requiring local tutorial data."""

import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
EXPECTED = [
    "01_认识数据.ipynb",
    "02_栅格对齐.ipynb",
    "03_构建训练表.ipynb",
    "04_空间验证.ipynb",
    "05_回归基线.ipynb",
    "06_决策树与随机森林.ipynb",
    "07_XGBoost与SHAP.ipynb",
    "08_KNN_SVR_MLP.ipynb",
]


def main():
    problems = []
    for name in EXPECTED:
        path = NOTEBOOKS / name
        if not path.exists():
            problems.append(f"missing notebook: {name}")
            continue

        notebook = nbformat.read(path, as_version=4)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        if not code_cells:
            problems.append(f"{name}: no code cells")
        if notebook.metadata.get("kernelspec", {}).get("name") != "geoml":
            problems.append(f"{name}: kernelspec is not geoml")

        for cell in code_cells:
            if cell.execution_count is not None or cell.outputs:
                problems.append(f"{name}: public notebook contains execution output")
                break

        source = "\n".join(cell.source for cell in notebook.cells)
        if "F:\\" in source or "D:\\DeepSeekHarnessDesktop" in source:
            problems.append(f"{name}: contains a local absolute path")

        print(f"{name}: {len(notebook.cells)} cells, {len(code_cells)} code cells")

    for relative in ("data/processed", "outputs", "tmp", ".jupyter"):
        if (ROOT / relative).exists():
            problems.append(f"local-only path included: {relative}")

    print("RESULT:", "PASS" if not problems else f"{len(problems)} problem(s)")
    for problem in problems:
        print(" -", problem)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
