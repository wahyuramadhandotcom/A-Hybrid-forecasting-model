"""
Rebuild the exp06b outputs (Table 3, Table 4 DM tests, Table 5 comparison) from
the SAVED baseline results, without Jupyter and without TensorFlow.

Why this exists: after revision R1-1 only the AR-LRX rows of exp06b change; the
eight reference baselines do not use residuals and are reloaded from
`results/exp06b_rossmann_baselines_strong.csv` and `..._predictions.npz`
(REUSE_SAVED_BASELINES = True in the notebook). On some Windows machines the
notebook stalls in its TensorFlow smoke test before reaching that point. This
script executes the notebook's own cells - so the logic cannot drift from the
notebook - but skips the three cells that need TensorFlow or a display:

    * the keras/tensorflow/lightgbm imports (replaced by stubs; the reuse path
      never calls them),
    * the TensorFlow smoke test (`B.smoke_test_v2()`),
    * the final bar chart.

Run from the repository root, AFTER exp05b and exp05d have been re-run:

    python tools/run_exp06b_reuse.py

It rewrites results/exp06b_rossmann_baselines_strong_{table_utama,dm,exp06_vs_exp06b}.csv
and never touches the saved baseline csv/npz. It refuses to run if the notebook
would retrain (REUSE_SAVED_BASELINES not True).
"""
from __future__ import annotations

import json
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK = os.path.join(ROOT, "notebooks", "exp06b_rossmann_baselines_strong.ipynb")


def main():
    os.chdir(os.path.join(ROOT, "notebooks"))          # notebook paths are relative to notebooks/
    for name in ("keras", "tensorflow", "lightgbm"):     # not used on the reuse path
        mod = types.ModuleType(name)
        mod.__version__ = "not loaded (reuse path)"
        sys.modules.setdefault(name, mod)
    import matplotlib
    matplotlib.use("Agg")

    cells = [("".join(c["source"]), c["cell_type"])
             for c in json.load(open(NOTEBOOK, encoding="utf-8"))["cells"]]
    code = [(i, s) for i, (s, t) in enumerate(cells) if t == "code"]

    if not any("REUSE_SAVED_BASELINES = True" in s for _, s in code):
        sys.exit("REUSE_SAVED_BASELINES is not True in the notebook - refusing to retrain from a script.")

    ns = {"display": lambda *a, **k: print(*a), "__name__": "__main__"}
    for i, src in code:
        if "smoke_test_v2()" in src:
            print(f"[cell {i}] skipped: TensorFlow smoke test (not needed when baselines are reused)")
            continue
        if "plt.subplots" in src and "barh" in src:
            print(f"[cell {i}] skipped: bar chart")
            continue
        print(f"\n[cell {i}] running", flush=True)
        exec(compile(src, f"exp06b cell {i}", "exec"), ns)
        if "REUSE_SAVED_BASELINES" in ns and ns["REUSE_SAVED_BASELINES"] is not True:
            sys.exit("REUSE_SAVED_BASELINES was switched off - stopping before any retraining.")
    print("\nDone. Updated: results/exp06b_rossmann_baselines_strong_table_utama.csv, "
          "_dm.csv, _exp06_vs_exp06b.csv")


if __name__ == "__main__":
    main()
