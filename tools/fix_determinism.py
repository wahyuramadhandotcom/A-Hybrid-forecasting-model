#!/usr/bin/env python3
"""Audit dan perbaiki determinisme pada notebook repo ini.

Masalah yang ditangani
----------------------
1. Instansiasi ``XGBRegressor`` tanpa ``random_state``. Beberapa di antaranya
   adalah model FINAL yang angkanya masuk ke naskah, sehingga hasil yang
   dilaporkan tidak dapat direproduksi persis.
2. Tidak adanya seed global di awal notebook.
3. ``our_study_rosman.ipynb``: model di-``fit`` pada ``X_train`` (mentah) tetapi
   ``predict`` pada ``X_test_scaled`` (ter-skala).

Pemakaian
---------
    python tools/fix_determinism.py --audit          # hanya melaporkan
    python tools/fix_determinism.py --apply          # menulis perubahan

Skrip ini idempoten: menjalankannya dua kali tidak menambah patch ganda.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
TARGETS = ["our_study_pharma_daily.ipynb",
           "our_study_pharma_weekly.ipynb",
           "our_study_rosman.ipynb"]

CALL = re.compile(r'(?<![\w.])(?:xgb\.)?XGBRegressor\s*\(')

SEED_CELL = (
    "# === Reproducibility (ditambahkan pada revisi) ===\n"
    "# Satu seed global untuk seluruh notebook. SETIAP estimator stokastik\n"
    "# di bawah menerima random_state=SEED secara eksplisit.\n"
    "import os, random\n"
    "SEED = 42\n"
    "os.environ['PYTHONHASHSEED'] = str(SEED)\n"
    "random.seed(SEED)\n"
    "np.random.seed(SEED)\n"
)


def _in_string_or_comment(src: str, pos: int) -> bool:
    line = src[src.rfind("\n", 0, pos) + 1 : pos]
    return (line.count('"') % 2 == 1 or line.count("'") % 2 == 1
            or line.lstrip().startswith("#"))


def _closing_paren(src: str, open_paren: int) -> int:
    depth = 0
    for j in range(open_paren, len(src)):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return len(src) - 1


def process(path: Path, apply: bool):
    nb = json.loads(path.read_text(encoding="utf-8"))
    findings, changed = [], False

    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        original = src
        targets = []
        for m in CALL.finditer(src):
            if _in_string_or_comment(src, m.start()):
                continue
            op = m.end() - 1
            args = src[op + 1 : _closing_paren(src, op)]
            if "random_state" in args or "seed=" in args:
                continue
            targets.append((op, "\n" in args))
        for op, multiline in reversed(targets):
            findings.append(f"cell {ci}: XGBRegressor(...) tanpa random_state")
            insert = "random_state=SEED," + ("" if multiline else " ")
            src = src[: op + 1] + insert + src[op + 1 :]

        if path.name == "our_study_rosman.ipynb" and \
                "xgb_model.fit(X_train, y_train)" in src:
            findings.append(f"cell {ci}: fit pada X_train mentah, predict pada "
                            f"X_test_scaled (scaling mismatch)")
            src = src.replace(
                "xgb_model.fit(X_train, y_train)",
                "# FIX: sebelumnya fit pada X_train (mentah) tetapi predict pada\n"
                "# X_test_scaled (ter-skala) -- train/predict scaling mismatch.\n"
                "xgb_model.fit(X_train_scaled, y_train)")

        if src != original:
            changed = True
            if apply:
                nb["cells"][ci]["source"] = src.splitlines(keepends=True)

    sources = ["".join(c["source"]) for c in nb["cells"]]
    if not any("SEED = 42" in s for s in sources):
        anchor = max([i for i, s in enumerate(sources)
                      if "import numpy" in s or "import xgboost" in s] or [0])
        findings.append(f"tidak ada seed global (akan disisipkan setelah cell {anchor})")
        changed = True
        if apply:
            nb["cells"].insert(anchor + 1, {
                "cell_type": "code", "execution_count": None, "metadata": {},
                "outputs": [], "source": SEED_CELL.splitlines(keepends=True)})

    if apply and changed:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return findings, changed


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--audit", action="store_true")
    group.add_argument("--apply", action="store_true")
    ap.add_argument("--notebooks", nargs="*", default=TARGETS)
    args = ap.parse_args()

    total = 0
    for name in args.notebooks:
        path = NOTEBOOK_DIR / name
        if not path.exists():
            print(f"[lewati] {name} tidak ditemukan")
            continue
        findings, changed = process(path, apply=args.apply)
        total += len(findings)
        status = "DIPERBAIKI" if (args.apply and changed) else \
                 ("PERLU PERBAIKAN" if findings else "BERSIH")
        print(f"\n{name}  [{status}]")
        for f in findings:
            print(f"   - {f}")
    print(f"\nTotal temuan: {total}")
    if args.audit and total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
