#!/usr/bin/env python3
"""
Affiche les runs incomplets qui seraient supprimés par cleanup_incomplete_runs.py.
Aucune suppression n'est effectuée.
"""

import time
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent.parent / "results" / "benchmark"
MAX_AGE_DAYS = 1.0


def find_run_dirs(benchmark_dir: Path):
    for network_dir in sorted(benchmark_dir.iterdir()):
        if not network_dir.is_dir():
            continue
        for run_dir in sorted(network_dir.iterdir()):
            if run_dir.is_dir():
                yield run_dir


def has_nonempty_csv(run_dir: Path) -> bool:
    for csv_file in run_dir.rglob("*.csv"):
        try:
            with csv_file.open() as f:
                header = f.readline()
                if header and f.readline():
                    return True
        except OSError:
            continue
    return False


now = time.time()
to_delete = []
for run_dir in find_run_dirs(BENCHMARK_DIR):
    age_h = (now - run_dir.stat().st_mtime) / 3600
    old_enough = age_h > MAX_AGE_DAYS * 24
    if not has_nonempty_csv(run_dir) and old_enough:
        to_delete.append((run_dir, age_h))

if not to_delete:
    print("Aucun run incomplet éligible.")
else:
    print(f"{len(to_delete)} run(s) sans CSV non vide et âgés de plus de {MAX_AGE_DAYS} jour(s) :\n")
    for run_dir, age_h in to_delete:
        rel = run_dir.relative_to(BENCHMARK_DIR)
        print(f"  {rel}  (âge : {age_h:.1f}h)")
