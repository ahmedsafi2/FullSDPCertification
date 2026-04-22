#!/usr/bin/env python3
"""
Supprime les sous-dossiers de runs incomplets dans results/benchmark/.

Un run est considéré incomplet si :
  - il ne contient aucun fichier CSV non vide (header seul = vide)
  - il date de plus d'un jour (mtime du dossier)

Usage :
  python scripts/cleanup_incomplete_runs.py [--dry-run] [--max-age-days N] [--yes]

Rappel mensuel conseillé : ajoute une entrée dans ton calendrier le 1er de chaque mois.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent.parent / "results" / "benchmark"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Affiche ce qui serait supprimé sans rien effacer.",
    )
    parser.add_argument(
        "--max-age-days", type=float, default=1.0, metavar="N",
        help="Ancienneté minimale (en jours) pour qu'un run soit éligible à la suppression (défaut : 1).",
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Supprime sans demander de confirmation (pour usage non interactif).",
    )
    return parser.parse_args()


def find_run_dirs(benchmark_dir: Path):
    """Yield tous les sous-dossiers de runs (deux niveaux : réseau/run)."""
    for network_dir in sorted(benchmark_dir.iterdir()):
        if not network_dir.is_dir():
            continue
        for run_dir in sorted(network_dir.iterdir()):
            if run_dir.is_dir():
                yield run_dir


def has_nonempty_csv(run_dir: Path) -> bool:
    """Retourne True si le dossier contient au moins un CSV avec plus d'une ligne (header + données)."""
    for csv_file in run_dir.rglob("*.csv"):
        try:
            with csv_file.open() as f:
                header = f.readline()
                if header and f.readline():
                    return True
        except OSError:
            continue
    return False


def is_old_enough(path: Path, max_age_days: float) -> bool:
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds > max_age_days * 86400


def confirm(n: int) -> bool:
    """Demande une confirmation interactive. Retourne True si l'utilisateur confirme."""
    print(f"\n{n} dossier(s) vont être supprimés définitivement.")
    try:
        answer = input("Confirmer la suppression ? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAnnulé.")
        return False
    return answer == "y"


def main():
    args = parse_args()

    if not BENCHMARK_DIR.exists():
        print(f"Dossier benchmark introuvable : {BENCHMARK_DIR}")
        sys.exit(1)

    print("Analyse des runs incomplets en cours...")
    to_delete = []
    for run_dir in find_run_dirs(BENCHMARK_DIR):
        if not has_nonempty_csv(run_dir) and is_old_enough(run_dir, args.max_age_days):
            to_delete.append(run_dir)

    if not to_delete:
        print("Aucun run incomplet éligible à la suppression.")
        return

    print(f"\n{len(to_delete)} run(s) sans CSV non vide, âgés de plus de {args.max_age_days} jour(s) :\n")
    for run_dir in to_delete:
        age_h = (time.time() - run_dir.stat().st_mtime) / 3600
        print(f"  {run_dir.relative_to(BENCHMARK_DIR.parent.parent)}  (âge : {age_h:.1f}h)")

    if args.dry_run:
        print("\nMode dry-run : rien n'a été supprimé.")
        return

    if not args.yes and not confirm(len(to_delete)):
        print("Suppression annulée.")
        return

    print()
    deleted, errors = 0, 0
    for run_dir in to_delete:
        try:
            shutil.rmtree(run_dir)
            print(f"  Supprimé : {run_dir.name}")
            deleted += 1
        except Exception as e:
            print(f"  ERREUR sur {run_dir.name} : {e}")
            errors += 1

    print(f"\n{deleted} dossier(s) supprimé(s), {errors} erreur(s).")


if __name__ == "__main__":
    main()
