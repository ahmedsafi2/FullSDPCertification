#!/usr/bin/env python3
"""
Rename the model value `MdSDP` to `UntargetedSDP` in all YAML files under all_product_yamls/.

Examples:
    python scripts/rename_mdsdp_to_mzbarsdp.py --dry-run
    python scripts/rename_mdsdp_to_mzbarsdp.py --backup
"""

import argparse
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
YAML_DIR = PROJECT_ROOT / "all_product_yamls"

VALUE_PATTERN = re.compile(r"(?m)^(?P<prefix>\s*(?:-\s*)?certification_model_type:\s*)MdSDP\s*$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rename MdSDP to UntargetedSDP in all_product_yamls/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the files that would be updated without modifying anything.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Keep a .bak copy next to each modified YAML file.",
    )
    return parser.parse_args()


def iter_yaml_files(directory: Path):
    for yaml_file in sorted(directory.glob("*.yaml")):
        if yaml_file.is_file():
            yield yaml_file


def replace_model_value(text: str) -> tuple[str, int]:
    updated_text, count = VALUE_PATTERN.subn(r"\g<prefix>UntargetedSDP", text)
    return updated_text, count


def main():
    args = parse_args()

    if not YAML_DIR.exists():
        raise SystemExit(f"YAML directory not found: {YAML_DIR}")

    changed_files = []
    for yaml_file in iter_yaml_files(YAML_DIR):
        original_text = yaml_file.read_text(encoding="utf-8")
        updated_text, replacements = replace_model_value(original_text)

        if replacements == 0:
            continue

        changed_files.append((yaml_file, replacements))
        if args.dry_run:
            continue

        if args.backup:
            backup_path = yaml_file.with_suffix(yaml_file.suffix + ".bak")
            backup_path.write_text(original_text, encoding="utf-8")

        yaml_file.write_text(updated_text, encoding="utf-8")

    if not changed_files:
        print("No files needed changes.")
        return 0

    print(f"{len(changed_files)} file(s) contain the legacy model name.")
    for yaml_file, replacements in changed_files:
        action = "Would update" if args.dry_run else "Updated"
        print(f"{action}: {yaml_file.relative_to(PROJECT_ROOT)} ({replacements} replacement(s))")

    if args.dry_run:
        print("\nDry-run only: no files were modified.")
    elif args.backup:
        print("\nBackups were written with the .bak suffix.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())