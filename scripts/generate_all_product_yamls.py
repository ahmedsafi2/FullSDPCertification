"""Generate the `all_product_yamls` family from `config/blob_4x10.yaml`.

The generated layout matches the existing repository contents:

- `combo_0000__baseline__all_one_variable.yaml` keeps every product as
  `one_variable`.
- `combo_0001..combo_0330` each flip exactly one product entry to
  `composed`.

This script preserves the source file verbatim for the baseline and only
patches the `type:` field inside the selected product block for the
variants.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = PROJECT_ROOT / "config" / "blob_4x10.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "all_product_yamls"


def load_source() -> tuple[str, dict]:
    source_text = SOURCE_CONFIG.read_text(encoding="utf-8")
    source_data = yaml.safe_load(source_text)
    return source_text, source_data


def iter_products(source_data: dict):
    bound_strategy = source_data["models"][0]["bound_strategy"]
    for product_name, product_config in bound_strategy.items():
        yield product_name, product_config


def product_label(product_name: str, product_config: dict) -> str:
    layer, neuron, next_layer, next_neuron = product_config["key"]
    return f"{product_name}__l{layer}_u{neuron}_k{next_layer}_j{next_neuron}"


def replace_product_type(source_text: str, product_name: str, new_type: str) -> str:
    pattern = re.compile(
        rf"(^    {re.escape(product_name)}:\n(?:      .*\n)*?      type: )one_variable",
        re.MULTILINE,
    )
    replacement = rf"\1{new_type}"
    updated_text, count = pattern.subn(replacement, source_text, count=1)
    if count != 1:
        raise ValueError(f"Could not patch {product_name} to {new_type}")
    return updated_text


def write_text(path: Path, content: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(content, encoding="utf-8")


def generate(output_dir: Path, *, force: bool, dry_run: bool) -> int:
    source_text, source_data = load_source()
    products = list(iter_products(source_data))

    if dry_run:
        print(f"source: {SOURCE_CONFIG}")
        print(f"output: {output_dir}")
        print(f"products: {len(products)}")
        print("files: 1 baseline + one variant per product")
        return len(products) + 1

    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = output_dir / "combo_0000__baseline__all_one_variable.yaml"
    write_text(baseline_path, source_text, force=force)

    for index, (product_name, product_config) in enumerate(products, start=1):
        variant_name = product_label(product_name, product_config)
        variant_path = output_dir / f"combo_{index:04d}__{variant_name}__composed.yaml"
        variant_text = replace_product_type(source_text, product_name, "composed")
        write_text(variant_path, variant_text, force=force)

    return len(products) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate the all_product_yamls directory from config/blob_4x10.yaml.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the generated YAML files into.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the number of files that would be generated without writing them.",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Move an existing output directory aside before generating new files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.backup_existing and args.output_dir.exists():
        backup_path = args.output_dir.with_name(f"{args.output_dir.name}.bak")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(str(args.output_dir), str(backup_path))

    count = generate(args.output_dir, force=args.force, dry_run=args.dry_run)
    if args.dry_run:
        print(f"would_generate: {count}")
    else:
        print(f"generated: {count} files")


if __name__ == "__main__":
    main()