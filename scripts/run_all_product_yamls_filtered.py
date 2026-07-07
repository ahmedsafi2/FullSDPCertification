#!/usr/bin/env python3
"""
Run certification_problem.py on every YAML file in all_product_yamls/.

The underlying solver keeps writing its normal outputs in results/benchmark/,
and this wrapper only filters what is shown in the terminal. It also writes a
raw per-YAML log to make failures easier to inspect.

Example:
  python scripts/run_all_product_yamls_filtered.py --network blob_4x10 --title test
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML_DIR = PROJECT_ROOT / "all_product_yamls"
DEFAULT_SCRIPT = PROJECT_ROOT / "src" / "certification_problem.py"
DEFAULT_PATTERN = re.compile(
    r"ERROR|Running|CALLBACK|Traceback|Exception|FileNotFoundError|ValueError|TypeError|RuntimeError|KeyError|AssertionError"
)
DEFAULT_RAW_LOG_DIR = PROJECT_ROOT / "results" / "batch_logs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run certification_problem.py over every YAML in all_product_yamls and filter its output.",
    )
    parser.add_argument(
        "--network",
        default="blob_4x10",
        help="Network name passed to certification_problem.py (default: blob_4x10).",
    )
    parser.add_argument(
        "--title",
        default="test",
        help="Base run title passed to certification_problem.py (default: test).",
    )
    parser.add_argument(
        "--yaml-dir",
        type=Path,
        default=DEFAULT_YAML_DIR,
        help="Directory containing the YAML files to process.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=DEFAULT_SCRIPT,
        help="Path to certification_problem.py.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN.pattern,
        help="Regex used to filter lines from the subprocess output.",
    )
    parser.add_argument(
        "--raw-log-dir",
        type=Path,
        default=DEFAULT_RAW_LOG_DIR,
        help="Directory where the complete stdout/stderr of each run is saved.",
    )
    return parser.parse_args()


def iter_yaml_files(yaml_dir: Path):
    for yaml_file in sorted(yaml_dir.glob("*.yaml")):
        if yaml_file.is_file():
            yield yaml_file


def run_for_yaml(
    script: Path,
    network: str,
    title: str,
    yaml_file: Path,
    pattern: re.Pattern,
    raw_log_dir: Path,
):
    run_title = f"{title}__{yaml_file.stem}"
    command = [
        sys.executable,
        str(script),
        network,
        run_title,
        "--config",
        str(yaml_file),
    ]

    print(f"\n=== {yaml_file.name} -> title {run_title} ===")
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = raw_log_dir / f"{run_title}.log"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    with raw_log_path.open("w", encoding="utf-8", errors="replace") as raw_log_file:
        for raw_line in process.stdout:
            raw_log_file.write(raw_line)
            line = raw_line.rstrip("\n")
            if pattern.search(line):
                print(f"[{yaml_file.name}] {line}")

    return_code = process.wait()
    print(f"[{yaml_file.name}] exit code: {return_code}")
    print(f"[{yaml_file.name}] raw log: {raw_log_path}")
    return return_code


def main():
    args = parse_args()
    yaml_dir = args.yaml_dir.resolve()
    script = args.script.resolve()
    pattern = re.compile(args.pattern)
    raw_log_dir = args.raw_log_dir.resolve()

    if not yaml_dir.exists():
        print(f"YAML directory not found: {yaml_dir}", file=sys.stderr)
        return 1

    if not script.exists():
        print(f"certification_problem.py not found: {script}", file=sys.stderr)
        return 1

    yaml_files = list(iter_yaml_files(yaml_dir))
    if not yaml_files:
        print(f"No YAML files found in: {yaml_dir}", file=sys.stderr)
        return 1

    print(f"Processing {len(yaml_files)} YAML file(s) from {yaml_dir}")
    failures = []
    for yaml_file in yaml_files:
        return_code = run_for_yaml(script, args.network, args.title, yaml_file, pattern, raw_log_dir)
        if return_code != 0:
            failures.append((yaml_file.name, return_code))

    if failures:
        print("\nFailures:")
        for file_name, return_code in failures:
            print(f"  {file_name}: exit code {return_code}")
        return 1

    print("\nAll YAML runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())