#!/usr/bin/env python3
"""
Run certification_problem.py on every YAML file in all_product_yamls/.

The underlying solver keeps writing its normal outputs in results/benchmark/,
and this wrapper only filters what is shown in the terminal. It also writes a
raw per-YAML log to make failures easier to inspect.

Examples:
  # Run all yamls for blob_4x10 on the first 100 samples
  python scripts/run_all_product_yamls_filtered.py --network blob_4x10 --title test_100 --start 0 --end 100
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import random
import tempfile
import yaml
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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
    parser.add_argument(
        "--start", type=int, default=None, help="Start index for data samples (inclusive)."
    )
    parser.add_argument(
        "--end", type=int, default=None, help="End index for data samples (exclusive)."
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
    start: int | None,
    end: int | None,
):
    run_title = f"{title}__{yaml_file.stem}"
    result_dir = PROJECT_ROOT / "results" / "benchmark" / network / run_title
    if result_dir.exists():
        print(f"\n--- Skipping {yaml_file.name} (result directory exists) ---")
    network_results_dir = PROJECT_ROOT / "results" / "benchmark" / network
    network_results_dir.mkdir(parents=True, exist_ok=True)

    # Check if a result directory for this base title (with any timestamp) already exists.
    existing_runs = list(network_results_dir.glob(f"*{run_title}"))
    if existing_runs:
        print(f"\n--- Skipping {yaml_file.name} (found existing result: {existing_runs[0].name}) ---")
        return 99  # Special code for skipped

    command = [
        sys.executable,
        str(script),
        network,
        run_title,
        "--config",
        str(yaml_file),
    ]
    if start is not None:
        command.extend(["--start", str(start)])
    if end is not None:
        command.extend(["--end", str(end)])
    temp_yaml_path = None
    try:
        # To ensure certification_problem.py adds a timestamp, we must not use --start/--end args.
        # Instead, we inject the range into a temporary YAML file.
        if start is not None or end is not None:
            with yaml_file.open('r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if 'data' not in config: config['data'] = {}
            if start is not None: config['data']['start'] = start
            if end is not None: config['data']['end'] = end

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
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml", prefix="tmp_", dir=yaml_file.parent, encoding='utf-8') as tmp:
                yaml.dump(config, tmp, default_flow_style=False, sort_keys=False)
                temp_yaml_path = Path(tmp.name)
            config_path_for_run = temp_yaml_path
        else:
            config_path_for_run = yaml_file

    assert process.stdout is not None
    with raw_log_path.open("w", encoding="utf-8", errors="replace") as raw_log_file:
        for raw_line in process.stdout:
            raw_log_file.write(raw_line)
            line = raw_line.rstrip("\n")
            if pattern.search(line):
                print(f"[{yaml_file.name}] {line}")
        command = [
            sys.executable,
            str(script),
            network,
            run_title,  # certification_problem.py will prepend the timestamp to this
            "--config",
            str(config_path_for_run),
        ]

    return_code = process.wait()
    print(f"[{yaml_file.name}] exit code: {return_code}")
    print(f"[{yaml_file.name}] raw log: {raw_log_path}")
    return return_code
        print(f"\n=== {yaml_file.name} -> title {run_title} ===")
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        raw_log_path = raw_log_dir / f"{run_title}.log"
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
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
    finally:
        if temp_yaml_path and temp_yaml_path.exists():
            temp_yaml_path.unlink()


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

    random.shuffle(yaml_files)

    print(f"Found {len(yaml_files)} YAML file(s) in {yaml_dir}. Will process in random order, skipping existing runs.")
    failures = []
    skipped_count = 0
    for yaml_file in yaml_files:
        return_code = run_for_yaml(
            script,
            args.network,
            args.title,
            yaml_file,
            pattern,
            raw_log_dir,
            args.start,
            args.end,
        )
        if return_code == 99:
            skipped_count += 1
        elif return_code != 0:
            failures.append((yaml_file.name, return_code))

    processed_count = len(yaml_files) - skipped_count
    print(f"\n--- Run Summary ---")
    print(f"Total YAMLs found:      {len(yaml_files)}")
    print(f"Skipped (results exist): {skipped_count}")
    print(f"Attempted to process:    {processed_count}")

    if failures:
        print(f"Failures:                {len(failures)}")
        for file_name, return_code in failures:
            print(f"  - {file_name}: exit code {return_code}")
        return 1

    if processed_count > 0:
        print("\nAll new runs completed successfully.")
    else:
        print("\nNo new runs were needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())