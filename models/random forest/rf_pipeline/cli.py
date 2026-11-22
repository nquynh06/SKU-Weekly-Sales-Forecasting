"""Run from any working directory; paths passed on the command line stay explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import Config
from .data import DataContractError
from .pipeline import predict_pipeline, run_pipeline, validate_pipeline, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Random Forest: gross completed SKU sales units, weekly H=1")
    subcommands = parser.add_subparsers(dest="command", required=True)
    default_config = Path(__file__).resolve().parents[1] / "configs" / "random_forest.json"
    for name in ("run", "validate", "predict"):
        sub = subcommands.add_parser(name)
        sub.add_argument("--config", type=Path, default=default_config)
        sub.add_argument("--data-dir", required=True, type=Path, help="Directory containing raw CSV tables")
        sub.add_argument("--output-dir", required=True, type=Path, help="NEW directory; existing directories are never overwritten")
        if name == "predict":
            sub.add_argument("--artifact", required=True, type=Path, help="TRUSTED local model.joblib only: unpickling may execute code")
            sub.add_argument("--origin-week", required=True, help="Monday of last closed week; >= model training cutoff")
    args = parser.parse_args(argv)
    # Check before entering pipeline, so diagnostics can never overwrite an old run.
    if args.output_dir.expanduser().resolve().exists():
        print("ERROR: output directory already exists; choose a new run directory", file=sys.stderr)
        return 2
    try:
        config = Config.load(args.config)
        if args.command == "run":
            result = run_pipeline(config, args.data_dir, args.output_dir)
        elif args.command == "validate":
            result = validate_pipeline(config, args.data_dir, args.output_dir)
        else:
            result = predict_pipeline(config, args.data_dir, args.output_dir, args.artifact, args.origin_week)
    except (ValueError, OSError, KeyError) as exc:
        output = args.output_dir.expanduser().resolve()
        if output.is_dir():
            diagnostic = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            if isinstance(exc, DataContractError):
                diagnostic["dq_report"] = exc.report
                write_json(output / "dq_report.json", exc.report)
            write_json(output / "failure.json", diagnostic)
            manifest_path = output / "run_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest.update(diagnostic)
                write_json(manifest_path, manifest)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
