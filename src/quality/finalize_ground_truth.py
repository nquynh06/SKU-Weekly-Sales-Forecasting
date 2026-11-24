from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def existing(root: Path, names: list[str], label: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {label}; tried: " + ", ".join(names))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_metrics(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, dtype="string").fillna("")
    if not {"metric", "value"}.issubset(df.columns):
        raise ValueError(f"Expected metric and value columns in {path}")
    return dict(zip(df["metric"], df["value"]))


def run_step(name: str, script: Path, args: list[str], logfile: Path) -> None:
    cmd = [sys.executable, str(script), *args]
    with logfile.open("w", encoding="utf-8") as f:
        f.write("Command: " + " ".join(cmd) + "\n\n")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", check=False)
        f.write(result.stdout)
    if result.returncode != 0:
        excerpt = result.stdout[-2500:]
        raise RuntimeError(f"{name} failed (exit {result.returncode}). See {logfile}\n{excerpt}")


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def make_archive(output_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=f"DS317_GroundTruth_Handoff/{path.relative_to(output_dir).as_posix()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    p.add_argument("--candidates", type=Path,
                   help="Reviewed SKU-week candidate CSV (default: reports/tables/sku_week_activity_candidates_reviewed.csv)")
    p.add_argument("--clean-file", type=Path, help="Clean transactions CSV")
    p.add_argument("--output-dir", type=Path, default=Path("deliverables/ground_truth_review"),
                   help="Destination folder, relative to repo root unless absolute")
    a = p.parse_args()
    root = a.root.resolve()
    if not root.is_dir():
        p.error(f"Repository root does not exist: {root}")

    def resolve_override(path: Path) -> Path:
        return path if path.is_absolute() else (root / path)

    candidates = resolve_override(a.candidates) if a.candidates else existing(
        root, ["reports/tables/sku_week_activity_candidates_reviewed.csv"],
        "Step 9 reviewed candidates")
    clean = resolve_override(a.clean_file) if a.clean_file else existing(
        root, ["data/processed/clean_sales_transactions.csv",
               "src/data/processed/clean_sales_transactions.csv"], "clean transactions")
    if not candidates.is_file() or not clean.is_file():
        p.error("Candidates or clean transactions CSV is missing")
    scripts = {
        "build": existing(root, ["src/data/build_ground_truth.py"], "Step 4 script"),
        "quality": existing(root, ["src/quality/data_quality.py"], "Step 5 script"),
        "eda": existing(root, ["src/quality/ground_truth_eda.py"], "Step 6 script"),
        "validation": existing(root, ["src/quality/validate_ground_truth.py"], "Step 7 script"),
    }
    reports = {name: existing(root, ["reports/tables/" + name], name) for name in [
        "raw_quality_report.csv", "key_relationship_report.csv",
        "transaction_cleaning_report.csv", "sku_lifecycle_summary.csv"]}

    dest = resolve_override(a.output_dir).resolve()
    if dest == root or root in dest.parents and dest.name in {"src", "data", "reports"}:
        p.error("Choose a dedicated handoff output directory, not an existing source folder")
    tables = dest / "tables"
    figures = dest / "figures"
    logs = dest / "logs"
    data = dest / "data"
    for d in (tables, figures, logs, data):
        d.mkdir(parents=True, exist_ok=True)
    ground_truth = data / "sku_week_ground_truth_provisional.csv"

    run_step("Step 4 - Ground truth", scripts["build"], [
        "--candidates", str(candidates), "--clean-file", str(clean),
        "--out-file", str(ground_truth),
        "--report-file", str(tables / "ground_truth_build_report.csv")],
        logs / "step4.log")
    run_step("Step 5 - Quality", scripts["quality"], [
        "--raw-quality", str(reports["raw_quality_report.csv"]),
        "--key-report", str(reports["key_relationship_report.csv"]),
        "--clean-report", str(reports["transaction_cleaning_report.csv"]),
        "--lifecycle-summary", str(reports["sku_lifecycle_summary.csv"]),
        "--candidates", str(candidates),
        "--ground-truth", str(ground_truth), "--report-dir", str(tables)],
        logs / "step5.log")
    run_step("Step 6 - EDA", scripts["eda"], [
        "--ground-truth", str(ground_truth),
        "--dq-file", str(tables / "data_quality_6_dimensions.csv"),
        "--figure-dir", str(figures), "--table-dir", str(tables)],
        logs / "step6.log")
    run_step("Step 7 - Validation", scripts["validation"], [
        "--ground-truth", str(ground_truth), "--candidates", str(candidates),
        "--clean-file", str(clean), "--out-dir", str(tables)],
        logs / "step7.log")

    builder = read_metrics(tables / "ground_truth_build_report.csv")
    validator = read_metrics(tables / "ground_truth_validation_summary.csv")
    checks = pd.read_csv(tables / "ground_truth_validation_checks.csv", low_memory=False)
    failed = checks[checks["status"].astype(str).eq("FAIL")]
    if not failed.empty:
        raise RuntimeError(f"Validation has {len(failed)} FAIL checks; no handoff ZIP was made. Review {tables}")
    if int(builder["gross_quantity_in_panel"]) != int(builder["scoped_clean_source_gross_qty"]):
        raise RuntimeError("Quantity mismatch between labels and cleaned source")
    if int(builder["unverified_no_sale_sku_weeks"]) > 0 and str(validator["validation_status"]).startswith("FINAL"):
        raise RuntimeError("Validation marked FINAL despite unverified no-sale weeks")

    # Do not claim full ground truth certification on the strength of automated audits alone.
    status = "PROVISIONAL_BLOCKED" if (
        int(builder["unverified_no_sale_sku_weeks"]) > 0 or
        str(validator["validation_status"]) != "FINAL_APPROVED"
    ) else "REQUIRES_MANUAL_SIGNOFF"
    for name in ["evidence_application_summary.csv", "evidence_application_decisions.csv"]:
        candidate = root / "reports" / "tables" / name
        if candidate.is_file():
            shutil.copy2(candidate, tables / name)

    checks_count = {str(k): int(v) for k, v in checks["status"].value_counts().items()}
    summary = {
        "handoff_status": status,
        "validation_status": validator.get("validation_status"),
        "dataset_status": builder.get("dataset_status"),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_candidates": rel(candidates, root),
        "source_transactions": rel(clean, root),
        "total_sku_weeks": int(builder["sku_week_rows"]),
        "observed_positive": int(builder["observed_positive_sku_weeks"]),
        "verified_zero": int(builder["verified_zero_sku_weeks"]),
        "unverified_no_sale": int(builder["unverified_no_sale_sku_weeks"]),
        "gross_units": int(builder["gross_quantity_in_panel"]),
        "product_date_conflict_skus": int(builder["skus_with_product_date_conflicts"]),
        "validation_check_counts": checks_count,
        "steps_reexecuted": [4, 5, 6, 7],
        "model_training_performed": False,
        "unknown_labels_filled_with_zero": False,
    }
    (dest / "handoff_status.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    readme = f"""# DS317 – SKU–Week Ground Truth Handoff

**Status: {status}** (NOT certified as complete forecasting ground truth)

## Scope and definition
- Target: `WeeklyQuantity` = gross units from valid completed orders by SKU and Monday–Sunday week.
- Returns are not subtracted; each observed positive SKU-week uses real transaction quantities.
- `verified_zero` is allowed ONLY with documented SKU activity, complete weekly order-log coverage and explicit approval.
- Unverified no-sale weeks stay `NA`, not `0`.
- Rolling one-step model feature training is **outside this handoff**.

## This run
- SKU-week rows: **{summary['total_sku_weeks']:,}**
- Observed positive rows: **{summary['observed_positive']:,}**
- Verified zero rows: **{summary['verified_zero']:,}**
- Unverified no-sale rows: **{summary['unverified_no_sale']:,}**
- Observed gross units: **{summary['gross_units']:,}**
- Product-date conflict SKUs: **{summary['product_date_conflict_skus']:,}**
- Validation: **{summary['validation_status']}** (status counts: {checks_count})

## Included files
- `data/sku_week_ground_truth_provisional.csv` – provisional SKU-week observations and evidence flags.
- `tables/` – cleaning reconciliation, six-dimensional quality review, EDA tables and validation checks.
- `figures/` – exploratory figures; missing labels never become zero-sales observations.
- `logs/` – complete commands/output for steps 4–7.
- `handoff_status.json` – machine-readable handoff status and metrics.
- `source_manifest.json` – SHA-256 checksums for scripts/input datasets used in the run.

## Unresolved limitations and next decision
1. Confirm SKU-level sellability / product effective dates and investigate product-date conflicts.
2. Establish authoritative completeness of weekly order logs before approving no-sale labels.
3. If the source data cannot provide this evidence, retain the dataset as **observed gross-sales ground truth with missing targets**; do not label it complete or use all 117,260 rows as supervised targets.
4. Run any MA4, RF, HGB feature pipeline with past-only features and separate leakage controls; this handoff does **not** certify model features.

**Important:** Automated tests passing does not imply the missing zero-sales evidence has been supplied.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")
    tracked = {"step9_candidates": candidates, "clean_transactions": clean}
    tracked.update({f"script_{name}": path for name, path in scripts.items()})
    tracked.update({f"source_{name}": path for name, path in reports.items()})
    manifest = {name: {"file": rel(path, root), "size_bytes": path.stat().st_size,
                       "sha256": sha256(path)} for name, path in tracked.items()}
    (dest / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    zip_path = dest.parent / "DS317_GroundTruth_Handoff_PROVISIONAL.zip"
    make_archive(dest, zip_path)
    print("\n=== HANDOFF COMPLETE ===")
    print(f"Status: {status}")
    print(f"Validation: {validator.get('validation_status')}; {checks_count}")
    print(f"Verified zero: {summary['verified_zero']}; NA: {summary['unverified_no_sale']}")
    print(f"Ground truth: {ground_truth}")
    print(f"Package: {zip_path}")


if __name__ == "__main__":
    main()
