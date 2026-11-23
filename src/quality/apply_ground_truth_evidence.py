from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEY = ["SKU_ID", "WeekStart"]
REVIEW_REQUIRED = {
    "SKU_ID", "ProductID", "WeekStart", "WeekEnd", "ProductWindowEvidence",
    "sku_activity_evidence_uri", "sku_activity_evidence_verified",
    "complete_week_order_log_evidence_uri", "complete_week_order_log_verified",
    "reviewer", "review_date", "decision", "notes",
}
CANDIDATE_REQUIRED = {
    "SKU_ID", "ProductID", "WeekStart", "WeekEnd", "ProductWindowEvidence",
    "HasObservedSale", "SkuWeekVerifiedActive", "SalesLogCoverageVerified",
    "ZeroLabelAuthorized", "DateConflictSalesLines", "observed_gross_qty",
}
DECISIONS = {"PENDING", "ELIGIBLE_ZERO", "UNVERIFIED", "OUT_OF_SCOPE"}
OUTSIDE = {"before_active_from", "after_active_to"}


def bool_series(s: pd.Series, column: str) -> pd.Series:
    values = s.astype("string").str.strip().str.lower()
    if (~values.isin(["true", "false", "0", "1"])).any():
        raise ValueError(f"{column} has missing/invalid booleans (only true/false or 1/0 allowed)")
    return values.isin(["true", "1"])


def optional_true(s: pd.Series) -> pd.Series:
    values = s.fillna("").astype(str).str.strip().str.lower()
    bad = ~values.isin(["true", "false", "", "1", "0"])
    if bad.any():
        raise ValueError(f"Review evidence verification contains {int(bad.sum())} invalid values")
    return values.isin(["true", "1"])


def text_present(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().ne("")


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required fields {missing}")


def apply(candidate_file: Path, review_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    panel = pd.read_csv(candidate_file, dtype={"SKU_ID": "string", "ProductID": "string"}, low_memory=False)
    rev = pd.read_csv(review_file, dtype={"SKU_ID": "string", "ProductID": "string"}, low_memory=False)
    require_columns(panel, CANDIDATE_REQUIRED, candidate_file.name)
    require_columns(rev, REVIEW_REQUIRED, review_file.name)
    if panel.duplicated(KEY).any() or rev.duplicated(KEY).any():
        raise ValueError("Duplicate (SKU_ID, WeekStart) in candidate panel or review file")

    panel["WeekStart"] = pd.to_datetime(panel["WeekStart"], errors="raise").dt.strftime("%Y-%m-%d")
    rev["WeekStart"] = pd.to_datetime(rev["WeekStart"], errors="raise").dt.strftime("%Y-%m-%d")
    panel["WeekEnd"] = pd.to_datetime(panel["WeekEnd"], errors="raise").dt.strftime("%Y-%m-%d")
    rev["WeekEnd"] = pd.to_datetime(rev["WeekEnd"], errors="raise").dt.strftime("%Y-%m-%d")

    for col in ["HasObservedSale", "SkuWeekVerifiedActive", "SalesLogCoverageVerified", "ZeroLabelAuthorized"]:
        panel[col] = bool_series(panel[col], col)
    panel["DateConflictSalesLines"] = pd.to_numeric(panel["DateConflictSalesLines"], errors="raise")
    panel["observed_gross_qty"] = pd.to_numeric(panel["observed_gross_qty"], errors="raise")
    if (panel["DateConflictSalesLines"] < 0).any():
        raise ValueError("Negative DateConflictSalesLines")
    if (panel["HasObservedSale"] & panel["observed_gross_qty"].le(0)).any():
        raise ValueError("Observed positive weeks must have positive sales quantity")

    rev["decision"] = rev["decision"].fillna("PENDING").astype(str).str.strip().str.upper()
    unknown_decisions = sorted(set(rev["decision"]) - DECISIONS)
    if unknown_decisions:
        raise ValueError(f"Unknown review decisions: {unknown_decisions}")

    keys = panel[KEY + ["ProductID", "WeekEnd", "ProductWindowEvidence", "HasObservedSale", "DateConflictSalesLines"]].merge(
        rev[KEY + ["ProductID", "WeekEnd", "ProductWindowEvidence", "decision"]],
        on=KEY, how="right", indicator=True, validate="one_to_one", suffixes=("_panel", "_review")
    )
    if (keys["_merge"] != "both").any():
        raise ValueError("Review contains keys not present in candidate panel")
    for field in ["ProductID", "WeekEnd", "ProductWindowEvidence"]:
        different = keys[f"{field}_panel"].fillna("").ne(keys[f"{field}_review"].fillna(""))
        if different.any():
            raise ValueError(f"Review {field} differs from candidates for {int(different.sum())} rows; regenerate review")
    if (keys["HasObservedSale"] & keys["decision"].ne("PENDING")).any():
        raise ValueError("Refusing manual review actions on weeks with observed sales")

    review_dates = pd.to_datetime(rev["review_date"], errors="coerce")
    has_reviewer = text_present(rev["reviewer"])
    has_date = review_dates.notna()
    activity_verified = optional_true(rev["sku_activity_evidence_verified"])
    coverage_verified = optional_true(rev["complete_week_order_log_verified"])
    has_activity_uri = text_present(rev["sku_activity_evidence_uri"])
    has_coverage_uri = text_present(rev["complete_week_order_log_evidence_uri"])

    approves = rev["decision"].eq("ELIGIBLE_ZERO")
    valid_approval = (activity_verified & coverage_verified & has_activity_uri & has_coverage_uri & has_reviewer & has_date)
    if (approves & ~valid_approval).any():
        problems = rev.loc[approves & ~valid_approval, KEY].head(8).to_dict("records")
        raise ValueError(f"ELIGIBLE_ZERO requires BOTH verified evidence sources, reviewer, and review date; examples: {problems}")
    in_conflict = keys["DateConflictSalesLines"].gt(0).reset_index(drop=True)
    if (approves & in_conflict).any():
        raise ValueError("Cannot approve zeros on SKU with unresolved product-date conflicts. Resolve and re-audit first")
    if (approves & rev["ProductWindowEvidence"].isin(OUTSIDE)).any():
        raise ValueError("Cannot approve zero outside recorded product activity window")

    scopes = rev["decision"].eq("OUT_OF_SCOPE")
    if (scopes & ~(has_activity_uri & has_reviewer & has_date & text_present(rev["notes"]))).any():
        raise ValueError("OUT_OF_SCOPE needs evidence reference, reviewer, date, and written reason")
    if (scopes & in_conflict).any():
        raise ValueError("Resolve conflicting lifecycle metadata before confirming OUT_OF_SCOPE")

    # Review data are attached only to candidate rows matching the exact SKU-week key.
    rev_sub = rev[KEY + ["decision", "reviewer", "review_date", "sku_activity_evidence_uri",
                          "complete_week_order_log_evidence_uri"]].rename(columns={"decision": "manual_decision"})
    out = panel.merge(rev_sub, on=KEY, how="left", validate="one_to_one")
    out["manual_decision"] = out["manual_decision"].fillna("NOT_REVIEWED")
    out["scope_review_status"] = "UNVERIFIED"
    out.loc[out["manual_decision"].eq("OUT_OF_SCOPE"), "scope_review_status"] = "REVIEWED_OUT_OF_SCOPE"

    # Existing verified zeros are preserved and checked; new zeros require explicit reviews.
    pos = out["HasObservedSale"]
    old_auth = out["ZeroLabelAuthorized"]
    if (old_auth & (pos | ~out["SkuWeekVerifiedActive"] | ~out["SalesLogCoverageVerified"])).any():
        raise ValueError("Existing candidate zero authorization is internally inconsistent")
    newly_approved = out["manual_decision"].eq("ELIGIBLE_ZERO") & ~pos
    out.loc[newly_approved, "SkuWeekVerifiedActive"] = True
    out.loc[newly_approved, "SalesLogCoverageVerified"] = True
    out.loc[newly_approved, "ZeroLabelAuthorized"] = True

    # Don't let a reviewed exclusion produce a zero or erase a real recorded sale.
    reviewed_out = out["manual_decision"].eq("OUT_OF_SCOPE")
    out.loc[reviewed_out, "SkuWeekVerifiedActive"] = False
    out.loc[reviewed_out, "SalesLogCoverageVerified"] = False
    out.loc[reviewed_out, "ZeroLabelAuthorized"] = False

    if out.duplicated(KEY).any() or len(out) != len(panel):
        raise AssertionError("Candidate count/key uniqueness changed after review")
    if (out["HasObservedSale"] != panel["HasObservedSale"]).any():
        raise AssertionError("Observed sales flags changed")
    if (out["observed_gross_qty"] != panel["observed_gross_qty"]).any():
        raise AssertionError("Observed gross quantities changed")

    result = rev[KEY + ["ProductWindowEvidence", "decision", "reviewer", "review_date"]].copy()
    result["result"] = "UNCHANGED"
    result.loc[approves, "result"] = "ZERO_AUTHORIZED_BY_DOCUMENTED_REVIEW"
    result.loc[scopes, "result"] = "OUT_OF_SCOPE_MARKED_ONLY"
    stats = {
        "source_candidate_rows": len(panel),
        "review_template_rows": len(rev),
        "approved_zero_reviews": int(approves.sum()),
        "reviewed_out_of_scope": int(scopes.sum()),
        "explicitly_unverified_reviews": int(rev["decision"].eq("UNVERIFIED").sum()),
        "pending_reviews": int(rev["decision"].eq("PENDING").sum()),
        "not_in_review_template": len(panel) - len(rev),
        "authorized_zero_candidate_rows": int((~out["HasObservedSale"] & out["ZeroLabelAuthorized"]).sum()),
        "observed_gross_units_unchanged": int(out["observed_gross_qty"].sum()),
    }
    return out, result, stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, default=Path("reports/tables/sku_week_activity_candidates.csv"))
    p.add_argument("--review-file", type=Path, default=Path("reports/tables/zero_evidence_review_template.csv"))
    p.add_argument("--out-candidates", type=Path, default=Path("reports/tables/sku_week_activity_candidates_reviewed.csv"))
    p.add_argument("--out-report", type=Path, default=Path("reports/tables/evidence_application_summary.csv"))
    p.add_argument("--out-decisions", type=Path, default=Path("reports/tables/evidence_application_decisions.csv"))
    args = p.parse_args()
    for source in [args.candidates, args.review_file]:
        if not source.is_file():
            p.error(f"Missing input: {source}")
    for dest in [args.out_candidates, args.out_report, args.out_decisions]:
        if dest.resolve() in {args.candidates.resolve(), args.review_file.resolve()}:
            p.error(f"Refusing to overwrite source input: {dest}")
    updated, decisions, counts = apply(args.candidates, args.review_file)
    for path in [args.out_candidates, args.out_report, args.out_decisions]:
        path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.out_candidates, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"metric": name, "value": value} for name, value in counts.items()]).to_csv(
        args.out_report, index=False, encoding="utf-8-sig")
    decisions.to_csv(args.out_decisions, index=False, encoding="utf-8-sig")
    print("\nStep 9 – Evidence review application")
    for k, v in counts.items():
        print(f"{k:38s}: {v:,}")
    print(f"\nSaved candidate panel: {args.out_candidates}")
    if counts["approved_zero_reviews"] == 0:
        print("No verified zero labels were newly approved. Ground truth remains provisional.")
    else:
        print("Human-reviewed zero authorization applied; independently verify evidence before certification.")


if __name__ == "__main__":
    main()
