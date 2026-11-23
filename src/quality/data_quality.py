"""DS317 

Outputs:
  data_quality_6_dimensions.csv
  ground_truth_quality_summary.csv
  no_sale_evidence_breakdown.csv
  data_quality_findings.md

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path, encoding='utf-8-sig', low_memory=False, **kwargs)


def lookup(report: pd.DataFrame, metric: str, default=None):
    if not {'metric', 'value'}.issubset(report.columns):
        raise ValueError('Summary files must contain metric,value columns')
    x = report.loc[report.metric.astype(str) == metric, 'value']
    if len(x) > 1:
        raise ValueError(f'Duplicate summary metric: {metric}')
    return x.iloc[0] if len(x) else default


def num(val, label: str) -> float:
    try:
        x = float(val)
    except (TypeError, ValueError):
        raise ValueError(f'Expected numeric {label}, got {val!r}')
    if not np.isfinite(x):
        raise ValueError(f'Nonfinite numeric {label}: {val!r}')
    return x


def raw_check(report: pd.DataFrame, table: str, check: str):
    matches = report.loc[(report.table == table) & (report.check == check)]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        raise ValueError(f'Duplicate raw-quality check for {table}, {check}')
    row = matches.iloc[0]
    return int(num(row.affected_rows, check)), int(num(row.total_rows, 'total_rows'))


def compute(raw: pd.DataFrame, keys: pd.DataFrame, clean: pd.DataFrame,
            lifecycle: pd.DataFrame, candidates: pd.DataFrame, gt: pd.DataFrame):
    required_candidate = {'SKU_ID','WeekStart','ProductWindowEvidence','HasObservedSale',
                          'observed_gross_qty','SkuWeekVerifiedActive',
                          'SalesLogCoverageVerified','ZeroLabelAuthorized'}
    required_gt = {'SKU_ID','WeekStart','WeeklyQuantity','label_status','ZeroLabelAuthorized',
                   'SkuWeekVerifiedActive','SalesLogCoverageVerified'}
    for label, actual, required in [('candidates', candidates, required_candidate),
                                    ('ground truth', gt, required_gt)]:
        missing = required - set(actual.columns)
        if missing:
            raise ValueError(f'{label} missing {sorted(missing)}')
    if len(gt) == 0 or len(candidates) == 0:
        raise ValueError('Input candidate or ground-truth table is empty')
    for label, df in [('candidates', candidates), ('ground truth', gt)]:
        if df[['SKU_ID','WeekStart']].isna().any().any():
            raise ValueError(f'{label} has missing SKU/week IDs')
        if df.duplicated(['SKU_ID','WeekStart']).any():
            raise ValueError(f'{label} has duplicate SKU-week keys')
    if len(gt) != len(candidates):
        raise ValueError('Candidate panel and ground truth have unequal row count')
    if not (gt[['SKU_ID','WeekStart']].reset_index(drop=True).equals(
            candidates[['SKU_ID','WeekStart']].reset_index(drop=True))):
        # Row ordering differences are legal; key sets must still match.
        a = pd.MultiIndex.from_frame(gt[['SKU_ID','WeekStart']])
        b = pd.MultiIndex.from_frame(candidates[['SKU_ID','WeekStart']])
        if not a.symmetric_difference(b).empty:
            raise ValueError('Candidate / ground truth SKU-week keys differ')

    valid_status = {'observed_positive','verified_zero','unverified_no_sale'}
    if not set(gt.label_status.dropna().unique()).issubset(valid_status):
        raise ValueError('Unexpected label_status')
    qty = pd.to_numeric(gt['WeeklyQuantity'], errors='coerce')
    candidate_qty = pd.to_numeric(candidates['observed_gross_qty'], errors='coerce')
    if candidate_qty.isna().any():
        raise ValueError('Invalid observed_gross_qty in candidate table')
    positive = gt.label_status.eq('observed_positive')
    zero = gt.label_status.eq('verified_zero')
    unverified = gt.label_status.eq('unverified_no_sale')
    if not (positive | zero | unverified).all():
        raise ValueError('Missing or invalid label_status')
    if qty.loc[positive].isna().any() or qty.loc[positive].le(0).any():
        raise ValueError('Observed positive labels are not positive quantities')
    if qty.loc[zero].isna().any() or qty.loc[zero].ne(0).any():
        raise ValueError('Verified zero labels must be numeric 0')
    if qty.loc[unverified].notna().any():
        raise ValueError('Unverified no-sale weeks must remain NA')
    if not np.isclose(qty.loc[positive].sum(), candidate_qty.loc[candidates.HasObservedSale.astype(str).str.lower().eq('true')].sum()):
        raise ValueError('Positive label totals differ from positive candidate gross quantity')
    if zero.any():
        for col in ['ZeroLabelAuthorized','SkuWeekVerifiedActive','SalesLogCoverageVerified']:
            if not gt.loc[zero, col].astype(str).str.lower().isin(['true','1']).all():
                raise ValueError(f'Zero labels without {col} verification')

    gross = int(qty.sum(skipna=True))
    eligible = int(positive.sum() + zero.sum())
    n = len(gt)
    clean_qty = num(lookup(clean, 'gross_quantity_accepted'), 'gross_quantity_accepted')
    in_scope = num(lookup(lifecycle, 'candidate_sku_weeks'), 'candidate_sku_weeks')
    if n != in_scope:
        raise ValueError(f'Ground truth {n} rows versus lifecycle {in_scope} candidate rows')
    scoped_summ = clean_qty - 0  # Full-date clean gross; separate from study-window gross.
    out_rows = lookup(lifecycle, 'sales_lines_outside_study_weeks')
    if out_rows is not None:
        out_rows = int(num(out_rows, 'sales_lines_outside_study_weeks'))
    src_panel_gross = int(candidate_qty.sum())
    if gross != src_panel_gross:
        raise ValueError('Gross label total does not reconcile to candidate panel')

    metrics=[]
    def add(dimension, test, affected, total, stage, status, note):
        metrics.append(dict(dimension=dimension, check=test,
                            affected_count=int(affected) if affected is not None else None,
                            inspected_count=int(total) if total is not None else None,
                            rate_pct=round(100*affected/total, 4) if total else None,
                            stage=stage, status=status, notes=note))
    def source(dimension, table, check, status_if_nonzero='ISSUE', note=''):
        value = raw_check(raw, table, check)
        if value is None:
            add(dimension, f'{table}.{check}', None, None, 'raw', 'NOT_REPORTED', note)
        else:
            affected,total=value
            add(dimension, f'{table}.{check}', affected,total,'raw',status_if_nonzero if affected else 'PASS',note)

    # Completeness: optional columns do not all have equal importance for gross sales.
    source('Completeness','SalesOrderLines','missing:SKU_ID', note='Target-critical key')
    source('Completeness','SalesOrderLines','missing:Quantity', note='Target-critical quantity')
    source('Completeness','SalesOrders','missing:OrderDate', note='Target-critical date')
    source('Completeness','Products','missing:ActiveTo', note='Missing can represent open-ended status; cannot infer SKU activation')
    add('Completeness','unverified_no_sale_sku_weeks',int(unverified.sum()),n,'ground_truth',
        'BLOCKED' if unverified.any() else 'PASS','NA is intentional, not a zero sale; model training must not fill these with 0')
    # Consistency: FK and product-date checks.
    if {'status','orphan_rows','nonnull_child_rows','child','parent'}.issubset(keys):
        for _,r in keys.iterrows():
            if str(r['status']).upper() == 'CHECKED':
                x=int(num(r['orphan_rows'],'orphan_rows')); total=int(num(r['nonnull_child_rows'],'nonnull_child_rows'))
                add('Consistency',f'FK:{r["child"]}->{r["parent"]}',x,total,'raw','ISSUE' if x else 'PASS','Source relationship consistency')
    conflicted=int(num(lookup(lifecycle,'skus_with_sales_product_date_conflicts'),'date_conflict_skus'))
    add('Consistency','skus_with_sales_product_date_conflicts',conflicted,gt.SKU_ID.nunique(),
        'lifecycle','ISSUE' if conflicted else 'PASS','Investigate transactions outside product-level recorded window')

    # Accuracy: internal reconciliation only; truth of real demand cannot be verified.
    add('Accuracy','gross_label_to_candidate_quantity_difference',abs(gross-src_panel_gross),
        max(src_panel_gross,1),'ground_truth','PASS' if gross == src_panel_gross else 'ISSUE',
        f'Both totals={gross}; this checks internal arithmetic, not real-world demand accuracy')
    add('Accuracy','external_ground_truth_verification',None,None,'source','NOT_VERIFIED',
        'No external source verifying true demand or all missing sales was provided')

    # Timeliness: temporal scope is fixed, but transaction-log completeness is not proven.
    add('Timeliness','sales_lines_outside_study_weeks',out_rows,
        int(num(lookup(clean,'sales_lines_accepted'),'sales_lines_accepted')),
        'study_scope','REVIEW' if out_rows else 'PASS','Outside 2023-07-03 to 2025-12-22 week-start scope, not automatically errors')
    verified_logs=gt.SalesLogCoverageVerified.astype(str).str.lower().isin(['true','1']).sum()
    add('Timeliness','weeks_without_verified_sales_log_coverage',n-int(verified_logs),n,
        'ground_truth','NOT_VERIFIED' if verified_logs<n else 'PASS',
        'A sales event on a day cannot prove a complete week of order ingestion')

    # Validity.
    source('Validity','SalesOrderLines','nonpositive_quantity')
    source('Validity','SalesOrderLines','non_numeric:Quantity')
    source('Validity','SalesOrders','invalid_date:OrderDate')
    source('Validity','Products','invalid_date:ActiveFrom')
    add('Validity','observed_labels_nonpositive',int((qty.loc[positive]<=0).sum()),int(positive.sum()),
        'ground_truth','PASS','Positive sales quantities must exceed 0')

    # Uniqueness.
    source('Uniqueness','SalesOrders','exact_duplicate_rows',note='Deduplicated before joining to sales lines')
    dedup=int(num(lookup(clean,'orders_exact_duplicates_removed'),'orders_exact_duplicates_removed'))
    add('Uniqueness','sales_orders_exact_duplicates_removed',dedup,
        int(num(lookup(clean,'orders_raw_rows'),'orders_raw_rows')),
        'cleaning','RESOLVED','Duplicates removed as part of clean transactions')
    add('Uniqueness','duplicated_sku_week_keys',int(gt.duplicated(['SKU_ID','WeekStart']).sum()),n,
        'ground_truth','PASS','One row per SKU and calendar week')

    quality = pd.DataFrame(metrics)
    candidate_no_sale = candidates.loc[~candidates.HasObservedSale.astype(str).str.lower().eq('true')]
    breakdown = (candidate_no_sale.groupby('ProductWindowEvidence',dropna=False)
                 .agg(sku_weeks=('SKU_ID','size'), distinct_skus=('SKU_ID','nunique'))
                 .reset_index().sort_values('sku_weeks',ascending=False))
    breakdown['pct_of_no_sale']= (100*breakdown.sku_weeks/len(candidate_no_sale)).round(2)
    breakdown['zero_label_authorized']='NO - evidence flags unverified'

    summary = pd.DataFrame([
        ('dataset_status','PROVISIONAL_NOT_FINAL'),
        ('sku_weeks',n),('distinct_skus',gt.SKU_ID.nunique()),
        ('observed_positive_sku_weeks',int(positive.sum())),
        ('verified_zero_sku_weeks',int(zero.sum())),
        ('unverified_no_sale_sku_weeks',int(unverified.sum())),
        ('verified_label_coverage_pct',round(100*eligible/n,2)),
        ('gross_quantity_in_panel',gross),
        ('gross_quantity_candidate_reconciliation_delta',gross-src_panel_gross),
        ('clean_gross_all_dates',int(clean_qty)),
        ('clean_gross_outside_study_scope',int(clean_qty-gross)),
        ('clean_sales_lines_outside_study_scope',out_rows),
        ('skus_with_product_date_conflicts',conflicted),
        ('weeks_with_verified_sales_log_coverage',int(verified_logs)),
        ('zero_labels_ready_for_training',False),
    ],columns=['metric','value'])
    return quality, summary, breakdown


def main():
    p=argparse.ArgumentParser(description='Six-dimension quality review (does not manufacture zero labels)')
    p.add_argument('--raw-quality',type=Path,default=Path('reports/tables/raw_quality_report.csv'))
    p.add_argument('--key-report',type=Path,default=Path('reports/tables/key_relationship_report.csv'))
    p.add_argument('--clean-report',type=Path,default=Path('reports/tables/transaction_cleaning_report.csv'))
    p.add_argument('--lifecycle-summary',type=Path,default=Path('reports/tables/sku_lifecycle_summary.csv'))
    p.add_argument('--candidates',type=Path,default=Path('reports/tables/sku_week_activity_candidates.csv'))
    p.add_argument('--ground-truth',type=Path,default=None,
                   help='Ground truth CSV; auto-detected from data/processed or src/data/processed if omitted')
    p.add_argument('--report-dir',type=Path,default=Path('reports/tables'))
    a=p.parse_args()
    if a.ground_truth is None:
        options = [Path('data/processed/sku_week_ground_truth_provisional.csv'),
                   Path('src/data/processed/sku_week_ground_truth_provisional.csv')]
        found = [candidate for candidate in options if candidate.is_file()]
        if not found:
            p.error('Missing provisional ground truth; checked: ' + ', '.join(map(str, options)))
        a.ground_truth = found[0]
    print('Using ground truth:', a.ground_truth)
    quality,summary,breakdown = compute(
        load_csv(a.raw_quality),load_csv(a.key_report),load_csv(a.clean_report),
        load_csv(a.lifecycle_summary),load_csv(a.candidates,dtype={'SKU_ID':'string'}),
        load_csv(a.ground_truth,dtype={'SKU_ID':'string'}))
    a.report_dir.mkdir(parents=True,exist_ok=True)
    for name,table in [
        ('data_quality_6_dimensions.csv',quality),
        ('ground_truth_quality_summary.csv',summary),
        ('no_sale_evidence_breakdown.csv',breakdown),
    ]:
        table.to_csv(a.report_dir/name,index=False,encoding='utf-8-sig')
    vals=dict(zip(summary.metric,summary.value))
    md=['# DS317 - Data Quality Review (provisional)',
        '', '> This is a preliminary audit, not certification of final ground truth.',
        '',f'- Panel: **{vals["sku_weeks"]:,}** SKU-week rows across **{vals["distinct_skus"]:,}** SKUs.',
        f'- Observed positives: **{vals["observed_positive_sku_weeks"]:,}**; verified zeros: **{vals["verified_zero_sku_weeks"]:,}**; unverified no-sales: **{vals["unverified_no_sale_sku_weeks"]:,}**.',
        f'- Labeled coverage: **{vals["verified_label_coverage_pct"]}%**; total gross units (study scope): **{vals["gross_quantity_in_panel"]:,}**.',
        f'- Candidate quantity reconciliation difference: **{vals["gross_quantity_candidate_reconciliation_delta"]}**.',
        f'- Product-date conflict SKUs: **{vals["skus_with_product_date_conflicts"]}**.',
        '', '## No-sale weeks by product window evidence', '',
        '| Evidence | SKU-weeks | Share |','|---|---:|---:|']
    for _,r in breakdown.iterrows():
        md.append(f'| {r.ProductWindowEvidence} | {int(r.sku_weeks):,} | {r.pct_of_no_sale:.2f}% |')
    md += ['', '## Important limitations',
          '- ActiveFrom/ActiveTo are product-level metadata, not proof of SKU-level sellability.',
          '- A week with other recorded sales does not prove that order logs are complete.',
          '- InventoryBalances are snapshots; inventory movements do not establish continuous stock availability without a verified reconstruction.',
          '- Gross sales are observed transactions, not uncensored customer demand.',
          '- **Do not fill unverified NA values with zero**; do not present this dataset as final training/evaluation truth.',
          '', '## Follow-up',
          '1. Confirm SKU active status with documented product/variant availability rules.',
          '2. Prove or document complete order-log coverage for each week from source/export metadata.',
          '3. Review SKU/product-date conflicts and temporal outliers.',
          '4. Only then authorize verified zero labels and rerun the Step 4 ground truth script.',
          '']
    (a.report_dir/'data_quality_findings.md').write_text('\n'.join(md),encoding='utf-8')
    print(summary.to_string(index=False))
    print('\nNo-sale evidence:')
    print(breakdown.to_string(index=False))
    print('\nOutputs written to',a.report_dir)

if __name__=='__main__':
    main()
