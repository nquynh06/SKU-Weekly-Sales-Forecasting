"""DS317

Policy:
- Gross quantity from cleaned completed-order lines, no return deductions.
- Observed positive weeks receive summed gross sales quantity.
- No-sale weeks become 0 ONLY when all three evidence flags from the
  lifecycle/coverage audit are explicitly True.
- Otherwise target remains NA, not zero.
- The output is provisional until SKU activity and weekly log coverage are
  independently verified; no model training is performed.

"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_CANDIDATES = {
    'SKU_ID', 'WeekStart', 'WeekEnd', 'observed_sales_lines',
    'observed_gross_qty', 'HasObservedSale', 'SkuWeekVerifiedActive',
    'SalesLogCoverageVerified', 'ZeroLabelAuthorized', 'ProductWindowEvidence',
    'DateConflictSalesLines'
}


def parse_bool(s: pd.Series, name: str) -> pd.Series:
    val = s.astype('string').str.strip().str.lower()
    allowed = {'true', 'false', '1', '0'}
    bad = val.isna() | (~val.isin(allowed))
    if bad.any():
        raise ValueError(f'{name}: found {bad.sum()} missing/invalid boolean values')
    return val.isin({'true', '1'})


def dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors='coerce').dt.normalize()


def build(candidates: Path, clean_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not candidates.is_file() or not clean_file.is_file():
        raise FileNotFoundError(f'Input missing: {candidates} or {clean_file}')
    df = pd.read_csv(candidates, dtype={'SKU_ID': 'string'}, low_memory=False)
    missing = REQUIRED_CANDIDATES - set(df.columns)
    if missing:
        raise ValueError(f'Candidate file missing columns: {sorted(missing)}')
    df['WeekStart'] = dates(df['WeekStart'])
    df['WeekEnd'] = dates(df['WeekEnd'])
    if df['SKU_ID'].isna().any() or df[['WeekStart', 'WeekEnd']].isna().any().any():
        raise ValueError('Missing SKU or week dates')
    if df.duplicated(['SKU_ID', 'WeekStart']).any():
        raise ValueError('Duplicate SKU_ID / WeekStart pairs')
    if ((df.WeekStart.dt.dayofweek != 0) | (df.WeekEnd != df.WeekStart + pd.Timedelta(days=6))).any():
        raise ValueError('Weekly grid must be Monday through Sunday')
    if df.SKU_ID.nunique() * df.WeekStart.nunique() != len(df):
        raise ValueError('Candidate panel is not a complete SKU-week cross product')

    for col in ['HasObservedSale', 'SkuWeekVerifiedActive', 'SalesLogCoverageVerified', 'ZeroLabelAuthorized']:
        df[col] = parse_bool(df[col], col)
    df['observed_sales_lines'] = pd.to_numeric(df['observed_sales_lines'], errors='raise')
    df['observed_gross_qty'] = pd.to_numeric(df['observed_gross_qty'], errors='raise')
    if (df['observed_sales_lines'] < 0).any() or (df['observed_gross_qty'] < 0).any():
        raise ValueError('Negative observed sales quantities/counts')
    positive = df.HasObservedSale
    if ((positive & ((df.observed_sales_lines <= 0) | (df.observed_gross_qty <= 0))) |
        (~positive & ((df.observed_sales_lines != 0) | (df.observed_gross_qty != 0)))).any():
        raise ValueError('HasObservedSale is inconsistent with aggregated sales')

    # Independent reconciliation of the SKU-week aggregation against clean source.
    source = pd.read_csv(clean_file, usecols=['SKU_ID','SalesOrderLineID','OrderDate','Quantity'],
                         dtype={'SKU_ID': 'string', 'SalesOrderLineID': 'string'}, low_memory=False)
    source['OrderDate'] = dates(source['OrderDate'])
    source['Quantity'] = pd.to_numeric(source['Quantity'], errors='coerce')
    if (source.OrderDate.isna() | source.Quantity.isna() | source.Quantity.le(0) | source.SKU_ID.isna()).any():
        raise ValueError('Clean transactions contain invalid dates/quantity/SKU')
    if source.SalesOrderLineID.isna().any() or source.SalesOrderLineID.duplicated().any():
        raise ValueError('Clean sales have invalid or duplicate line IDs')
    source['WeekStart'] = source.OrderDate - pd.to_timedelta(source.OrderDate.dt.dayofweek, unit='D')
    scope_start, scope_end = df.WeekStart.min(), df.WeekStart.max()
    scoped = source.loc[source.WeekStart.between(scope_start, scope_end)].copy()
    agg = scoped.groupby(['SKU_ID','WeekStart'], as_index=False, sort=False).agg(
        _lines=('Quantity','size'), _qty=('Quantity','sum'))
    check = df[['SKU_ID','WeekStart','observed_sales_lines','observed_gross_qty']].merge(
        agg, on=['SKU_ID','WeekStart'], how='outer', validate='one_to_one', indicator=True)
    check['_lines'] = check['_lines'].fillna(0)
    check['_qty'] = check['_qty'].fillna(0)
    check['observed_sales_lines'] = check['observed_sales_lines'].fillna(0)
    check['observed_gross_qty'] = check['observed_gross_qty'].fillna(0)
    mismatch = (check['_merge'] != 'both') & (check['_merge'] != 'left_only')
    mismatch |= (check.observed_sales_lines != check._lines)
    mismatch |= ~np.isclose(check.observed_gross_qty, check._qty, atol=1e-8, rtol=0)
    if mismatch.any():
        examples = check.loc[mismatch, ['SKU_ID', 'WeekStart']].head(5).to_dict('records')
        raise ValueError(f'Weekly candidate totals do not reconcile with cleaned source ({int(mismatch.sum())} rows), e.g. {examples}')

    # Defensive: a future audit must prove both component conditions and its authorization flag.
    verified_zero = (~positive & df.SkuWeekVerifiedActive &
                     df.SalesLogCoverageVerified & df.ZeroLabelAuthorized)
    inconsistent_authorization = (~positive & df.ZeroLabelAuthorized & ~verified_zero)
    if inconsistent_authorization.any():
        raise ValueError('ZeroLabelAuthorized=True without verified SKU activity and log coverage')

    out = df[['SKU_ID','WeekStart','WeekEnd','ProductID','ProductWindowEvidence',
              'observed_sales_lines','DateConflictSalesLines',
              'SkuWeekVerifiedActive','SalesLogCoverageVerified','ZeroLabelAuthorized']].copy()
    if 'ProductID' not in df.columns:
        raise ValueError('Candidates missing ProductID')
    out['WeeklyQuantity'] = pd.Series(pd.NA, index=out.index, dtype='Int64')
    if (df.observed_gross_qty % 1 != 0).any():
        raise ValueError('Expected integer unit quantities, found fractional gross values')
    out.loc[positive, 'WeeklyQuantity'] = df.loc[positive, 'observed_gross_qty'].astype('int64')
    out.loc[verified_zero, 'WeeklyQuantity'] = 0
    out['label_status'] = np.select([positive, verified_zero],
                                    ['observed_positive', 'verified_zero'],
                                    default='unverified_no_sale')
    out['reason_code'] = np.select(
        [positive, verified_zero, ~df.SkuWeekVerifiedActive, ~df.SalesLogCoverageVerified],
        ['observed_completed_sale','all_zero_conditions_verified',
         'sku_week_activity_not_verified','sales_log_coverage_not_verified'],
        default='no_sale_unverified')
    out['quality_flag_product_date_conflict'] = df.DateConflictSalesLines.gt(0)
    out['target_definition'] = 'gross_completed_sales_quantity'
    out['dataset_status'] = 'PROVISIONAL_NOT_FINAL'

    metrics = [
        ('dataset_status', 'PROVISIONAL_NOT_FINAL'),
        ('first_week', scope_start.date().isoformat()),
        ('last_week', scope_end.date().isoformat()),
        ('distinct_skus', int(df.SKU_ID.nunique())),
        ('distinct_weeks', int(df.WeekStart.nunique())),
        ('sku_week_rows', len(df)),
        ('observed_positive_sku_weeks', int(positive.sum())),
        ('verified_zero_sku_weeks', int(verified_zero.sum())),
        ('unverified_no_sale_sku_weeks', int((~positive & ~verified_zero).sum())),
        ('gross_quantity_in_panel', int(out.WeeklyQuantity.sum(skipna=True))),
        ('scoped_clean_source_sales_lines', len(scoped)),
        ('scoped_clean_source_gross_qty', int(scoped.Quantity.sum())),
        ('clean_source_rows_outside_panel_weeks', len(source) - len(scoped)),
        ('clean_source_qty_outside_panel_weeks', int(source.Quantity.sum()-scoped.Quantity.sum())),
        ('skus_with_product_date_conflicts', int(df.loc[df.DateConflictSalesLines.gt(0),'SKU_ID'].nunique())),
    ]
    return out, pd.DataFrame(metrics, columns=['metric','value'])


def main():
    parser = argparse.ArgumentParser(description='Build provisional SKU-week observed gross-sales labels (no guessed zeros)')
    parser.add_argument('--candidates', type=Path, default=Path('reports/tables/sku_week_activity_candidates.csv'))
    parser.add_argument('--clean-file', type=Path, default=Path('src/data/processed/clean_sales_transactions.csv'))
    parser.add_argument('--out-file', type=Path, default=Path('src/data/processed/sku_week_ground_truth_provisional.csv'))
    parser.add_argument('--report-file', type=Path, default=Path('reports/tables/ground_truth_build_report.csv'))
    args = parser.parse_args()
    ground_truth, summary = build(args.candidates, args.clean_file)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(args.out_file, index=False, encoding='utf-8-sig', date_format='%Y-%m-%d', na_rep='')
    summary.to_csv(args.report_file, index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
