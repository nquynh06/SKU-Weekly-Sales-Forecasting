"""DS317 

Produces: sku_lifecycle_audit.csv (one row per SKU),
          sku_week_activity_candidates.csv (one row per SKU/full week),
          sku_lifecycle_summary.csv (audit metrics).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def find_csv(folder: Path, stem: str, optional: bool = False) -> Path | None:
    if not folder.exists():
        raise FileNotFoundError(f"Raw directory not found: {folder}")
    matches = [p for p in folder.rglob('*.csv') if p.stem.lower() == stem.lower()]
    if len(matches) == 0 and optional:
        return None
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {stem}.csv under {folder}; found {len(matches)}")
    return matches[0]


def load_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype='string', encoding='utf-8-sig', usecols=columns, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype='string', encoding='cp1252', usecols=columns, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def dates(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors='coerce', format='mixed').dt.normalize()
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors='coerce').dt.normalize()


def normalize(series: pd.Series) -> pd.Series:
    out = series.astype('string').str.strip()
    return out.mask(out.eq(''))


def deduplicate_keys(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, int, set]:
    """Remove exact duplicate records; quarantine conflicting primary keys."""
    df = df.drop_duplicates().copy()
    df[key] = normalize(df[key])
    conflicts = set(df.loc[df[key].notna() & df[key].duplicated(keep=False), key].tolist())
    df = df.loc[df[key].notna() & ~df[key].isin(conflicts)].copy()
    return df, len(conflicts), conflicts


def prepare_source_tables(raw_dir: Path) -> tuple[pd.DataFrame, dict]:
    products = load_csv(find_csv(raw_dir, 'Products'))
    variants = load_csv(find_csv(raw_dir, 'ProductVariants'))
    required_products = {'ProductID', 'ActiveFrom', 'ActiveTo'}
    required_variants = {'SKU_ID', 'ProductID'}
    if not required_products.issubset(products):
        raise ValueError(f"Products missing: {sorted(required_products - set(products))}")
    if not required_variants.issubset(variants):
        raise ValueError(f"ProductVariants missing: {sorted(required_variants - set(variants))}")
    products, product_conflicts, _ = deduplicate_keys(products, 'ProductID')
    variants, sku_conflicts, _ = deduplicate_keys(variants, 'SKU_ID')
    variants['ProductID'] = normalize(variants['ProductID'])
    products['ActiveFromParsed'] = dates(products['ActiveFrom'])
    products['ActiveToParsed'] = dates(products['ActiveTo'])
    bad_range = products.ActiveFromParsed.notna() & products.ActiveToParsed.notna() & (products.ActiveFromParsed > products.ActiveToParsed)
    products['BadProductDates'] = bad_range
    merged = variants[['SKU_ID','ProductID']].merge(
        products[['ProductID','ActiveFrom','ActiveTo','ActiveFromParsed','ActiveToParsed','BadProductDates']],
        on='ProductID', how='left', validate='many_to_one', indicator=True
    )
    merged['MissingProduct'] = merged['_merge'].ne('both')
    merged = merged.drop(columns='_merge')
    info = {'product_key_conflicts': product_conflicts, 'sku_key_conflicts': sku_conflicts,
            'distinct_products_after_audit': len(products), 'distinct_skus_after_audit': len(variants),
            'missing_product_mapping_skus': int(merged.MissingProduct.sum()),
            'bad_product_date_skus': int(merged.BadProductDates.fillna(False).sum())}
    return merged, info


def prepare_sales(clean_file: Path) -> pd.DataFrame:
    needed = ['SKU_ID', 'OrderDate', 'Quantity', 'SalesOrderLineID']
    clean = load_csv(clean_file, columns=needed)
    if not {'SKU_ID','OrderDate','Quantity','SalesOrderLineID'}.issubset(clean):
        raise ValueError('Clean transactions file is missing required fields')
    clean['SKU_ID'] = normalize(clean['SKU_ID'])
    clean['OrderDateParsed'] = dates(clean['OrderDate'])
    clean['QuantityParsed'] = pd.to_numeric(clean.Quantity, errors='coerce')
    if (clean['OrderDateParsed'].isna() | clean['SKU_ID'].isna() | clean['QuantityParsed'].isna() | clean['QuantityParsed'].le(0)).any():
        raise ValueError('Clean sales file contains invalid dates, SKU IDs or Quantity; rerun Step 2')
    if clean['SalesOrderLineID'].duplicated().any():
        raise ValueError('Duplicate SalesOrderLineID in clean data')
    return clean


def make_audit(raw_dir: Path, clean_file: Path, first: pd.Timestamp, last: pd.Timestamp):
    variants, info = prepare_source_tables(raw_dir)
    sales = prepare_sales(clean_file)
    sales['WeekStart'] = sales.OrderDateParsed - pd.to_timedelta(sales.OrderDateParsed.dt.dayofweek, unit='D')
    sale_agg = (sales.groupby('SKU_ID', as_index=False)
                .agg(ObservedSalesLines=('SKU_ID','size'), FirstObservedSale=('OrderDateParsed','min'),
                     LastObservedSale=('OrderDateParsed','max'), ObservedGrossQty=('QuantityParsed','sum')))
    scoped = sales.loc[sales.WeekStart.between(first, last)].copy()
    sale_week = scoped.groupby(['SKU_ID', 'WeekStart'], as_index=False).agg(
        observed_sales_lines=('SKU_ID','size'), observed_gross_qty=('QuantityParsed','sum'))
    audit = variants.merge(sale_agg, on='SKU_ID', how='left', validate='one_to_one')
    audit['ObservedSalesLines'] = audit.ObservedSalesLines.fillna(0).astype(int)
    audit['ObservedGrossQty'] = audit.ObservedGrossQty.fillna(0)
    # Cross-check sales against product-level dates; flags are NOT automatic exclusions.
    joined = sales[['SKU_ID','OrderDateParsed']].merge(
        audit[['SKU_ID','ActiveFromParsed','ActiveToParsed']], on='SKU_ID', how='left', validate='many_to_one')
    joined['before_active_from'] = joined.ActiveFromParsed.notna() & (joined.OrderDateParsed < joined.ActiveFromParsed)
    joined['after_active_to'] = joined.ActiveToParsed.notna() & (joined.OrderDateParsed > joined.ActiveToParsed)
    conflicts = joined.groupby('SKU_ID', as_index=False)[['before_active_from','after_active_to']].sum()
    audit = audit.merge(conflicts, on='SKU_ID', how='left')
    for col in ['before_active_from','after_active_to']:
        audit[col] = audit[col].fillna(0).astype(int)
    audit['DateConflictSalesLines'] = audit.before_active_from + audit.after_active_to
    # InventoryMovements is only evidence of an event, NOT weekly stock availability.
    movements_path = find_csv(raw_dir, 'InventoryMovements', optional=True)
    if movements_path is not None:
        mv = load_csv(movements_path, ['SKU_ID','MovementDateTime'])
        mv['SKU_ID'] = normalize(mv['SKU_ID'])
        mv['MovementDateParsed'] = dates(mv['MovementDateTime'])
        mv = mv.loc[mv.SKU_ID.notna() & mv.MovementDateParsed.notna()].copy()
        mv_agg = mv.groupby('SKU_ID', as_index=False).agg(
            MovementEventCount=('SKU_ID','size'), FirstMovement=('MovementDateParsed','min'), LastMovement=('MovementDateParsed','max'))
        audit = audit.merge(mv_agg, on='SKU_ID', how='left', validate='one_to_one')
    else:
        audit['MovementEventCount'] = pd.NA
        audit['FirstMovement'] = pd.NaT
        audit['LastMovement'] = pd.NaT
    balance_path = find_csv(raw_dir, 'InventoryBalances', optional=True)
    if balance_path is not None:
        balance = load_csv(balance_path, ['SKU_ID','AvailableQuantity'])
        balance['SKU_ID'] = normalize(balance['SKU_ID'])
        balance['AvailableQuantity'] = pd.to_numeric(balance.AvailableQuantity, errors='coerce')
        balance_agg = balance.groupby('SKU_ID', as_index=False).agg(
            SnapshotLocations=('SKU_ID','size'), SnapshotAvailableQty=('AvailableQuantity','sum'))
        audit = audit.merge(balance_agg, on='SKU_ID', how='left', validate='one_to_one')
    else:
        audit['SnapshotLocations'] = pd.NA
        audit['SnapshotAvailableQty'] = pd.NA

    audit['ReviewReason'] = ''
    def reason(mask, code):
        audit.loc[mask.fillna(False), 'ReviewReason'] = audit.loc[mask.fillna(False),'ReviewReason'].map(lambda x: x + (';' if x else '') + code)
    reason(audit.MissingProduct, 'missing_product_mapping')
    reason(audit.BadProductDates.fillna(False), 'invalid_product_dates')
    reason(audit.ActiveFromParsed.isna(), 'missing_or_invalid_active_from')
    reason(audit.ActiveToParsed.isna(), 'open_end_or_missing_active_to')
    reason(audit.DateConflictSalesLines.gt(0), 'sales_outside_product_lifecycle')
    reason(audit.ObservedSalesLines.eq(0), 'no_observed_sales')
    audit['ReviewReason'] = audit.ReviewReason.replace('', 'none')

    weeks = pd.date_range(first, last, freq='W-MON')
    calendar = pd.DataFrame({'WeekStart': weeks})
    grid = audit[['SKU_ID','ProductID','ActiveFromParsed','ActiveToParsed','MissingProduct','BadProductDates','DateConflictSalesLines']].merge(calendar, how='cross')
    grid['WeekEnd'] = grid.WeekStart + pd.Timedelta(days=6)
    before = grid.ActiveFromParsed.notna() & (grid.WeekEnd < grid.ActiveFromParsed)
    after = grid.ActiveToParsed.notna() & (grid.WeekStart > grid.ActiveToParsed)
    full_start = grid.ActiveFromParsed.notna() & (grid.WeekStart >= grid.ActiveFromParsed)
    full_end = grid.ActiveToParsed.notna() & (grid.WeekEnd <= grid.ActiveToParsed)
    supported_closed = full_start & full_end
    supported_open = full_start & grid.ActiveToParsed.isna()
    partial = (~before & ~after) & (grid.ActiveFromParsed.notna()) & (grid.ActiveToParsed.notna()) & (~supported_closed)
    grid['ProductWindowEvidence'] = np.select(
        [grid.MissingProduct.fillna(False).to_numpy(), grid.BadProductDates.fillna(False).to_numpy(),
         before.to_numpy(), after.to_numpy(), supported_closed.to_numpy(), supported_open.to_numpy(), partial.to_numpy()],
        ['missing_product', 'invalid_product_dates', 'before_active_from', 'after_active_to',
         'within_recorded_product_window', 'open_end_unverified', 'partial_week_overlap'],
        default='unverified_product_window')
    grid = grid.merge(sale_week, on=['SKU_ID','WeekStart'], how='left', validate='one_to_one')
    grid['observed_sales_lines'] = grid.observed_sales_lines.fillna(0).astype(int)
    grid['observed_gross_qty'] = grid.observed_gross_qty.fillna(0)
    grid['HasObservedSale'] = grid.observed_sales_lines.gt(0)
    # No proof of complete source logs or SKU-specific selling status; do not issue zero labels.
    grid['SkuWeekVerifiedActive'] = False
    grid['SalesLogCoverageVerified'] = False
    grid['ZeroLabelAuthorized'] = False
    grid['ReviewPriority'] = np.where(grid.HasObservedSale, 'observed_positive_sales',
                                      np.where(grid.ProductWindowEvidence.eq('within_recorded_product_window'),
                                               'verify_no_sale_week', 'review_metadata_or_coverage'))
    grid = grid.drop(columns=['ActiveFromParsed','ActiveToParsed','MissingProduct','BadProductDates'])
    summ = dict(info)
    summ.update({
        'study_start_week': first.date().isoformat(), 'study_end_week': last.date().isoformat(),
        'weeks_in_panel': len(weeks), 'candidate_sku_weeks': len(grid),
        'distinct_skus_with_sales': sales.SKU_ID.nunique(),
        'observed_sales_lines_all_dates': len(sales),
        'sales_lines_outside_study_weeks': int((~sales.WeekStart.between(first,last)).sum()),
        'skus_with_sales_product_date_conflicts': int(audit.DateConflictSalesLines.gt(0).sum()),
        'sku_weeks_with_observed_sales': int(grid.HasObservedSale.sum()),
        'sku_weeks_without_observed_sales': int((~grid.HasObservedSale).sum()),
        'zero_labels_generated': 0,
    })
    return audit, grid, pd.DataFrame([{'metric': k, 'value': v} for k, v in summ.items()])


def main():
    parser = argparse.ArgumentParser(description='Step 3: SKU lifecycle audit (no zero labels generated)')
    parser.add_argument('--raw-dir', type=Path, default=Path('src/data/raw'))
    parser.add_argument('--clean-file', type=Path, default=Path('data/processed/clean_sales_transactions.csv'))
    parser.add_argument('--report-dir', type=Path, default=Path('reports/tables'))
    parser.add_argument('--start-week', default='2023-07-03')
    parser.add_argument('--end-week', default='2025-12-22')
    args = parser.parse_args()
    first, last = pd.Timestamp(args.start_week), pd.Timestamp(args.end_week)
    if first > last or first.dayofweek != 0 or last.dayofweek != 0:
        parser.error('--start-week and --end-week must be Mondays in ascending order')
    if not args.clean_file.is_file():
        parser.error(f'Missing clean file: {args.clean_file}')
    audit, weeks, summary = make_audit(args.raw_dir, args.clean_file, first, last)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.report_dir / 'sku_lifecycle_audit.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    weeks.to_csv(args.report_dir / 'sku_week_activity_candidates.csv', index=False, encoding='utf-8-sig', date_format='%Y-%m-%d')
    summary.to_csv(args.report_dir / 'sku_lifecycle_summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
