"""
Outputs (reports/tables):
    ground_truth_validation_checks.csv
    ground_truth_validation_summary.csv
    ground_truth_validation_findings.md


"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


VALID_STATUSES = {'observed_positive', 'verified_zero', 'unverified_no_sale'}


def locate(user_path: Path | None, candidates: list[str], description: str) -> Path:
    if user_path is not None:
        if not user_path.is_file():
            raise FileNotFoundError(f'{description} not found: {user_path}')
        return user_path
    for name in candidates:
        p = Path(name)
        if p.is_file():
            return p
    raise FileNotFoundError(f'{description} not found. Checked: {candidates}')


def parse_true(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = series.astype('string').str.strip().str.lower()
    valid = text.isin(['true', 'false', '1', '0'])
    return text.isin(['true', '1']), ~valid


def examples(df: pd.DataFrame, mask: pd.Series | np.ndarray, limit: int = 3) -> str:
    part = df.loc[mask, [c for c in ('SKU_ID', 'WeekStart') if c in df.columns]].head(limit)
    if 'WeekStart' in part.columns:
        part = part.copy()
        part['WeekStart'] = part['WeekStart'].astype('string')
    return '; '.join(' / '.join(map(str, row)) for row in part.itertuples(index=False, name=None))


class Auditor:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, key: str, group: str, issues: int, total: int, description: str,
               severity: str = 'FAIL', sample: str = '') -> None:
        count = int(issues)
        self.rows.append({
            'check_id': key, 'group': group,
            'status': 'PASS' if count == 0 else severity,
            'affected_rows': count, 'inspected_rows': int(total),
            'description': description, 'examples': sample
        })

    def note(self, key: str, group: str, status: str, description: str) -> None:
        self.rows.append({
            'check_id': key, 'group': group, 'status': status,
            'affected_rows': '', 'inspected_rows': '',
            'description': description, 'examples': ''
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def required(df: pd.DataFrame, cols: list[str], label: str) -> None:
    absent = set(cols) - set(df.columns)
    if absent:
        raise ValueError(f'{label} missing required columns: {sorted(absent)}')


def run(gt_path: Path, cand_path: Path, clean_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.read_csv(gt_path, dtype={'SKU_ID': 'string'}, low_memory=False)
    cand = pd.read_csv(cand_path, dtype={'SKU_ID': 'string'}, low_memory=False)
    src = pd.read_csv(
        clean_path, usecols=['SKU_ID', 'SalesOrderLineID', 'OrderDate', 'Quantity',
                             'OrderStatusCanonical'] if 'OrderStatusCanonical' in pd.read_csv(clean_path, nrows=0).columns
                             else ['SKU_ID', 'SalesOrderLineID', 'OrderDate', 'Quantity'],
        dtype={'SKU_ID': 'string', 'SalesOrderLineID': 'string'}, low_memory=False)

    required(gt, ['SKU_ID', 'WeekStart', 'WeekEnd', 'WeeklyQuantity', 'label_status',
                  'SkuWeekVerifiedActive', 'SalesLogCoverageVerified',
                  'ZeroLabelAuthorized', 'observed_sales_lines', 'dataset_status',
                  'target_definition'], 'Ground truth')
    required(cand, ['SKU_ID', 'WeekStart', 'observed_gross_qty', 'observed_sales_lines',
                    'HasObservedSale'], 'Activity candidates')
    audit = Auditor()

    for frame in (gt, cand):
        frame['WeekStart'] = pd.to_datetime(frame.WeekStart, errors='coerce').dt.normalize()
    gt['WeekEnd'] = pd.to_datetime(gt.WeekEnd, errors='coerce').dt.normalize()
    src['OrderDate'] = pd.to_datetime(src.OrderDate, errors='coerce').dt.normalize()
    gt['WeeklyQuantity'] = pd.to_numeric(gt.WeeklyQuantity, errors='coerce')
    cand['observed_gross_qty'] = pd.to_numeric(cand.observed_gross_qty, errors='coerce')
    cand['observed_sales_lines'] = pd.to_numeric(cand.observed_sales_lines, errors='coerce')
    gt['observed_sales_lines'] = pd.to_numeric(gt.observed_sales_lines, errors='coerce')
    src['Quantity'] = pd.to_numeric(src.Quantity, errors='coerce')

    gt_bad_keys = gt.SKU_ID.isna() | gt.WeekStart.isna()
    audit.record('T01', 'structure', gt_bad_keys.sum(), len(gt), 'Missing SKU ID or invalid WeekStart.', sample=examples(gt, gt_bad_keys))
    dup_gt = gt.duplicated(['SKU_ID', 'WeekStart'], keep=False)
    audit.record('T02', 'structure', dup_gt.sum(), len(gt), 'SKU-week key must be unique.', sample=examples(gt, dup_gt))
    bad_week = gt.WeekStart.dt.dayofweek.ne(0) | gt.WeekEnd.ne(gt.WeekStart + pd.Timedelta(days=6))
    audit.record('T03', 'time', bad_week.sum(), len(gt), 'Each week must run Monday through Sunday.', sample=examples(gt, bad_week))

    dup_cand = cand.duplicated(['SKU_ID', 'WeekStart'], keep=False)
    audit.record('T04', 'structure', dup_cand.sum(), len(cand), 'Candidate panel keys must be unique.', sample=examples(cand, dup_cand))
    if not (dup_gt.any() or dup_cand.any() or gt_bad_keys.any() or cand.WeekStart.isna().any()):
        keys = gt[['SKU_ID', 'WeekStart']].merge(cand[['SKU_ID', 'WeekStart']],
                                                on=['SKU_ID', 'WeekStart'], how='outer', indicator=True)
        mismatch = keys['_merge'].ne('both')
        audit.record('T05', 'structure', mismatch.sum(), len(keys),
                     'Ground truth and activity panel must use identical SKU-week keys.', sample=examples(keys, mismatch))
        panel_keys_same = not mismatch.any()
    else:
        panel_keys_same = False
        audit.note('T05', 'structure', 'NOT_ASSESSED', 'Cannot compare full key sets until duplicate/invalid keys are resolved.')

    valid_source = src.SKU_ID.notna() & src.OrderDate.notna() & src.Quantity.notna() & src.Quantity.gt(0)
    source_bad = ~valid_source
    audit.record('T06', 'source', source_bad.sum(), len(src), 'Clean source needs valid SKU, date and positive Quantity.',
                 sample=examples(src, source_bad))
    dup_line = src.SalesOrderLineID.isna() | src.SalesOrderLineID.duplicated(keep=False)
    audit.record('T07', 'source', dup_line.sum(), len(src), 'SalesOrderLineID in clean source must be present and unique.')
    if 'OrderStatusCanonical' in src.columns:
        not_completed = src.OrderStatusCanonical.astype('string').str.lower().ne('completed').fillna(True)
        audit.record('T08', 'source', not_completed.sum(), len(src), 'Clean source must only contain completed orders.')
    else:
        audit.note('T08', 'source', 'NOT_ASSESSED', 'OrderStatusCanonical absent; cannot recheck completed-order filter.')

    statuses = gt.label_status.astype('string')
    unknown_status = ~statuses.isin(VALID_STATUSES)
    audit.record('T09', 'labels', unknown_status.sum(), len(gt), 'Only observed_positive, verified_zero, unverified_no_sale are allowed.',
                 sample=examples(gt, unknown_status))
    positive = statuses.eq('observed_positive').fillna(False)
    zero = statuses.eq('verified_zero').fillna(False)
    unverified = statuses.eq('unverified_no_sale').fillna(False)
    q = gt.WeeklyQuantity
    incorrect_labels = ((positive & (~q.gt(0).fillna(False))) |
                        (zero & (~q.eq(0).fillna(False))) |
                        (unverified & q.notna()))
    audit.record('T10', 'labels', incorrect_labels.sum(), len(gt),
                 'Positive >0, verified_zero ==0, and unverified_no_sale must stay NA.',
                 sample=examples(gt, incorrect_labels))
    fractional = q.notna() & ((q < 0) | (~np.isclose(q.fillna(0), np.round(q.fillna(0)))))
    audit.record('T11', 'labels', fractional.sum(), len(gt), 'Gross unit counts must be nonnegative whole numbers.',
                 sample=examples(gt, fractional))

    flags = {}
    bad_bool = pd.Series(False, index=gt.index)
    for field in ['SkuWeekVerifiedActive', 'SalesLogCoverageVerified', 'ZeroLabelAuthorized']:
        flags[field], wrong = parse_true(gt[field])
        bad_bool |= wrong
    audit.record('T12', 'labels', bad_bool.sum(), len(gt), 'Evidence flags must contain valid booleans.',
                 sample=examples(gt, bad_bool))
    unauth_zero = zero & ~(flags['SkuWeekVerifiedActive'] & flags['SalesLogCoverageVerified'] & flags['ZeroLabelAuthorized'])
    audit.record('T13', 'labels', unauth_zero.sum(), int(zero.sum()),
                 'Every zero label needs verified SKU activity, complete logs and explicit authorization.',
                 sample=examples(gt, unauth_zero))
    incorrect_target = gt.target_definition.astype('string').ne('gross_completed_sales_quantity').fillna(True)
    audit.record('T14', 'policy', incorrect_target.sum(), len(gt),
                 'Target must remain gross completed-sales quantity (returns are not deducted).')
    incorrect_status = gt.dataset_status.astype('string').ne('PROVISIONAL_NOT_FINAL').fillna(True)
    audit.record('T15', 'policy', incorrect_status.sum(), len(gt),
                 'Current file must be marked PROVISIONAL_NOT_FINAL.', sample=examples(gt, incorrect_status))

    scoped = pd.DataFrame()
    source_gross = np.nan
    off_scope_rows, off_scope_qty = 0, 0
    if valid_source.all() and gt.WeekStart.notna().all() and not dup_gt.any():
        src['WeekStart'] = src.OrderDate - pd.to_timedelta(src.OrderDate.dt.dayofweek, unit='D')
        period = gt.WeekStart.dropna()
        scoped = src[src.WeekStart.between(period.min(), period.max())].copy()
        source_gross = float(scoped.Quantity.sum())
        off_scope_rows = len(src) - len(scoped)
        off_scope_qty = float(src.Quantity.sum() - source_gross)
        by_week = scoped.groupby(['SKU_ID','WeekStart'], as_index=False).agg(
            source_qty=('Quantity','sum'), source_lines=('Quantity','size'))
        joined = gt[['SKU_ID','WeekStart','WeeklyQuantity','label_status', 'observed_sales_lines']].merge(
            by_week, on=['SKU_ID','WeekStart'], how='left', validate='one_to_one')
        joined['source_qty'] = joined.source_qty.fillna(0)
        joined['source_lines'] = joined.source_lines.fillna(0)
        qty_mismatch = positive & ~np.isclose(joined.source_qty, q.fillna(0), rtol=0, atol=1e-9)
        unexpected_sale = ~positive & joined.source_qty.gt(0)
        audit.record('T16', 'reconciliation', qty_mismatch.sum() + unexpected_sale.sum(), len(gt),
                     'Ground truth must preserve every observed positive source quantity, no invented positives.',
                     sample=examples(gt, (qty_mismatch | unexpected_sale)))
        bad_line_counts = gt.observed_sales_lines.ne(joined.source_lines).fillna(True)
        audit.record('T17', 'reconciliation', bad_line_counts.sum(), len(gt),
                     'Observed transaction-line counts must match the clean source.',
                     sample=examples(gt, bad_line_counts))
        bad_zero_sale = zero & joined.source_qty.gt(0)
        audit.record('T18', 'reconciliation', bad_zero_sale.sum(), int(zero.sum()),
                     'No verified zero may have a source sale.', sample=examples(gt, bad_zero_sale))
        source_keys = pd.MultiIndex.from_frame(by_week[['SKU_ID','WeekStart']])
        gt_keys = pd.MultiIndex.from_frame(gt[['SKU_ID','WeekStart']])
        absent_gt = (~source_keys.isin(gt_keys)).sum()
        audit.record('T19', 'reconciliation', absent_gt, len(by_week),
                     'Every in-scope source SKU-week must exist in the ground truth panel.')
        source_sum_delta = float(q.sum(skipna=True) - source_gross)
        audit.record('T20', 'reconciliation', int(not np.isclose(source_sum_delta, 0, rtol=0, atol=1e-9)), 1,
                     f'Ground truth total must match in-scope clean source. Delta={source_sum_delta:,.0f}.')
    else:
        for id_ in ('T16','T17','T18','T19','T20'):
            audit.note(id_, 'reconciliation', 'NOT_ASSESSED',
                       'Clean source or ground truth keys have invalid/duplicate values; reconciliation skipped.')

    if panel_keys_same:
        cg = cand[['SKU_ID','WeekStart','observed_gross_qty','observed_sales_lines','HasObservedSale']]
        cross = gt[['SKU_ID','WeekStart']].merge(cg,on=['SKU_ID','WeekStart'],how='left',validate='one_to_one')
        mismatch_lines = ~cross.observed_sales_lines.eq(gt.observed_sales_lines).fillna(False)
        audit.record('T21', 'reconciliation', mismatch_lines.sum(), len(gt),
                     'Candidate observed_sales_lines must agree with ground truth.')
        candidate_diff = ~np.isclose(cross.observed_gross_qty.fillna(-1), q.fillna(0), rtol=0, atol=1e-9)
        # Candidate has zero for all no-sale weeks; ground truth deliberately uses NA there.
        audit.record('T22', 'reconciliation', candidate_diff.sum(), len(gt),
                     'Candidate gross quantity must equal observed labels, treating NA targets as candidate zero.')
        has_sale, invalid_flag = parse_true(cross.HasObservedSale)
        mismatch_sale = invalid_flag | (has_sale != cross.observed_gross_qty.gt(0))
        audit.record('T23', 'reconciliation', mismatch_sale.sum(), len(gt),
                     'Candidate HasObservedSale must agree with candidate gross Quantity.')
    else:
        for id_ in ('T21','T22','T23'):
            audit.note(id_, 'reconciliation', 'NOT_ASSESSED', 'Candidate/ground-truth keys not aligned.')

    weeks = pd.DatetimeIndex(gt.WeekStart.dropna().unique()).sort_values()
    discontinuity = (weeks[1:] - weeks[:-1] != pd.Timedelta(days=7)).sum()
    audit.record('T24', 'time', discontinuity, max(len(weeks)-1,0), 'Study weeks must progress in 7-day steps.')
    conflict_col = gt.get('quality_flag_product_date_conflict')
    if conflict_col is not None:
        conflict, invalid_conflict = parse_true(conflict_col)
        conflict_skus = int(gt.loc[conflict, 'SKU_ID'].nunique())
        audit.record('T25', 'metadata', int(invalid_conflict.sum()), len(gt), 'Product-date conflict flags must be booleans.')
        audit.note('T26', 'metadata', 'WARN' if conflict_skus else 'PASS',
                   f'{conflict_skus} SKU(s) have source/product lifecycle date conflicts; review before final release.')
    else:
        conflict_skus = None
        audit.note('T25', 'metadata', 'NOT_ASSESSED', 'No product-date conflict flag in ground truth.')
        audit.note('T26', 'metadata', 'NOT_ASSESSED', 'Unable to assess product-date conflict coverage.')

    no_sale = int(unverified.sum())
    if no_sale > 0:
        audit.note('T27', 'readiness', 'BLOCKED',
                   f'{no_sale:,} no-sale SKU-weeks remain unverified (NA). Cannot certify full ground truth or zero-sale distribution.')
    else:
        audit.note('T27', 'readiness', 'PASS', 'No unverified no-sale weeks remain.')
    if not (flags['SkuWeekVerifiedActive'] & flags['SalesLogCoverageVerified']).any():
        audit.note('T28', 'readiness', 'BLOCKED',
                   'No SKU-weeks currently have BOTH verified SKU activity and verified complete transaction logs.')
    else:
        audit.note('T28', 'readiness', 'PASS', 'Some weeks have both evidence flags; verify their provenance before final release.')
    audit.note('T29', 'leakage', 'NOT_ASSESSED',
               'Rolling one-step feature availability and training/test leakage cannot be certified without feature/model pipeline.')

    # MA4 historical label availability only; this is NOT evidence that forecasting can be trained yet.
    ordered = gt.sort_values(['SKU_ID', 'WeekStart']).copy()
    prev4 = pd.Series(True, index=ordered.index)
    for offset in range(1,5):
        group = ordered.groupby('SKU_ID', sort=False)
        lag = group.WeeklyQuantity.shift(offset)
        lag_date = group.WeekStart.shift(offset)
        prev4 &= lag.notna() & lag_date.eq(ordered.WeekStart - pd.Timedelta(days=7*offset))
    ma4_ready = int((prev4 & ordered.WeeklyQuantity.notna()).sum())
    audit.note('T30', 'readiness', 'INFO',
               f'{ma4_ready:,} SKU-weeks have 4 consecutive prior labeled weeks AND an observed current target; this is not a training approval.')

    check_statuses = set(audit.frame()['status'])
    final_review_status = ('VALIDATION_FAILED' if 'FAIL' in check_statuses
                           else 'PROVISIONAL_BLOCKED' if 'BLOCKED' in check_statuses
                           else 'REVIEW_REQUIRED')
    counts = pd.Series({
        'validation_status': final_review_status,
        'ground_truth_file': str(gt_path),
        'source_file': str(clean_path),
        'candidate_file': str(cand_path),
        'sku_weeks': len(gt),
        'distinct_skus': int(gt.SKU_ID.nunique()),
        'weeks': int(len(weeks)),
        'observed_positive_sku_weeks': int(positive.sum()),
        'verified_zero_sku_weeks': int(zero.sum()),
        'unverified_no_sale_sku_weeks': no_sale,
        'labeled_coverage_pct': round(100 * int((positive|zero).sum()) / max(len(gt),1), 2),
        'observed_gross_quantity': int(q.sum(skipna=True)),
        'scoped_source_gross_quantity': '' if pd.isna(source_gross) else int(source_gross),
        'source_rows_outside_panel_weeks': off_scope_rows,
        'source_qty_outside_panel_weeks': int(off_scope_qty),
        'product_date_conflict_skus': '' if conflict_skus is None else conflict_skus,
        'sku_weeks_eligible_for_ma4_with_observed_target': ma4_ready,
        'note': 'Not approved as complete training/evaluation ground truth until unknown-week evidence is resolved.'
    })
    return audit.frame(), counts.rename_axis('metric').reset_index(name='value')


def main() -> None:
    parser = argparse.ArgumentParser(description='Step 7: validate provisional SKU-week ground truth; never fill missing zeros.')
    parser.add_argument('--ground-truth', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--clean-file', type=Path)
    parser.add_argument('--out-dir', type=Path, default=Path('reports/tables'))
    args = parser.parse_args()
    gt_path = locate(args.ground_truth, [
        'data/processed/sku_week_ground_truth_provisional.csv',
        'src/data/processed/sku_week_ground_truth_provisional.csv'], 'Ground truth')
    cand_path = locate(args.candidates, [
        'reports/tables/sku_week_activity_candidates.csv',
        'src/data/processed/sku_week_activity_candidates.csv'], 'SKU-week candidates')
    clean_path = locate(args.clean_file, [
        'data/processed/clean_sales_transactions.csv',
        'src/data/processed/clean_sales_transactions.csv'], 'Clean transactions')
    print('Validating:', gt_path, '\nCandidates:', cand_path, '\nSource:', clean_path)
    checks, summary = run(gt_path, cand_path, clean_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(args.out_dir/'ground_truth_validation_checks.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(args.out_dir/'ground_truth_validation_summary.csv', index=False, encoding='utf-8-sig')
    summary_dict = dict(zip(summary.metric, summary.value))
    sections = ['# DS317 – Step 7: Ground Truth Validation', '',
                '> Ground truth is provisional; successful reconciliation is not final certification.', '',
                '## Summary', '',
                '| Metric | Value |', '|---|---|']
    for _,r in summary.iterrows():
        if r.metric.endswith('_file'):
            continue
        sections.append(f'| {r.metric} | {r.value} |')
    sections.extend(['', '## Findings requiring action', ''])
    findings = checks[checks.status.isin(['FAIL','WARN','BLOCKED','NOT_ASSESSED'])]
    if findings.empty:
        sections.append('No unresolved automatic checks (manual source verification is still required).')
    else:
        for _,r in findings.iterrows():
            sections.append(f'- **{r.check_id} [{r.status}]** {r.description}' +
                            (f' Examples: {r.examples}' if isinstance(r.examples,str) and r.examples else ''))
    sections.extend(['', '## Decision and next action', '',
       '- **Do not** replace `NA` with `0` unless SKU-week activity and order-log completeness are independently verified.',
       '- Revisit `ActiveFrom`/`ActiveTo` at the SKU level and review product-date conflicts.',
       '- Validate per-week export completeness from authoritative source information, not merely the existence of other orders.',
       '- Once evidence is available, update the lifecycle flags, rerun Step 4 and this Step 7 script.',
       '- The feature pipeline must separately enforce rolling one-step and no future leakage.', ''])
    (args.out_dir/'ground_truth_validation_findings.md').write_text('\n'.join(sections), encoding='utf-8')
    print('\n' + checks.groupby('status').size().to_string())
    print('\nValidation:', summary_dict['validation_status'])
    print('Label coverage:', summary_dict['labeled_coverage_pct'], '%')
    print('Gross quantity:', summary_dict['observed_gross_quantity'])
    print('Reports saved in:', args.out_dir)


if __name__ == '__main__':
    main()
