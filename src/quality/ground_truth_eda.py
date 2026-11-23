from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_existing(explicit: Path | None, alternatives: list[str], label: str) -> Path:
    if explicit is not None:
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f'{label} not found: {explicit}')
    for name in alternatives:
        p = Path(name)
        if p.is_file():
            return p
    raise FileNotFoundError(f'{label} not found. Tried: {alternatives}')


def save(fig: plt.Figure, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def bool_flag(data: pd.Series, name: str) -> pd.Series:
    x = data.astype('string').str.strip().str.lower()
    good = x.isin(['true', 'false', '1', '0'])
    if (~good).any():
        raise ValueError(f'Invalid values in {name}: {int((~good).sum())} rows')
    return x.isin(['true', '1'])


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'SKU_ID': 'string'}, low_memory=False)
    need = {'SKU_ID','WeekStart','WeeklyQuantity','label_status'}
    if not need.issubset(df.columns):
        raise ValueError(f'{path}: missing columns {sorted(need-set(df.columns))}')
    df['WeekStart'] = pd.to_datetime(df['WeekStart'], errors='coerce')
    if df[['SKU_ID','WeekStart']].isna().any().any():
        raise ValueError('Missing SKU ID or invalid WeekStart')
    if df.duplicated(['SKU_ID','WeekStart']).any():
        raise ValueError('Duplicate SKU/WeekStart combinations')
    if (df.WeekStart.dt.dayofweek != 0).any():
        raise ValueError('WeekStart must be a Monday')
    df['WeeklyQuantity'] = pd.to_numeric(df['WeeklyQuantity'], errors='coerce')
    allowed = {'observed_positive','verified_zero','unverified_no_sale'}
    if not df.label_status.isin(allowed).all():
        raise ValueError(f'Unexpected label_status: {sorted(set(df.label_status.dropna()) - allowed)}')
    p = df.label_status.eq('observed_positive')
    z = df.label_status.eq('verified_zero')
    u = df.label_status.eq('unverified_no_sale')
    if not (df.loc[p, 'WeeklyQuantity'].gt(0).all()
            and df.loc[z,'WeeklyQuantity'].eq(0).all()
            and df.loc[u,'WeeklyQuantity'].isna().all()):
        raise ValueError('Label state / quantity mismatch. Refusing to treat unknown as zero.')
    if 'dataset_status' in df.columns and not df['dataset_status'].eq('PROVISIONAL_NOT_FINAL').all():
        raise ValueError('Unexpected dataset status: this script is for the provisional ground truth')
    return df


def to_csv(frame: pd.DataFrame, p: Path) -> None:
    frame.to_csv(p, index=False, encoding='utf-8-sig', float_format='%.5f')


def run(gt_file: Path, dq_file: Path | None, figures: Path, tables: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    df = load_ground_truth(gt_file)
    total = len(df)
    labeled_mask = df.WeeklyQuantity.notna()
    p = df.label_status.eq('observed_positive')
    z = df.label_status.eq('verified_zero')
    u = df.label_status.eq('unverified_no_sale')

    # EDA 1: Sales over time. A sum over positive observed labels (never over invented zeros).
    weekly = df.groupby('WeekStart', as_index=False).agg(
        observed_gross_units=('WeeklyQuantity','sum'),
        observed_sale_skus=('label_status', lambda x: int(x.eq('observed_positive').sum())),
        verified_zero_sku_weeks=('label_status', lambda x: int(x.eq('verified_zero').sum())),
        unverified_sku_weeks=('label_status', lambda x: int(x.eq('unverified_no_sale').sum())),
        sku_week_rows=('label_status', 'size'),
    )
    weekly['labeled_share_pct'] = 100 * (weekly.sku_week_rows-weekly.unverified_sku_weeks)/weekly.sku_week_rows
    to_csv(weekly.assign(WeekStart=weekly.WeekStart.dt.strftime('%Y-%m-%d')),
           tables/'eda_01_weekly_sales.csv')
    fig, ax = plt.subplots(figsize=(11,4.6))
    ax.plot(weekly.WeekStart, weekly.observed_gross_units, linewidth=1.6)
    ax.set(title='Observed gross sales by week (provisional)',
           xlabel='Week', ylabel='Gross units from observed completed sales')
    ax.grid(axis='y', alpha=0.25)
    save(fig, figures/'eda_01_weekly_sales.png')

    # EDA 2: Distribution of VALID observed labels. Here verified zeros may be absent.
    vals = df.loc[labeled_mask,'WeeklyQuantity'].astype(float)
    positive_vals = df.loc[p, 'WeeklyQuantity'].astype(float)
    desc = vals.describe(percentiles=[.25,.5,.75,.9,.95,.99])
    distribution = pd.DataFrame([{'metric':str(key),'value':float(value)} for key,value in desc.items()]
                                + [{'metric':'observed_positive_sku_weeks','value':int(p.sum())},
                                   {'metric':'verified_zero_sku_weeks','value':int(z.sum())},
                                   {'metric':'unverified_no_sale_sku_weeks','value':int(u.sum())}])
    to_csv(distribution, tables/'eda_02_target_distribution.csv')
    fig, ax = plt.subplots(figsize=(9,4.6))
    ax.hist(np.log1p(positive_vals), bins=40)
    ax.set(title='Distribution of observed POSITIVE SKU-week sales',
           xlabel='ln(1 + WeeklyQuantity); excludes unverified no-sales', ylabel='SKU-week count')
    ax.grid(axis='y', alpha=0.25)
    save(fig, figures/'eda_02_target_distribution.png')

    # EDA 3: Coverage of actually labeled observations by SKU.
    sku = df.groupby('SKU_ID', as_index=False).agg(
        total_weeks=('label_status','size'),
        observed_positive_weeks=('label_status', lambda x: int(x.eq('observed_positive').sum())),
        verified_zero_weeks=('label_status', lambda x: int(x.eq('verified_zero').sum())),
        unknown_weeks=('label_status', lambda x: int(x.eq('unverified_no_sale').sum())),
        observed_gross_units=('WeeklyQuantity','sum'))
    sku['labeled_share_pct'] = 100*(sku.observed_positive_weeks+sku.verified_zero_weeks)/sku.total_weeks
    to_csv(sku, tables/'eda_03_sku_coverage.csv')
    fig, ax = plt.subplots(figsize=(9,4.6))
    ax.hist(sku.labeled_share_pct, bins=np.arange(0,105,5))
    ax.set(title='SKU weekly-label coverage (not zero-sales rate)',
           xlabel='Share of SKU-weeks with verified labels (%)', ylabel='SKU count')
    ax.grid(axis='y', alpha=0.25)
    save(fig, figures/'eda_03_sku_coverage.png')

    # EDA 4: Observed gap between sale EVENTS, not standard ADI of true demand.
    # Missing weekly sales cannot be used as 0 in intermittent-demand classification.
    positive_dates = df.loc[p, ['SKU_ID','WeekStart']].sort_values(['SKU_ID','WeekStart'])
    weeks_between = positive_dates.groupby('SKU_ID').WeekStart.diff().dt.days.div(7).dropna().astype('int64')
    gaps = weeks_between.value_counts().rename_axis('weeks_between_observed_sales').reset_index(name='count')
    gaps = gaps.sort_values('weeks_between_observed_sales')
    to_csv(gaps, tables/'eda_04_observed_sale_gaps.csv')
    fig, ax = plt.subplots(figsize=(9,4.6))
    if not weeks_between.empty:
        ax.hist(weeks_between.clip(upper=20), bins=np.arange(0.5,21.5,1))
    ax.set(title='Gap between recorded positive-sale weeks (descriptive only)',
           xlabel='Weeks between observed positive sales (20 = 20+)', ylabel='Interval count')
    ax.grid(axis='y', alpha=0.25)
    save(fig, figures/'eda_04_observed_sale_gaps.png')

    # Status: ADI/CV² demand pattern classification is only defensible for
    # fully verified consecutive SKU history; none is assumed present.
    complete = sku.loc[sku.unknown_weeks.eq(0)].copy()
    if complete.empty:
        pattern = pd.DataFrame([{'status':'BLOCKED','qualified_skus':0,
            'reason':'No SKU has a fully verified weekly panel; ADI/CV2 are not computed from unverified NA.'}])
    else:
        # No automatic classification: method needs an explicit active-life window and
        # defensible handling of short histories, not just a whole-grid completeness check.
        pattern = pd.DataFrame([{'status':'REVIEW_REQUIRED','qualified_skus':len(complete),
            'reason':'Complete grid exists for some SKUs; confirm active-life window and ADI/CV2 thresholds before classification.'}])
    to_csv(pattern, tables/'eda_04_adi_cv2_readiness.csv')

    # EDA 5: Six-dimensional data quality, counts of issues, not a misleading composite score.
    dq_notice = 'not supplied'
    if dq_file is not None:
        dq = pd.read_csv(dq_file, low_memory=False)
        required = {'dimension','check','affected_count','inspected_count','rate_pct','status','notes'}
        if not required.issubset(dq.columns):
            raise ValueError(f'DQ report missing columns {sorted(required-set(dq.columns))}')
        to_csv(dq, tables/'eda_05_quality_checks.csv')
        flag = dq.assign(needs_review=~dq.status.isin(['PASS','RESOLVED']))
        issues = (flag.groupby('dimension').needs_review.sum()
                  .reindex(['Completeness','Consistency','Accuracy','Timeliness','Validity','Uniqueness'],fill_value=0))
        fig, ax = plt.subplots(figsize=(10,4.6))
        ax.bar(issues.index, issues.values)
        ax.set(title='Data-quality checks needing review by dimension',
               ylabel='Count of flagged / not-verified checks', xlabel='Quality dimension')
        ax.tick_params(axis='x', labelrotation=25)
        ax.grid(axis='y', alpha=0.25)
        save(fig, figures/'eda_05_data_quality.png')
        dq_notice = f'{int(flag.needs_review.sum())} of {len(dq)} checks flagged or not verified; see eda_05_quality_checks.csv'

    trend = 'Trend is a sum of observed sales, NOT proof of complete weekly demand.'
    pct = 100*labeled_mask.sum()/total
    lines = [
        '# DS317 | Step 6 - Preliminary EDA of provisional ground truth',
        '',
        f'- Source: `{gt_file.as_posix()}`',
        '- Status: **PROVISIONAL_NOT_FINAL; NOT ready for model training/evaluation**.',
        f'- Rows: **{total:,}**; SKUs: **{df.SKU_ID.nunique():,}**; weeks: **{df.WeekStart.nunique():,}**.',
        f'- Observed positive labels: **{int(p.sum()):,}**; verified zero labels: **{int(z.sum()):,}**; unverified no-sales: **{int(u.sum()):,}**.',
        f'- Labeled share: **{pct:.2f}%**; sum of observed gross units: **{int(vals.sum()):,}**.',
        '',
        '## 5 outputs',
        '1. `eda_01_weekly_sales.png/.csv`: observed gross units by week. ' + trend,
        '2. `eda_02_target_distribution.png/.csv`: distribution of **observed positive** SKU-week quantities, not the population with true zero weeks.',
        '3. `eda_03_sku_coverage.png/.csv`: fraction of weeks with known labels per SKU; cannot be called a verified no-sale rate.',
        '4. `eda_04_observed_sale_gaps.png/.csv`: descriptive gap between **recorded positive-sale events**. **Not ADI**. True ADI/CV² readiness: `eda_04_adi_cv2_readiness.csv`.',
        f'5. `eda_05_data_quality.png/.csv`: 6 dimensions; {dq_notice}.',
        '',
        '## Blockers before final forecasting experiments',
        '- Verify SKU availability at weekly level and complete order-log coverage.',
        '- Do NOT impute NA as zero or derive canonical ADI/CV² from missing no-sale labels.',
        '- Review product date conflicts and source period scope.',
        '- For rolling one-step, compute features strictly from data available before the forecast week.',
        '',
        'The outputs here are exploratory and should be described as preliminary in Lab 3.'
    ]
    (tables/'eda_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:7]))
    print(f'\nGenerated EDA files under: {figures} and {tables}')


def main():
    parser = argparse.ArgumentParser(description='EDA of provisional SKU-week ground truth')
    parser.add_argument('--ground-truth', type=Path, default=None)
    parser.add_argument('--dq-file', type=Path, default=None)
    parser.add_argument('--figure-dir', type=Path, default=Path('reports/figures'))
    parser.add_argument('--table-dir', type=Path, default=Path('reports/tables'))
    args = parser.parse_args()
    gt = find_existing(args.ground_truth, [
        'data/processed/sku_week_ground_truth_provisional.csv',
        'src/data/processed/sku_week_ground_truth_provisional.csv'
    ], 'Ground truth CSV')
    # Optional report; EDA still runs without it (only the fifth image is skipped).
    dq = None
    if args.dq_file is not None:
        dq = find_existing(args.dq_file, [], 'Data quality CSV')
    else:
        for name in ['reports/tables/data_quality_6_dimensions.csv',
                     'data_quality_6_dimensions.csv']:
            if Path(name).is_file():
                dq = Path(name)
                break
    run(gt, dq, args.figure_dir, args.table_dir)


if __name__ == '__main__':
    main()
