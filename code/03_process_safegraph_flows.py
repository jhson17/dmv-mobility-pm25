# 03_process_safegraph_flows.py
# Processes raw SafeGraph Patterns CSV files and produces
# aggregated OD flow data for flow map visualization.
#
# Input:
#   data/raw/safegraph/YYYY/places_Washington-Baltimore-Arlington*.csv
#
# Output:
#   data/processed/flow_map.csv
#
# Columns:
#   home_cbg          - origin CBG (12-digit FIPS)
#   poi_cbg           - destination CBG (12-digit FIPS)
#   period            - "All Years", "Pre-COVID (2018-2019)", "COVID Era (2020-2021)"
#   avg_visitor_count - mean visitor count across months in period
#   avg_median_dwell  - mean median dwell time (minutes) across months in period
#
# Usage:
#   python code/03_process_safegraph_flows.py
#   python code/03_process_safegraph_flows.py --year 2019

import argparse
import ast
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------
# PARAMETERS
# ------------------------------
RAW_DIR = Path("data/raw/safegraph")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2018, 2019, 2020, 2021]

DMV_COUNTIES = [
    '11001',
    '24031', '24033',
    '51013', '51059',
    '51510', '51600', '51610',
]


# ------------------------------
# FUNCTIONS
# ------------------------------
def parse_cbg_json(val):
    """Parse visitor_home_cbgs JSON string to dict."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def unpack_visitor_home_cbgs(df):
    """
    Unpack visitor_home_cbgs column into one row per origin CBG.
    No minimum visitor threshold applied beyond SafeGraph's own
    differential privacy suppression (which excludes counts below 2).
    """
    df['cbg_dict'] = df['visitor_home_cbgs'].apply(parse_cbg_json)
    df = df[df['cbg_dict'].map(len) > 0].copy()
    if df.empty:
        return None

    df['cbg_items'] = df['cbg_dict'].apply(lambda d: list(d.items()))
    df = df.explode('cbg_items')
    df[['home_cbg', 'visitor_count']] = pd.DataFrame(
        df['cbg_items'].tolist(), index=df.index
    )

    df['home_cbg']      = df['home_cbg'].astype(str).str.zfill(12)
    df['visitor_count'] = (
        pd.to_numeric(df['visitor_count'], errors='coerce')
        .fillna(0).astype(int)
    )

    # Keep only DMV origins and destinations
    df = df[df['home_cbg'].str[:5].isin(DMV_COUNTIES)].copy()
    if df.empty:
        return None

    return df[['home_cbg', 'poi_cbg', 'year', 'month',
               'visitor_count', 'median_dwell']].copy()


def process_file(filepath):
    """Load one SafeGraph part file and return unpacked flows."""
    try:
        df = pd.read_csv(filepath, dtype={'poi_cbg': str}, low_memory=False)
    except Exception as e:
        print(f"    ERROR reading {filepath.name}: {e}")
        return None

    required = ['poi_cbg', 'date_range_start', 'visitor_home_cbgs']
    if not all(c in df.columns for c in required):
        print(f"    SKIP {filepath.name} — missing columns")
        return None

    df['poi_cbg'] = df['poi_cbg'].str.split('.').str[0].str.zfill(12)
    df = df[df['poi_cbg'].str[:5].isin(DMV_COUNTIES)].copy()
    if df.empty:
        return None

    dates       = pd.to_datetime(df['date_range_start'], errors='coerce')
    df['year']  = dates.dt.year
    df['month'] = dates.dt.month

    if 'median_dwell' in df.columns:
        df['median_dwell'] = pd.to_numeric(df['median_dwell'], errors='coerce')
        df.loc[df['median_dwell'] < 0, 'median_dwell'] = np.nan
    else:
        df['median_dwell'] = np.nan

    return unpack_visitor_home_cbgs(df)


def aggregate_flows(flows):
    """Aggregate to one row per home_cbg x poi_cbg x year x month."""
    return (
        flows
        .groupby(['home_cbg', 'poi_cbg', 'year', 'month'], as_index=False)
        .agg(
            visitor_count=('visitor_count', 'sum'),
            median_dwell =('median_dwell',  'mean')
        )
    )


def build_flow_map(combined):
    """
    Aggregate flows into three periods for flow map visualization:
      - All Years (2018-2021)
      - Pre-COVID (2018-2019)
      - COVID Era (2020-2021)
    """
    periods = [
        ('All Years',              combined),
        ('Pre-COVID (2018-2019)',   combined[combined['year'].isin([2018, 2019])]),
        ('COVID Era (2020-2021)',   combined[combined['year'].isin([2020, 2021])]),
    ]

    all_periods = []
    for label, df in periods:
        if df.empty:
            continue
        agg = (
            df.groupby(['home_cbg', 'poi_cbg'], as_index=False)
            .agg(
                avg_visitor_count=('visitor_count', 'mean'),
                avg_median_dwell =('median_dwell',  'mean')
            )
        )
        agg['period'] = label
        all_periods.append(agg)
        print(f"    {label:30s} : {len(agg):,} OD pairs")

    flow_map = pd.concat(all_periods, ignore_index=True)
    flow_map['avg_visitor_count'] = flow_map['avg_visitor_count'].round(1)
    flow_map['avg_median_dwell']  = flow_map['avg_median_dwell'].round(1)
    flow_map = flow_map[['home_cbg', 'poi_cbg', 'period',
                          'avg_visitor_count', 'avg_median_dwell']]
    flow_map.sort_values(['period', 'avg_visitor_count'],
                         ascending=[True, False], inplace=True)
    flow_map.reset_index(drop=True, inplace=True)
    return flow_map


# ------------------------------
# MAIN
# ------------------------------
def run(years):
    print("-" * 55)
    print("Processing SafeGraph Flows for Flow Map")
    print("-" * 55)

    all_flows = []

    for year in years:
        year_dir   = RAW_DIR / str(year)
        part_files = sorted(
            year_dir.glob("places_Washington-Baltimore-Arlington*.csv")
        )

        if not part_files:
            print(f"\n  [{year}] No files found — skipping")
            continue

        print(f"\n  [{year}] — {len(part_files)} files")

        year_flows = []
        for filepath in part_files:
            print(f"    {filepath.name}...", end=' ')
            flows = process_file(filepath)
            if flows is None:
                print("skipped")
                continue
            year_flows.append(flows)
            print(f"{len(flows):,} flows")

        if not year_flows:
            continue

        year_agg = aggregate_flows(pd.concat(year_flows, ignore_index=True))
        all_flows.append(year_agg)
        print(f"\n    [{year}] total: {len(year_agg):,} OD pairs  "
              f"visitors={year_agg['visitor_count'].sum():,}")

    if not all_flows:
        print("\nERROR: No data processed.")
        return

    print("\nBuilding flow map periods...")
    combined = aggregate_flows(pd.concat(all_flows, ignore_index=True))
    flow_map = build_flow_map(combined)

    out = OUT_DIR / "flow_map.csv"
    flow_map.to_csv(out, index=False)

    print(f"\n{'-' * 55}")
    print(f"  Periods    : {flow_map['period'].unique().tolist()}")
    print(f"  Total rows : {len(flow_map):,}")
    print(f"  Output     : {out}")
    print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process SafeGraph flows for flow map visualization"
    )
    parser.add_argument('--year', type=int, help='Single year (e.g. 2019)')
    args  = parser.parse_args()
    years = [args.year] if args.year else YEARS
    run(years)
