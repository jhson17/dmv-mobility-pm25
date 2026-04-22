# 02_process_acs.py
# Processes downloaded 2020 5-year ACS CSV files for DMV Census Block Groups.
# Computes demographic variables needed for mobility and EJ analysis.
#
# Input:  data/raw/acs/ACSDT5Y2020.B*-Data.csv
# Output: data/processed/cbg_demographics.csv
#
# Usage:
#   python code/02_process_acs.py

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------
# PARAMETERS
# ------------------------------
ACS_DIR = Path("data/raw/acs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DMV_COUNTIES = [
    '11001',
    '24031', '24033',
    '51013', '51059',
    '51510', '51600', '51610',
]


# ------------------------------
# FUNCTIONS
# ------------------------------
def load_table(filepath):
    """
    Load one ACS CSV file. Skips the label row (row 1),
    builds GEOID from GEO_ID, filters to DMV counties.
    Returns DataFrame indexed by GEOID or None on error.
    """
    try:
        df = pd.read_csv(filepath, dtype=str, skiprows=[1])
        df.columns = df.columns.str.strip()

        if 'GEO_ID' not in df.columns:
            print(f"  WARNING: No GEO_ID in {filepath.name}")
            return None

        df['GEOID'] = df['GEO_ID'].str[-12:].str.zfill(12)
        df = df[df['GEOID'].str[:5].isin(DMV_COUNTIES)].copy()

        if df.empty:
            print(f"  WARNING: No DMV rows in {filepath.name}")
            return None

        print(f"  {filepath.name}  →  {len(df):,} CBGs")
        return df.set_index('GEOID')

    except Exception as e:
        print(f"  ERROR loading {filepath.name}: {e}")
        return None


def safe_numeric(series):
    """Convert to numeric and replace suppressed values (negative) with NaN."""
    s = pd.to_numeric(series, errors='coerce')
    s[s < 0] = np.nan
    return s


def get_col(df, prefix):
    """Get first column matching a variable prefix."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    return safe_numeric(df[cols[0]]) if cols else None


# ------------------------------
# MAIN
# ------------------------------
def run():
    print("-" * 55)
    print("Processing ACS Demographics")
    print("-" * 55)

    # Load tables
    tables = {}
    for table in ['B01003', 'B03002', 'B19013', 'B17017',
                  'B25044', 'B15003', 'B25003']:
        matches = list(ACS_DIR.glob(f"*{table}*-Data.csv"))
        if matches:
            df = load_table(matches[0])
            if df is not None:
                tables[table] = df
        else:
            print(f"  WARNING: No file found for {table}")

    if not tables:
        print("\nERROR: No ACS tables loaded.")
        return

    # Common GEOID index
    geoids = None
    for df in tables.values():
        geoids = df.index if geoids is None else geoids.intersection(df.index)
    print(f"\n  Common CBGs: {len(geoids):,}")

    # Compute demographics
    result = pd.DataFrame(index=geoids)

    # Population
    if 'B01003' in tables:
        result['total_pop'] = get_col(tables['B01003'], 'B01003_001')

    # Race
    if 'B03002' in tables:
        t = tables['B03002']
        race_total    = get_col(t, 'B03002_001')
        result['pct_white']    = get_col(t, 'B03002_003') / race_total * 100
        result['pct_black']    = get_col(t, 'B03002_004') / race_total * 100
        result['pct_asian']    = get_col(t, 'B03002_006') / race_total * 100
        result['pct_hispanic'] = get_col(t, 'B03002_012') / race_total * 100
        result['pct_nonwhite'] = 100 - result['pct_white']

    # Income
    if 'B19013' in tables:
        result['median_income'] = get_col(tables['B19013'], 'B19013_001')

    # Poverty
    if 'B17017' in tables:
        t = tables['B17017']
        result['pct_poverty'] = get_col(t, 'B17017_002') / get_col(t, 'B17017_001') * 100

    # Vehicles
    if 'B25044' in tables:
        t = tables['B25044']
        no_veh = get_col(t, 'B25044_003').fillna(0) + get_col(t, 'B25044_010').fillna(0)
        result['pct_no_vehicle'] = no_veh / get_col(t, 'B25044_001') * 100

    # Education
    if 'B15003' in tables:
        t = tables['B15003']
        college = (
            get_col(t, 'B15003_022').fillna(0) +
            get_col(t, 'B15003_023').fillna(0) +
            get_col(t, 'B15003_024').fillna(0) +
            get_col(t, 'B15003_025').fillna(0)
        )
        result['pct_college'] = college / get_col(t, 'B15003_001') * 100

    # Tenure
    if 'B25003' in tables:
        t = tables['B25003']
        result['pct_renter'] = get_col(t, 'B25003_003') / get_col(t, 'B25003_001') * 100

    # Cap percentages
    for col in [c for c in result.columns if c.startswith('pct_')]:
        result[col] = result[col].clip(0, 100)

    # Clean up
    result = result.reset_index()
    result = result[result['total_pop'] > 0].copy()
    result.sort_values('GEOID', inplace=True)
    result.reset_index(drop=True, inplace=True)

    out = OUT_DIR / "cbg_demographics.csv"
    result.to_csv(out, index=False)

    print(f"\n{'-' * 55}")
    print(f"  CBGs processed : {len(result):,}")
    print(f"  Output         : {out}")
    print(f"\n  Variable summary:")
    for col in ['pct_nonwhite', 'pct_black', 'pct_hispanic', 'pct_asian',
                'median_income', 'pct_poverty', 'pct_no_vehicle',
                'pct_college', 'pct_renter']:
        if col in result.columns:
            print(f"    {col:20s}  "
                  f"mean={result[col].mean():7.1f}  "
                  f"missing={result[col].isna().sum():,}")
    print("-" * 55)


if __name__ == "__main__":
    run()
