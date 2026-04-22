# 04_build_final_data.py
# Reads raw SafeGraph Patterns files and GHAP PM2.5 to compute
# mobility-based PM2.5 exposure differentials per origin CBG.
#
# Inputs:
#   data/raw/safegraph/YYYY/places_Washington-Baltimore-Arlington*.csv
#   data/processed/ghap_pm25_monthly.csv
#
# Output:
#   data/processed/final_data_monthly.csv
#
# Columns:
#   GEOID             - 12-digit CBG FIPS (origin)
#   year              - 4-digit year
#   month             - numeric month (1-12)
#   time_id           - integer time index for panel models (1 = Jan 2018)
#   mean_delta_pm25   - visitor-weighted mean (pm25_dest - pm25_origin)
#   pct_trips_worse   - % of visitor-weighted trips to higher-PM2.5 areas
#   mobility_adj_pm25 - visitor-weighted mean destination PM2.5
#   residential_pm25  - origin CBG PM2.5 that month
#   n_destinations    - unique destination CBGs visited that month
#   total_visitors    - sum of visitor counts (relative mobility volume)
#   pct_low_count     - % of flows with DP-rounded visitor count (quality flag)
#   covid             - 1 from March 2020 onward
#   covid_era         - 1 for all of 2020-2021
#
# Demographics (cbg_demographics.csv) are joined in R before modeling.
#
# Usage:
#   python code/04_build_final_data.py
#   python code/04_build_final_data.py --year 2019

import argparse
import ast
import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------
# PARAMETERS
# ------------------------------
RAW_DIR  = Path("data/raw/safegraph")
PROC_DIR = Path("data/processed")
OUT_DIR  = Path("data/processed")

YEARS = [2018, 2019, 2020, 2021]

DMV_COUNTIES = [
    '11001',
    '24031', '24033',
    '51013', '51059',
    '51510', '51600', '51610',
]

# Minimum sum of visitor counts for a CBG-month to be included.
# CBG-months below this threshold have too few observed trips
# to produce a reliable weighted mean delta_pm25.
MIN_WEIGHTED_TRIPS = 10


# ------------------------------
# SAFEGRAPH PARSING
# ------------------------------
def parse_cbg_json(val):
    """Parse visitor_home_cbgs JSON string to dict."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def unpack_flows(df):
    """Unpack visitor_home_cbgs into one row per origin CBG."""
    df['cbg_dict'] = df['visitor_home_cbgs'].apply(parse_cbg_json)
    df = df[df['cbg_dict'].map(len) > 0].copy()
    if df.empty:
        return None

    df['cbg_items'] = df['cbg_dict'].apply(lambda d: list(d.items()))
    df = df.explode('cbg_items')
    df[['origin_cbg', 'visitor_count']] = pd.DataFrame(
        df['cbg_items'].tolist(), index=df.index
    )

    df['origin_cbg']    = df['origin_cbg'].astype(str).str.zfill(12)
    df['visitor_count'] = (
        pd.to_numeric(df['visitor_count'], errors='coerce')
        .fillna(0).astype(int)
    )

    # Keep only DMV origins
    df = df[df['origin_cbg'].str[:5].isin(DMV_COUNTIES)].copy()
    if df.empty:
        return None

    return df[['origin_cbg', 'dest_cbg', 'year', 'month', 'visitor_count']].copy()


def load_safegraph_year(year):
    """Load all part files for one year, return OD flows at month grain."""
    year_dir   = RAW_DIR / str(year)
    part_files = sorted(
        year_dir.glob("places_Washington-Baltimore-Arlington*.csv")
    )

    if not part_files:
        print(f"  [{year}] No files found — skipping")
        return None

    year_flows = []
    for filepath in part_files:
        try:
            df = pd.read_csv(filepath, dtype={'poi_cbg': str}, low_memory=False)
        except Exception as e:
            print(f"  [{year}] Error reading {filepath.name}: {e}")
            continue

        required = ['poi_cbg', 'date_range_start', 'visitor_home_cbgs']
        if not all(c in df.columns for c in required):
            continue

        df['dest_cbg'] = df['poi_cbg'].str.split('.').str[0].str.zfill(12)
        df = df[df['dest_cbg'].str[:5].isin(DMV_COUNTIES)].copy()
        if df.empty:
            continue

        dates       = pd.to_datetime(df['date_range_start'], errors='coerce')
        df['year']  = dates.dt.year
        df['month'] = dates.dt.month

        flows = unpack_flows(df)
        if flows is not None:
            year_flows.append(flows)

    if not year_flows:
        return None

    combined = pd.concat(year_flows, ignore_index=True)
    combined = (
        combined
        .groupby(['origin_cbg', 'dest_cbg', 'year', 'month'], as_index=False)
        .agg(visitor_count=('visitor_count', 'sum'))
    )

    print(f"  [{year}] {len(combined):,} OD-month pairs")
    return combined


# ------------------------------
# MAIN
# ------------------------------
def run(years):
    print("-" * 55)
    print("Building Final Data (CBG x Month)")
    print("-" * 55)

    # 1. Load SafeGraph flows
    print("\nLoading SafeGraph flows...")
    all_flows = [load_safegraph_year(y) for y in years]
    all_flows = [f for f in all_flows if f is not None]

    if not all_flows:
        print("ERROR: No SafeGraph data loaded.")
        return

    flows = pd.concat(all_flows, ignore_index=True)

    # Flag DP-rounded records (SafeGraph rounds counts of 2-4 up to 4)
    flows['is_low_count'] = flows['visitor_count'] <= 4

    print(f"  Total OD-month pairs : {len(flows):,}")
    print(f"  Low-count flag       : {flows['is_low_count'].mean()*100:.1f}% of flows")

    # 2. Load GHAP PM2.5
    print("\nLoading GHAP PM2.5...")
    ghap = pd.read_csv(
        PROC_DIR / "ghap_pm25_monthly.csv",
        dtype={"GEOID": str}
    )
    ghap["GEOID"] = ghap["GEOID"].str.zfill(12)
    ghap = ghap[ghap["year"].isin(years)].copy()
    print(f"  {len(ghap):,} CBG-month rows  |  {ghap['GEOID'].nunique():,} CBGs")

    # 3. Join PM2.5 to flows at monthly grain
    print("\nJoining PM2.5 to flows...")

    flows = flows.merge(
        ghap.rename(columns={"GEOID": "origin_cbg", "pm25_avg": "pm25_origin"}),
        on=["origin_cbg", "year", "month"], how="left"
    )
    flows = flows.merge(
        ghap.rename(columns={"GEOID": "dest_cbg", "pm25_avg": "pm25_dest"}),
        on=["dest_cbg", "year", "month"], how="left"
    )

    n_before = len(flows)
    flows    = flows.dropna(subset=["pm25_origin", "pm25_dest"])
    n_dropped = n_before - len(flows)
    print(f"  Dropped {n_dropped:,} flows with missing PM2.5 "
          f"({n_dropped / n_before * 100:.1f}%)")

    flows["delta_pm25"]  = flows["pm25_dest"] - flows["pm25_origin"]
    flows["trips_worse"] = (flows["delta_pm25"] > 0).astype(int)

    print(f"  delta_pm25 : mean={flows['delta_pm25'].mean():.3f}  "
          f"std={flows['delta_pm25'].std():.3f}")

    # 4. Aggregate to CBG x month
    print("\nAggregating to CBG x month...")

    records = []
    for (origin_cbg, year, month), grp in flows.groupby(
            ["origin_cbg", "year", "month"]):
        w = grp["visitor_count"].values.astype(float)
        if w.sum() < MIN_WEIGHTED_TRIPS:
            continue

        records.append({
            "GEOID":             origin_cbg,
            "year":              year,
            "month":             month,
            "mean_delta_pm25":   np.average(grp["delta_pm25"].values,  weights=w),
            "pct_trips_worse":   np.average(grp["trips_worse"].values, weights=w) * 100,
            "mobility_adj_pm25": np.average(grp["pm25_dest"].values,   weights=w),
            "residential_pm25":  grp["pm25_origin"].iloc[0],
            "n_destinations":    grp["dest_cbg"].nunique(),
            "total_visitors":    w.sum(),
            "pct_low_count":     grp["is_low_count"].mean() * 100,
        })

    data = pd.DataFrame(records)
    data.sort_values(["GEOID", "year", "month"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    # Time index for panel models — 1 = Jan of first year, sequential
    base_year       = data["year"].min()
    data["time_id"] = (data["year"] - base_year) * 12 + data["month"]

    # COVID flags
    data["covid"]     = (
        (data["year"] == 2020) & (data["month"] >= 3)
    ).astype(int)
    data["covid_era"] = (data["year"].isin([2020, 2021])).astype(int)

    print(f"  CBG-month rows  : {len(data):,}")
    print(f"  Unique CBGs     : {data['GEOID'].nunique():,}")
    print(f"  Time periods    : {data['time_id'].nunique()} months")
    print(f"  mean_delta_pm25 : mean={data['mean_delta_pm25'].mean():.3f}  "
          f"std={data['mean_delta_pm25'].std():.3f}")
    print(f"  pct_trips_worse : {data['pct_trips_worse'].mean():.1f}%")

    # 5. Save
    out = OUT_DIR / "final_data_monthly.csv"
    data.to_csv(out, index=False)

    print(f"\n{'-' * 55}")
    print(f"  Output : {out}")
    print(f"  Rows   : {len(data):,}")
    print(f"  Cols   : {list(data.columns)}")
    print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build monthly CBG panel with PM2.5 exposure differentials"
    )
    parser.add_argument('--year', type=int, help='Single year (e.g. 2019)')
    args  = parser.parse_args()
    years = [args.year] if args.year else YEARS
    run(years)
