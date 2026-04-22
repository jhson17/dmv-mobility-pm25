# 01_process_ghap.py
# Extracts monthly PM2.5 values from GHAP 1km NetCDF files
# and aggregates to DMV Census Block Group level.
#
# Input:  data/raw/ghap/GHAP_PM2.5_M1K_YYYYMM_V1.nc
# Output: data/processed/ghap_pm25_monthly.csv
#
# Columns:
#   GEOID      - 12-digit CBG FIPS
#   year       - 4-digit year
#   month      - numeric month (1-12)
#   pm25_avg   - mean PM2.5 within CBG (ug/m3)
#
# Usage:
#   python code/01_process_ghap.py
#   python code/01_process_ghap.py --year 2019
#   python code/01_process_ghap.py --year 2019 --month 7

import argparse
import tempfile
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import pygris
import rasterio
from rasterio.transform import from_bounds
from rasterstats import zonal_stats
from pathlib import Path

# ------------------------------
# PARAMETERS
# ------------------------------
GHAP_DIR = Path("data/raw/ghap")
OUT_DIR  = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS  = [2018, 2019, 2020, 2021]
MONTHS = list(range(1, 13))

DMV_COUNTIES = [
    '11001',
    '24031', '24033',
    '51013', '51059',
    '51510', '51600', '51610',
]

LAT_MAX, LAT_MIN =  39.40,  38.50
LON_MIN, LON_MAX = -77.60, -76.60


# ------------------------------
# FUNCTIONS
# ------------------------------
def load_cbgs():
    """Load and filter CBG boundaries to DMV counties."""

    parts = [pygris.block_groups(state=s, year=2019, cache=True)
             for s in ['DC', 'MD', 'VA']]
    
    cbgs = pd.concat(parts, ignore_index=True)
    cbgs = gpd.GeoDataFrame(cbgs, crs=cbgs.crs).to_crs('EPSG:4326')
    cbgs = cbgs[cbgs['GEOID'].str[:5].isin(DMV_COUNTIES)][['GEOID', 'geometry']].copy()
    cbgs.reset_index(drop=True, inplace=True)
    return cbgs


def process_month(year, month, cbgs):
    """
    Load GHAP NetCDF for one month, clip to DMV bounding box,
    extract mean PM2.5 per CBG using zonal stats.
    Returns DataFrame with one row per CBG, or None if file missing/error.
    """

    fname = f"GHAP_PM2.5_M1K_{year}{month:02d}_V1.nc"
    fpath = GHAP_DIR / fname

    if not fpath.exists():
        print(f"  [{year}-{month:02d}] File not found — skipping")
        return None

    print(f"  [{year}-{month:02d}] Processing...", end=' ')

    try:
        ds  = xr.open_dataset(fpath, engine='netcdf4')
        dmv = ds['PM2.5'].sel(
            lat=slice(LAT_MAX, LAT_MIN),
            lon=slice(LON_MIN, LON_MAX)
        )

        arr  = dmv.values.astype('float32')
        h, w = arr.shape
        lons  = dmv.coords['lon'].values
        lats  = dmv.coords['lat'].values

        transform = from_bounds(
            lons.min(), lats.min(),
            lons.max(), lats.max(),
            w, h
        )

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name

        with rasterio.open(
            tmp_path, 'w', driver='GTiff',
            height=h, width=w, count=1,
            dtype='float32', crs='EPSG:4326',
            transform=transform, nodata=np.nan
        ) as dst:
            dst.write(arr, 1)

        stats = zonal_stats(
            cbgs, tmp_path,
            stats=['mean'],
            nodata=np.nan,
            geojson_out=False,
            all_touched=True
        )

        Path(tmp_path).unlink(missing_ok=True)
        ds.close()

        result = cbgs[['GEOID']].copy()
        result['year']     = year
        result['month']    = month
        result['pm25_avg'] = [s['mean'] for s in stats]

        n_valid = result['pm25_avg'].notna().sum()
        print(f"{n_valid:,} CBGs  "
              f"mean={result['pm25_avg'].mean():.2f}  "
              f"std={result['pm25_avg'].std():.2f}  "
              f"range={result['pm25_avg'].min():.2f}-{result['pm25_avg'].max():.2f}")

        return result

    except Exception as e:
        print(f"ERROR: {e}")
        return None


# ------------------------------
# MAIN
# ------------------------------
def run(years, months):
    print("-" * 55)
    print("Processing GHAP PM2.5")
    print("-" * 55)

    cbgs    = load_cbgs()
    results = []
    skipped = []

    for year in years:
        for month in months:
            result = process_month(year, month, cbgs)
            if result is not None:
                results.append(result)
            else:
                skipped.append({'year': year, 'month': month})

    if not results:
        print("\nERROR: No data processed. Check GHAP files in data/raw/ghap/")
        return

    print("\nCombining results...")
    combined = pd.concat(results, ignore_index=True)
    combined.sort_values(['GEOID', 'year', 'month'], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    out = OUT_DIR / "ghap_pm25_monthly.csv"
    combined.to_csv(out, index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(OUT_DIR / "ghap_skipped.csv", index=False)

    print(f"\n{'-' * 55}")
    print(f"  Months processed : {len(results)}")
    print(f"  Months skipped   : {len(skipped)}")
    print(f"  Unique CBGs      : {combined['GEOID'].nunique():,}")
    print(f"  Total rows       : {len(combined):,}")
    print(f"  PM2.5 overall    : mean={combined['pm25_avg'].mean():.2f}  "
          f"std={combined['pm25_avg'].std():.2f}")
    print(f"  Output           : {out}")
    print("-" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract GHAP PM2.5 to DMV CBGs")
    parser.add_argument('--year',  type=int, help='Single year (e.g. 2019)')
    parser.add_argument('--month', type=int, help='Single month (e.g. 7)')
    args = parser.parse_args()

    years  = [args.year]  if args.year  else YEARS
    months = [args.month] if args.month else MONTHS

    run(years, months)
