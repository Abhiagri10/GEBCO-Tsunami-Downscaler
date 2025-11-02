#!/usr/bin/env python3
"""
GEBCO-Tsunami-Downscaler: Core processing module
Author: Abhishek
Version: 1.2.0
"""

from __future__ import annotations
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

DEFAULT_CONFIG = {
    'input_file': None,
    'coastline_shapefile': None,
    'source_resolution_arcsec': 15,
    'target_resolution_arcmin': 2.0,
    'coarsen_factor': 8,
    'output_dir': './outputs',
    'quality_threshold_r': 0.95,
    'extreme_shallow': 0.25,
    'extreme_shelf': 0.15,
    'extreme_deep': 0.08,
    'extreme_land': 0.12,
    'coastal_boost': 0.15,
    'land_threshold': 0.5,
    'coastal_buffer': 5,
    'column_nan_threshold': 0.15,
    'preserve_full_terrain': True,
    'compression_level': 6,
    'chunk_size': 500
}

def progress_bar(current: int, total: int, prefix: str = '', bar_len: int = 50):
    pct = current / total
    bar = '█' * int(pct * bar_len) + '░' * (bar_len - int(pct * bar_len))
    sys.stdout.write(f'\r{prefix}: [{bar}] {pct*100:.1f}%')
    sys.stdout.flush()
    if current == total:
        sys.stdout.write('\n')

def get_coords(da: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    lon_keys = ['lon', 'longitude', 'x', 'X']
    lat_keys = ['lat', 'latitude', 'y', 'Y']
    lon = next((da.coords[k].values for k in lon_keys if k in da.coords), None)
    lat = next((da.coords[k].values for k in lat_keys if k in da.coords), None)
    if lon is None or lat is None:
        raise ValueError(f"Coordinates not found. Available: {list(da.coords.keys())}")
    return np.asarray(lon), np.asarray(lat)

def fix_column_nans(data: np.ndarray, threshold: float = 0.15) -> Tuple[np.ndarray, bool, int]:
    col_nan_frac = np.isnan(data).mean(axis=0)
    bad_cols = np.where(col_nan_frac > threshold)[0]
    if bad_cols.size == 0:
        nans = np.isnan(data).sum()
        msg = "  ✓ No problematic columns"
        if nans > 0:
            msg += f" ({nans:,} NaNs present)"
        print(msg)
        return data, False, 0
    print(f"  ⚠ Inpainting {bad_cols.size:,} columns...")
    data_fixed = data.copy().astype(np.float32)
    for i, c in enumerate(bad_cols):
        if i % 100 == 0:
            progress_bar(i, len(bad_cols), '    Progress')
        col = data_fixed[:, c]
        valid = ~np.isnan(col)
        if valid.sum() >= 2:
            data_fixed[:, c] = np.interp(np.arange(len(col)), np.where(valid)[0], col[valid])
        else:
            window = min(5, c, data.shape[1] - c - 1)
            if window > 0:
                data_fixed[:, c] = np.nanmean(data_fixed[:, max(0, c-window):min(data.shape[1], c+window+1)], axis=1)
    progress_bar(len(bad_cols), len(bad_cols), '    Progress')
    print(f"  ✓ NaN reduction: {np.isnan(data).sum():,} → {np.isnan(data_fixed).sum():,}")
    return data_fixed, True, len(bad_cols)

def load_coastline_mask(lon: np.ndarray, lat: np.ndarray, shapefile: str, buffer: int = 5) -> Tuple[Optional[np.ndarray], bool]:
    try:
        gdf = gpd.read_file(shapefile)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326) if gdf.crs else gdf.set_crs(epsg=4326)
        lat_ordered = lat if lat[0] < lat[-1] else lat[::-1]
        flip_needed = lat[0] > lat[-1]
        gdf_clip = gdf.cx[float(lon.min()):float(lon.max()), float(lat_ordered.min()):float(lat_ordered.max())]
        if len(gdf_clip) == 0:
            return None, False
        ny, nx = len(lat_ordered), len(lon)
        transform = from_bounds(float(lon.min()), float(lat_ordered.min()), float(lon.max()), float(lat_ordered.max()), nx, ny)
        coastal_line = features.rasterize(((g, 1) for g in gdf_clip.geometry if g is not None), out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8)
        coastal_zone = binary_dilation(coastal_line, structure=np.ones((buffer*2+1, buffer*2+1)))
        if flip_needed:
            coastal_zone = coastal_zone[::-1, :]
        coverage = coastal_zone.sum() / coastal_zone.size * 100
        print(f"  ✓ Coastline: {len(gdf_clip)} segments, {coverage:.2f}% coverage")
        return coastal_zone.astype(np.float32), flip_needed
    except Exception as e:
        print(f"  ⚠ Coastline load failed: {e}")
        return None, False

def smart_coarsen(elev_da: xr.DataArray, factor: int, land_thresh: float = 0.5, extreme_factors: Optional[Dict[str, float]] = None) -> Tuple[xr.DataArray, np.ndarray]:
    if extreme_factors is None:
        extreme_factors = {'shallow': 0.25, 'shelf': 0.15, 'deep': 0.08, 'land': 0.12}
    arr = elev_da.values.astype(np.float32)
    ny, nx = arr.shape
    py, px = (ny // factor) * factor, (nx // factor) * factor
    arr = arr[:py, :px]
    blocks = arr.reshape(py//factor, factor, px//factor, factor)
    land_mask = blocks >= 0
    land_frac = land_mask.sum(axis=(1,3)) / (factor * factor)
    coarsened = np.zeros((py//factor, px//factor), dtype=np.float32)
    print(f"  Processing {py//factor} × {px//factor} blocks...")
    for i in range(py//factor):
        if i % 50 == 0:
            progress_bar(i, py//factor, '    Blocks')
        for j in range(px//factor):
            block = blocks[i, :, j, :]
            lf = land_frac[i, j]
            if lf >= land_thresh:
                land_vals = block[block >= 0]
                if land_vals.size > 0:
                    mean_land = np.mean(land_vals)
                    max_land = np.max(land_vals)
                    coarsened[i, j] = mean_land + extreme_factors['land'] * (max_land - mean_land)
                else:
                    coarsened[i, j] = 1.0
            elif lf > 0.1:
                ocean_vals = block[block < 0]
                if ocean_vals.size > 0:
                    mean_ocean = np.mean(ocean_vals)
                    if mean_ocean > -50:
                        coarsened[i, j] = np.percentile(ocean_vals, 35)
                    elif mean_ocean > -100:
                        coarsened[i, j] = np.percentile(ocean_vals, 40)
                    else:
                        coarsened[i, j] = mean_ocean
                else:
                    coarsened[i, j] = np.mean(block)
            else:
                ocean_vals = block[block < 0]
                if ocean_vals.size > 0:
                    mean_ocean = np.mean(ocean_vals)
                    min_ocean = np.min(ocean_vals)
                    if mean_ocean > -50:
                        factor_use = extreme_factors['shallow']
                    elif mean_ocean > -200:
                        factor_use = extreme_factors['shelf']
                    else:
                        factor_use = extreme_factors['deep']
                    coarsened[i, j] = mean_ocean + factor_use * (min_ocean - mean_ocean)
                else:
                    coarsened[i, j] = np.mean(block)
    progress_bar(py//factor, py//factor, '    Blocks')
    lat_coarse = elev_da.lat.values[::factor][:coarsened.shape[0]]
    lon_coarse = elev_da.lon.values[::factor][:coarsened.shape[1]]
    result = xr.DataArray(coarsened, dims=('lat', 'lon'), coords={'lat': lat_coarse, 'lon': lon_coarse}, name='elevation')
    land_cells = (land_frac >= land_thresh).sum()
    ocean_cells = (land_frac < land_thresh).sum()
    coastal_cells = ((land_frac > 0.1) & (land_frac < land_thresh)).sum()
    print(f"  ✓ Land: {land_cells:,} ({land_cells/coarsened.size*100:.1f}%)")
    print(f"  ✓ Ocean: {ocean_cells:,} ({ocean_cells/coarsened.size*100:.1f}%)")
    print(f"  ✓ Coastal: {coastal_cells:,} ({coastal_cells/coarsened.size*100:.1f}%)")
    return result, land_frac

@dataclass
class BathymetryProcessor:
    config: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG.copy())
    ds_orig: Optional[xr.Dataset] = None
    ds_proc: Optional[xr.DataArray] = None
    land_frac: Optional[np.ndarray] = None
    coastal_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    phase_times: list = field(default_factory=list)
    
    def load_data(self):
        print(f"{'='*70}\nPHASE 1: DATA LOADING\n{'='*70}")
        self.phase_start = time.time()
        if not os.path.exists(self.config['input_file']):
            raise FileNotFoundError(f"File not found: {self.config['input_file']}")
        ds = xr.open_dataset(self.config['input_file'], decode_times=False)
        bathy_var = next((v for v in ['elevation', 'Band1', 'z', 'depth'] if v in ds.variables and len(ds[v].dims) >= 2), None)
        if not bathy_var:
            raise ValueError(f"No bathymetry variable found")
        if bathy_var != 'elevation':
            ds = ds.rename({bathy_var: 'elevation'})
        for old, new in [(['lat', 'latitude', 'y'], 'lat'), (['lon', 'longitude', 'x'], 'lon')]:
            coord = next((c for c in old if c in ds.coords), None)
            if coord and coord != new:
                ds = ds.rename({coord: new})
        self.ds_orig = ds
        lon, lat = get_coords(ds)
        elev = ds.elevation
        self.metadata['original_shape'] = elev.shape
        print(f"✓ Loaded: {elev.shape}")
        elapsed = time.time() - self.phase_start
        self.phase_times.append(("Load", elapsed))
        print(f"  → Load: {elapsed:.1f}s\n")
    
    def process(self):
        print(f"{'='*70}\nPHASE 2: PROCESSING\n{'='*70}")
        self.phase_start = time.time()
        factor = self.config['coarsen_factor']
        arr = self.ds_orig.elevation.values.astype(np.float32)
        arr, _, _ = fix_column_nans(arr, self.config['column_nan_threshold'])
        self.ds_orig['elevation'].values = arr
        extreme_factors = {k: self.config[f'extreme_{k}'] for k in ['shallow', 'shelf', 'deep', 'land']}
        self.ds_proc, self.land_frac = smart_coarsen(self.ds_orig.elevation, factor, self.config['land_threshold'], extreme_factors)
        elapsed = time.time() - self.phase_start
        self.phase_times.append(("Process", elapsed))
        print(f"  → Process: {elapsed:.1f}s\n")
    
    def validate(self) -> float:
        print(f"{'='*70}\nPHASE 3: VALIDATION\n{'='*70}")
        self.phase_start = time.time()
        factor = self.config['coarsen_factor']
        ref, _ = smart_coarsen(self.ds_orig.elevation, factor, self.config['land_threshold'])
        proc, ref_vals = self.ds_proc.values, ref.values
        valid = ~(np.isnan(proc) | np.isnan(ref_vals))
        if valid.sum() == 0:
            return 0.0
        p, r = proc[valid], ref_vals[valid]
        rmse = np.sqrt(np.mean((p - r)**2))
        pearson_r, _ = pearsonr(p, r) if len(p) > 1 else (0.0, 0.0)
        print(f"  RMSE: {rmse:.2f} m | r: {pearson_r:.6f}")
        self.metadata['validation'] = {'rmse': float(rmse), 'pearson_r': float(pearson_r)}
        elapsed = time.time() - self.phase_start
        self.phase_times.append(("Validate", elapsed))
        print(f"  → Validate: {elapsed:.1f}s\n")
        return pearson_r
    
    def save(self, filename: Optional[str] = None) -> str:
        print(f"{'='*70}\nPHASE 4: SAVING\n{'='*70}")
        self.phase_start = time.time()
        os.makedirs(self.config['output_dir'], exist_ok=True)
        if filename is None:
            filename = f"gebco_downscaled_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
        output_path = os.path.join(self.config['output_dir'], filename)
        ds_out = self.ds_proc.to_dataset(name='elevation')
        ds_out.attrs.update({'title': 'GEBCO Downscaled', 'version': '1.2.0', 'Conventions': 'CF-1.8'})
        encoding = {'elevation': {'zlib': True, 'complevel': 4, 'dtype': 'float32', '_FillValue': -9999.0}}
        ds_out.to_netcdf(output_path, encoding=encoding, format='NETCDF4')
        print(f"✓ Saved: {output_path}")
        elapsed = time.time() - self.phase_start
        self.phase_times.append(("Save", elapsed))
        print(f"  → Save: {elapsed:.1f}s\n")
        return output_path

def main(cfg: Optional[Dict[str, Any]] = None):
    config = {**DEFAULT_CONFIG, **(cfg or {})}
    processor = BathymetryProcessor(config=config)
    processor.load_data()
    processor.process()
    processor.validate()
    return processor.save()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GEBCO-Tsunami-Downscaler')
    parser.add_argument('--input', required=True, help='Input NetCDF')
    parser.add_argument('--output-dir', default='./outputs', help='Output directory')
    parser.add_argument('--coarsen-factor', type=int, default=8, help='Coarsening factor')
    args = parser.parse_args()
    cfg = {'input_file': args.input, 'output_dir': args.output_dir, 'coarsen_factor': args.coarsen_factor}
    main(cfg)

