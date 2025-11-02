#!/usr/bin/env python3
"""Test suite for GEBCO-Tsunami-Downscaler"""

import os
import tempfile
import pytest
import numpy as np
import xarray as xr
from src.downscaler import BathymetryProcessor, get_coords, smart_coarsen, DEFAULT_CONFIG

@pytest.fixture
def synthetic_dataset():
    nx, ny = 64, 48
    lon = np.linspace(120.0, 125.0, nx)
    lat = np.linspace(-10.0, -5.0, ny)
    LON, LAT = np.meshgrid(lon, lat)
    elev = -1000 + 10*(LON - 120) + 5*(LAT + 10)
    trench_mask = (LON > 122) & (LON < 123) & (LAT > -8) & (LAT < -7)
    elev[trench_mask] = -5000
    land_mask = (LON > 123.5) & (LAT > -7)
    elev[land_mask] = 50 + 100*np.random.rand(*elev[land_mask].shape)
    da = xr.DataArray(elev.astype(np.float32), dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}, name='elevation')
    return da.to_dataset()

def test_get_coords(synthetic_dataset):
    lon, lat = get_coords(synthetic_dataset.elevation)
    assert lon.shape[0] == 64
    assert lat.shape[0] == 48

def test_smart_coarsen_basic(synthetic_dataset):
    da = synthetic_dataset.elevation
    coarsened, land_frac = smart_coarsen(da, factor=4, land_thresh=0.5)
    expected_rows = (da.shape[0] // 4) * 4 // 4
    expected_cols = (da.shape[1] // 4) * 4 // 4
    assert coarsened.shape == (expected_rows, expected_cols)
    assert land_frac.shape == coarsened.shape

def test_processor_full_pipeline(synthetic_dataset, tmp_path):
    input_file = tmp_path / 'test_input.nc'
    synthetic_dataset.to_netcdf(input_file)
    config = DEFAULT_CONFIG.copy()
    config.update({'input_file': str(input_file), 'output_dir': str(tmp_path), 'coarsen_factor': 4})
    processor = BathymetryProcessor(config=config)
    processor.load_data()
    processor.process()
    quality = processor.validate()
    output_file = processor.save()
    assert os.path.exists(output_file)
    assert quality > 0.80

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


