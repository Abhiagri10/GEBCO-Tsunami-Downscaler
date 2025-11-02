# Methodology

## Land-Aware Depth-Stratified Coarsening

Physics-informed downscaling for tsunami modeling.

### Algorithm

1. **Depth-Stratified Preservation**
   - Shallow (> -50m): 25% extreme retention
   - Shelf (-50 to -200m): 15% retention
   - Deep (< -200m): 8% retention

2. **Land-Ocean Separation**
   - Prevents topographic mixing
   - Preserves coastal features

3. **Validation**
   - Pearson r > 0.95
   - RMSE < 50m

### References

- GEBCO: https://www.gebco.net/
- xarray: https://docs.xarray.dev/
