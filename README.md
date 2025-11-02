# GEBCO-Tsunami-Downscaler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**Production-grade bathymetry downscaling toolkit optimized for tsunami modeling.**

## Features

- 🌊 **Land-Aware Processing**: Intelligent ocean/land separation
- �� **Depth-Stratified Preservation**: Retains critical features (25%/15%/8%)
- ✅ **Validated**: Pearson r > 0.98 on GEBCO 2025
- 🔧 **CF-Compliant**: NetCDF4 output

## Quick Start
```bash
git clone https://github.com/Abhiagri10/GEBCO-Tsunami-Downscaler.git
cd GEBCO-Tsunami-Downscaler
pip install -r requirements.txt

# Run
python src/downscaler.py --input data.nc --output-dir ./outputs --coarsen-factor 8
```
## Installation
```bash
pip install -r requirements.txt
---

### 🧪 Usage
```markdown
## Usage
```python
from src.downscaler import BathymetryProcessor

config = {
    'input_file': 'gebco_input.nc',
    'coarsen_factor': 8,
    'output_dir': './outputs'
}

processor = BathymetryProcessor(config=config)
processor.load_data()
processor.process()
processor.validate()
processor.save()
---

### 🧪 Testing
```markdown
## Testing
```bash
pytest tests/ -v
---

### 📊 Algorithm
```markdown
## Algorithm

Land-aware depth-stratified coarsening:
- **Shallow** (> -50m): 25% extreme retention
- **Shelf** (-50 to -200m): 15% retention
- **Deep** (< -200m): 8% retention

See `docs/methodology.md` for details.

## Citation
```bibtex
@software{gebco_tsunami_downscaler,
  author = {Abhishek},
  title = {GEBCO-Tsunami-Downscaler},
  year = {2025},
  url = {https://github.com/Abhiagri10/GEBCO-Tsunami-Downscaler}
}

## Citation
```bibtex
@software{gebco_tsunami_downscaler,
  author = {Abhishek},
  title = {GEBCO-Tsunami-Downscaler},
  year = {2025},
  url = {https://github.com/Abhiagri10/GEBCO-Tsunami-Downscaler}
}


---

### 📄 License
```markdown
## License

MIT License - see LICENSE file.
EOF

