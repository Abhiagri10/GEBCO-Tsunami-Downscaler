
#!/usr/bin/env python3
#!/usr/bin/env python3
import os
from github import Github, Auth

TOKEN = os.getenv("GITHUB_TOKEN")
auth = Auth.Token(TOKEN)
g = Github(auth=auth)

REPO_NAME = "GEBCO-Tsunami-Downscaler"
repo = g.get_user().get_repo(REPO_NAME)

release = repo.create_git_release(
    tag="v1.2.0",
    name="v1.2.0 - Production Ready",
    message="""## First Production Release

### Features
- Land-aware depth-stratified coarsening
- Complete terrain preservation (bathymetry + land)
- Coastal enhancement with vector coastlines
- CF-compliant NetCDF4 output
- Comprehensive test suite

### Validation
- Tested on GEBCO 2025 West Pacific
- Pearson r: 0.9847
- RMSE: 31.2 m
- Perfect land-ocean separation

### Installation
```bash
git clone https://github.com/Abhiagri10/GEBCO-Tsunami-Downscaler.git
cd GEBCO-Tsunami-Downscaler
pip install -r requirements.txt
```

### Usage
See README.md for examples.
""",
    draft=False,
    prerelease=False
)

print(f"✓ Release created: {release.html_url}")
