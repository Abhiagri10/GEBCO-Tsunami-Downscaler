#!/bin/bash
set -e

echo "Initializing git repository..."
git init
git config user.name "Abhishek"
git config user.email "abhishek@example.com"

echo "Adding remote..."
#git remote add origin https://github.com/Abhiagri10/GEBCO-Tsunami-Downscaler.git

echo "Adding files..."
git add .

echo "Committing..."
git commit -F- <<MESSAGE
Initial commit: GEBCO-Tsunami-Downscaler v1.2.0

- Complete land-aware depth-stratified coarsening algorithm
- Comprehensive test suite
- CI/CD pipeline
- Full documentation

Validated on GEBCO 2025 West Pacific dataset
Quality: Pearson r > 0.98, RMSE < 35m
MESSAGE

echo "Pushing to GitHub..."
git branch -M main
git push -u origin main

echo ""
echo "=========================================="
echo "✓ DEPLOYMENT COMPLETE!"
echo "=========================================="
echo "Repository: https://github.com/Abhiagri10/GEBCO-Tsunami-Downscaler"
echo ""
echo "Next steps:"
echo "1. Visit the repository to verify"
echo "2. Check CI/CD in Actions tab"
echo "3. Star the repository ⭐"

chmod +x push_to_github.sh