#!/usr/bin/env python3
"""Automated GitHub Repository Creator"""

from github import Github, GithubException
import os
import time

TOKEN = os.environ.get('GITHUB_TOKEN', '')
REPO_NAME = "GEBCO-Tsunami-Downscaler"

print("="*70)
print("CREATING GITHUB REPOSITORY")
print("="*70)

g = Github(TOKEN)
user = g.get_user()
print(f"✓ User: {user.login}\n")

# Create repository
try:
    repo = user.create_repo(
        name=REPO_NAME,
        description="Production-grade bathymetry downscaling for tsunami modeling",
        private=False,
        has_issues=True,
        auto_init=False
    )
    print(f"✓ Created: {repo.html_url}")
except GithubException as e:
    if e.status == 422:
        print(f"⚠ Repository exists, using it...")
        repo = user.get_repo(REPO_NAME)
    else:
        raise

print(f"\n✓ Repository ready: {repo.clone_url}")
print("\nNext: Run 'bash push_to_github.sh' to upload files")
