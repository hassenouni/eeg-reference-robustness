# Preparing a GitHub release

The source folder is ready to import into a Git repository. Tests and documentation can be reviewed before publishing. No remote repository is created by this package.

1. Inspect README, methods, authors, citation metadata, and the license with the authors.
2. Install and run the technical tests.
3. Create an empty repository on GitHub, then add its actual URL as the remote.
4. Commit source files and push using your GitHub account.
5. Tag the reviewed version and attach the source archive if desired.

```bash
git init -b main
git add .
git commit -m "Add manuscript EEG methods implementation"
git remote add origin https://github.com/hassenouni/eeg-reference-robustness.git
git push -u origin main
```

Dataset files, generated outputs, virtual environments, and caches are ignored. The current license reserves authors' rights; select an open-source license if public reuse is intended. Update CITATION.cff with the final repository URL and manuscript DOI only when those identifiers exist.

Build a clean shareable source ZIP:

```bash
python scripts/package_release.py
```

The script writes the archive and SHA-256 checksum to `dist/`, excluding data, outputs, caches, and Git metadata.
