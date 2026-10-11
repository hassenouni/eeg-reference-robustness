"""Build a clean source archive and its SHA-256 digest using an allowlist."""
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root/'dist'
    destination.mkdir(exist_ok=True)
    archive = destination/'eeg-reference-robustness-v0.1.0.zip'
    directories = {'src', 'tests', 'config', 'docs', 'manifests', 'scripts', '.github'}
    root_files = {'README.md', 'pyproject.toml', 'requirements.txt', 'CITATION.cff',
                  'LICENSE', 'MANIFEST.in', 'CHANGELOG.md', 'CONTRIBUTING.md', '.gitignore', '.gitattributes'}
    blocked = {'__pycache__', '.pytest_cache', '.git', 'data', 'outputs', '.venv'}
    with ZipFile(archive, 'w', ZIP_DEFLATED) as stream:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if (not path.is_file() or blocked.intersection(relative.parts)
                    or any(part.endswith('.egg-info') for part in relative.parts)):
                continue
            if relative.parts[0] not in directories and relative.as_posix() not in root_files:
                continue
            if path.suffix in {'.pyc', '.edf', '.EDF', '.npz', '.log'}:
                continue
            stream.write(path, Path(root.name)/relative)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n')
    print(archive)
    print('SHA-256: '+digest)


if __name__ == '__main__':
    main()
