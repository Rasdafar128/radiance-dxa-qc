"""Получить и проверить фиксированный E5 — только стандартная библиотека Python.

python -m src.utils.prepare_model --from-directory artifacts/ml-final
python -m src.utils.prepare_model --verify-only
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from .. import config as C


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(destination=C.MODEL, source=None, verify_only=False):
    manifest = json.loads((destination / 'weights.json').read_text())
    checks = json.loads((destination / 'selection.json').read_text())['files_sha256']
    for relative, entry in manifest['files'].items():
        if relative not in ('member_0/encoder.pt', 'member_1/encoder.pt') or checks[relative] != entry['sha256']:
            raise ValueError('Inconsistent E5 weights manifest')
        target = destination / relative
        if target.exists():
            if target.stat().st_size != entry['bytes'] or digest(target) != entry['sha256']:
                raise ValueError(f'Corrupted weights: {target}; remove this file before retrying')
            continue
        if verify_only:
            raise FileNotFoundError(f'Missing weights: {target}')
        if source is None and not entry['url']:
            raise ValueError('Public weights URL is not configured; use --from-directory with the E5 bundle')
        if source is None and not entry['url'].startswith('https://'):
            raise ValueError('Weights URL must use HTTPS')
        print(f'Preparing {relative} ({entry["bytes"] / 1024**3:.2f} GiB)', flush=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=target.parent, suffix='.download', delete=False) as output:
                temporary = Path(output.name)
                with ((source / relative).open('rb') if source else urlopen(entry['url'], timeout=60)) as incoming:
                    shutil.copyfileobj(incoming, output, 8 * 1024**2)
            if temporary.stat().st_size != entry['bytes'] or digest(temporary) != entry['sha256']:
                raise ValueError(f'Checksum or size mismatch: {relative}')
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    for relative, checksum in checks.items():
        if digest(destination / relative) != checksum:
            raise ValueError(f'Model file checksum mismatch: {relative}')
    print(f'E5 verified: {len(checks)} files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=C.MODEL)
    parser.add_argument('--from-directory', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    prepare(args.model, args.from_directory, args.verify_only)
