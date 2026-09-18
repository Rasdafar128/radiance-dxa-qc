"""Необязательная сегментация: python -m src.utils.prepare_segmentation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2] / 'models/anatomy'


def prepare(verify_only=False):
    entry = json.loads((ROOT / 'weights.json').read_text())
    target = ROOT / entry['file']
    if not target.is_file():
        if verify_only:
            raise FileNotFoundError(f'Missing optional segmentation weights: {target}')
        temporary = None
        try:
            with NamedTemporaryFile(dir=ROOT, suffix='.download', delete=False) as output:
                temporary = Path(output.name)
                with urlopen(entry['url'], timeout=60) as incoming:
                    shutil.copyfileobj(incoming, output, 8 * 1024**2)
            verify(temporary, entry)
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
    verify(target, entry)
    print('Radiance Anatomy weights verified')


def verify(path, entry):
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    if path.stat().st_size != entry['bytes'] or checksum != entry['sha256']:
        raise ValueError(f'Segmentation weights checksum mismatch: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    prepare(parser.parse_args().verify_only)
