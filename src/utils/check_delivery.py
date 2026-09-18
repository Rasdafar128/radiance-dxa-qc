"""Проверить подготовку весов и автономный контейнер без подключения репозитория.

python -m src.utils.check_delivery --image lct26-dxa:e5-cpu --input data/test
Без --image выполняются только быстрые проверки подготовки комплекта.
"""

import argparse
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import uuid
from unittest.mock import patch

from .. import config as C
from .prepare_model import digest, prepare


def check_prepare():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        source, target = root / 'source', root / 'target'
        entries, checks = {}, {}
        for index in range(2):
            relative = f'member_{index}/encoder.pt'
            (source / relative).parent.mkdir(parents=True)
            (target / relative).parent.mkdir(parents=True)
            (source / relative).write_bytes(f'fixture-{index}'.encode())
            checks[relative] = digest(source / relative)
            entries[relative] = dict(sha256=checks[relative], bytes=(source / relative).stat().st_size, url=None)
        (target / 'weights.json').write_text(json.dumps(dict(files=entries)))
        (target / 'selection.json').write_text(json.dumps(dict(files_sha256=checks)))
        try:
            prepare(target, verify_only=True)
            raise AssertionError('Missing weights accepted')
        except FileNotFoundError:
            pass
        prepare(target, source)
        prepare(target, verify_only=True)
        damaged = target / 'member_0/encoder.pt'
        damaged.write_bytes(b'wrong')
        try:
            prepare(target, source)
            raise AssertionError('Corrupted weights accepted')
        except ValueError as error:
            assert 'Corrupted' in str(error)
        damaged.unlink()
        (source / 'member_0/encoder.pt').write_bytes(b'wrong')
        try:
            prepare(target, source)
            raise AssertionError('Wrong source accepted')
        except ValueError as error:
            assert 'mismatch' in str(error)
        assert not damaged.exists() and not list(target.rglob('*.download'))
        for index, (relative, entry) in enumerate(entries.items()):
            (source / relative).write_bytes(f'fixture-{index}'.encode())
            (target / relative).unlink(missing_ok=True)
            entry['url'] = 'https://weights.example/' + relative
        (target / 'weights.json').write_text(json.dumps(dict(files=entries)))
        def download(url, timeout):
            return (source / url.removeprefix('https://weights.example/')).open('rb')
        with patch('src.utils.prepare_model.urlopen', side_effect=download):
            prepare(target)
        assert all((target / p).stat().st_mode & 0o444 == 0o444 for p in entries)


# Выполняется stdlib-Python внутри контейнера; данные доступны только через /input.
CONTAINER_CHECK = r'''
import csv, io, json, os, time, urllib.request, urllib.error, zipfile
from pathlib import Path
started = time.monotonic()
for attempt in range(180):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3) as response:
            health = json.load(response)
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(1)
else:
    raise AssertionError('API did not become ready')
assert health['backbones'] == ['dinov3-large', 'medimageinsight'], health
assert health['device'] == 'cpu'
with urllib.request.urlopen('http://127.0.0.1:8080/openapi.json') as response:
    endpoint = json.load(response)['paths']['/batch']['post']
assert endpoint['requestBody']['content']['application/zip']['schema']['format'] == 'binary'
assert 'text/csv' in endpoint['responses']['200']['content']
assert os.getuid() == 10001
assert not Path('/app/data').exists() and not Path('/app/src/utils').exists()
assert not Path('/app/.git').exists()
assert Path('/app/models/e5/licenses/DINOv3.md').exists()
files = sorted(Path('/input').rglob('*.dcm'))
assert files, 'No DICOM test files'

def archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name, content in members:
            z.writestr(name, content)
    return stream.getvalue()

def request(content, kind='application/zip'):
    req = urllib.request.Request('http://127.0.0.1:8080/batch', data=content,
                                 headers={'Content-Type': kind})
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()

members = [(str(p.relative_to('/input')), p.read_bytes()) for p in files]
code, body = request(archive(members))
assert code == 200, body
baseline = list(csv.DictReader(io.StringIO(body)))
assert len(baseline) == len(files)
assert all(r['processing_status'] == 'Success' for r in baseline)
assert all(0 <= float(r['quality_prob']) <= 1 for r in baseline)
assert all(int(r['quality_class']) == bool(r['violation_type']) for r in baseline)
# Один второй пакет одновременно проверяет перестановку, дубль и частичный сбой.
code, mixed_body = request(archive(list(reversed(members)) + [members[0], ('broken.dcm', b'broken')]))
assert code == 200, mixed_body
mixed = list(csv.DictReader(io.StringIO(mixed_body)))
stable = lambda r: {k:v for k,v in r.items() if k != 'time_of_processing'}
assert list(map(stable, mixed[:-2])) == list(map(stable, reversed(baseline)))
assert stable(mixed[-2]) == stable(baseline[0])
assert mixed[-1]['processing_status'] == 'Failure'
assert mixed[-1]['quality_class'] == mixed[-1]['quality_prob'] == ''
for invalid in [b'invalid', archive([]), archive([('../escape.dcm', b'x')])]:
    assert request(invalid)[0] == 400
assert request(archive(members), 'text/plain')[0] == 415
print(json.dumps(dict(health=health, images=len(files), mixed_rows=len(mixed),
    offline=True, readonly=True, nonroot=True, seconds=time.monotonic()-started,
    checks='HTTP, order, duplicates, partial failure, invalid ZIP/path/content type, isolated model')))
'''


def check_container(image, inputs, output):
    name = 'lct26-check-' + uuid.uuid4().hex[:8]
    command = ['docker', 'run', '-d', '--platform', 'linux/amd64', '--name', name,
               '--network', 'none', '--read-only', '--tmpfs', '/tmp:rw,nosuid,size=1g',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
               '--mount', f'type=bind,src={inputs.resolve()},dst=/input,readonly', image]
    subprocess.run(command, check=True, capture_output=True, text=True)
    try:
        result = subprocess.run(['docker', 'exec', '-i', name, 'python', '-'], input=CONTAINER_CHECK,
                                capture_output=True, text=True, timeout=1800)
        if result.returncode:
            logs = subprocess.run(['docker', 'logs', '--tail', '40', name], capture_output=True, text=True)
            raise RuntimeError(result.stderr + logs.stderr + logs.stdout)
        report = json.loads(result.stdout)
        info = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
        assert info['HostConfig']['NetworkMode'] == 'none' and info['HostConfig']['ReadonlyRootfs']
        report.update(image=image, image_id=info['Image'], preparation_checks=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'container.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        subprocess.run(['docker', 'rm', '-f', name], check=True, capture_output=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image')
    parser.add_argument('--input', type=Path, default=C.TEST)
    parser.add_argument('--output', type=Path, default=C.ARTIFACTS / 'delivery-checks')
    args = parser.parse_args()
    check_prepare()
    if args.image:
        check_container(args.image, args.input, args.output)
    else:
        print('Preparation checks passed')
