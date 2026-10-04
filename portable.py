"""Project-local setup and launch: Python 3.11, Windows/macOS/Linux."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent


def check_data(root=ROOT):
    missing = []
    count = 0
    for filename, folder in [('train_1.csv', 'train_images'), ('valid.csv', 'val_images'), ('test.csv', 'test_images')]:
        base = root / 'dataset' / 'aptos2019'
        path = base / filename
        if not path.is_file():
            missing.append(str(path))
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                count += 1
                image = base / folder / (row['id_code'] + '.png')
                if not image.is_file():
                    missing.append(str(image))
    return count, missing


def relocated_path(value, root):
    """Map persisted project paths from either OS to this copy, if present."""
    if not isinstance(value, str):
        return value
    normalized = value.replace('\\', '/')
    for directory in ('dataset', 'artifacts'):
        marker = directory + '/'
        if normalized.startswith(marker):
            suffix = normalized
        elif '/' + marker in normalized:
            suffix = directory + '/' + normalized.split('/' + marker, 1)[1]
        else:
            continue
        candidate = (root / suffix).resolve()
        if candidate.is_relative_to(root.resolve()) and candidate.exists():
            return str(candidate)
    return value


def repair_artifact_paths(root=ROOT):
    """Repair only stored references, preserving model weights and metrics."""
    changed = 0

    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return relocated_path(value, root)

    for path in (root / 'artifacts').rglob('*'):
        if path.suffix not in ('.csv', '.json') or not path.is_file():
            continue
        original = path.read_text(encoding='utf-8-sig')
        if path.suffix == '.json':
            value = json.loads(original)
            updated = relocate(value)
            # JSON serialization also avoids NaN equality causing false changes.
            if json.dumps(value, sort_keys=True) == json.dumps(updated, sort_keys=True):
                continue
            content = json.dumps(updated, indent=2) + '\n'
        else:
            import io
            rows = list(csv.reader(io.StringIO(original)))
            updated = [[relocated_path(v, root) for v in row] for row in rows]
            if rows == updated:
                continue
            output = io.StringIO(newline='')
            csv.writer(output).writerows(updated)
            content = output.getvalue()
        partial = path.with_suffix(path.suffix + '.relocating')
        partial.write_text(content, encoding='utf-8')
        partial.replace(path)
        changed += 1
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'launch', 'check'])
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == 'check':
        count, missing = check_data()
        checkpoints = list((ROOT / 'artifacts' / 'checkpoints').glob('*.pt'))
        print(f'Project: {ROOT}\nDataset rows: {count}; missing files: {len(missing)}; models: {len(checkpoints)}')
        if missing:
            print('First missing file:', missing[0])
        raise SystemExit(1 if missing or not checkpoints else 0)

    environment = ROOT / '.venv'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    marker = environment / 'project-location.json'
    identity = {'root': str(ROOT), 'platform': platform.system(), 'machine': platform.machine(),
                'host': platform.node()}
    if environment.exists() and (not marker.exists() or json.loads(marker.read_text()) != identity):
        raise SystemExit('This .venv belongs to another location or machine type. Remove only .venv from the copied folder, then run setup again. Do not copy environments between computers.')
    if args.action == 'setup':
        if sys.version_info[:2] != (3, 11):
            raise SystemExit('Run setup with 64-bit Python 3.11 (Windows: py -3.11 portable.py setup).')
        if not environment.exists():
            venv.EnvBuilder(with_pip=True).create(environment)
            marker.write_text(json.dumps(identity))
        subprocess.run([str(python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        torch_args = [str(python), '-m', 'pip', 'install', 'torch==2.9.1', 'torchvision==0.24.1']
        if platform.system() in ('Windows', 'Linux'):
            torch_args += ['--index-url', 'https://download.pytorch.org/whl/cpu']
        subprocess.run(torch_args, check=True)
        subprocess.run([str(python), '-m', 'pip', 'install', '--only-binary=shap', '-r', str(ROOT/'requirements.txt')], check=True)
        subprocess.run([str(python), '-m', 'pip', 'check'], check=True)
        subprocess.run([str(python), '-c', 'import torch, torchvision, cv2, captum, shap, lime, skimage; print("Dependencies ready")'], check=True)
        subprocess.run([str(python), '-m', 'ipykernel', 'install', '--sys-prefix', '--name', 'medical-dipatie', '--display-name', 'Python (Medical Dipatie)'], check=True)
        print('Setup complete. Run the launch script next.')
    else:
        if not python.is_file():
            raise SystemExit('Run setup first on this computer.')
        print('Repaired artifact files:', repair_artifact_paths())
        subprocess.run([str(python), '-m', 'notebook', str(ROOT/'project_demo_windows.ipynb')], check=True)


if __name__ == '__main__':
    main()
