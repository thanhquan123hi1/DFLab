"""Create a standalone training environment without changing Colab's kernel packages."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-dir', type=Path, default=ROOT / '.venv-colab')
    args = parser.parse_args()
    if not (3, 8) <= sys.version_info[:2] <= (3, 12):
        parser.error('Use Python 3.8-3.12.')
    target = args.env_dir.resolve()
    if target.exists() and not (target / 'pyvenv.cfg').exists():
        parser.error(f'{target} exists and is not a virtual environment; choose another directory.')
    env = dict(os.environ)
    for name in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_PREFIX', 'PIP_USER'):
        env.pop(name, None)
    env['PYTHONNOUSERSITE'] = '1'
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(target)
    python = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    def run(*arguments):
        subprocess.run([str(python), '-I', *arguments], cwd=ROOT, env=env, check=True)
    req_file = ROOT / 'requirements.txt'
    run('-m', 'pip', 'install', '-r', str(req_file),
        '--extra-index-url', 'https://download.pytorch.org/whl/cu113')
    run('-m', 'pip', 'check')
    run(str(ROOT / 'scripts/check_environment.py'))
    print(f'Environment ready. Train with: {python} -I training/train.py', flush=True)


if __name__ == '__main__':
    main()
