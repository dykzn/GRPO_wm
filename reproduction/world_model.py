#!/usr/bin/env python3
"""Run the existing collector/update tools with the restored bundle's paths."""
import argparse
import os
from pathlib import Path
import sys

from portable import environment, read, select_model_pair

ENTRIES = {
    'check': 'wm_update.py',
    'collect': 'wm_update_reserved_collect.py',
    'prepare': 'wm_update.py',
    'inspect': 'wm_update.py',
    'train': 'wm_update_until_cosmos.py',
    'package': 'wm_update_until_cosmos.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument('command', choices=list(ENTRIES))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--init-pair', type=Path, help='Initialize this update from a previously packaged pair')
    if len(sys.argv) == 1 or sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        return 0
    args, native_args = parser.parse_known_args()
    root = args.root.resolve()
    ready = root / '.restore/ready.json'
    if not ready.is_file() or read(ready).get('root') != str(root):
        raise RuntimeError('Run reproduction/portable.py restore and check first')
    env = environment(root, root / 'runs/wm_update')
    if args.init_pair:
        select_model_pair(root, env, args.init_pair)
    env.update(PYTHONPATH=os.pathsep.join([str(root / 'project'), str(root / 'project/experiments')]),
        WM_UPDATE_RESERVE_MEMORY='1', WM_UPDATE_MEMORY_HEADROOM_GIB='3.5')
    if args.command == 'check':
        env['CUDA_VISIBLE_DEVICES'] = ''
    python = root / 'envs/rlinf/bin/python'
    command = [str(python), str(root / 'project/experiments' / ENTRIES[args.command]),
        args.command, *native_args]
    os.execve(str(python), command, env)


if __name__ == '__main__':
    raise SystemExit(main())
