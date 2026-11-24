"""Use the unchanged training launcher and an explicit evaluation-only driver."""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'project/experiments'))
import spatial_wovr_pair_launch as pair


def main():
    if os.environ.get('SPATIAL_REPRO_EVAL_STEP'):
        original_train = pair.native.train

        def evaluate(*args, **kwargs):
            original_popen = pair.native.subprocess.Popen

            def start(command, *arguments, **options):
                if isinstance(command, list) and len(command) > 1 and Path(command[1]).name == 'train_spatial_wovr.py':
                    command = [command[0], str(ROOT / 'reproduction/eval_train.py'), *command[2:]]
                return original_popen(command, *arguments, **options)

            pair.native.subprocess.Popen = start
            try:
                return original_train(*args, **kwargs)
            finally:
                pair.native.subprocess.Popen = original_popen

        pair.native.train = evaluate
    return pair.main()


if __name__ == '__main__':
    raise SystemExit(main())
