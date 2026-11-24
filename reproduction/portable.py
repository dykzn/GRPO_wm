#!/usr/bin/env python3
"""Restore and run the self-contained Linux/x86_64 Spatial GRPO release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe release path: {relative}')
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Release path escapes its root: {relative}')
    return target


def verify(root, *, full=True):
    manifest = read(root / 'release_manifest.json')
    for row in manifest['files']:
        path = inside(root, row['path'])
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError(f'Missing or incomplete release file: {row["path"]}')
        if full and checksum(path) != row['sha256']:
            raise ValueError(f'Checksum mismatch: {row["path"]}')
    print(f'Verified {len(manifest["files"])} release files (SHA256={full})', flush=True)
    return manifest


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    # Reading through stdin also supports local Hub cache symlinks: zstd's
    # command-line filename handling otherwise silently skips symlink inputs.
    with archive.open('rb') as compressed:
        process = subprocess.Popen(['zstd', '-q', '-d', '-c'], stdin=compressed, stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as tar:
            for member in tar:
                target = inside(destination, member.name)
                # Release archives contain only real files and directories;
                # external symlinks are materialized by the exporter.
                if not (member.isdir() or member.isfile()):
                    raise ValueError(f'Unsupported archive member: {member.name}')
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output, length=8 * 1024**2)
                target.chmod(member.mode & 0o777)
                os.utime(target, (member.mtime, member.mtime))
        # tar stops at its end marker before zstd necessarily finishes writing
        # the final padding. Drain the pipe before wait(), including on hosts
        # whose default pipe capacity is smaller than tar's record padding.
        while process.stdout.read(1024 * 1024):
            pass
        if process.wait():
            raise RuntimeError(f'zstd extraction failed: {archive}')
    finally:
        if process.stdout:
            process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def link(path, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        raise FileExistsError(f'Refusing to replace a real file: {path}')
    path.symlink_to(os.path.relpath(target, path.parent))


def relocate_pool(root):
    source = root / 'config/kir_manifest.original.json'
    value = read(source)
    original_digest = value['pool_sha256']
    raw = root / 'data/kir_sources'
    old_source = value['source']
    def replace(item):
        if isinstance(item, dict):
            result = {k: replace(v) for k, v in item.items()}
            if {'path', 'size', 'mtime_ns'} <= result.keys():
                path = Path(result['path'])
                if not path.is_file() or path.stat().st_size != result['size']:
                    raise ValueError(f'KIR source missing or changed: {path.name}')
                result['mtime_ns'] = path.stat().st_mtime_ns
            return result
        if isinstance(item, list):
            return [replace(v) for v in item]
        if isinstance(item, str):
            if item == old_source:
                return str(raw)
            if item.startswith(old_source.rstrip('/') + '/'):
                return str(raw / Path(item).name)
        return item
    value = replace(value)
    value.pop('pool_sha256')
    value['relocated_from_pool_sha256'] = original_digest
    value['pool_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
    output = root / 'config/kir_pool'
    output.mkdir(exist_ok=True)
    shutil.copyfile(root / 'config/kir_indices.npz', output / 'indices.npz')
    write(output / 'manifest.json', value)


def configure(root):
    template = (root / 'config/spatial.reference.yaml.in').read_text()
    (root / 'config/spatial.reference.yaml').write_text(template.replace('@BUNDLE@', str(root)))
    baseline = root / 'assets/wan_baseline'
    wan = root / 'weights/wan_step1000'
    deployment = root / 'config/updated_wan_deployment'
    for name in ['Wan2.2_VAE.pth', 'resnet_rm.pth']:
        link(deployment / name, baseline / name)
    link(deployment / 'model-00001.safetensors', wan / 'model-00001.safetensors')
    link(deployment / 'dataset', root / 'data/wan_initial')
    link(baseline / 'dataset', root / 'data/wan_initial')
    link(wan / 'Wan2.2_VAE.pth', baseline / 'Wan2.2_VAE.pth')
    pair = read(root / 'config/model_pair.original.json')
    pair.update(wan=str(wan), cosmos=str(root / 'weights/cosmos_step2000/iter_000002000'),
                manifest=str(root / 'evidence/world_model_training_manifest.json'))
    write(root / 'config/model_pair.json', pair)
    relocate_pool(root)


def runtime_wrapper(root, role, minor):
    """Set role-specific shared-library paths before the interpreter starts."""
    if (role, minor) not in {('rlinf', '3.11'), ('cosmos', '3.10')}:
        raise ValueError('Unknown binary runtime')
    script = r'''#!/usr/bin/env bash
set -euo pipefail
env_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
bundle_root="$(cd -- "$env_root/../.." && pwd -P)"
library_dirs=()
cudnn_root=""
cuda_root=""
for layer in "$bundle_root/runtime/@ROLE@"/packages_*; do
  for directory in "$layer"/nvidia/*/lib "$layer/torch/lib" "$layer/transformer_engine" "$layer/transformer_engine/wheel_lib"; do
    if [[ -d "$directory" ]]; then library_dirs+=("$directory"); fi
  done
  if [[ -z "$cudnn_root" && -d "$layer/nvidia/cudnn" ]]; then cudnn_root="$layer/nvidia/cudnn"; fi
  if [[ -z "$cuda_root" && -d "$layer/nvidia/cuda_nvrtc" ]]; then cuda_root="$layer/nvidia/cuda_nvrtc"; fi
done
IFS=: read -r -a inherited_dirs <<< "${LD_LIBRARY_PATH:-}"
for directory in "${inherited_dirs[@]}"; do
  if [[ -n "$directory" && "$directory" != "$bundle_root/runtime/"* ]]; then library_dirs+=("$directory"); fi
done
printf -v LD_LIBRARY_PATH '%s:' "${library_dirs[@]}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH%:}"
if [[ -n "$cudnn_root" ]]; then export CUDNN_HOME="$cudnn_root" CUDNN_PATH="$cudnn_root"; fi
if [[ -n "$cuda_root" ]]; then export CUDA_HOME="$cuda_root" CUDA_PATH="$cuda_root"; fi
exec "$env_root/bin/python@MINOR@" "$@"
'''
    path = root / 'envs' / role / 'bin/python'
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_text(script.replace('@ROLE@', role).replace('@MINOR@', minor))
    path.chmod(0o755)


def restore(root, quick=False, refresh_source=False):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('The binary runtime release targets Linux x86_64')
    manifest = verify(root, full=not quick)
    for row in manifest['archives']:
        receipt = root / '.restore' / (Path(row['path']).name + '.json')
        refresh = refresh_source and row['path'].startswith('source/')
        if not refresh and receipt.exists() and read(receipt).get('sha256') == row['sha256']:
            continue
        print('Extracting', row['path'], flush=True)
        extract(root / row['path'], inside(root, row['destination']))
        write(receipt, dict(sha256=row['sha256']))
    runtime = read(root / 'config/runtime_layout.json')
    for role, cfg in runtime.items():
        interpreter = root / cfg['python']
        env = root / 'envs' / role
        subprocess.run([str(interpreter), '-m', 'venv', '--without-pip', '--copies', str(env)], check=True)
        site = env / 'lib' / ('python' + cfg['python_minor']) / 'site-packages'
        site.mkdir(parents=True, exist_ok=True)
        paths = [str(root / p) for p in cfg['import_paths']]
        (site / 'grpo_release.pth').write_text('\n'.join(paths) + '\n')
        runtime_wrapper(root, role, cfg['python_minor'])
    configure(root)
    write(root / '.restore/ready.json', dict(root=str(root), release=manifest['release_id']))
    print('Restored. Next: python reproduction/portable.py check', flush=True)


def environment(root, output):
    project = root / 'project'
    env = dict(os.environ)
    env.update(PROJECT_ROOT=str(project),
        RLINF_BASE_ROOT=str(root / 'sources/rlinf_backend'),
        RLINF_ENV=str(root / 'envs/rlinf'),
        COSMOS_PYTHON=str(root / 'envs/cosmos/bin/python'),
        COSMOS_BASE_ROOT=str(root / 'sources/cosmos_base'),
        LIBERO_ROOT=str(root / 'sources/libero'),
        LIBERO_BENCHMARK_ROOT=str(root / 'sources/libero/libero/libero'),
        OPENPI_ROOT=str(root / 'sources/openpi'),
        DIFFSYNTH_ROOT=str(root / 'sources/diffsynth'),
        WORLDPILOT_ROOT=str(project / 'third_party/WorldPilot'),
        PI05_CHECKPOINT_ROOT=str(root / 'weights/pi05/global_step_12000'),
        WAN_SPATIAL_CHECKPOINT=str(root / 'config/updated_wan_deployment'),
        COSMOS_WRIST_CHECKPOINT=str(root / 'weights/cosmos_step2000/iter_000002000'),
        COSMOS_VAE=str(root / 'assets/cosmos/tokenizer.pth'),
        COSMOS_DATASET_STATS=str(root / 'assets/cosmos/libero_dataset_statistics.json'),
        COSMOS_T5_EMBEDDINGS=str(root / 'assets/cosmos/libero_t5_embeddings.pkl'),
        TOKENIZER_DIR=str(root / 'assets/tokenizer'),
        SPATIAL_KIR_FRAME_POOL=str(root / 'config/kir_pool'),
        SPATIAL_WOVR_MODEL_PAIR=str(root / 'config/model_pair.json'),
        SPATIAL_WOVR_BASELINE_WAN=str(root / 'assets/wan_baseline'),
        SPATIAL_WOVR_BASELINE_COSMOS=str(root / 'weights/cosmos_step2000/iter_000002000'),
        SPATIAL_WOVR_RESOLVED_TEMPLATE=str(root / 'config/spatial.reference.yaml'),
        SPATIAL_WOVR_LAUNCHER=str(root / 'reproduction/launcher.py'),
        SPATIAL_WOVR_OUTPUT=str(output),
        SPATIAL_WOVR_RUNTIME=str(root / 'runs/runtime'),
        SPATIAL_WOVR_TMP_ROOT=os.environ.get('SPATIAL_WOVR_TMP_ROOT', '/tmp'),
        SPATIAL_WOVR_GPUS=os.environ.get('SPATIAL_WOVR_GPUS', '0,1,2,3,4,5,6,7'),
        SPATIAL_COSMOS_PORT=os.environ.get('SPATIAL_COSMOS_PORT', '18759'))
    for key in ['PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
                'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'WM_UPDATE_HANDOFF_ROOT',
                'WM_UPDATE_HANDOFF_OWNER', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN']:
        env.pop(key, None)
    return env


def select_model_pair(root, env, pair_path):
    """Select updated model weights while retaining the frozen VAE and assets."""
    pair_path = pair_path.resolve()
    pair = read(pair_path)
    if pair.get('schema') != 'spatial-wm-pair-v1':
        raise ValueError('Expected a packaged world-model pair.json')
    wan, cosmos = Path(pair['wan']), Path(pair['cosmos'])
    if not wan.is_absolute() or not cosmos.is_absolute():
        raise ValueError('Package the model pair using an absolute --output directory')
    baseline = root / 'assets/wan_baseline'
    if (wan / 'Wan2.2_VAE.pth').resolve() != (baseline / 'Wan2.2_VAE.pth').resolve():
        raise ValueError('The updated pair must retain the frozen Wan VAE')
    deployment = root / 'runs/wm_deployments' / checksum(pair_path)[:16]
    link(deployment / 'model-00001.safetensors', wan / 'model-00001.safetensors')
    for name in ['Wan2.2_VAE.pth', 'resnet_rm.pth']:
        link(deployment / name, baseline / name)
    link(deployment / 'dataset', root / 'data/wan_initial')
    env.update(SPATIAL_WOVR_MODEL_PAIR=str(pair_path),
        WAN_SPATIAL_CHECKPOINT=str(deployment), COSMOS_WRIST_CHECKPOINT=str(cosmos))
    return pair


def runtime_compile_checks(root, env, *, triton=False):
    """Fail before GPU services start if either bundled Python cannot JIT."""
    directory = root / 'runs/compile_checks'
    directory.mkdir(parents=True, exist_ok=True)
    for role, minor in [('rlinf', '311'), ('cosmos', '310')]:
        command = [str(root / f'envs/{role}/bin/python'),
                   str(root / 'reproduction/compile_smoke.py'),
                   '--runtime-prefix', str(root / f'runtime/python{minor}'),
                   '--output', str(directory / f'{role}.json')]
        if triton:
            command.append('--triton')
        log = directory / f'{role}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, env=dict(env, CUDA_VISIBLE_DEVICES=''),
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=240)
        if result.returncode:
            print(log.read_text(errors='replace')[-12000:], file=sys.stderr)
            raise RuntimeError(f'{role} runtime compile check failed (exit={result.returncode}): {log}')
        print(f'{role} Python ABI' + (' / fresh Triton cuda_utils' if triton else '')
              + f' compile passed: {log}', flush=True)


def run(root, args):
    ready = root / '.restore/ready.json'
    if not ready.exists() or read(ready).get('root') != str(root):
        raise RuntimeError('Run restore first; rerun restore after moving the release directory')
    output = (args.output or root / 'runs' / args.command).resolve()
    if args.command != 'check' and output.exists():
        raise FileExistsError(f'Choose a new run output: {output}')
    env = environment(root, output)
    runtime_compile_checks(root, env, triton=getattr(args, 'jit_smoke', False))
    selected = None
    if getattr(args, 'model_pair', None):
        selected = select_model_pair(root, env, args.model_pair)
    project = root / 'project'
    command = ['bash', str(project / 'experiments/run_spatial_wovr_shared_pool_8gpu.sh'),
               'rollout.model.is_lora=false']
    if args.command == 'check':
        env['CUDA_VISIBLE_DEVICES'] = ''
        command += ['--check-assets']
    else:
        target = args.target or (int(args.resume or 0) + args.steps)
        command += [f'runner.max_steps={target}', f'runner.max_epochs={target}',
                    'runner.logger.experiment_name=spatial_reproduction',
                    'runner.save_interval=5', 'runner.val_check_interval=5']
        if args.command == 'eval':
            checkpoint = root / 'weights/policy' / f'global_step_{args.checkpoint}'
            if not (checkpoint / 'actor/model_state_dict/full_weights.pt').is_file():
                raise FileNotFoundError(checkpoint)
            # Keep the validated training-side initialization and both ends of
            # weight_syncer. eval_train.py replaces run(), so no optimization
            # or training rollout is entered despite this internal flag.
            command += ['runner.only_eval=false', 'runner.resume_dir=null', 'runner.ckpt_path=null',
                        f'actor.model.model_path={checkpoint}', f'rollout.model.model_path={checkpoint}']
            env['SPATIAL_REPRO_EVAL_STEP'] = str(args.checkpoint)
        elif args.resume:
            checkpoint = root / 'weights/policy' / f'global_step_{args.resume}'
            if not (checkpoint / 'actor/dcp_checkpoint/.metadata').is_file():
                raise ValueError('Resume needs a full DCP checkpoint; use policy step40, step50 or step60')
            if target <= args.resume:
                raise ValueError('The target must exceed the resumed step')
            command += [f'runner.resume_dir={checkpoint}']
        else:
            command += ['runner.resume_dir=null', 'runner.ckpt_path=null']
        saves = [s for s in range(int(args.resume or 0) + 1, target + 1) if s % 10 == 0 or s == target]
        command += ['actor.fsdp_config.save_training_state_at_steps=' + json.dumps(saves, separators=(',', ':'))]
        write(root / 'runs' / f'launch_{args.command}.json', dict(command=command,
            output=str(output), source_policy_step=args.checkpoint if args.command == 'eval' else args.resume,
            model_pair=env['SPATIAL_WOVR_MODEL_PAIR'],
            wan_step=selected['wan_step'] if selected else 1000,
            cosmos_step=selected['cosmos_step'] if selected else 2000))
    print('Command:', ' '.join(command), flush=True)
    return subprocess.call(command, env=env, cwd=project)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['restore', 'verify', 'check', 'train', 'eval'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--quick', action='store_true', help='Only check sizes, without hashing payloads')
    parser.add_argument('--refresh-source', action='store_true',
                        help='With restore: re-extract all source layers, replacing manual source edits')
    parser.add_argument('--jit-smoke', action='store_true',
                        help='With check: freshly compile Triton cuda_utils in both runtimes (no GPU context)')
    parser.add_argument('--resume', type=int, choices=[40, 50, 60])
    parser.add_argument('--checkpoint', type=int, choices=[40, 45, 50, 60], default=60)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--target', type=int)
    parser.add_argument('--model-pair', type=Path, help='Use the newly packaged world-model pair.json')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.jit_smoke and args.command != 'check':
        parser.error('--jit-smoke is used with check')
    if args.refresh_source and args.command != 'restore':
        parser.error('--refresh-source is used with restore')
    root = args.root.resolve()
    if args.command == 'restore':
        restore(root, args.quick, args.refresh_source)
    elif args.command == 'verify':
        verify(root, full=not args.quick)
    else:
        return run(root, args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
