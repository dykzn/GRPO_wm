"""Train Wan/Cosmos concurrently; finish both as soon as Cosmos completes.

Use the same ``train --model both`` arguments as wm_update.py. The actual Wan
workers observe a checkpoint request at their next complete optimizer update.
Cosmos and Wan keep their reusable CUDA caches until both full checkpoints and
Wan's inference export are verified. No controller is paused or replaced.

``package --selection TRAIN/selected_world_models.json --output PAIR`` then
packages exactly these weights, independently of their optimizer step counts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from queue_cosmos_first_wan8 import identity, latest_wan_checkpoint
from queue_spatial_wm_update import process_alive, utc, write_json
from spatial_followup_email import read_json
from stop_spatial_grpo import read_process


def checkpoint_path(output, model, step):
    if model == 'wan':
        return output / 'wan' / f'step_{step:06d}'
    return output / 'cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints' / f'iter_{step:09d}'


def completed_checkpoint(output, control, model, manifest, ranks):
    rows = [read_json(control / f'{model}_rank{rank}.json') for rank in range(ranks)]
    if not all(rows):
        return None
    step = rows[0].get('step')
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f'Invalid completed {model} step')
    for rank, row in enumerate(rows):
        if (row.get('rank') != rank or row.get('step') != step or row.get('model') != model
                or row.get('manifest_sha256') != manifest or row.get('reserved_gib', 0) < 1
                or not process_alive(row)):
            raise RuntimeError(f'A {model} worker did not retain its matching checkpoint CUDA cache')
    checkpoint = checkpoint_path(output, model, step)
    if any(Path(row['checkpoint']) != checkpoint for row in rows):
        raise ValueError(f'{model} ranks recorded different checkpoints')
    if model == 'wan':
        if latest_wan_checkpoint(output / 'wan', manifest) != checkpoint:
            raise ValueError('Wan has no matching complete and exported checkpoint')
        record = read_json(checkpoint / 'complete.json')
    else:
        record = read_json(checkpoint / 'wm_update_complete.json')
        required = [checkpoint / name / '.metadata' for name in ('model', 'optim', 'scheduler', 'trainer')]
        required += [checkpoint / 'master_weights.pt']
        required += [checkpoint / f'rng_rank{rank}.pt' for rank in range(ranks)]
        if not all(path.is_file() for path in required):
            raise ValueError('Cosmos full checkpoint is incomplete')
    if (record.get('step') != step or record.get('world_size') != ranks
            or record.get('identity', {}).get('manifest') != manifest):
        raise ValueError(f'{model} checkpoint does not match its completion receipts')
    return checkpoint


def wait_for_pair(args, children, token):
    from wm_update_data import load_manifest

    control = args.output / 'cosmos_first'
    manifest = load_manifest(args.manifest)['manifest_sha256']
    requested = False
    while True:
        for model, child in children.items():
            if child.poll() is not None:
                raise RuntimeError(f'{model} exited before coordinated checkpoint release: {child.returncode}')
        cosmos = completed_checkpoint(args.output, control, 'cosmos', manifest, args.ranks_per_model)
        if cosmos is not None:
            if read_json(cosmos / 'wm_update_complete.json')['step'] != args.steps:
                raise ValueError('Cosmos did not reach its configured final step')
            if not requested:
                write_json(control / 'checkpoint_request.json', dict(token=token,
                    manifest_sha256=manifest, utc=utc(),
                    reason='Cosmos finished; save latest Wan and continue GRPO directly'))
                write_json(control / 'state.json', dict(status='saving_wan', updated_utc=utc(),
                    cosmos_checkpoint=str(cosmos)))
                requested = True
            wan = completed_checkpoint(args.output, control, 'wan', manifest, args.ranks_per_model)
            if wan is not None:
                selected = dict(schema='spatial-cosmos-first-selection-v1', manifest=str(args.manifest.resolve()),
                    manifest_sha256=manifest, wan_checkpoint=str(wan), cosmos_checkpoint=str(cosmos),
                    wan_step=read_json(wan / 'complete.json')['step'], cosmos_step=args.steps,
                    policy='Cosmos finished -> latest full Wan checkpoint -> GRPO', updated_utc=utc())
                # Persist the exact model pair before either computing process
                # releases its cache. The caller can now package and run GRPO.
                write_json(args.output / 'selected_world_models.json', selected)
                return selected
        time.sleep(.5)


def supervise_until_cosmos(args, commands):
    from spatial_wovr_launch import stop_process

    if args.model != 'both' or len(commands) != 2 or args.ranks_per_model != 4:
        raise ValueError('This entry requires two four-rank groups on eight GPUs')
    if not args.reserve_memory:
        raise ValueError('Use --reserve-memory for the actual training workers')
    if args.steps == 2000 and args.save_interval != 1000:
        raise ValueError('A 2000-step Cosmos run saves only full 1000/2000 checkpoints')
    control = args.output / 'cosmos_first'
    control.mkdir(exist_ok=False)
    token = uuid.uuid4().hex
    owner = identity(read_process(os.getpid()))
    write_json(control / 'owner.json', dict(owner=owner, token=token, created_utc=utc()))
    children, streams = {}, []
    try:
        for command, environment, log in commands:
            model = command[command.index('--model')+1]
            env = dict(environment, WM_COSMOS_FIRST_CONTROL=str(control),
                       WM_COSMOS_FIRST_OWNER=json.dumps(owner), WM_COSMOS_FIRST_TOKEN=token)
            # This supervisor owns the cache handoff; never retain workers for
            # a separate outer controller that waits for a fixed Wan target.
            env.pop('WM_UPDATE_HANDOFF_ROOT', None)
            env.pop('WM_UPDATE_HANDOFF_OWNER', None)
            stream = log.open('a')
            streams.append(stream)
            children[model] = subprocess.Popen(command, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        selected = wait_for_pair(args, children, token)
        write_json(control / 'release.json', dict(utc=utc(), reason='Both selected full checkpoints verified'))
        for model, child in children.items():
            if child.wait(timeout=90):
                raise RuntimeError(f'{model} did not exit cleanly after checkpoint handoff')
        write_json(control / 'state.json', dict(selected, status='complete', updated_utc=utc()))
    finally:
        for child in children.values():
            stop_process(child)
        for stream in streams:
            stream.close()


def controlled_wan(args, native_train):
    """Arm the same native save boundary for fresh and resumed Wan training."""
    import torch
    import torch.distributed as dist
    import wm_update_wan as native
    from wm_update_data import load_manifest
    from wm_update_wan_controlled import CheckpointRequest

    args.control_dir = Path(os.environ['WM_COSMOS_FIRST_CONTROL'])
    args.control_token = os.environ['WM_COSMOS_FIRST_TOKEN']
    reset, original_load = torch.cuda.reset_peak_memory_stats, native.load_state
    manifest = load_manifest(args.manifest)['manifest_sha256']
    restored, control = 0, None

    def load(path, model, optimizer, *, identity):
        nonlocal restored
        restored = original_load(path, model, optimizer, identity=identity)
        return restored

    def begin_update():
        nonlocal control
        if control is None:
            # Called inside the initialized native loop, after selecting the
            # local CUDA device and loading any full optimizer/RNG checkpoint.
            control = CheckpointRequest(args, restored, manifest, torch, dist, reset)
        control.begin_update()

    native.load_state = load
    torch.cuda.reset_peak_memory_stats = begin_update
    try:
        return native_train(args)
    finally:
        native.load_state = original_load
        torch.cuda.reset_peak_memory_stats = reset


def retain_cache(args):
    import torch
    from wm_update_data import load_manifest

    model, rank = args.model, int(os.environ['RANK'])
    control = Path(os.environ['WM_COSMOS_FIRST_CONTROL'])
    torch.cuda.synchronize()
    write_json(control / f'{model}_rank{rank}.json', dict(identity(read_process(os.getpid())),
        model=model, rank=rank, step=args.steps,
        manifest_sha256=load_manifest(args.manifest)['manifest_sha256'],
        checkpoint=str(checkpoint_path(args.output.parent, model, args.steps)),
        reserved_gib=torch.cuda.memory_reserved()/2**30,
        allocated_gib=torch.cuda.memory_allocated()/2**30))
    owner = json.loads(os.environ['WM_COSMOS_FIRST_OWNER'])
    while not (control / 'release.json').exists():
        if not process_alive(owner):
            raise RuntimeError('The Cosmos-first checkpoint owner exited')
        time.sleep(.5)


def package_selection(args):
    from wm_update_accept import package_pair
    from wm_update_data import load_manifest

    selected = read_json(args.selection)
    if not selected or load_manifest(Path(selected['manifest']))['manifest_sha256'] != selected['manifest_sha256']:
        raise ValueError('The selected paired training manifest changed')
    wan, cosmos = Path(selected['wan_checkpoint']), Path(selected['cosmos_checkpoint'])
    for path, marker, step in ((wan, 'complete.json', selected['wan_step']),
                               (cosmos, 'wm_update_complete.json', selected['cosmos_step'])):
        record = read_json(path / marker)
        if record.get('step') != step or record.get('identity', {}).get('manifest') != selected['manifest_sha256']:
            raise ValueError('The selected world-model checkpoint changed')
    return package_pair(SimpleNamespace(manifest=Path(selected['manifest']), wan_checkpoint=wan,
                                       cosmos_checkpoint=cosmos, output=args.output))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'package':
        parser = argparse.ArgumentParser(description='Package the Cosmos-first selected checkpoints')
        parser.add_argument('command', choices=['package'])
        parser.add_argument('--selection', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        package_selection(parser.parse_args())
        return
    import wm_update
    import wm_eval

    wm_update.SCRIPT = Path(__file__).resolve()
    plan, supervise = wm_update.training_plan, wm_eval.supervise
    captured = {}
    def training_plan(args):
        if args.model != 'both' or args.gpus.count(',') != 7:
            raise ValueError('Choose --model both and eight GPUs for concurrent four-card model updates')
        if args.ranks_per_model is None:
            args.ranks_per_model = 4
        args.output = args.output.resolve()
        captured['args'] = args
        return plan(args)
    wm_update.training_plan = training_plan
    wm_eval.supervise = lambda commands: supervise_until_cosmos(captured['args'], commands)
    worker_args = {}
    if len(sys.argv) > 1 and sys.argv[1] == 'train-worker':
        import wm_update_memory
        import wm_update_wan
        native_wan = wm_update_wan.train
        train_with_memory = wm_update_memory.train_with_memory
        def train_and_hold(args, model, train):
            worker_args['args'] = args
            if model == 'wan':
                train = lambda value: controlled_wan(value, native_wan)
            return train_with_memory(args, model, train)
        wm_update_memory.train_with_memory = train_and_hold
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, lambda n, f: (_ for _ in ()).throw(KeyboardInterrupt(f'Signal {n}')))
    try:
        result = wm_update.main()
        if worker_args:
            retain_cache(worker_args['args'])
        return result
    finally:
        wm_update.training_plan, wm_eval.supervise = plan, supervise


if __name__ == '__main__':
    raise SystemExit(main())
