"""Save the current Wan update when Cosmos finishes, then continue GRPO directly.

This entry can adopt a live controlled Wan worker without restarting it. The
old controller stays paused, and therefore alive, until Wan has saved its full
native checkpoint and inference export. Only recorded processes are stopped.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from queue_cosmos_first_wan8 import identity, latest_wan_checkpoint, stop_owned_tree
from queue_iterative_cosmos_metrics_wan8 import (
    controlled_checkpoint, cosmos_checkpoint, cosmos_ready,
    verify as verify_previous,
)
from queue_spatial_iterative_update import assert_idle
from queue_spatial_wm_update import gpu_compute_pids, gpu_usage, process_alive, sha256, utc, write_json
from spatial_followup_email import OutcomeMailer, read_json
from spatial_wovr_memory_guard import checkpoint_complete
from stop_spatial_grpo import read_process, signal_process
from watch_training_email import DEFAULT_CONFIG

PROJECT = Path(__file__).resolve().parents[1]


def verify(plan):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != plan['boot_id']:
        raise RuntimeError('Run this continuation on its recorded GPU node')
    for filename, expected in plan['files'].items():
        if sha256(Path(filename)) != expected:
            raise RuntimeError(f'Pinned continuation code/input changed: {filename}')
    if not checkpoint_complete(Path(plan['source']['policy_checkpoint']), len(plan['gpus'])):
        raise RuntimeError('The full policy and optimizer resume checkpoint is incomplete')


def emit(plan, status, **details):
    root = Path(plan['output'])
    value = dict(status=status, updated_utc=utc(), pid=os.getpid(), **details)
    write_json(root / 'state.json', value)
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
    if (root / 'ownership_transferred.json').exists():
        source = Path(plan['source']['output'])
        write_json(source / 'state.json', dict(value, continuation_plan=str(root / 'plan.json')))
        pointer = PROJECT / 'runtime_artifacts/training_queue/iterative_policy40_current.json'
        current = read_json(pointer)
        if current.get('output') == str(source):
            current.pop('error', None)
            current.update(value, continuation_plan=str(root / 'plan.json'),
                           pending_policy_update=None, screen=plan['screen'],
                           notifier_screen=plan['notifier_screen'])
            write_json(pointer, current)
    print(json.dumps(value), flush=True)


def prepare(args):
    previous = read_json(args.active_plan)
    verify_previous(previous)
    old = Path(previous['output'])
    if not (old / 'ownership_transferred.json').exists():
        raise ValueError('Adoption requires the existing controlled Wan continuation')
    worker = read_json(old / 'worker_identity.json')
    wan = read_json(old / 'wan_child.json')
    notifier = read_json(old / 'notifier_identity.json')
    live = [process_alive(row) for row in (worker, wan)]
    if any(live) and not all(live):
        raise ValueError('An incomplete live handoff needs inspection before adoption')
    adoption = 'live_controlled_wan' if all(live) else 'completed_checkpoint'
    selected = latest_committed_wan(previous) if adoption == 'completed_checkpoint' else None
    if not cosmos_ready(previous):
        raise ValueError('Cosmos has not completed its full final checkpoint')
    source = read_json(Path(previous['source_plan']))
    if (Path(source['output']) / 'model_pair').exists():
        raise FileExistsError('A world-model pair already exists; do not replace a live deployment')
    output = args.output.resolve()
    (output / 'logs').mkdir(parents=True, exist_ok=False)
    code = set(source['code']) | {
        str(PROJECT / 'experiments' / name) for name in previous['code']
    } | {str(Path(__file__).resolve()), str(PROJECT / 'experiments/spatial_followup_watchdogs.py')}
    files = code | set(source['inputs']) | {
        str(args.active_plan.resolve()), previous['source_plan'],
        str(Path(previous['wm_output']) / 'manifest.json'),
        str(cosmos_checkpoint(previous, 2000) / 'wm_update_complete.json'),
    }
    source.update(screen='spatial-policy40-direct-grpo', notifier_screen='spatial-policy40-direct-mail')
    plan = dict(schema='spatial-cosmos-complete-latest-wan-grpo-v1', created_utc=utc(),
        output=str(output), source=source, previous_plan=str(args.active_plan.resolve()),
        old_controller=worker, old_notifier=notifier, wan_child=wan,
        old_metric_child=read_json(old / 'active_child.json'),
        adoption=adoption, selected_wan_checkpoint=str(selected) if selected else None,
        boot_id=source['boot_id'], gpus=source['gpus'],
        screen=source['screen'], notifier_screen=source['notifier_screen'],
        policy='Cosmos final checkpoint -> save latest Wan -> GRPO; no video-metric or Wan-target wait',
        files={filename: sha256(Path(filename)) for filename in sorted(files)})
    verify(plan)
    write_json(output / 'plan.json', plan)
    emit(plan, 'prepared', stage='save_latest_wan')
    print(output / 'plan.json', flush=True)


def latest_committed_wan(previous):
    candidates = []
    root = Path(previous['output'])
    for directory in (root / 'train4/wan', root / 'train8/wan',
                      Path(previous['wm_output']) / 'train/wan'):
        try:
            checkpoint = latest_wan_checkpoint(directory, previous['manifest_sha256'])
        except ValueError as error:
            if str(error).startswith('No complete, exported and logged'):
                continue
            raise
        candidates.append((read_json(checkpoint / 'complete.json')['step'], checkpoint))
    if not candidates:
        raise ValueError('No full current-iteration Wan checkpoint is available')
    return max(candidates, key=lambda row: row[0])[1]


def resume(args):
    """Keep the immutable packaged pair after a GPU-availability-only failure."""
    previous = read_json(args.plan)
    old = Path(previous['output'])
    if read_json(old / 'state.json').get('status') != 'failed':
        raise ValueError('Resume requires a failed attempt')
    for name in ('worker_identity.json', 'notifier_identity.json'):
        row = read_json(old / name)
        if row and process_alive(row):
            raise ValueError('The previous controller/notifier is still running')
    own_file = str(Path(__file__).resolve())
    for filename, expected in previous['files'].items():
        if filename != own_file and sha256(Path(filename)) != expected:
            raise ValueError(f'Unexpected code/input change: {filename}')
    source = dict(previous['source'])
    pair_file = Path(source['output']) / 'model_pair/pair.json'
    pair = read_json(pair_file)
    if not pair or (Path(source['output']) / 'grpo_primary').exists():
        raise ValueError('Reuse only a prepared pair before GRPO has started')
    root = args.output.resolve()
    (root / 'logs').mkdir(parents=True, exist_ok=False)
    source.update(screen='spatial-policy40-direct-grpo-wait', notifier_screen='spatial-policy40-direct-wait-mail')
    files = dict(previous['files'])
    files.update({own_file: sha256(Path(own_file)), str(args.plan.resolve()): sha256(args.plan),
                  str(pair_file): sha256(pair_file)})
    plan = dict(previous, output=str(root), source=source, created_utc=utc(),
                restart_of=str(old), reuse_packaged=True, files=files,
                screen=source['screen'], notifier_screen=source['notifier_screen'])
    verify(plan)
    write_json(root / 'plan.json', plan)
    emit(plan, 'prepared', stage='wait_for_gpus', wan_step=pair['wan_step'], cosmos_step=pair['cosmos_step'])
    print(root / 'plan.json', flush=True)


def wait_for_gpus(plan):
    last = float('-inf')
    while True:
        used, active = gpu_usage(), gpu_compute_pids(plan['gpus'])
        if not active and all(used.get(device, 81920) <= 1024 for device in plan['gpus']):
            assert_idle(plan['gpus'])
            return
        if time.monotonic() - last >= 30:
            emit(plan, 'waiting_for_gpus', stage='wait_for_gpus', used_mib=used, compute_pids=active)
            last = time.monotonic()
        time.sleep(2)


def grpo_stage(source, pair, expected_wan_step):
    from spatial_followup_watchdogs import create_wan_deployment, grpo_command

    if (pair['wan_step'] != expected_wan_step or pair['cosmos_step'] != source['wm_steps']
            or pair['manifest_sha256'] != source['wm_manifest_sha256']):
        raise ValueError('GRPO must use the newly saved Wan and completed Cosmos from the current data')
    if source.get('reuse_packaged'):
        deployment = Path(source['output']) / 'updated_wan_deployment'
        for name, expected in {'model-00001.safetensors': Path(pair['wan']) / 'model-00001.safetensors',
                               'Wan2.2_VAE.pth': Path(pair['wan']) / 'Wan2.2_VAE.pth',
                               'resnet_rm.pth': Path(source['baseline_wan']) / 'resnet_rm.pth'}.items():
            if not (deployment / name).is_file() or (deployment / name).resolve() != expected.resolve():
                raise ValueError('Existing Wan deployment does not match the packaged pair')
    else:
        create_wan_deployment(source, pair)
    command, env, output, target = grpo_command(source, 'primary', Path(source['policy_checkpoint']),
        Path(source['output']) / 'model_pair/pair.json')
    command = [arg for arg in command if not arg.startswith('actor.fsdp_config.save_training_state_at_steps=')]
    command.append(f'actor.fsdp_config.save_training_state_at_steps=[{target-10},{target}]')
    env.update(SPATIAL_WOVR_LAUNCHER=str(Path(source['output']) / 'inputs/fast_relaunch.py'),
               SPATIAL_COSMOS_PORT='18759', TORCHINDUCTOR_COMPILE_THREADS='1')
    return command, env, output, target


def save_wan(plan):
    previous = read_json(Path(plan['previous_plan']))
    old = Path(previous['output'])
    if plan.get('adoption') == 'completed_checkpoint':
        if any(process_alive(plan[key]) for key in ('old_controller', 'wan_child')):
            raise RuntimeError('A stopped predecessor restarted before direct GRPO adoption')
        checkpoint = latest_committed_wan(previous)
        if str(checkpoint) != plan['selected_wan_checkpoint']:
            raise RuntimeError('The selected latest Wan checkpoint changed after review')
        record = read_json(checkpoint / 'complete.json')
        write_json(Path(plan['output']) / 'selected_wan_checkpoint.json',
            dict(path=str(checkpoint), record=record, utc=utc(), selection='latest complete current-iteration checkpoint'))
        emit(plan, 'stage_complete', stage='save_latest_wan', checkpoint=str(checkpoint),
             step=record['step'], selection='latest complete checkpoint after predecessor stopped')
        return previous, checkpoint
    process = read_process(plan['old_controller']['pid'])
    if process is None or process.started != plan['old_controller']['start_ticks']:
        raise RuntimeError('The old Wan cache owner exited before checkpoint handoff')
    signal_process(process, signal.SIGSTOP)
    # Stop only the metric group; the live Wan workers keep their CUDA caches.
    stop_owned_tree(plan['old_metric_child'])
    write_json(old / 'wan_control/checkpoint_request.json', dict(
        token=previous['wan_control_token'], manifest_sha256=previous['manifest_sha256'],
        utc=utc(), reason='Cosmos complete: save latest Wan and start GRPO directly'))
    emit(plan, 'running', stage='save_latest_wan')
    while True:
        if not process_alive(plan['wan_child']):
            raise RuntimeError('Wan exited before its requested full checkpoint was verified')
        checkpoint = controlled_checkpoint(previous)
        if checkpoint is not None:
            break
        time.sleep(.5)
    record = read_json(checkpoint / 'complete.json')
    write_json(Path(plan['output']) / 'selected_wan_checkpoint.json',
        dict(path=str(checkpoint), record=record, utc=utc()))
    emit(plan, 'stage_complete', stage='save_latest_wan', checkpoint=str(checkpoint), step=record['step'])
    return previous, checkpoint


def take_ownership(plan, previous, checkpoint):
    root, source = Path(plan['output']), Path(plan['source']['output'])
    write_json(Path(previous['output']) / 'wan_control/release.json',
               dict(utc=utc(), checkpoint=str(checkpoint), reason='Release for direct eight-GPU GRPO'))
    deadline = time.monotonic() + 90
    while process_alive(plan['wan_child']):
        if time.monotonic() > deadline:
            raise RuntimeError('Completed Wan did not release its training process group')
        time.sleep(.25)
    stop_owned_tree(plan['old_notifier'], immediate=True)
    process = read_process(plan['old_controller']['pid'])
    if process is not None and process.started == plan['old_controller']['start_ticks']:
        signal_process(process, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while process_alive(plan['old_controller']):
        if time.monotonic() > deadline:
            raise RuntimeError('The old CPU controller did not release its workflow lock')
        time.sleep(.1)
    lock = (source / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    own = identity(read_process(os.getpid()))
    for filename in ('worker_identity.json', 'primary_identity.json'):
        write_json(source / filename, own)
    write_json(root / 'ownership_transferred.json', dict(utc=utc(), identity=own))
    write_json(Path(previous['output']) / 'state.json', dict(status='superseded',
        updated_utc=utc(), reason=plan['policy'], replacement_plan=str(root / 'plan.json')))
    emit(plan, 'stage_complete', stage='direct_grpo_handoff')
    return lock


def run(plan):
    from spatial_followup_watchdogs import cleanup_workload, finished_grpo, run_child
    from wm_update import environment

    root, source = Path(plan['output']), dict(plan['source'])
    ownership = None
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / 'worker_identity.json').exists():
            raise FileExistsError('This continuation already ran; inspect its recorded stages')
        write_json(root / 'worker_identity.json', identity(read_process(os.getpid())))
        try:
            verify(plan)
            previous, checkpoint = save_wan(plan)
            os.environ.update(source['environment'], CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
            environment()
            env = dict(os.environ)
            for name in ('WM_UPDATE_HANDOFF_ROOT', 'WM_UPDATE_HANDOFF_OWNER', 'RANK', 'LOCAL_RANK',
                         'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
                env.pop(name, None)
            emit(plan, 'running', stage='package_latest_world_models')
            if not plan.get('reuse_packaged'):
                with (root / 'logs/package.log').open('a') as stream:
                    subprocess.run([source['python'], str(PROJECT / 'experiments/wm_update.py'), 'package',
                        '--manifest', str(Path(previous['wm_output']) / 'manifest.json'),
                        '--wan-checkpoint', str(checkpoint),
                        '--cosmos-checkpoint', str(cosmos_checkpoint(previous, 2000)),
                        '--output', str(Path(source['output']) / 'model_pair')],
                        cwd=PROJECT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            pair = read_json(Path(source['output']) / 'model_pair/pair.json')
            source['wm_manifest_sha256'] = previous['manifest_sha256']
            source['reuse_packaged'] = plan.get('reuse_packaged', False)
            wan_step = read_json(checkpoint / 'complete.json')['step']
            command, grpo_env, output, target = grpo_stage(source, pair, wan_step)
            emit(plan, 'stage_complete', stage='package_latest_world_models',
                 wan_step=wan_step, cosmos_step=pair['cosmos_step'])
            ownership = take_ownership(plan, previous, checkpoint)
            write_json(Path(source['output']) / 'grpo_launch.json',
                dict(command=command, output=str(output), target_step=target))
            pointer = PROJECT / 'runtime_artifacts/training_queue/grpo_latestwm_current.json'
            latest = dict(status='waiting_for_gpus', output=source['output'], plan=str(root / 'plan.json'),
                grpo_output=str(output), resume_step=source['policy_step'], target_step=target,
                last_completed_step=source['policy_step'], wan_step=wan_step, cosmos_step=pair['cosmos_step'],
                pair_manifest=str(Path(source['output']) / 'model_pair/pair.json'),
                gpus=[int(g) for g in plan['gpus']], gpu_count=len(plan['gpus']),
                screen=plan['screen'], notifier_screen=plan['notifier_screen'], updated_utc=utc())
            write_json(pointer, latest)
            wait_for_gpus(plan)
            latest.update(status='launching', updated_utc=utc())
            write_json(pointer, latest)
            emit(plan, 'running', stage='grpo', grpo_output=str(output), target_step=target,
                 wan_step=wan_step, cosmos_step=pair['cosmos_step'])
            run_child(source, 'primary', 'grpo', command, grpo_env,
                resume=source['policy_checkpoint'], target_step=target,
                grpo_output=str(output), updated_world_model=True)
            if not finished_grpo(output, target):
                raise RuntimeError('GRPO did not finish with a complete final policy/optimizer checkpoint')
            cleanup_workload(source, 'primary')
            latest.update(status='complete', last_completed_step=target, updated_utc=utc())
            write_json(pointer, latest)
            emit(plan, 'complete', stage='grpo', grpo_output=str(output), target_step=target)
        except BaseException as error:
            if ownership is not None:
                cleanup_workload(source, 'primary')
            current = read_json(root / 'state.json')
            emit(plan, 'failed', stage=current.get('stage'), error=f'{type(error).__name__}: {error}')
            raise
        finally:
            if ownership is not None:
                ownership.close()


def notify(plan):
    root = Path(plan['output'])
    with (root / 'notifier.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(root / 'notifier_identity.json', identity(read_process(os.getpid())))
        mailer = OutcomeMailer(plan, DEFAULT_CONFIG)
        for file in (root / 'outcome_email/outbox').glob('*.json'):
            mailer.events[file.stem] = read_json(file)
            mailer.deliveries[file.stem] = read_json(root / 'outcome_email/delivery' / file.name)
        mailer.enqueue('armed', dict(outcome='armed', subject='[Spatial] Cosmos结束→保存Wan→八卡GRPO 队列启用',
            body=f"{plan['policy']}\n计划：{root / 'plan.json'}"))
        while True:
            result = read_json(root / 'state.json')
            worker = read_json(root / 'worker_identity.json')
            unexpected = worker and not process_alive(worker) and result.get('status') not in {'complete', 'failed'}
            if unexpected:
                result = dict(status='failed', stage=result.get('stage'), error='Direct GRPO controller exited unexpectedly')
            events = root / 'events.jsonl'
            for line in events.read_text().splitlines() if events.exists() else []:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('status') == 'stage_complete':
                    name = event['stage']
                    mailer.enqueue('stage-' + name, dict(outcome='complete', subject=f'[Spatial 接续] {name} 完成',
                        body=json.dumps(event, ensure_ascii=False)))
            terminal = result.get('status') in {'complete', 'failed'}
            if terminal:
                mailer.enqueue('primary-outcome', dict(outcome=result['status'],
                    subject=f"[Spatial 接续] Cosmos→Wan保存→GRPO {result['status']}",
                    body=f"输出：{root}\n{json.dumps(result, ensure_ascii=False)}"))
            mailer.send_due()
            sent = terminal and mailer.deliveries.get('primary-outcome', {}).get('status') == 'sent'
            mailer.persist(root / 'outcome_email/state.json', dict(status='notified' if sent else 'watching',
                updated_utc=utc(), deliveries=mailer.deliveries))
            if unexpected and not (root / 'orphan_cleanup.json').exists():
                if (root / 'ownership_transferred.json').exists():
                    from spatial_followup_watchdogs import cleanup_workload
                    cleanup_workload(plan['source'], 'primary')
                write_json(root / 'orphan_cleanup.json', dict(status='complete', utc=utc()))
            if sent and all(row.get('status') == 'sent' for row in mailer.deliveries.values()):
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'resume', 'verify', 'run', 'notify'])
    for name in ('plan', 'active-plan', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.mode == 'resume':
        if args.plan is None or args.output is None:
            parser.error('resume requires --plan and a new --output')
        resume(args)
        return
    if args.mode == 'prepare':
        if args.active_plan is None or args.output is None:
            parser.error('prepare requires --active-plan and a new --output')
        prepare(args)
        return
    if args.plan is None:
        parser.error('--plan is required')
    plan = read_json(args.plan)
    if args.mode == 'run':
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, lambda n, f: (_ for _ in ()).throw(KeyboardInterrupt(f'Signal {n}')))
        run(plan)
    elif args.mode == 'notify':
        notify(plan)
    else:
        verify(plan)
        print('Direct Cosmos-to-GRPO continuation verified; no GPU work started')


if __name__ == '__main__':
    main()
