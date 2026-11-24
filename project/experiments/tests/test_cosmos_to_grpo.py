"""Cosmos completion accepts a fresh Wan checkpoint at a different step."""

import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import queue_cosmos_to_grpo as queue


def source_at(tmp_path):
    root = tmp_path / 'source'
    (root / 'model_pair').mkdir(parents=True)
    return dict(output=str(root), wm_steps=2000, wm_manifest_sha256='current-data',
                policy_checkpoint='/policy/global_step_40')


def test_grpo_uses_newest_wan_without_waiting_for_wan_2000(tmp_path):
    source = source_at(tmp_path)
    pair = dict(wan_step=1135, cosmos_step=2000, manifest_sha256='current-data')
    command = ['launch', 'runner.max_steps=60', 'actor.fsdp_config.save_training_state_at_steps=[45,50,55,60]']
    with patch('spatial_followup_watchdogs.create_wan_deployment') as deploy, \
         patch('spatial_followup_watchdogs.grpo_command', return_value=(command, {}, Path('/grpo'), 60)):
        argv, env, output, target = queue.grpo_stage(source, pair, 1135)
    deploy.assert_called_once_with(source, pair)
    assert target == 60
    assert 'actor.fsdp_config.save_training_state_at_steps=[50,60]' in argv
    assert 'actor.fsdp_config.save_training_state_at_steps=[45,50,55,60]' not in argv
    assert env['SPATIAL_WOVR_LAUNCHER'].endswith('/inputs/fast_relaunch.py')


@pytest.mark.parametrize('changes', [dict(wan_step=1000), dict(cosmos_step=1000), dict(manifest_sha256='old-data')])
def test_stale_or_unmatched_pair_never_starts_grpo(tmp_path, changes):
    source = source_at(tmp_path)
    pair = dict(wan_step=1135, cosmos_step=2000, manifest_sha256='current-data', **{})
    pair.update(changes)
    with patch('spatial_followup_watchdogs.create_wan_deployment') as deploy, \
         pytest.raises(ValueError, match='newly saved Wan'):
        queue.grpo_stage(source, pair, 1135)
    deploy.assert_not_called()


def test_live_switch_requests_full_save_and_keeps_wan_running_until_verified(tmp_path):
    old, output = tmp_path / 'old', tmp_path / 'new'
    old.mkdir(); output.mkdir()
    previous = dict(output=str(old), wan_control_token='current-token', manifest_sha256='current-data')
    previous_file = old / 'plan.json'
    previous_file.write_text(json.dumps(previous))
    checkpoint = old / 'train4/wan/step_001135'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'complete.json').write_text(json.dumps(dict(step=1135, world_size=4)))
    plan = dict(output=str(output), previous_plan=str(previous_file),
                old_controller=dict(pid=100, start_ticks=200),
                old_metric_child=dict(pid=101, start_ticks=201), wan_child=dict(pid=102, start_ticks=202))
    checks = []
    def committed(_):
        request = json.loads((old / 'wan_control/checkpoint_request.json').read_text())
        assert request['token'] == 'current-token'
        assert request['manifest_sha256'] == 'current-data'
        assert not (old / 'wan_control/release.json').exists()
        checks.append(True)
        return checkpoint if len(checks) == 2 else None
    process = SimpleNamespace(pid=100, started=200)
    with patch.object(queue, 'read_process', return_value=process), \
         patch.object(queue, 'signal_process') as send, \
         patch.object(queue, 'stop_owned_tree') as stop, \
         patch.object(queue, 'process_alive', return_value=True), \
         patch.object(queue, 'controlled_checkpoint', side_effect=committed), \
         patch.object(queue.time, 'sleep'):
        restored, saved = queue.save_wan(plan)
    assert saved == checkpoint and restored == previous
    send.assert_called_once_with(process, signal.SIGSTOP)
    stop.assert_called_once_with(plan['old_metric_child'])
    assert not (old / 'wan_control/release.json').exists()
    assert json.loads((output / 'selected_wan_checkpoint.json').read_text())['record']['step'] == 1135


def test_busy_foreign_gpu_keeps_cpu_queue_waiting_until_all_devices_are_free(tmp_path):
    plan = dict(output=str(tmp_path), gpus=['0', '1'])
    with patch.object(queue, 'gpu_usage', side_effect=[{'0': 13000, '1': 0}, {'0': 0, '1': 0}]), \
         patch.object(queue, 'gpu_compute_pids', side_effect=[{'0': [900]}, {}]), \
         patch.object(queue, 'assert_idle') as check, \
         patch.object(queue, 'stop_owned_tree') as stop, \
         patch.object(queue.time, 'sleep'):
        queue.wait_for_gpus(plan)
    check.assert_called_once_with(['0', '1'])
    stop.assert_not_called()
    state = json.loads((tmp_path / 'state.json').read_text())
    assert state['status'] == 'waiting_for_gpus'
    assert state['compute_pids'] == {'0': [900]}
