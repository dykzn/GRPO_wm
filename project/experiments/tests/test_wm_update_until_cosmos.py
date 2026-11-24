"""A completed Cosmos triggers Wan's full save; neither model is killed early."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wm_update_until_cosmos as update


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def args_at(tmp_path):
    output = tmp_path / 'train'
    output.mkdir()
    return SimpleNamespace(output=output, manifest=tmp_path / 'manifest.json',
                           steps=2000, ranks_per_model=4)


def test_cosmos_completion_requests_wan_save_before_selecting_and_releasing(tmp_path):
    args = args_at(tmp_path)
    cosmos = update.checkpoint_path(args.output, 'cosmos', 2000)
    wan = update.checkpoint_path(args.output, 'wan', 1234)
    write(cosmos / 'wm_update_complete.json', dict(step=2000))
    write(wan / 'complete.json', dict(step=1234))
    control = args.output / 'cosmos_first'
    observed = []
    def completed(output, directory, model, manifest, ranks):
        assert manifest == 'paired-data'
        if model == 'cosmos':
            observed.append('cosmos')
            if len(observed) == 1:
                assert not (control / 'checkpoint_request.json').exists()
                return None
            return cosmos
        request = json.loads((control / 'checkpoint_request.json').read_text())
        assert request['token'] == 'this-run'
        assert request['manifest_sha256'] == 'paired-data'
        assert not (control / 'release.json').exists()
        observed.append('wan')
        return wan
    child = SimpleNamespace(poll=lambda: None)
    with patch('wm_update_data.load_manifest', return_value=dict(manifest_sha256='paired-data')), \
         patch.object(update, 'completed_checkpoint', side_effect=completed), \
         patch.object(update.time, 'sleep'):
        selected = update.wait_for_pair(args, {'wan': child, 'cosmos': child}, 'this-run')
    assert observed == ['cosmos', 'cosmos', 'wan']
    assert selected['wan_step'] == 1234 and selected['cosmos_step'] == 2000
    assert json.loads((args.output / 'selected_world_models.json').read_text()) == selected
    assert not (control / 'release.json').exists()


def test_intermediate_cosmos_checkpoint_does_not_cut_wan_short(tmp_path):
    args = args_at(tmp_path)
    cosmos = update.checkpoint_path(args.output, 'cosmos', 1000)
    write(cosmos / 'wm_update_complete.json', dict(step=1000))
    child = SimpleNamespace(poll=lambda: None)
    with patch('wm_update_data.load_manifest', return_value=dict(manifest_sha256='paired-data')), \
         patch.object(update, 'completed_checkpoint', return_value=cosmos), \
         pytest.raises(ValueError, match='configured final step'):
        update.wait_for_pair(args, {'wan': child, 'cosmos': child}, 'this-run')
    assert not (args.output / 'cosmos_first/checkpoint_request.json').exists()


def test_cosmos_receipts_require_full_optimizer_rng_and_master_weights(tmp_path):
    args = args_at(tmp_path)
    checkpoint = update.checkpoint_path(args.output, 'cosmos', 2000)
    control = args.output / 'cosmos_first'
    for rank in range(4):
        write(control / f'cosmos_rank{rank}.json', dict(model='cosmos', rank=rank,
            step=2000, manifest_sha256='paired-data', reserved_gib=75,
            checkpoint=str(checkpoint), pid=100+rank, start_ticks=200+rank))
    write(checkpoint / 'wm_update_complete.json', dict(step=2000, world_size=4,
        identity=dict(manifest='paired-data')))
    for name in ('model', 'optim', 'scheduler', 'trainer'):
        path = checkpoint / name / '.metadata'
        path.parent.mkdir(); path.touch()
    for rank in range(4):
        (checkpoint / f'rng_rank{rank}.pt').touch()
    with patch.object(update, 'process_alive', return_value=True):
        with pytest.raises(ValueError, match='full checkpoint is incomplete'):
            update.completed_checkpoint(args.output, control, 'cosmos', 'paired-data', 4)
        (checkpoint / 'master_weights.pt').touch()
        assert update.completed_checkpoint(args.output, control, 'cosmos', 'paired-data', 4) == checkpoint


@pytest.mark.parametrize('restored_step', [0, 1000])
def test_native_wan_hook_saves_next_update_for_fresh_and_full_resumed_runs(tmp_path, monkeypatch, restored_step):
    import torch
    import wm_update_wan as native
    from wm_update_wan_controlled import CheckpointRequest

    control = tmp_path / 'control'
    control.mkdir()
    monkeypatch.setenv('WM_COSMOS_FIRST_CONTROL', str(control))
    monkeypatch.setenv('WM_COSMOS_FIRST_TOKEN', 'this-run')
    args = SimpleNamespace(manifest=tmp_path / 'manifest.json', steps=2000)
    calls = []
    dist = SimpleNamespace(get_rank=lambda: 0, broadcast=lambda value, src: None)
    reset = lambda: calls.append(True)
    original_load = lambda *args, **kwargs: restored_step
    def factory(args, step, manifest, torch_module, dist_module, reset_function):
        return CheckpointRequest(args, step, manifest, torch_module, dist, reset_function, device='cpu')
    def train(value):
        if restored_step:
            assert native.load_state('/resume', None, None, identity={}) == restored_step
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_peak_memory_stats()
        assert value.steps == 2000
        write(control / 'checkpoint_request.json', dict(token='this-run', manifest_sha256='paired-data'))
        torch.cuda.reset_peak_memory_stats()
        assert value.steps == restored_step + 3
    with patch('wm_update_data.load_manifest', return_value=dict(manifest_sha256='paired-data')), \
         patch('wm_update_wan_controlled.CheckpointRequest', side_effect=factory), \
         patch.object(native, 'load_state', original_load), \
         patch.object(torch.cuda, 'reset_peak_memory_stats', reset):
        update.controlled_wan(args, train)
        assert native.load_state is original_load
        assert torch.cuda.reset_peak_memory_stats is reset
    assert len(calls) == 3


def test_failed_training_group_does_not_export_a_selection(tmp_path):
    args = args_at(tmp_path)
    child = SimpleNamespace(poll=lambda: 1, returncode=1)
    with patch('wm_update_data.load_manifest', return_value=dict(manifest_sha256='paired-data')), \
         pytest.raises(RuntimeError, match='before coordinated checkpoint release'):
        update.wait_for_pair(args, {'cosmos': child}, 'this-run')
    assert not (args.output / 'selected_world_models.json').exists()


def test_same_controller_releases_both_groups_only_after_selection_is_durable(tmp_path):
    args = args_at(tmp_path)
    args.model, args.reserve_memory, args.save_interval = 'both', True, 1000
    cosmos = update.checkpoint_path(args.output, 'cosmos', 2000)
    wan = update.checkpoint_path(args.output, 'wan', 1234)
    write(cosmos / 'wm_update_complete.json', dict(step=2000))
    write(wan / 'complete.json', dict(step=1234))
    control = args.output / 'cosmos_first'
    order = []
    class Child:
        def __init__(self, model): self.model = model
        def poll(self): return None
        def wait(self, timeout):
            assert (control / 'release.json').is_file()
            assert (args.output / 'selected_world_models.json').is_file()
            order.append('released-' + self.model)
            return 0
    def launch(command, **kwargs):
        model = command[-1]
        assert 'WM_UPDATE_HANDOFF_ROOT' not in kwargs['env']
        assert json.loads(kwargs['env']['WM_COSMOS_FIRST_OWNER']) == dict(pid=100, start_ticks=200)
        order.append('started-' + model)
        return Child(model)
    def completed(output, directory, model, manifest, ranks):
        assert order[:2] == ['started-wan', 'started-cosmos']
        assert not (control / 'release.json').exists()
        return wan if model == 'wan' else cosmos
    commands = [(['worker', '--model', model], {'WM_UPDATE_HANDOFF_ROOT':'/outer-owner'}, args.output / f'{model}.log')
                for model in ('wan', 'cosmos')]
    own = SimpleNamespace(pid=100, started=200)
    with patch('wm_update_data.load_manifest', return_value=dict(manifest_sha256='paired-data')), \
         patch.object(update, 'completed_checkpoint', side_effect=completed), \
         patch.object(update, 'read_process', return_value=own), \
         patch.object(update.subprocess, 'Popen', side_effect=launch), \
         patch('spatial_wovr_launch.stop_process') as stop:
        update.supervise_until_cosmos(args, commands)
    assert order == ['started-wan', 'started-cosmos', 'released-wan', 'released-cosmos']
    assert stop.call_count == 2
    result = json.loads((control / 'state.json').read_text())
    assert result['status'] == 'complete' and result['wan_step'] == 1234
