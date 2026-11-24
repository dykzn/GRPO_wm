"""The next iteration must use the final policy and fully initialized models."""

import json
from pathlib import Path
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from queue_spatial_iterative_update import stages, upstream_ready, grpo_stage, prepare_data, release_training_handoff
from wm_update_reserved_collect import reserved_collect


def make_plan(root):
    upstream=root/'previous'
    run=upstream/'grpo_primary'
    run.mkdir(parents=True)
    root_output=root/'next'
    (root_output/'inputs').mkdir(parents=True)
    checkpoint=run/'experiment/checkpoints/global_step_40'
    evaluation=run/'experiment/evaluations/global_step_40/metrics.json'
    evaluation.parent.mkdir(parents=True)
    rows=[dict(task_id=task,trial_id=trial,success=True) for task in range(10) for trial in range(50)]
    evaluation.write_text(json.dumps(dict(episode_results=rows)))
    (run/'exit_code').write_text('0\n')
    (run/'metrics.log').write_text('Global Step: 40/40\n')
    (upstream/'primary_state.json').write_text(json.dumps(dict(status='complete')))
    return dict(upstream=str(upstream),upstream_run=str(run),upstream_identity=dict(pid=1,start_ticks=2),
                output=str(root_output),wm_output=str(root/'large_storage'),policy_step=40,
                policy_checkpoint=str(checkpoint),policy_evaluation=str(evaluation),
                python='/python',gpus=[str(i) for i in range(8)],per_task=50,seed=20261021,
                wm_steps=2000,wm_save_interval=1000,additional_grpo_steps=20)


def test_waits_for_supervisor_cleanup_even_after_final_policy_and_eval(tmp_path):
    plan=make_plan(tmp_path)
    with patch('queue_spatial_iterative_update.process_alive',return_value=True):
        assert not upstream_ready(plan)


def test_requires_full_final_optimizer_checkpoint_and_complete_evaluation(tmp_path):
    plan=make_plan(tmp_path)
    with patch('queue_spatial_iterative_update.process_alive',return_value=False), \
         patch('queue_spatial_iterative_update.checkpoint_complete',return_value=True):
        assert upstream_ready(plan)
        path=Path(plan['policy_evaluation'])
        value=json.loads(path.read_text());value['episode_results'].pop();path.write_text(json.dumps(value))
        with pytest.raises(RuntimeError,match='evaluation is incomplete'):
            upstream_ready(plan)


@pytest.mark.parametrize('status',['failed','cancelled'])
def test_failed_predecessor_never_starts_next_collection(tmp_path,status):
    plan=make_plan(tmp_path)
    (Path(plan['upstream'])/'primary_state.json').write_text(json.dumps(dict(status=status)))
    with pytest.raises(RuntimeError,match='did not finish successfully'):
        upstream_ready(plan)


def test_missing_optimizer_cannot_fall_back_to_an_earlier_policy(tmp_path):
    plan=make_plan(tmp_path)
    with patch('queue_spatial_iterative_update.process_alive',return_value=False), \
         patch('queue_spatial_iterative_update.checkpoint_complete',return_value=False):
        with pytest.raises(RuntimeError,match='checkpoint is incomplete'):
            upstream_ready(plan)


def test_new_iteration_keeps_500_raw_episodes_and_two_disjoint_model_groups(tmp_path):
    plan=make_plan(tmp_path)
    commands={stage['name']:stage['command'] for stage in stages(plan)}
    collect=commands['collect']
    assert collect[collect.index('--per-task')+1]=='50'
    assert collect[collect.index('--policy-checkpoint')+1]==plan['policy_checkpoint']
    assert Path(collect[1]).name=='wm_update_reserved_collect.py'
    train=commands['train']
    assert train[train.index('--cosmos-recipe')+1]=='v3-reconstructed'
    assert train[train.index('--ranks-per-model')+1]=='4'
    assert train[train.index('--accumulation')+1]=='4'
    assert train[train.index('--save-interval')+1]=='1000'
    assert '--resume' not in train
    assert '--reserve-memory' in train
    assert Path(train[1]).name=='wm_update_reserved_collect.py'


def test_continuation_runs_40_to_60_and_saves_full_state_at_50_60(tmp_path):
    plan=make_plan(tmp_path)
    pair=Path(plan['output'])/'model_pair/pair.json';pair.parent.mkdir()
    pair.write_text(json.dumps(dict(wan_step=2000,cosmos_step=2000)))
    def command(plan,role,checkpoint,pair):
        assert checkpoint.name=='global_step_40'
        return ['launch','runner.max_steps=60','actor.fsdp_config.save_training_state_at_steps=[45,50,55,60]'],{},Path(plan['output'])/'grpo_primary',60
    with patch('spatial_followup_watchdogs.create_wan_deployment'), \
         patch('spatial_followup_watchdogs.grpo_command',side_effect=command):
        argv,env,output,target=grpo_stage(plan)
    assert target==60
    assert 'actor.fsdp_config.save_training_state_at_steps=[50,60]' in argv
    assert 'actor.fsdp_config.save_training_state_at_steps=[45,50,55,60]' not in argv
    assert env['TORCHINDUCTOR_COMPILE_THREADS']=='1'


def test_collector_cache_belongs_to_the_actual_worker_on_its_selected_gpu(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','7')
    for name in ['LOCAL_RANK','RANK','LOCAL_WORLD_SIZE']:
        monkeypatch.setenv(name,'9')
    fake_torch=SimpleNamespace(cuda=SimpleNamespace(set_device=MagicMock()))
    pool=MagicMock()
    args=SimpleNamespace(output=tmp_path,reserve_memory=True)
    collect=MagicMock()
    with patch.dict(sys.modules,torch=fake_torch), \
         patch('wm_update_memory.TrainingMemoryReservation',return_value=pool) as constructor, \
         patch('wm_update_memory.memory_hooks',return_value=nullcontext(pool)):
        reserved_collect(args,collect)
    fake_torch.cuda.set_device.assert_called_once_with(0)
    assert constructor.call_args.args[1]==tmp_path/'memory/gpu_7'
    pool.initialize.assert_called_once()
    collect.assert_called_once_with(args)
    pool.record.assert_called_once_with('collection_complete')


def test_replay_preserves_old_holdouts_and_groups_new_policy_by_the_same_states(tmp_path):
    from wm_update_data import build_manifest,load_manifest

    def rows(prefix):
        return [dict(trajectory_id=f'{prefix}-{task}-{state}',sha256=f'{prefix}-sha-{task}-{state}',
                     path=f'/{prefix}/{task}-{state}.hdf5',suite='libero_spatial',task_id=task,
                     initial_state_id=state,horizon=16,role='new',success=True)
                for task in range(10) for state in range(5)]
    wm=tmp_path/'wm';(wm/'collection').mkdir(parents=True)
    replay=tmp_path/'replay.json'
    original=rows('old');new=rows('new')
    with patch('wm_update_data.unchanged'):
        baseline=build_manifest(original,replay,seed=20261021)
    (wm/'collection/collection.json').write_text(json.dumps(dict(episodes=new)))
    by_path={row['path']:row for row in new}
    plan=dict(wm_output=str(wm),replay_manifest=str(replay),per_task=5,seed=20261021)
    with patch('wm_update_data.unchanged'), \
         patch('wm_update_data.inventory',side_effect=lambda path,**kwargs:by_path[str(path)]):
        prepare_data(plan)
    result=load_manifest(wm/'manifest.json')
    assert len(result['episodes'])==100
    assert sum(row['role']=='old' for row in result['episodes'])==50
    before={row['trajectory_id']:row['split'] for row in baseline['episodes']}
    assert all(row['split']==before[row['trajectory_id']] for row in result['episodes'] if row['role']=='old')
    train={row['split_group'] for row in result['episodes'] if row['split']=='train'}
    val={row['split_group'] for row in result['episodes'] if row['split']=='val'}
    assert train.isdisjoint(val)


def test_completed_cosmos_keeps_its_cache_until_wan_also_finishes(tmp_path):
    plan=make_plan(tmp_path)
    plan['model_allocation']=dict(wan=['0','1','2','3'],cosmos=['4','5','6','7'])
    root=Path(plan['output'])/'training_handoff';root.mkdir()
    for rank in range(4):
        (root/f'cosmos_rank{rank}.json').write_text(json.dumps(dict(model='cosmos',rank=rank,reserved_gib=75.0)))
    assert not release_training_handoff(plan)
    assert not (root/'release.json').exists()
    for rank in range(4):
        (root/f'wan_rank{rank}.json').write_text(json.dumps(dict(model='wan',rank=rank,reserved_gib=75.0)))
    wm=Path(plan['wm_output'])
    checkpoints=[wm/'train/wan/step_002000/complete.json',
        wm/'train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints/iter_000002000/wm_update_complete.json']
    for path in checkpoints:
        path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(step=2000,world_size=4)))
    assert release_training_handoff(plan)
    receipt=json.loads((root/'release.json').read_text())
    assert len(receipt['workers'])==8


def test_handoff_requires_both_complete_checkpoints_not_just_worker_receipts(tmp_path):
    plan=make_plan(tmp_path)
    plan['model_allocation']=dict(wan=['0','1','2','3'],cosmos=['4','5','6','7'])
    root=Path(plan['output'])/'training_handoff';root.mkdir()
    for model in ['wan','cosmos']:
        for rank in range(4):
            (root/f'{model}_rank{rank}.json').write_text(json.dumps(dict(model=model,rank=rank,reserved_gib=75.0)))
    with pytest.raises(FileNotFoundError):release_training_handoff(plan)
    assert not (root/'release.json').exists()
