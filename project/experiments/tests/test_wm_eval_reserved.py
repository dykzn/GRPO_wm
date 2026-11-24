"""Parallel feature aggregation must reproduce a single global metric pass."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm_eval_data import FeatureMoments, atomic_json, frechet_distance
from wm_eval_reserved import MOMENTS, merge_moments, merge_scores, worker_command


def test_parallel_covariance_includes_between_shard_mean_differences():
    random = np.random.default_rng(7)
    values = random.normal(size=(43, 5)) + np.arange(43)[:, None]
    expected = FeatureMoments()
    expected.add(values)
    actual = FeatureMoments()
    merge_moments(actual, FeatureMoments())
    for chunk in np.array_split(values, 8):
        partial = FeatureMoments()
        partial.add(chunk)
        merge_moments(actual, partial)
    assert actual.count == expected.count
    np.testing.assert_allclose(actual.mean, expected.mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual.m2, expected.m2, rtol=1e-12, atol=1e-10)


def fixture_scores(root, shards=3):
    random = np.random.default_rng(20261023)
    rows = [dict(trajectory_id=str(i), task_id=i, horizon=16+8*i) for i in range(5)]
    manifest = dict(episodes=rows, manifest_sha256="frozen", protocol="test", split_status="held-out")
    args = argparse.Namespace(output=root, model="cosmos", reference_comparison=None)
    prediction_root = root / "cosmos"
    prediction_root.mkdir()
    partial_root = root / "metric_shards/cosmos"
    partial_root.mkdir(parents=True)
    full = {name: FeatureMoments() for name in MOMENTS}
    data = {}
    for row in rows:
        length = row["horizon"]
        data[row["trajectory_id"]] = dict(
            fid_real=random.normal(size=(length, 4)), fid_pred=random.normal(size=(length, 4))+0.5,
            fvd_real=random.normal(size=(1, 3)), fvd_pred=random.normal(size=(1, 3))+0.3)
        (prediction_root / f"{row['trajectory_id']}.npz").touch()
        atomic_json(prediction_root / f"{row['trajectory_id']}.json", dict(
            episode=row, model="cosmos", settings_sha256="one_model"))
        for name in MOMENTS:
            full[name].add(data[row["trajectory_id"]][name])
    for rank in range(shards):
        partial = {name: FeatureMoments() for name in MOMENTS}
        subset = rows[rank::shards]
        episodes = []
        for row in subset:
            for name in MOMENTS:
                partial[name].add(data[row["trajectory_id"]][name])
            episodes.append(dict(trajectory_id=row["trajectory_id"], task_id=row["task_id"],
                                 frames=row["horizon"], LPIPS=0.25, FloLPIPS=0.5, FPS=4.0))
        frames = sum(row["horizon"] for row in subset)
        payload = {}
        for name, moment in partial.items():
            payload.update({name+"_count":np.array(moment.count), name+"_mean":moment.mean,
                            name+"_m2":moment.m2})
        path = partial_root / f"shard_{rank:03d}.npz"
        np.savez(path, **payload)
        atomic_json(path.with_suffix(".json"), dict(
            manifest_sha256="frozen", settings_sha256="one_model", shard_index=rank,
            num_shards=shards, model="cosmos", episodes=episodes, frames=frames,
            generation_seconds=frames/4, lpips_sum=frames*0.25, lpips_count=frames,
            motion_sum=(frames-len(subset))*0.5, motion_count=frames-len(subset)))
    return args, manifest, full


def test_sharded_metrics_match_global_statistics_and_frame_weights(tmp_path):
    args, manifest, full = fixture_scores(tmp_path)
    actual = merge_scores(args, manifest, 3)
    assert actual["episodes"] == 5
    assert actual["frames"] == 160
    assert actual["FPS"] == 4.0
    assert actual["LPIPS"] == 0.25
    assert actual["FloLPIPS"] == 0.5
    assert actual["FID"] == pytest.approx(frechet_distance(full["fid_real"],full["fid_pred"]),abs=1e-10)
    assert actual["FVD_variable_length"] == pytest.approx(frechet_distance(full["fvd_real"],full["fvd_pred"]),abs=1e-10)
    episode_rows=json.loads((tmp_path/'cosmos_episode_metrics.json').read_text())
    assert [row['trajectory_id'] for row in episode_rows] == [str(i) for i in range(5)]


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'checkpoint', 'weight'])
def test_merge_rejects_partial_or_mixed_metric_shards(tmp_path, corruption):
    args, manifest, _ = fixture_scores(tmp_path)
    path=tmp_path/'metric_shards/cosmos/shard_000.json'
    data=json.loads(path.read_text())
    if corruption == 'missing': data['episodes'].pop()
    if corruption == 'duplicate': data['episodes'].append(data['episodes'][0])
    if corruption == 'checkpoint': data['settings_sha256']='other_model'
    if corruption == 'weight': data['lpips_count']-=1
    atomic_json(path,data)
    with pytest.raises(ValueError): merge_scores(args,manifest,3)


def test_worker_keeps_shard_device_and_uses_no_compile_sidecars():
    original=['/python','/wm_eval.py','predict','--num-shards','8','--shard-index','7']
    command, env, log=worker_command((original,dict(CUDA_VISIBLE_DEVICES='7'),Path('/log')))
    assert original[1]=='/wm_eval.py'
    assert Path(command[1]).name=='wm_eval_reserved.py'
    assert command[2:]==original[2:]
    assert env['CUDA_VISIBLE_DEVICES']=='7'
    assert env['TORCHINDUCTOR_COMPILE_THREADS']=='1'
    assert env['LOCAL_WORLD_SIZE']=='1'
