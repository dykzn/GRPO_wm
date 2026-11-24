"""Causal wrist anchors, reconstructed augmentations and safe recipe resumes."""

import copy
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cosmos_recipe_data import (
    NOOP,
    RECIPE_VERSION,
    RecipeSettings,
    ReconstructedCosmosDataset,
    aligned_window,
    degrade_memory_frames,
    fixed_cpu_rng,
)
from cosmos_recipe_train import (
    CORRECTED,
    LEGACY,
    apply_training_settings,
    augmented_identity,
    checkpoint_model_inventory,
    corrected_training,
    guarded_model_load,
    isolated_loader_rng,
    model_load_coverage,
    select_recipe,
    strict_native_model_load,
    train,
)
from wm_update import training_plan
from wm_update_data import (
    PairedWindowDataset,
    atomic_json,
    build_manifest,
    inventory,
    training_code_identity,
    window_indices,
    write_episode,
)


def pattern(value, size=32):
    y, x = np.mgrid[:size, :size]
    return np.stack(
        [(x + value) % 256, (y * 3 + value) % 256, ((x + y) * 2 + value) % 256], axis=-1
    ).astype(np.uint8)


def memory_window(step=8, count=32):
    frames, incoming = window_indices(step, count)
    actions = np.tile(NOOP, (13, 1))
    for slot, source in enumerate(incoming):
        if source >= 0:
            actions[slot, :6] = (source + 1) / 100
            actions[slot, 6] = 1 if source >= 10 else -1
    return {
        "wrist": np.stack([pattern(i) for i in frames]),
        "primary": np.stack([pattern(i + 100) for i in frames]),
        "actions": actions,
        "action_indices": incoming,
        "step": step,
        "episode": {"actions": count, "trajectory_id": "test"},
    }


@pytest.fixture
def data_factory(tmp_path):
    # Constructor uses a real immutable episode/manifest; controlled window
    # tests below exercise the new adapter without JPEG quantization ambiguity.
    count = 32
    path = tmp_path / "episode.hdf5"
    write_episode(
        path,
        primary=[pattern(i + 100) for i in range(count + 1)],
        wrist=[pattern(i) for i in range(count + 1)],
        actions=np.tile(NOOP, (count, 1)),
        states=np.zeros((count + 1, 2)),
        proprio=np.zeros((count + 1, 9)),
        successes=np.zeros(count, bool),
        rewards=np.zeros(count),
        dones=np.zeros(count, bool),
        metadata=dict(
            task="test",
            episode_id="test",
            task_id=0,
            suite="libero_spatial",
            initial_state_id=0,
            policy_id="policy20",
            seed=42,
            success=False,
        ),
    )
    row = inventory(path)
    # The production splitter needs at least two distinct state groups.
    other_path = tmp_path / "other.hdf5"
    write_episode(
        other_path,
        primary=[pattern(i + 120) for i in range(count + 1)],
        wrist=[pattern(i + 20) for i in range(count + 1)],
        actions=np.tile(NOOP, (count, 1)),
        states=np.zeros((count + 1, 2)),
        proprio=np.zeros((count + 1, 9)),
        successes=np.zeros(count, bool),
        rewards=np.zeros(count),
        dones=np.zeros(count, bool),
        metadata=dict(
            task="test",
            episode_id="other",
            task_id=0,
            suite="libero_spatial",
            initial_state_id=1,
            policy_id="policy20",
            seed=43,
            success=False,
        ),
    )
    other = inventory(other_path)
    manifest = tmp_path / "manifest.json"
    build_manifest([row, other], manifest)
    stats = tmp_path / "stats.json"
    atomic_json(stats, dict(actions_min=[-2] * 7, actions_max=[4] * 7))

    def factory(*, recipe=None, split="train"):
        return ReconstructedCosmosDataset(
            str(manifest),
            stats_path=str(stats),
            split=split,
            recipe=recipe
            or RecipeSettings(
                memory_degrade_prob=0, static_video_probability=0, use_image_aug=False
            ),
        )

    return factory


def sample_window(data, window, index=17):
    with patch.object(PairedWindowDataset, "__getitem__", return_value=window):
        return data[index]


def model_video(rgb):
    from world_model_grpo.cosmos_wrist_adapter import (
        _preprocess_wrist_model_space,
        _runtime_to_cosmos_model,
    )

    return torch.from_numpy(
        np.stack(
            [
                _runtime_to_cosmos_model(
                    _preprocess_wrist_model_space(frame, (224, 224))
                )
                for frame in rgb
            ]
        )
        .transpose(3, 0, 1, 2)
        .copy()
    )


def test_all_frame_and_action_positions_include_fixed_wrist_anchor():
    window = memory_window(8)
    original = copy.deepcopy(window)
    rgb, actions, missing = aligned_window(window)
    expected = [window["primary"][0], pattern(0), pattern(6), pattern(7), pattern(8)]
    expected += [pattern(i) for i in range(9, 17)]
    np.testing.assert_array_equal(rgb, np.stack(expected))
    np.testing.assert_array_equal(actions[:2], [NOOP, NOOP])
    np.testing.assert_allclose(actions[2:, 0], np.arange(6, 17) / 100)
    np.testing.assert_array_equal(np.flatnonzero(missing), [0, 1])
    for key in ["wrist", "primary", "actions", "action_indices"]:
        np.testing.assert_array_equal(window[key], original[key])


def test_bootstrap_preserves_exact_normalized_sentinels(data_factory):
    window = memory_window(0)
    sample = sample_window(data_factory(), window)
    np.testing.assert_array_equal(sample["action"][:5], np.tile(NOOP, (5, 1)))
    np.testing.assert_allclose(
        sample["action"][5:, 0], 2 * (np.arange(1, 9) / 100 + 2) / 6 - 1, atol=1e-7
    )
    expected, _, _ = aligned_window(window)
    assert torch.equal(sample["video"], model_video(expected))


def test_nonbootstrap_action_normalization_leaves_only_references_as_sentinels(
    data_factory,
):
    window = memory_window(8)
    sample = sample_window(data_factory(), window)
    np.testing.assert_array_equal(sample["action"][:2], [NOOP, NOOP])
    np.testing.assert_allclose(
        sample["action"][2:, 0], 2 * (np.arange(6, 17) / 100 + 2) / 6 - 1, atol=1e-7
    )
    assert sample["video"].dtype == torch.uint8
    assert sample["video"].shape == (3, 13, 224, 224)
    assert sample["action"].shape == (13, 7)
    assert sample["t5_text_embeddings"].shape == (512, 1024)
    assert sample["num_conditional_frames"] == 2


@pytest.mark.parametrize("step,count", [(1, 32), (2, 32), (1, 9), (2, 10)])
def test_early_history_resamples_valid_causal_window_or_bootstrap(
    data_factory, step, count
):
    window = memory_window(step, count)
    with (
        patch.object(PairedWindowDataset, "__getitem__", return_value=window),
        patch(
            "cosmos_recipe_data.read_window",
            side_effect=lambda row, t: memory_window(t, count),
        ) as reader,
    ):
        sample = data_factory()[17]
    chosen = reader.call_args.args[1]
    assert chosen >= 3 if count >= 11 else chosen == 0
    assert sample["__key__"] == f"test:{chosen}"


def test_memory_corruption_keeps_reference_anchor_and_targets_clean(data_factory):
    window = memory_window(8)
    clean = sample_window(data_factory(), window)
    recipe = RecipeSettings(
        memory_degrade_prob=1, static_video_probability=0, use_image_aug=False
    )
    dirty = sample_window(data_factory(recipe=recipe), window)
    assert torch.equal(clean["video"][:, :2], dirty["video"][:, :2])
    assert torch.equal(clean["video"][:, 5:], dirty["video"][:, 5:])
    assert not torch.equal(clean["video"][:, 2:5], dirty["video"][:, 2:5])
    assert torch.equal(clean["action"], dirty["action"])
    assert torch.equal(
        dirty["video"], sample_window(data_factory(recipe=recipe), window)["video"]
    )


def test_static_branch_holds_future_and_gripper_but_keeps_observed_history(
    data_factory,
):
    window = memory_window(16)
    clean = sample_window(data_factory(), window)
    recipe = RecipeSettings(
        memory_degrade_prob=0, static_video_probability=1, use_image_aug=False
    )
    static = sample_window(data_factory(recipe=recipe), window)
    assert torch.equal(static["video"][:, :5], clean["video"][:, :5])
    assert torch.equal(
        static["video"][:, 5:], clean["video"][:, 4:5].expand(-1, 8, -1, -1)
    )
    assert torch.equal(static["action"][:5], clean["action"][:5])
    np.testing.assert_allclose(static["action"][5:, :6], -1 / 3, atol=1e-7)
    np.testing.assert_allclose(
        static["action"][5:, 6], 0, atol=1e-7
    )  # held +1, normalized using [-2,4]


def test_validation_disables_all_augmentations(data_factory):
    window = memory_window(8)
    recipe = RecipeSettings(
        memory_degrade_prob=1, static_video_probability=1, use_image_aug=True
    )
    with (
        patch(
            "cosmos_recipe_data.native_image_aug",
            side_effect=AssertionError("validation image aug"),
        ),
        patch(
            "cosmos_recipe_data.degrade_memory_frames",
            side_effect=AssertionError("validation corruption"),
        ),
    ):
        sample = sample_window(data_factory(recipe=recipe, split="val"), window)
    clean = sample_window(data_factory(split="val"), window)
    assert torch.equal(sample["video"], clean["video"])
    assert torch.equal(sample["action"], clean["action"])


def test_native_image_augmentation_receives_the_complete_clip(data_factory):
    window = memory_window(8)
    recipe = RecipeSettings(
        memory_degrade_prob=0, static_video_probability=0, use_image_aug=True
    )
    with patch(
        "cosmos_recipe_data.native_image_aug",
        side_effect=lambda video, **kw: video.clone(),
    ) as aug:
        sample = sample_window(data_factory(recipe=recipe), window)
    assert aug.call_count == 1
    assert aug.call_args.args[0].shape == (3, 13, 224, 224)
    assert aug.call_args.kwargs["stronger"] is True
    assert torch.equal(sample["video"], sample_window(data_factory(), window)["video"])


def test_cpu_augmentation_context_preserves_model_rng_without_cuda():
    random.seed(1)
    np.random.seed(2)
    torch.random.default_generator.manual_seed(3)
    py, np_state, cpu = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    with (
        patch.object(
            torch.cuda,
            "manual_seed_all",
            side_effect=AssertionError("CUDA RNG touched"),
        ),
        patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA initialized")
        ),
    ):
        with fixed_cpu_rng(123):
            first = (random.random(), np.random.rand(), torch.rand(3))
        with fixed_cpu_rng(123):
            second = (random.random(), np.random.rand(), torch.rand(3))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
    assert random.getstate() == py
    assert np.random.get_state()[0] == np_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), cpu)


def test_dataset_is_deterministic_and_does_not_consume_training_rng(data_factory):
    data = data_factory(recipe=RecipeSettings(use_image_aug=False))
    window = memory_window(8)
    state = torch.get_rng_state().clone()
    first = sample_window(data, window)
    second = sample_window(data, window)
    assert torch.equal(first["video"], second["video"])
    assert torch.equal(first["action"], second["action"])
    assert torch.equal(state, torch.get_rng_state())


def test_reconstructed_augmentation_probabilities_are_applied(data_factory):
    data = data_factory(recipe=RecipeSettings(use_image_aug=False))
    window = memory_window(8)
    clean = sample_window(data_factory(), window)
    corrupted, static = 0, 0
    for index in range(200):
        sample = sample_window(data, window, index=index)
        corrupted += not torch.equal(sample["video"][:, 2:5], clean["video"][:, 2:5])
        static += torch.equal(sample["video"][:, 5], sample["video"][:, 4])
    assert 65 < corrupted < 135
    assert 1 <= static <= 25


def test_memory_corruption_uint8_contract():
    inputs = np.stack([pattern(40), pattern(60), pattern(80)])
    output = degrade_memory_frames(inputs, np.random.default_rng(123))
    assert output.shape == inputs.shape and output.dtype == np.uint8
    assert not np.array_equal(output, inputs)


def test_real_manifest_dataset_can_sample_without_patched_reader(data_factory):
    data = data_factory()
    sample = data[19]
    assert sample["video"].shape == (3, 13, 224, 224)
    assert torch.isfinite(sample["action"]).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"memory_degrade_prob": -0.1},
        {"memory_degrade_prob": float("nan")},
        {"static_video_probability": 2},
        {"weight_decay": -1},
        {"use_image_aug": "false"},
        {"typo_prob": 0.5},
    ],
)
def test_bad_recipe_settings_fail_before_training(overrides):
    with pytest.raises(ValueError):
        RecipeSettings.from_settings({"cosmos_recipe": overrides})


def test_recipe_defaults_and_weight_decay_reach_native_config():
    recipe = RecipeSettings.from_settings(
        dict(weight_decay=0.012, static_video_probability=0.08)
    )
    cfg = SimpleNamespace(optimizer=SimpleNamespace(lr=1e-6, weight_decay=0.1))
    assert apply_training_settings(cfg, recipe) is cfg
    assert cfg.optimizer.weight_decay == 0.012 and cfg.optimizer.lr == 1e-6
    assert recipe.memory_degrade_prob == 0.5 and recipe.static_video_probability == 0.08
    assert recipe.identity()["original_augmentation_implementation_available"] is False


def arguments(tmp_path, **kwargs):
    settings = tmp_path / "settings.json"
    atomic_json(settings, dict(sampling={}))
    return SimpleNamespace(
        resume=None,
        cosmos_recipe="auto",
        settings=settings,
        output=tmp_path / "output",
        **kwargs,
    )


def resume_marker(tmp_path, corrected):
    path = tmp_path / "checkpoint"
    identity = {"model": "cosmos"}
    if corrected:
        identity["cosmos_recipe"] = RecipeSettings().identity()
    atomic_json(path / "wm_update_complete.json", dict(identity=identity, step=200))
    return path


def test_auto_selects_corrected_fresh_but_preserves_both_checkpoint_recipes(tmp_path):
    args = arguments(tmp_path)
    assert select_recipe(args) == CORRECTED
    args.resume = resume_marker(tmp_path, False)
    assert select_recipe(args) == LEGACY
    args.resume = resume_marker(tmp_path, True)
    assert select_recipe(args) == CORRECTED


@pytest.mark.parametrize("saved,requested", [(False, CORRECTED), (True, LEGACY)])
def test_optimizer_resume_cannot_silently_switch_recipe(tmp_path, saved, requested):
    args = arguments(tmp_path)
    args.resume, args.cosmos_recipe = resume_marker(tmp_path, saved), requested
    with pytest.raises(ValueError, match="Cannot resume optimizer"):
        select_recipe(args)


def test_new_identity_records_effective_recipe_dropout_and_distinct_code(tmp_path):
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                conditioner=SimpleNamespace(action=SimpleNamespace(dropout_rate=0.1))
            )
        )
    )
    base = dict(code_sha256=training_code_identity(), model="cosmos")
    identity = augmented_identity(base, RecipeSettings(), cfg)
    assert identity["cosmos_recipe"]["version"] == RECIPE_VERSION
    assert identity["effective_action_dropout"] == 0.1
    assert len(identity["recipe_code_sha256"]) == 64
    assert "cosmos_recipe" not in base


def test_scoped_native_overrides_restore_even_after_failure(tmp_path):
    import wm_update_cosmos as legacy

    args = arguments(tmp_path)
    original = (
        legacy.CosmosUpdateDataset,
        legacy.make_training_config,
        legacy.identity_for,
    )
    with pytest.raises(RuntimeError, match="test failure"):
        with corrected_training(args):
            assert legacy.CosmosUpdateDataset is not original[0]
            raise RuntimeError("test failure")
    assert original == (
        legacy.CosmosUpdateDataset,
        legacy.make_training_config,
        legacy.identity_for,
    )


def test_scoped_wrapper_changes_only_dataset_decay_and_identity(tmp_path):
    import wm_update_cosmos as legacy

    args = arguments(tmp_path)
    cfg = SimpleNamespace(
        optimizer=SimpleNamespace(lr=1e-6, weight_decay=0.1),
        model=SimpleNamespace(
            config=SimpleNamespace(
                conditioner=SimpleNamespace(action=SimpleNamespace(dropout_rate=0.1)),
                noise="native-sde",
                loss="native-rf",
                state_t=4,
            )
        ),
        checkpoint=SimpleNamespace(
            load_path="original-v3-20k", load_training_state=False
        ),
    )
    original_model_config = copy.deepcopy(cfg.model.config)
    with (
        patch.object(legacy, "make_training_config", return_value=cfg),
        patch.object(legacy, "identity_for", return_value={"model": "cosmos"}),
    ):
        with corrected_training(args):
            effective = legacy.make_training_config(args)
            identity = legacy.identity_for(args)
            assert legacy.CosmosUpdateDataset.func is ReconstructedCosmosDataset
            assert (
                legacy.CosmosUpdateDataset.keywords["recipe"].memory_degrade_prob == 0.5
            )
    assert effective.optimizer.weight_decay == 0.01
    assert effective.model.config == original_model_config
    assert effective.checkpoint.load_path == "original-v3-20k"
    assert effective.checkpoint.load_training_state is False
    assert identity["cosmos_recipe"]["version"] == RECIPE_VERSION


def test_initializer_preflight_validates_real_dcp_and_detects_truncation(tmp_path):
    import torch.distributed.checkpoint as dcp

    checkpoint = tmp_path / "initializer"
    dcp.save(
        {"net.weight": torch.arange(12).reshape(3, 4)},
        checkpoint_id=checkpoint / "model",
    )
    inventory = checkpoint_model_inventory(checkpoint)
    assert inventory["state_entries"] == 1
    assert len(inventory["model_metadata_sha256"]) == 64
    # Packaging uses file symlinks; these remain valid initializers.
    package = tmp_path / "package/model"
    package.mkdir(parents=True)
    for path in (checkpoint / "model").iterdir():
        (package / path.name).symlink_to(path)
    assert (
        checkpoint_model_inventory(package.parent)["referenced_shards"]
        == inventory["referenced_shards"]
    )
    shard = next((checkpoint / "model").glob("*.distcp"))
    shard.write_bytes(shard.read_bytes()[:100])
    with pytest.raises(ValueError, match="Truncated Cosmos model shard"):
        checkpoint_model_inventory(checkpoint)


def test_missing_initializer_fails_before_model_memory_allocation(tmp_path):
    args = arguments(tmp_path)
    with (
        patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(tmp_path / "missing")),
        patch("wm_update_cosmos.train") as native,
        patch.object(
            torch.cuda, "set_device", side_effect=AssertionError("CUDA initialized")
        ),
    ):
        with pytest.raises(FileNotFoundError, match="model/.metadata"):
            train(args)
    native.assert_not_called()


def test_rank_binds_its_gpu_before_native_backend_imports(tmp_path):
    from contextlib import contextmanager
    import torch.distributed.checkpoint as dcp

    checkpoint = tmp_path / "step200"
    dcp.save({"net.weight": torch.ones(1)}, checkpoint_id=checkpoint / "model")
    args = arguments(tmp_path)
    with (
        patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(checkpoint), LOCAL_RANK="3"),
        patch.object(torch.cuda, "set_device") as bind,
        patch("wm_update_cosmos.train") as native,
    ):

        @contextmanager
        def checked_context(*unused):
            bind.assert_called_once_with(3)
            yield

        with patch("cosmos_recipe_train.strict_native_model_load", checked_context):
            train(args)
        native.assert_called_once_with(args)


def test_strict_model_load_copies_exact_weights_and_skips_source_optimizer(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    checkpoint = tmp_path / "step200"
    source = {
        "net.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "net.action_mlp1.0.weight": torch.arange(14, dtype=torch.float32).reshape(2, 7),
    }
    dcp.save(source, checkpoint_id=checkpoint / "model")
    dcp.save({"optimizer_lr": torch.tensor(99.0)}, checkpoint_id=checkpoint / "optim")
    target = {key: torch.full_like(value, -100) for key, value in source.items()}
    audit = tmp_path / "load_audit.json"

    def native(state, reader, planner):
        dcp.load(state, storage_reader=reader, planner=planner)

    reader = dcp.FileSystemReader(checkpoint / "model")
    guarded_model_load(
        native,
        target,
        reader,
        DefaultLoadPlanner(allow_partial_load=True),
        checkpoint=checkpoint,
        audit_path=audit,
    )
    for key in source:
        assert torch.equal(source[key], target[key])
    import json

    result = json.loads(audit.read_text())
    assert result["status"] == "loaded" and result["matched_tensors"] == 2
    assert result["action_tensors"] == ["net.action_mlp1.0.weight"]
    assert "optimizer_lr" not in target


@pytest.mark.parametrize("failure", ["missing_action", "wrong_shape", "wrong_path"])
def test_strict_model_load_rejects_partial_or_wrong_initializer(tmp_path, failure):
    import torch.distributed.checkpoint as dcp
    from unittest.mock import Mock

    checkpoint = tmp_path / "source"
    dcp.save({"net.weight": torch.ones(2, 3)}, checkpoint_id=checkpoint / "model")
    target = {"net.weight": torch.zeros(2, 3)}
    if failure == "missing_action":
        target["net.action_mlp1.0.weight"] = torch.zeros(2, 7)
    elif failure == "wrong_shape":
        target["net.weight"] = torch.zeros(3, 2)
    native = Mock()
    with pytest.raises(ValueError, match="Incomplete Cosmos|checkpoint path changed"):
        guarded_model_load(
            native,
            target,
            dcp.FileSystemReader(checkpoint / "model"),
            None,
            checkpoint=checkpoint if failure != "wrong_path" else tmp_path / "other",
            audit_path=tmp_path / "audit.json",
        )
    native.assert_not_called()


def test_strict_model_load_ignores_te_extra_state_but_not_action_parameters():
    target = {"net.weight": torch.ones(2), "net.attn._extra_state": torch.empty(0)}
    metadata = SimpleNamespace(
        state_dict_metadata={"net.weight": SimpleNamespace(size=[2])}
    )
    assert model_load_coverage(target, metadata)["matched_tensors"] == 1
    target["net.action_mlp2.0.weight"] = torch.zeros(1, 7)
    with pytest.raises(ValueError, match="action_mlp2"):
        model_load_coverage(target, metadata)


def test_native_load_guard_is_scoped_and_restored_after_exception(tmp_path):
    args = arguments(tmp_path)
    native = lambda *a: None
    backend = SimpleNamespace(dcp_load_state_dict=native)
    with patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(tmp_path / "step200")):
        with pytest.raises(RuntimeError, match="load failed"):
            with strict_native_model_load(args, backend=backend):
                assert backend.dcp_load_state_dict is not native
                raise RuntimeError("load failed")
    assert backend.dcp_load_state_dict is native


def test_resume_next_update_matches_continuous_with_real_data_loader():
    from torch.utils.data import DataLoader

    class IndexedData(ReconstructedCosmosDataset):
        def __init__(self):
            self.seed, self.split = 42, "train"

        def __getitem__(self, index):
            return torch.tensor([index + 1.0, 0.5])

        def __len__(self):
            return 4

    def update(model, optimizer, batch):
        optimizer.zero_grad(set_to_none=True)
        # CPU stochastic loss makes any hidden DataLoader RNG draw observable.
        target = torch.randn(1)
        loss = (model(batch).flatten() - target).square().mean()
        loss.backward()
        optimizer.step()

    torch.random.default_generator.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    previous = DataLoader.__iter__
    with isolated_loader_rng():
        loader = iter(
            DataLoader(IndexedData(), batch_size=1, sampler=range(4), num_workers=0)
        )
        update(model, optimizer, next(loader))
        saved_model, saved_opt = (
            copy.deepcopy(model.state_dict()),
            copy.deepcopy(optimizer.state_dict()),
        )
        saved_rng = torch.get_rng_state().clone()
        update(model, optimizer, next(loader))
        expected, expected_rng = (
            copy.deepcopy(model.state_dict()),
            torch.get_rng_state().clone(),
        )
        resumed = torch.nn.Linear(2, 1)
        resumed.load_state_dict(saved_model)
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_opt.load_state_dict(saved_opt)
        torch.set_rng_state(saved_rng)
        resumed_loader = iter(
            DataLoader(IndexedData(), batch_size=1, sampler=range(1, 4), num_workers=0)
        )
        update(resumed, resumed_opt, next(resumed_loader))
        for key, value in expected.items():
            assert torch.equal(value, resumed.state_dict()[key])
        assert torch.equal(expected_rng, torch.get_rng_state())
    assert DataLoader.__iter__ is previous
    assert not torch.cuda.is_initialized()


def test_legacy_route_never_applies_new_overrides(tmp_path):
    args = arguments(tmp_path)
    args.resume = resume_marker(tmp_path, False)
    with (
        patch("wm_update_cosmos.train") as native,
        patch("cosmos_recipe_train.corrected_training") as scoped,
    ):
        train(args)
    native.assert_called_once_with(args)
    scoped.assert_not_called()


def test_driver_pins_recipe_flag_without_changing_batch_or_gpu_groups(tmp_path):
    args = arguments(
        tmp_path,
        manifest=tmp_path / "manifest.json",
        steps=2000,
        save_interval=200,
        accumulation=4,
        lr=1e-6,
        seed=42,
        gpus="0,1,2,3,4,5,6,7",
        model="both",
    )
    with patch.dict(os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"):
        jobs = training_plan(args)
    assert jobs[0]["gpus"] == ["0", "1", "2", "3"]
    assert jobs[1]["gpus"] == ["4", "5", "6", "7"]
    assert all(job["effective_batch"] == 16 for job in jobs)
    assert jobs[1]["cosmos_recipe"] == CORRECTED
    command = jobs[1]["command"]
    assert command[command.index("--cosmos-recipe") + 1] == "auto"
    assert "--reserve-memory" in command
    assert "--cosmos-recipe" not in jobs[0]["command"]


@pytest.mark.parametrize("ranks,accumulation", [(8, 2), (4, 4), (2, 8)])
def test_cosmos_single_model_keeps_batch16_on_each_supported_gpu_count(
    tmp_path, ranks, accumulation
):
    args = arguments(
        tmp_path,
        manifest=tmp_path / "manifest.json",
        steps=2000,
        save_interval=200,
        accumulation=accumulation,
        lr=1e-6,
        seed=42,
        gpus=",".join(map(str, range(ranks))),
        model="cosmos",
        ranks_per_model=ranks,
    )
    with patch.dict(os.environ, RLINF_ENV="/runtime", COSMOS_PYTHON="/cosmos/python"):
        jobs = training_plan(args)
    assert len(jobs) == 1
    assert jobs[0]["num_processes"] == ranks
    assert jobs[0]["effective_batch"] == 16
    assert jobs[0]["cosmos_recipe"] == CORRECTED
    assert "--resume" not in jobs[0]["command"]
    assert "--reserve-memory" in jobs[0]["command"]


def test_legacy_training_code_identity_still_matches_saved_checkpoints():
    assert (
        training_code_identity()
        == "0a8fc3f839108a91d7af9599a61db49d590416ba5d269b22a2027b6b39968c40"
    )
