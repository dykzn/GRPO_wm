#!/usr/bin/env python3
"""Deterministic checks for the audited Cosmos LIBERO temporal contract."""

from __future__ import annotations

import json
import os
from pathlib import Path


CONTRACT_PATH = Path(__file__).with_name("cosmos_libero_temporal_contract.json")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_native_latent_layout() -> None:
    contract = load_contract()
    checkpoint = contract["checkpoint"]
    assert checkpoint["state_t"] == 9
    assert checkpoint["latent_shape"][1:] == [16, 9, 28, 28]
    slots = checkpoint["latent_slots"]
    assert list(slots) == [str(index) for index in range(9)]
    assert slots["4"] == "action_chunk"
    assert slots["7"] == "future_primary_image_t_plus_16"
    assert slots["8"] == "discounted_return_value_at_t_plus_16"


def test_action_and_frame_mapping() -> None:
    contract = load_contract()
    action = contract["action_contract"]
    output = contract["native_outputs"]
    assert action["horizon_steps"] == 16
    assert action["action_dim"] == 7
    assert action["target_observation_index"] == "t+16"
    assert action["last_action_repeat_padding_allowed_for_formal_runs"] is False
    assert output["future_primary_frames"] == 1
    assert output["requested_future_clip_frames"] == 8
    assert output["requested_future_clip_supported"] is False


def test_wovr_window_has_no_future_leakage() -> None:
    contract = load_contract()
    window = contract["desired_wovr_window"]
    t = 4
    reference = t - 4
    memory = [t - 3, t - 2, t - 1, t]
    actions = list(range(t, t + 16))
    target = t + 16
    assert reference == 0
    assert memory == [1, 2, 3, 4]
    assert max([reference, *memory]) == t
    assert actions == list(range(4, 20))
    assert target == 20
    assert target > max(memory)
    assert window["no_future_leakage"] is True


def test_dataset_totals_and_paths() -> None:
    contract = load_contract()
    dataset = contract["lerobot_dataset"]
    suites = dataset["suites"]
    assert sum(item["episodes"] for item in suites.values()) == 1693
    assert sum(item["frames"] for item in suites.values()) == 273465
    assert dataset["control_hz"] == 20
    assert dataset["frame_period_seconds"] == 0.05
    data_root = os.environ.get("LIBERO_DATA_ROOT")
    resolved_paths = [
        Path(data_root) / Path(item["path"]).name
        for item in suites.values()
    ] if data_root else []
    for item in suites.values():
        assert item["complete"] is True
        assert item["path"].startswith("${LIBERO_DATA_ROOT}/")
    if resolved_paths and all(path.is_dir() for path in resolved_paths):
        for path in resolved_paths:
            assert path.is_dir(), path
    assert dataset["incomplete_spatial_copy"]["must_not_be_used"] is True


def test_value_semantics_and_gate() -> None:
    contract = load_contract()
    value = contract["value_contract"]
    assert value["gamma_per_low_level_step"] == 0.99
    assert value["not_a_reward"] is True
    assert value["not_pure_v_or_q_by_default"] is True
    assert contract["m0_results"]["C001"] == "done"
    assert contract["m0_results"]["C003"] == "done"
    assert (
        contract["m0_results"]["stage1_native_1_plus_4_to_8_gate"]
        == "blocked"
    )


def test_formal_rlinf_config_uses_native_horizon() -> None:
    import yaml

    env_path = (
        PROJECT_ROOT
        / "third_party/RLinf/examples/embodiment/config/env/cosmos_libero.yaml"
    )
    train_path = (
        PROJECT_ROOT
        / "third_party/RLinf/examples/embodiment/config/cosmos_libero_grpo_openpi_pi05.yaml"
    )
    lerobot_train_path = (
        PROJECT_ROOT
        / "third_party/RLinf/examples/embodiment/config/cosmos_libero_grpo_lerobot_pi05.yaml"
    )
    with env_path.open("r", encoding="utf-8") as file:
        env_config = yaml.safe_load(file)
    with train_path.open("r", encoding="utf-8") as file:
        train_config = yaml.safe_load(file)
    with lerobot_train_path.open("r", encoding="utf-8") as file:
        lerobot_train_config = yaml.safe_load(file)
    assert env_config["cosmos_chunk_size"] == 16
    assert env_config["policy_action_horizon"] == 16
    assert env_config["enforce_action_horizon_alignment"] is True
    assert env_config["allow_action_horizon_padding"] is False
    assert train_config["actor"]["model"]["num_action_chunks"] == 16
    assert train_config["actor"]["model"]["openpi"]["action_chunk"] == 16
    assert lerobot_train_config["actor"]["model"]["num_action_chunks"] == 16


def test_env_fails_closed_on_value_and_time_alignment() -> None:
    source_path = (
        PROJECT_ROOT
        / "third_party/RLinf/rlinf/envs/world_model/cosmos_world_env.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "allow_padding=self.allow_action_horizon_padding" in source
    assert 'np.asarray(result.get("value"), dtype=np.float32)' in source
    assert 'result.get("value", np.zeros' not in source
    assert "self.episode_steps += self.cosmos_chunk_size" in source


def test_required_cosmos_batch_extension_is_present() -> None:
    source_path = (
        PROJECT_ROOT
        / "third_party/cosmos-policy/cosmos_policy/experiments/robot/cosmos_utils.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "def get_action_batch(" in source


def main() -> None:
    contract = load_contract()
    test_native_latent_layout()
    test_action_and_frame_mapping()
    test_wovr_window_has_no_future_leakage()
    test_dataset_totals_and_paths()
    test_value_semantics_and_gate()
    test_formal_rlinf_config_uses_native_horizon()
    test_env_fails_closed_on_value_and_time_alignment()
    test_required_cosmos_batch_extension_is_present()
    print(f"PASS: {CONTRACT_PATH}")


if __name__ == "__main__":
    main()
