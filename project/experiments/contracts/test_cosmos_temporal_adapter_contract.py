import json
from pathlib import Path
import sys

import pytest


CONTRACT_PATH = Path(__file__).with_name("cosmos_temporal_adapter_contract.json")
PROJECT_ROOT = CONTRACT_PATH.parents[2]
COSMOS_ROOT = PROJECT_ROOT / "third_party" / "cosmos-policy"
if str(COSMOS_ROOT) not in sys.path:
    sys.path.insert(0, str(COSMOS_ROOT))

try:
    from cosmos_policy.datasets.libero_temporal_adapter_dataset import (  # noqa: E402
        ACTION_HORIZON,
        FUTURE_PRIMARY_LATENT_INDICES,
        FUTURE_PRIMARY_OFFSETS,
        MEMORY_OFFSETS,
        NUM_CONDITIONAL_LATENT_FRAMES,
        STATE_T,
        TOKENIZER_CHUNK_DURATION,
    )
except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
    pytest.skip(
        f"Cosmos temporal adapter dependencies are unavailable: {error}",
        allow_module_level=True,
    )


def load_contract() -> dict:
    with CONTRACT_PATH.open("r", encoding="utf-8") as contract_file:
        return json.load(contract_file)


def test_temporal_time_mapping_is_unpadded_and_covers_eight_targets():
    contract = load_contract()
    time_contract = contract["time_contract"]
    assert time_contract["valid_t_min"] == 6
    assert time_contract["valid_t_max"] == "T-17"
    assert time_contract["memory_primary_indices"] == [
        f"t{offset:+d}" if offset else "t" for offset in MEMORY_OFFSETS
    ]
    assert time_contract["action_horizon_steps"] == ACTION_HORIZON
    assert time_contract["future_primary_indices"] == [
        f"t+{offset}" for offset in FUTURE_PRIMARY_OFFSETS
    ]
    assert time_contract["action_padding_allowed"] is False
    assert time_contract["future_target_padding_allowed"] is False
    assert time_contract["cross_episode_windows_allowed"] is False


def test_temporal_latent_layout_matches_active_dataset_constants():
    latent = load_contract()["latent_contract"]
    assert latent["state_t"] == STATE_T == 4
    assert latent["tokenizer_chunk_duration"] == TOKENIZER_CHUNK_DURATION == 13
    assert (
        latent["num_conditional_latent_frames"]
        == NUM_CONDITIONAL_LATENT_FRAMES
        == 2
    )
    assert latent["generated_latent_indices"] == list(
        FUTURE_PRIMARY_LATENT_INDICES
    )
    assert latent["actions_per_generated_latent"] == ACTION_HORIZON // 2
    assert len(latent["slots"]) == latent["state_t"]


def test_temporal_route_uses_real_data_and_native_lora():
    contract = load_contract()
    assert contract["route"]["selected"] is True
    assert contract["route"]["closed_loop_policy_training"] is False
    assert contract["initialization"]["backbone_frozen"] is True
    assert contract["initialization"]["lora_rank"] == 32
    assert contract["initialization"]["total_trainable_parameters"] == 113926144
    assert contract["data_contract"]["targets_must_be_real_observations"] is True
    assert contract["data_contract"]["generated_future_as_target_allowed"] is False
    assert contract["data_contract"]["manifest_schema"] == "2026-07-29.2"
    assert "trajectory_sha256" in contract["data_contract"]["split_hash"]


def test_contract_fails_closed_on_claim_scope_and_resume():
    contract = load_contract()
    assert contract["output_contract"]["value"] is None
    assert contract["output_contract"]["future_proprio"] is None
    assert (
        contract["evaluation_contract"][
            "closed_loop_libero_required_for_task_success_claim"
        ]
        is True
    )
    assert (
        contract["resume_contract"][
            "legacy_checkpoint_without_progress_is_exact_resume"
        ]
        is False
    )
