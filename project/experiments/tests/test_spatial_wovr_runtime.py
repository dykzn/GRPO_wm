"""Check that worker-failure cleanup targets the current Ray address."""

import ast


def test_private_ray_agent_defaults_to_ephemeral_and_preserves_explicit_port():
    from types import SimpleNamespace

    from spatial_wovr_runtime import bind_private_ray_agent_ports

    class Parameters:
        def __init__(self, address=None, dashboard_agent_listen_port=52365):
            self.port = dashboard_agent_listen_port

    module = SimpleNamespace(RayParams=Parameters)
    bind_private_ray_agent_ports(module)
    wrapper = Parameters.__init__
    bind_private_ray_agent_ports(module)
    assert Parameters.__init__ is wrapper
    assert Parameters().port == 0
    assert Parameters(dashboard_agent_listen_port=12345).port == 12345
    assert Parameters("local", 12346).port == 12346


import contextlib
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_wovr_runtime import (
    bind_ray_failure_cleanup,
    wait_for_env_offload_before_update,
)


class MemoryHandoffTests(unittest.TestCase):
    def native_source(self):
        path = Path("/data41/qianb2608/wan/RLinf_opd/rlinf/runners/embodied_runner.py")
        if not path.is_file():
            self.skipTest("Installed RLinf source is not present")
        return path.read_text()

    def run_native_iteration(self, source, *, fail_offload=False):
        tree = ast.parse(source)
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "EmbodiedRunner"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        namespace = {"time": time}
        exec(  # noqa: S102 -- run the installed native loop against fake workers
            compile(
                ast.Module(body=[method], type_ignores=[]), "<native-run-loop>", "exec"
            ),
            namespace,
        )
        events = []
        state = {"offloaded": False}

        class Handle:
            def __init__(self, name):
                self.name = name

            def wait(self):
                events.append(self.name)
                if self.name == "env_offload":
                    if fail_offload:
                        raise RuntimeError("Wan offload failed")
                    state["offloaded"] = True
                return {}

        def compute_advantages():
            if not state["offloaded"]:
                raise AssertionError("Wan still on CUDA when actor allocations begin")
            events.append("advantages")
            return Handle("advantages_done")

        def wait_for_memory():
            if not state["offloaded"]:
                raise AssertionError("Memory admission ran before Wan offload")
            events.append("update_memory_admitted")

        runner = SimpleNamespace(
            cfg=SimpleNamespace(runner={}),
            global_step=0,
            max_steps=1,
            preload_opd_teacher=False,
            overlap_env_bootstrap=False,
            reward=None,
            env_channel=None,
            rollout_channel=None,
            reward_channel=None,
            actor_channel=None,
            _should_profile_step=lambda step: False,
            _should_sync_rollout_weights=lambda step: False,
            timer=lambda *args, **kwargs: contextlib.nullcontext(),
            _maybe_eval_and_checkpoint=lambda step: {},
            _log_step_metrics=lambda **kwargs: None,
            _finish_run=lambda: None,
            _wait_for_update_memory=wait_for_memory,
            actor=SimpleNamespace(
                set_global_step=lambda step: None,
                recv_rollout_trajectories=lambda **kwargs: Handle(
                    "trajectories_received"
                ),
                compute_advantages_and_returns=compute_advantages,
                run_training=lambda: Handle("actor_update"),
            ),
            rollout=SimpleNamespace(
                set_global_step=lambda step: None,
                generate=lambda **kwargs: Handle("rollout_complete"),
            ),
            env=SimpleNamespace(interact=lambda **kwargs: Handle("env_offload")),
        )
        namespace["run"](runner)
        return events, runner

    def test_native_run_exposes_missing_offload_wait_and_overlay_prevents_overlap(self):
        source = self.native_source()
        with self.assertRaisesRegex(AssertionError, "Wan still on CUDA"):
            self.run_native_iteration(source)
        events, runner = self.run_native_iteration(
            wait_for_env_offload_before_update(source)
        )
        self.assertEqual(runner.global_step, 1)
        self.assertLess(events.index("rollout_complete"), events.index("env_offload"))
        self.assertLess(events.index("env_offload"), events.index("advantages"))
        self.assertLess(
            events.index("env_offload"), events.index("update_memory_admitted")
        )
        self.assertLess(
            events.index("update_memory_admitted"), events.index("advantages")
        )
        self.assertLess(events.index("advantages"), events.index("actor_update"))

    def test_failed_offload_is_raised_before_actor_training(self):
        with self.assertRaisesRegex(RuntimeError, "Wan offload failed"):
            self.run_native_iteration(
                wait_for_env_offload_before_update(self.native_source()),
                fail_offload=True,
            )

    def test_changed_native_runner_is_rejected_instead_of_applying_twice(self):
        patched = wait_for_env_offload_before_update(self.native_source())
        with self.assertRaisesRegex(ValueError, "already waits"):
            wait_for_env_offload_before_update(patched)


class RayFailureCleanupTests(unittest.TestCase):
    def test_state_api_uses_own_cluster_despite_local_ray_environment_and_restores_it(
        self,
    ):
        path = Path(
            "/data41/qianb2608/wan/RLinf/.venv-opd/lib/python3.11/site-packages/ray/dashboard/utils.py"
        )
        if not path.is_file():
            self.skipTest("Installed Ray State API source is not present")
        tree = ast.parse(path.read_text())
        method = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "get_address_for_submission_client"
        )
        namespace = {
            "Optional": __import__("typing").Optional,
            "os": os,
            "logger": Mock(),
            "ray_constants": SimpleNamespace(
                RAY_API_SERVER_ADDRESS_ENVIRONMENT_VARIABLE="RAY_API_SERVER_ADDRESS",
                RAY_ADDRESS_ENVIRONMENT_VARIABLE="RAY_ADDRESS",
            ),
            "ray_address_to_api_server_url": lambda address: address,
            "split_address": lambda address: address.split("://", 1),
        }
        exec(  # noqa: S102 -- execute the installed address resolver without networking
            compile(
                ast.Module(body=[method], type_ignores=[]),
                "<native-ray-state-address>",
                "exec",
            ),
            namespace,
        )
        resolver = namespace["get_address_for_submission_client"]
        cluster = SimpleNamespace(
            list_actors=lambda **kwargs: resolver(kwargs.get("address"))
        )
        ray = Mock()
        ray.is_initialized.return_value = True
        ray.get_runtime_context.return_value.gcs_address = "127.0.0.1:39741"
        with patch.dict(
            os.environ,
            {
                "RAY_ADDRESS": "local",
                "RAY_API_SERVER_ADDRESS": "http://other-cluster:8265",
            },
        ):
            self.assertEqual(resolver("127.0.0.1:39741"), "http://other-cluster:8265")
            bind_ray_failure_cleanup(cluster, ray)
            self.assertEqual(cluster.list_actors(), "127.0.0.1:39741")
            self.assertEqual(os.environ["RAY_ADDRESS"], "local")
            self.assertEqual(
                os.environ["RAY_API_SERVER_ADDRESS"], "http://other-cluster:8265"
            )

    def test_state_api_environment_override_is_removed_even_after_query_failure(self):
        cluster = SimpleNamespace(list_actors=Mock(side_effect=ConnectionError))
        ray = Mock()
        ray.is_initialized.return_value = True
        ray.get_runtime_context.return_value.gcs_address = "127.0.0.1:39741"
        with patch.dict(os.environ, {}, clear=True):
            bind_ray_failure_cleanup(cluster, ray)
            with self.assertRaises(ConnectionError):
                cluster.list_actors()
            self.assertNotIn("RAY_API_SERVER_ADDRESS", os.environ)

    def test_current_gcs_is_explicit_when_other_ray_clusters_exist(self):
        original = Mock(return_value=["own actor"])
        cluster = SimpleNamespace(list_actors=original)
        ray = Mock()
        ray.is_initialized.return_value = True
        ray.get_runtime_context.return_value.gcs_address = "127.0.0.1:39741"
        bind_ray_failure_cleanup(cluster, ray)
        self.assertEqual(
            cluster.list_actors(filters=[("STATE", "=", "ALIVE")]), ["own actor"]
        )
        original.assert_called_once_with(
            address="127.0.0.1:39741", filters=[("STATE", "=", "ALIVE")]
        )

    def test_no_initialized_cluster_means_no_actor_discovery(self):
        original = Mock()
        cluster = SimpleNamespace(list_actors=original)
        ray = Mock()
        ray.is_initialized.return_value = False
        bind_ray_failure_cleanup(cluster, ray)
        self.assertEqual(cluster.list_actors(), [])
        original.assert_not_called()

    def test_explicit_api_address_is_preserved(self):
        original = Mock()
        cluster = SimpleNamespace(list_actors=original)
        ray = Mock()
        ray.is_initialized.return_value = True
        bind_ray_failure_cleanup(cluster, ray)
        cluster.list_actors(address="http://127.0.0.1:8265")
        original.assert_called_once_with(address="http://127.0.0.1:8265")


if __name__ == "__main__":
    unittest.main()
