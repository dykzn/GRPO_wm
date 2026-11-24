"""CPU coverage for ordered fallback, exact resume state, and updated WM assets."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import spatial_followup_watchdogs as chain
import spatial_wovr_pair_launch as pair_launch
from spatial_followup_email import OutcomeMailer
from stop_spatial_grpo import Process


class FollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def checkpoint(self, root, step, complete=True):
        p = root / "experiment/checkpoints" / f"global_step_{step}"
        weights = p / "actor/model_state_dict/full_weights.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"policy")
        if complete:
            dcp = p / "actor/dcp_checkpoint"
            dcp.mkdir()
            (dcp / ".metadata").write_bytes(b"completed metadata")
            for rank in range(8):
                (dcp / f"__{rank}_0.distcp").write_bytes(b"optimizer shard")
        return p

    def test_fallback_selects_latest_logged_complete_optimizer_checkpoint(self):
        source, output = self.root / "source", self.root / "chain"
        initial = self.checkpoint(self.root / "original", 10)
        self.checkpoint(source, 20)
        (source / "metrics.log").write_text("Global Step: 20/20\n")
        primary = output / "grpo_primary"
        self.checkpoint(primary, 25, complete=False)
        complete = self.checkpoint(primary, 30)
        self.checkpoint(primary, 35)
        (primary / "metrics.log").write_text("Global Step: 34/40\n")
        plan = {
            "initial_resume": str(initial),
            "source_run": str(source),
            "output": str(output),
        }
        self.assertEqual(chain.latest_resume(plan), complete)

    def test_baseline_fallback_clears_new_model_selection_and_adds_twenty(self):
        plan = {
            "output": str(self.root),
            "additional_grpo_steps": 20,
            "gpus": list(map(str, range(8))),
            "source_config": "/snapshot/policy.yaml",
            "baseline_wan": "/baseline/wan",
            "baseline_cosmos": "/baseline/cosmos",
            "pi05_checkpoint_root": "/base/pi05",
        }
        with patch.dict(
            os.environ, SPATIAL_WOVR_MODEL_PAIR="/unexpected/new_pair.json"
        ):
            command, env, output, target = chain.grpo_command(
                plan, "fallback", Path("/checkpoints/global_step_30")
            )
        self.assertEqual(target, 50)
        self.assertIn("runner.max_steps=50", command)
        self.assertIn(
            "actor.fsdp_config.save_training_state_at_steps=[35,40,45,50]", command
        )
        self.assertNotIn("SPATIAL_WOVR_MODEL_PAIR", env)
        self.assertEqual(env["WAN_SPATIAL_CHECKPOINT"], "/baseline/wan")
        self.assertEqual(env["COSMOS_WRIST_CHECKPOINT"], "/baseline/cosmos")
        self.assertEqual(output, self.root / "grpo_fallback")

    def test_cleanup_selection_excludes_other_roles_and_other_jobs(self):
        processes = {}
        tags = {
            1: ("/this-chain", "primary_workload"),
            2: ("/this-chain", "fallback_workload"),
            3: ("/other-chain", "primary_workload"),
            4: ("/this-chain", "primary_notification"),
        }
        for pid, (root, role) in tags.items():
            processes[pid] = Process(pid, 0, os.geteuid(), 123, "worker", (), {})
            path = self.root / str(pid)
            path.mkdir()
            (path / "environ").write_bytes(
                f"SPATIAL_FOLLOWUP_CHAIN={root}\0SPATIAL_FOLLOWUP_ROLE={role}\0".encode()
            )
        with patch("spatial_followup_watchdogs.snapshot", return_value=processes):
            selected = chain.tagged_processes(
                {"output": "/this-chain"}, "primary", self.root
            )
        self.assertEqual(set(selected), {1})

    def test_cancel_marker_suppresses_fallback_even_if_primary_failed(self):
        (self.root / "disarmed.json").write_text("{}")
        (self.root / "primary_failed.json").write_text("{}")
        self.assertEqual(chain.primary_outcome({"output": str(self.root)}), "cancelled")

    def test_independent_watchdogs_handle_success_failure_and_supervisor_death(self):
        code = r"""
import json,os,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import spatial_followup_watchdogs as m
role,mode=sys.argv[2:4]
plan=json.loads(Path(sys.argv[4]).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
m.verify=lambda p:None
m.cleanup_workload=lambda p,r:None
m.wait_predecessor_stopped=lambda p,r:None
m.latest_resume=lambda p:Path('/saved/global_step_20')
m.create_wan_deployment=lambda p,q:None
def work(*args,**kwargs):
 if mode=='fail':raise RuntimeError('injected WM failure')
 if mode=='dead':os._exit(9)
m.run_child=work
def grpo(p,r,c,pair_path=None):
 Path(p['output'],r+'_grpo_called.json').write_text(json.dumps({'resume':str(c),'updated':pair_path is not None,'additional_steps':p['additional_grpo_steps']}))
m.run_grpo=grpo
(m.primary if role=='primary' else m.fallback)(plan)
"""
        for mode in ("success", "fail", "dead"):
            with self.subTest(mode=mode):
                output = self.root / mode
                output.mkdir()
                (output / "logs").mkdir()
                pair = output / "pair.json"
                pair.write_text("{}")
                wm = output / "wm_plan.json"
                wm.write_text('{"policy_checkpoint":"/saved/global_step_20"}')
                plan = {
                    "output": str(output),
                    "python": sys.executable,
                    "wm_plan": str(wm),
                    "pair_manifest": str(pair),
                    "additional_grpo_steps": 20,
                }
                path = output / "plan.json"
                path.write_text(json.dumps(plan))
                env = dict(
                    os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"
                )
                processes, streams = [], []
                try:
                    for role in ("fallback", "primary"):
                        log = (output / f"{role}.log").open("w")
                        streams.append(log)
                        processes.append(
                            subprocess.Popen(
                                [
                                    sys.executable,
                                    "-c",
                                    code,
                                    str(EXPERIMENTS),
                                    role,
                                    mode,
                                    str(path),
                                ],
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                            )
                        )
                    for p in reversed(processes):
                        p.wait(timeout=25)
                    self.assertEqual(processes[0].returncode, 0)
                    result = json.loads((output / "fallback_state.json").read_text())
                    mailer = OutcomeMailer(plan, output / "unused-smtp-config.json")
                    mailer.scan()
                    self.assertEqual(
                        mailer.events["primary-outcome"]["outcome"],
                        "complete" if mode == "success" else "failed",
                    )
                    self.assertEqual(
                        mailer.events["fallback-outcome"]["outcome"],
                        "not_needed" if mode == "success" else "complete",
                    )
                    if mode == "success":
                        self.assertEqual(result["status"], "not_needed")
                        self.assertTrue((output / "primary_grpo_called.json").exists())
                        self.assertFalse(
                            (output / "fallback_grpo_called.json").exists()
                        )
                    else:
                        self.assertEqual(result["status"], "complete")
                        call = json.loads(
                            (output / "fallback_grpo_called.json").read_text()
                        )
                        self.assertEqual(
                            call,
                            {
                                "resume": "/saved/global_step_20",
                                "updated": False,
                                "additional_steps": 20,
                            },
                        )
                        self.assertFalse((output / "primary_grpo_called.json").exists())
                finally:
                    for p in processes:
                        if p.poll() is None:
                            p.kill()
                            p.wait()
                    for stream in streams:
                        stream.close()


class UpdatedPairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "baseline_wan"
        self.baseline.mkdir()
        self.wan = self.root / "new_wan"
        self.wan.mkdir()
        (self.wan / "model-00001.safetensors").write_bytes(b"updated weights")
        for name in ("Wan2.2_VAE.pth", "resnet_rm.pth"):
            (self.baseline / name).write_bytes(b"frozen asset")
            (self.wan / name).symlink_to(self.baseline / name)
        self.cosmos = self.root / "cosmos/iter_000002000"
        (self.cosmos / "model").mkdir(parents=True)
        (self.cosmos / "model/.metadata").write_bytes(b"DCP metadata")
        (self.cosmos / "model/weights.distcp").write_bytes(b"trained cosmos")
        records = [
            {
                "path": str(p.relative_to(self.cosmos.parent)),
                "size": p.stat().st_size,
                "sha256": pair_launch.native.sha256(p),
            }
            for p in sorted((self.cosmos / "model").iterdir())
        ]
        tree = hashlib.sha256(
            "".join(f"{r['sha256']}  {r['path']}\n" for r in records).encode()
        ).hexdigest()
        (self.cosmos.parent / "checkpoint_manifest.json").write_text(
            json.dumps(
                {"checkpoint": self.cosmos.name, "files": records, "tree_sha256": tree}
            )
        )
        self.pair = self.root / "pair.json"
        self.pair.write_text(
            json.dumps(
                {
                    "schema": "spatial-wm-pair-v1",
                    "wan": str(self.wan),
                    "cosmos": str(self.cosmos),
                    "wan_step": 2000,
                    "cosmos_step": 2000,
                    "wan_weights_sha256": pair_launch.native.sha256(
                        self.wan / "model-00001.safetensors"
                    ),
                    "cosmos_tree_sha256": tree,
                }
            )
        )
        self.cfg = SimpleNamespace(
            env=SimpleNamespace(
                train=SimpleNamespace(
                    wan_wm_hf_ckpt_path=str(self.wan),
                    model_path=str(self.wan / "model-00001.safetensors"),
                )
            )
        )

    def test_new_checkpoint_number_and_safetensor_size_are_verified_by_checksums(self):
        with patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(self.cosmos)):
            report = pair_launch.validate_updated_pair(
                self.cfg, self.pair, self.baseline
            )
        self.assertEqual(report["cosmos_step"], 2000)

    def test_pipeline_cannot_silently_load_baseline_weights(self):
        self.cfg.env.train.model_path = str(self.baseline / "model-00001.safetensors")
        with patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(self.cosmos)):
            with self.assertRaisesRegex(ValueError, "pipeline model_path"):
                pair_launch.validate_updated_pair(self.cfg, self.pair, self.baseline)

    def test_modified_updated_weights_are_rejected(self):
        (self.wan / "model-00001.safetensors").write_bytes(b"partial weights")
        with (
            patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(self.cosmos)),
            self.assertRaisesRegex(ValueError, "checksum mismatch"),
        ):
            pair_launch.validate_updated_pair(self.cfg, self.pair, self.baseline)

    def test_updated_pair_does_not_require_the_original_cosmos_payload(self):
        run = self.root / "startup"
        run.mkdir()

        def unchanged_assets(cfg, full, run_dir, *, verify_wrist=True):
            self.assertFalse(verify_wrist)
            self.assertEqual(cfg.env.train.wan_wm_hf_ckpt_path, str(self.baseline))
            self.assertEqual(os.environ["COSMOS_WRIST_CHECKPOINT"], str(self.cosmos))
            (run_dir / "asset_check.json").write_text(json.dumps({"verified": []}))
            return True

        with (patch.object(pair_launch.native, "check_assets", side_effect=unchanged_assets),
              patch.dict(os.environ, COSMOS_WRIST_CHECKPOINT=str(self.cosmos))):
            pair_launch.install_pair_check(self.pair, self.baseline, self.root / "missing_original_cosmos")
            self.assertTrue(pair_launch.native.check_assets(self.cfg, True, run))
        report = json.loads((run / "asset_check.json").read_text())
        self.assertEqual(report["updated_pair"]["cosmos_step"], 2000)
        self.assertTrue(report["updated_pair_full_checksums"])

    def test_resolved_template_keeps_hypers_and_rebinds_shared_pool_and_output(self):
        source = self.root / "source.yaml"
        OmegaConf.save(
            OmegaConf.create(
                {
                    "runner": {
                        "max_steps": 20,
                        "logger": {"log_path": "/old"},
                        "memory_guard": {"enabled": False},
                    },
                    "actor": {
                        "micro_batch_size": 32,
                        "shared_wan_pool": {"rpc_root": "/old/socket"},
                    },
                    "env": {
                        "train": {
                            "shared_wan_pool": {"rpc_root": "/old/socket"},
                            "memory_guard": {},
                            "wan_wm_hf_ckpt_path": "/old/wan",
                            "cosmos_endpoints": [],
                            "video_cfg": {"video_base_dir": "/old/video"},
                        },
                        "eval": {"video_cfg": {"video_base_dir": "/old/eval"}},
                    },
                }
            ),
            source,
        )
        with patch.dict(
            os.environ,
            SPATIAL_WOVR_OUTPUT="/new/run",
            WAN_SPATIAL_CHECKPOINT="/new/wan",
            SPATIAL_WOVR_GPUS="0,1,2,3,4,5,6,7",
            SPATIAL_COSMOS_PORT="18159",
        ):
            cfg = pair_launch.compose_saved_config(source, ["runner.max_steps=40"])
        cfg.actor.shared_wan_pool.rpc_root = "/new/private/socket"
        self.assertEqual(cfg.env.train.shared_wan_pool.rpc_root, "/new/private/socket")
        self.assertEqual(cfg.actor.micro_batch_size, 32)
        self.assertEqual(cfg.runner.max_steps, 40)
        self.assertEqual(cfg.env.train.video_cfg.video_base_dir, "/new/run/video/train")
        self.assertEqual(cfg.env.train.cosmos_endpoints[0].port, 18159)
        self.assertEqual(cfg.env.train.model_path, "/new/wan/model-00001.safetensors")
        self.assertEqual(OmegaConf.load(source).runner.max_steps, 20)


if __name__ == "__main__":
    unittest.main()
