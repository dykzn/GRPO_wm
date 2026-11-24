"""Wait for completed GRPO, collect truth, update both WMs, then continue GRPO."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from queue_spatial_wm_update import completed_step, process_alive, sha256, utc, write_json
from spatial_wovr_memory_guard import checkpoint_complete

PROJECT = Path(__file__).resolve().parents[1]


def upstream_ready(plan):
    root, run = Path(plan["upstream"]), Path(plan["upstream_run"])
    state = json.loads((root / "primary_state.json").read_text())
    if state.get("status") in {"failed", "cancelled"} or (root / "disarmed.json").exists():
        raise RuntimeError("Preceding GRPO did not finish successfully")
    alive = process_alive(plan["upstream_identity"])
    if state.get("status") != "complete":
        if not alive:
            state = json.loads((root / "primary_state.json").read_text())
            if state.get("status") != "complete":
                raise RuntimeError("Preceding supervisor exited without successful completion")
        else:
            return False
    if alive:
        return False
    if (run / "exit_code").read_text().strip() != "0":
        raise RuntimeError("Preceding GRPO exited with an error")
    if completed_step(run) < plan["policy_step"]:
        raise RuntimeError("Preceding GRPO did not reach the required policy step")
    if not checkpoint_complete(Path(plan["policy_checkpoint"]), 8):
        raise RuntimeError("Final policy/optimizer checkpoint is incomplete")
    evaluation = json.loads(Path(plan["policy_evaluation"]).read_text())
    rows = evaluation["episode_results"]
    counts = Counter(row["task_id"] for row in rows)
    if (len(rows) != 500 or len({(r["task_id"],r["trial_id"]) for r in rows}) != 500
            or any(counts[i] != 50 for i in range(10))):
        raise RuntimeError("Final real Spatial evaluation is incomplete")
    return True


def verify(plan):
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != plan["boot_id"]:
        raise RuntimeError("The continuation belongs to another execution node")
    if plan["stages"] != stages(plan):
        raise RuntimeError("Frozen continuation commands changed")
    for model, variable in (("wan", "WAN_SPATIAL_CHECKPOINT"), ("cosmos", "COSMOS_WRIST_CHECKPOINT")):
        if plan["environment"].get(variable) != plan[f"initializer_{model}"]:
            raise RuntimeError(f"The {model} update must inherit the recorded current weights")
    for name, digest in plan["code"].items():
        if sha256(Path(name)) != digest:
            raise RuntimeError(f"Reviewed continuation code changed: {name}")
    for name, digest in plan["inputs"].items():
        if sha256(Path(name)) != digest:
            raise RuntimeError(f"Frozen continuation input changed: {name}")
    for record in plan["assets"]:
        path = Path(record["path"])
        if (not path.is_file() or path.stat().st_size != record["size"]
                or path.stat().st_mtime_ns != record["mtime_ns"]):
            raise RuntimeError(f"World-model initializer changed: {path}")


def state(plan, status, **details):
    root = Path(plan["output"])
    value = dict(status=status, updated_utc=utc(), pid=os.getpid(), **details)
    write_json(root / "state.json", value)
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    pointer = PROJECT / "runtime_artifacts/training_queue/iterative_policy40_current.json"
    if pointer.exists():
        current = json.loads(pointer.read_text())
        if current.get("output") == str(root):
            current.update(value)
            write_json(pointer, current)
    print(json.dumps(value), flush=True)


def stages(plan):
    root, wm = Path(plan["output"]), Path(plan["wm_output"])
    base = [plan["python"], str(PROJECT / "experiments/wm_update.py")]
    return [
        dict(name="collect", command=[plan["python"], str(PROJECT / "experiments/wm_update_reserved_collect.py"),
            "collect", "--gpus", ",".join(plan["gpus"]), "--per-task", str(plan["per_task"]),
            "--policy-config", str(root / "inputs/policy_config.yaml"),
            "--policy-checkpoint", plan["policy_checkpoint"], "--mode", "train",
            "--seed", str(plan["seed"]), "--max-steps", "240", "--settle-steps", "10",
            "--output", str(wm / "collection")]),
        dict(name="prepare", command=[plan["python"], str(Path(__file__).resolve()),
            "prepare-data", "--plan", str(root / "plan.json")]),
        dict(name="inspect", command=base + ["inspect", "--manifest", str(wm / "manifest.json"), "--verify-files"]),
        dict(name="train", command=[plan["python"],str(PROJECT / "experiments/wm_update_reserved_collect.py")] + ["train", "--model", "both", "--cosmos-recipe", "v3-reconstructed",
            "--gpus", ",".join(plan["gpus"]), "--ranks-per-model", "4", "--accumulation", "4",
            "--lr", "1e-6", "--seed", str(plan["seed"]), "--manifest", str(wm / "manifest.json"),
            "--settings", str(root / "inputs/settings.json"), "--steps", str(plan["wm_steps"]),
            "--save-interval", str(plan["wm_save_interval"]), "--reserve-memory",
            "--memory-headroom-gib", "3.5", "--output", str(wm / "train")]),
        dict(name="package", command=base + ["package", "--manifest", str(wm / "manifest.json"),
            "--wan-checkpoint", str(wm / f"train/wan/step_{plan['wm_steps']:06d}"),
            "--cosmos-checkpoint", str(wm / f"train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints/iter_{plan['wm_steps']:09d}"),
            "--output", str(root / "model_pair")]),
    ]


def prepare_data(plan):
    from wm_update_data import build_manifest, inventory, load_manifest, unchanged

    wm = Path(plan["wm_output"])
    collection = json.loads((wm / "collection/collection.json").read_text())
    if len(collection["episodes"]) != 10 * plan["per_task"]:
        raise RuntimeError("The new collection is incomplete")
    rows, excluded = [], []
    for row in collection["episodes"]:
        actual = inventory(Path(row["path"]), verify_rgb=True)
        if actual != row:
            raise RuntimeError("A collected trajectory changed before training")
        if row["horizon"] < 8:
            excluded.append(dict(path=row["path"], reason="fewer than eight observed successor frames"))
        else:
            rows.append(dict(row, role="new"))
    replay = load_manifest(Path(plan["replay_manifest"]))
    for row in replay["episodes"]:
        unchanged(row)
        rows.append(dict(row, role="old"))
    result = build_manifest(rows, wm / "manifest.json", seed=plan["seed"], val_fraction=0.2)
    replay_split = {row["trajectory_id"]:row["split"] for row in replay["episodes"]}
    if any(row["split"] != replay_split[row["trajectory_id"]] for row in result["episodes"] if row["role"] == "old"):
        raise RuntimeError("An earlier held-out state would enter training")
    write_json(wm / "data_audit.json", dict(raw_new_episodes=len(collection["episodes"]),
        replay_episodes=len(replay["episodes"]), excluded=excluded, summary=result["summary"],
        new_sampling_fraction=0.5, previous_holdout_preserved=True))


def assert_idle(devices):
    from wm_update_memory import device_process_snapshot
    for device in devices:
        snapshot = device_process_snapshot(device, os.getpid())
        if snapshot["foreign_pids"]:
            raise RuntimeError(f"GPU {device} was occupied before the next stage; no other job was stopped")


def release_training_handoff(plan):
    """Release only after all eight actual workers and both checkpoints finish."""
    root = Path(plan["output"]) / "training_handoff"
    release = root / "release.json"
    if release.exists():
        return True
    records = []
    for model in ("wan","cosmos"):
        for rank in range(len(plan["model_allocation"][model])):
            path = root / f"{model}_rank{rank}.json"
            if not path.exists():
                return False
            value = json.loads(path.read_text())
            if value.get("model") != model or value.get("rank") != rank or value.get("reserved_gib",0) < 1:
                raise RuntimeError("A completed training worker did not retain its own CUDA cache")
            records.append(value)
    wm = Path(plan["wm_output"])
    paths = [wm / f"train/wan/step_{plan['wm_steps']:06d}/complete.json",
             wm / f"train/cosmos/spatial_wmevo/cosmos_wrist_v3/train/checkpoints/iter_{plan['wm_steps']:09d}/wm_update_complete.json"]
    for path in paths:
        record = json.loads(path.read_text())
        if record.get("step") != plan["wm_steps"] or record.get("world_size") != 4:
            raise RuntimeError("Both complete final model checkpoints are required for handoff")
    write_json(release,dict(released_utc=utc(),workers=records,reason="Both model updates completed; release for packaging and GRPO"))
    return True


def launch_stage(plan, name, command, environment):
    from spatial_wovr_launch import stop_process

    root = Path(plan["output"])
    log = root / "logs" / f"{name}.log"
    state(plan, "running", stage=name, log=str(log))
    child = None
    try:
        with log.open("a") as stream:
            child = subprocess.Popen(command, env=environment, cwd=PROJECT,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            from stop_spatial_grpo import read_process
            process = read_process(child.pid)
            write_json(root / "active_child.json", dict(pid=process.pid, start_ticks=process.started, stage=name))
            if name == "train":
                while child.poll() is None:
                    release_training_handoff(plan)
                    time.sleep(.5)
                code = child.returncode
                if code == 0 and not (root / "training_handoff/release.json").exists():
                    raise RuntimeError("Training workers exited without the coordinated CUDA-cache handoff")
            else:
                code = child.wait()
        if code:
            raise RuntimeError(f"Stage {name} exited {code}: {log}")
    finally:
        if child is not None:
            stop_process(child)
    state(plan, "stage_complete", stage=name, log=str(log))


def grpo_stage(plan):
    from spatial_followup_watchdogs import create_wan_deployment, grpo_command

    root = Path(plan["output"])
    pair = json.loads((root / "model_pair/pair.json").read_text())
    if pair["wan_step"] != plan["wm_steps"] or pair["cosmos_step"] != plan["wm_steps"]:
        raise RuntimeError("Updated world-model pair is incomplete")
    create_wan_deployment(plan, pair)
    command, env, output, target = grpo_command(plan, "primary", Path(plan["policy_checkpoint"]), root / "model_pair/pair.json")
    command = [arg for arg in command if not arg.startswith("actor.fsdp_config.save_training_state_at_steps=")]
    command.append(f"actor.fsdp_config.save_training_state_at_steps=[{target-10},{target}]")
    env.update(SPATIAL_WOVR_LAUNCHER=str(root / "inputs/fast_relaunch.py"), SPATIAL_COSMOS_PORT="18759",
               TORCHINDUCTOR_COMPILE_THREADS="1")
    return command, env, output, target


def run(plan):
    from spatial_followup_watchdogs import cleanup_workload, finished_grpo, run_child
    from stop_spatial_grpo import read_process

    root = Path(plan["output"])
    lock = (root / "workflow.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "worker_identity.json").exists():
        raise RuntimeError("Do not replay an existing workflow; inspect its recorded stages")
    own = read_process(os.getpid())
    identity = dict(pid=own.pid, start_ticks=own.started)
    write_json(root / "worker_identity.json", identity)
    write_json(root / "primary_identity.json", identity)
    try:
        verify(plan)
        state(plan, "waiting_for_grpo", stage="wait_policy40", target_step=plan["policy_step"])
        while not upstream_ready(plan):
            time.sleep(2)
        verify(plan)
        state(plan, "upstream_complete", stage="wait_policy40")
        env = dict(os.environ, **plan["environment"], CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1",
                   SPATIAL_FOLLOWUP_CHAIN=str(root), SPATIAL_FOLLOWUP_ROLE="primary_workload")
        for stage in stages(plan):
            verify(plan)
            if stage["name"] in {"collect", "train"}:
                assert_idle(plan["gpus"])
            free = shutil.disk_usage(Path(plan["wm_output"])).free / 2**30
            if free < (plan["minimum_initial_free_gib"] if stage["name"] == "collect" else 32):
                raise RuntimeError(f"Insufficient model-update disk headroom: {free:.1f} GiB")
            stage_env = dict(env)
            if stage["name"] == "train":
                stage_env.update(WM_UPDATE_HANDOFF_ROOT=str(root / "training_handoff"),
                                 WM_UPDATE_HANDOFF_OWNER=json.dumps(identity))
            launch_stage(plan, stage["name"], stage["command"], stage_env)
        command, grpo_env, output, target = grpo_stage(plan)
        assert_idle(plan["gpus"])
        write_json(root / "grpo_launch.json", dict(command=command, output=str(output), target_step=target))
        write_json(PROJECT / "runtime_artifacts/training_queue/grpo_latestwm_current.json", dict(
            status="launching", output=str(root), plan=str(root / "plan.json"),
            grpo_output=str(output), gpus=[int(g) for g in plan["gpus"]], gpu_count=len(plan["gpus"]),
            resume_step=plan["policy_step"], target_step=target, last_completed_step=plan["policy_step"],
            wan_step=plan["wm_steps"], cosmos_step=plan["wm_steps"], cosmos_recipe="wrist-v3-reconstructed-v1",
            pair_manifest=str(root / "model_pair/pair.json"), screen=plan["screen"],
            notifier_screen=plan["notifier_screen"], updated_utc=utc()))
        state(plan, "running", stage="grpo", target_step=target, grpo_output=str(output))
        run_child(plan,"primary","grpo",command,grpo_env,resume=plan["policy_checkpoint"],
                  target_step=target,grpo_output=str(output),updated_world_model=True)
        if not finished_grpo(output,target):
            raise RuntimeError("Continuation did not save a complete final GRPO checkpoint")
        cleanup_workload(plan,"primary")
        state(plan,"complete",stage="grpo",target_step=target,grpo_output=str(output))
        latest = PROJECT / "runtime_artifacts/training_queue/grpo_latestwm_current.json"
        current = json.loads(latest.read_text())
        if current.get("output") == str(root):
            current.update(status="complete", phase="complete", last_completed_step=target, updated_utc=utc())
            write_json(latest, current)
    except BaseException as error:
        previous = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
        try:
            cleanup_workload(plan,"primary")
        finally:
            state(plan,"failed",stage=previous.get("stage","verify"),error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


def notify(plan):
    from spatial_followup_email import OutcomeMailer, read_json
    from watch_training_email import DEFAULT_CONFIG

    root = Path(plan["output"])
    mailer = OutcomeMailer(plan,DEFAULT_CONFIG)
    # Reload arbitrary stage receipts as well as the standard terminal events.
    for file in (root / "outcome_email/outbox").glob("*.json"):
        mailer.events[file.stem]=read_json(file)
        mailer.deliveries[file.stem]=read_json(root / "outcome_email/delivery" / file.name)
    mailer.enqueue("armed",dict(outcome="armed",subject="[Spatial] 第40步后采集/更新/GRPO接续已启用",
        body=f"等待策略第40步训练和真实评估结束。\n采集500条；Wan/Cosmos各四卡更新2000步；GRPO 40→60。\n输出：{root}"))
    while True:
        result=read_json(root / "state.json")
        identity=read_json(root / "worker_identity.json")
        if identity and not process_alive(identity) and result.get("status") not in {"complete","failed"}:
            result=read_json(root / "state.json")
            if result.get("status") not in {"complete","failed"}:
                result=dict(result,status="failed",error="Workflow supervisor exited unexpectedly")
        events=root / "events.jsonl"
        for line in events.read_text().splitlines() if events.exists() else []:
            try: event=json.loads(line)
            except ValueError: continue
            if event.get("status")=="stage_complete":
                stage=event["stage"]
                mailer.enqueue("stage-"+stage,dict(outcome="complete",subject=f"[Spatial 接续] {stage} 完成",
                    body=f"阶段：{stage}\n输出：{root}\n日志：{event.get('log','')}"))
        if result.get("status") in {"complete","failed"}:
            status=result["status"]
            mailer.enqueue("primary-outcome",dict(outcome=status,subject=f"[Spatial 接续] 采集/更新/GRPO {status}",
                body=f"输出：{root}\n结果：{json.dumps(result,ensure_ascii=False)}"))
        mailer.send_due()
        done=mailer.deliveries.get("primary-outcome",{}).get("status")=="sent"
        mailer.persist(root / "outcome_email/state.json",dict(status="notified" if done else "watching",updated_utc=utc(),deliveries=mailer.deliveries))
        if done and all(row.get("status")=="sent" for row in mailer.deliveries.values()):return
        time.sleep(2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["verify","run","notify","prepare-data"])
    parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    if args.mode=="run":
        for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGHUP]:
            signal.signal(sig,lambda number,frame: (_ for _ in ()).throw(KeyboardInterrupt(f"Signal {number}")))
        run(plan)
    elif args.mode=="notify":notify(plan)
    elif args.mode=="prepare-data":verify(plan);prepare_data(plan)
    else:verify(plan);print("Frozen continuation verified; no GPU workload started")


if __name__=="__main__":main()
