"""Serialize world-model memory handoff and scope cleanup to this Ray job."""

import ast
import functools
import inspect
import os


def bind_private_ray_agent_ports(parameter_module):
    """Let the kernel choose this cluster's agent port instead of shared 52365.

    ray.init does not expose dashboard_agent_listen_port in this installation.
    Its RayParams default is fixed even for a private address=local cluster.
    Patch only this driver process, preserving explicitly requested ports.
    """
    cls = parameter_module.RayParams
    original = cls.__init__
    if getattr(original, "_spatial_private_agent_ports", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        bound = signature.bind_partial(self, *args, **kwargs)
        if "dashboard_agent_listen_port" not in bound.arguments:
            kwargs["dashboard_agent_listen_port"] = 0
        original(self, *args, **kwargs)

    initialize._spatial_private_agent_ports = True
    cls.__init__ = initialize


def wait_for_env_offload_before_update(source: str) -> str:
    """Patch only this launch's runner to finish env.interact before updating.

    EnvWorker sends the last trajectory before it offloads Wan. Waiting for
    actor.receive/rollout.generate therefore does not imply Wan released CUDA.
    """
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedRunner"
    )
    run = next(
        node
        for node in runner.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    waits = [
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.attr == "wait"
    ]
    rollout_waits = [
        node for node in waits if node.value.func.value.id == "rollout_handle"
    ]
    if len(rollout_waits) != 1:
        raise ValueError(
            "Unsupported native runner: expected one rollout wait in run()"
        )
    anchor = rollout_waits[0]
    if any(
        node.value.func.value.id == "env_handle" and node.lineno < anchor.lineno
        for node in waits
    ):
        raise ValueError(
            "Unsupported native runner: env wait precedes rollout completion"
        )
    if any(node.value.func.value.id == "env_handle" for node in waits):
        raise ValueError(
            "Native runner already waits for env; review the local handoff patch"
        )
    lines = source.splitlines(keepends=True)
    indent = lines[anchor.lineno - 1][: anchor.col_offset]
    lines.insert(
        anchor.end_lineno,
        f"{indent}# Finish Wan offload before advantages or actor CUDA allocations.\n"
        f"{indent}env_handle.wait()\n"
        f"{indent}if hasattr(self, '_wait_for_update_memory'):\n"
        f"{indent}    self._wait_for_update_memory()\n",
    )
    patched = "".join(lines)
    compile(patched, "<spatial-wovr-runner>", "exec")
    return patched


def bind_ray_failure_cleanup(cluster_module, ray_module) -> None:
    """Keep RLinf's failure handler from resolving an ambiguous Ray address."""
    original = cluster_module.list_actors

    def list_current_cluster_actors(*args, **kwargs):
        if not ray_module.is_initialized():
            # There is no cluster owned by this process to inspect or clean up.
            return []
        if kwargs.get("address") in (None, "", "auto", "local"):
            kwargs["address"] = ray_module.get_runtime_context().gcs_address
        # Ray's State API overrides its address argument with RAY_ADDRESS.
        # The launcher sets RAY_ADDRESS=local when creating the private cluster,
        # so use the higher-priority API override for this call only.
        key = "RAY_API_SERVER_ADDRESS"
        previous = os.environ.get(key)
        os.environ[key] = kwargs["address"]
        try:
            return original(*args, **kwargs)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous

    cluster_module.list_actors = list_current_cluster_actors
