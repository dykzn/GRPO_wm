"""Preserve the first failed collective instead of decoding unwritten buffers."""

import ast
from pathlib import Path
import textwrap


def strict_broadcast_exceptions(source: str) -> str:
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'MultiChannelProcessGroup')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_broadcast')
    handlers = [handler for node in method.body if isinstance(node, ast.Try)
                for handler in node.handlers
                if isinstance(handler.type, ast.Name) and handler.type.id == 'Exception']
    if len(handlers) != 1 or handlers[0].name != 'error':
        raise ValueError('Unsupported collective backend: expected one named broadcast exception handler')
    handler = handlers[0]
    if isinstance(handler.body[-1], ast.Raise):
        return source
    if 'Broadcast failed on ProcessGroup' not in ast.get_source_segment(source, handler):
        raise ValueError('Unsupported collective backend: unknown broadcast failure handler')
    body = '''\
# Do not decode torch.empty() metadata after a failed transfer.
try:
    pg_name = dist._get_process_group_name(group)
except Exception:
    pg_name = '<unregistered>'
try:
    self._logger.error(
        'Broadcast failed on ProcessGroup %s rank %s with error: %s; '
        'src=%s async_op=%s shape=%s dtype=%s device=%s',
        pg_name, self._cur_rank, error, src, async_op,
        tuple(tensor.shape), tensor.dtype, tensor.device, exc_info=True,
    )
except Exception:
    # Diagnostics must not replace the original transport/registration error.
    pass
raise
'''
    lines = source.splitlines(keepends=True)
    indent = ' ' * handler.body[0].col_offset
    lines[handler.body[0].lineno - 1:handler.end_lineno] = [textwrap.indent(body, indent)]
    result = ''.join(lines)
    compile(result, '<spatial-collective-exceptions>', 'exec')
    return result


def install_collective_overlay(package: Path, baseline: Path, link_sources):
    native = baseline / 'rlinf/scheduler/collective/multi_channel_pg.py'
    if not native.is_file():
        return None
    scheduler = package / 'scheduler'
    scheduler.unlink()  # replace only the private launch's package link
    link_sources(scheduler, [native.parents[1]])
    collective = scheduler / 'collective'
    collective.unlink()
    link_sources(collective, [native.parent])
    target = collective / native.name
    target.unlink()
    target.write_text(strict_broadcast_exceptions(native.read_text()))
    return target
