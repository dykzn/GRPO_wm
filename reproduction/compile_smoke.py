"""Compile against the restored Python ABI, optionally including Triton's driver.

The Triton probe uses a fresh cache and compiles/loads cuda_utils. It never
creates a CUDA context, launches a kernel, or starts a world-model service.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile


def include_paths(prefix):
    scheme = sysconfig.get_default_scheme()
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    paths = sysconfig.get_paths(scheme=scheme)
    include = Path(paths['include']).resolve()
    if not include.is_relative_to(prefix.resolve()):
        raise RuntimeError(f'Python include path is outside the restored runtime: {include}')
    for name in ('Python.h', 'pyconfig.h'):
        if not (include / name).is_file():
            raise RuntimeError(f'Missing bundled Python header: {include / name}; download the latest release and rerun restore')
    return [include]


def compile_abi(prefix, directory):
    includes = include_paths(prefix)
    compiler = os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
    if not compiler:
        raise RuntimeError('C compiler missing; install build-essential (gcc and libc headers)')
    source = directory / 'python_abi.c'
    source.write_text('''#include <Python.h>
static PyObject *version(PyObject *self, PyObject *args) {
    return PyUnicode_FromString(PY_VERSION);
}
static PyMethodDef methods[] = {{"version", version, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_release_abi", NULL, -1, methods};
PyMODINIT_FUNC PyInit__release_abi(void) { return PyModule_Create(&module); }
''')
    output = directory / ('_release_abi' + sysconfig.get_config_var('EXT_SUFFIX'))
    command = [*shlex.split(compiler), '-shared', '-fPIC', '-O0', str(source),
               '-o', str(output), *[f'-I{path}' for path in includes]]
    print('Python ABI compile:', shlex.join(command), flush=True)
    subprocess.run(command, check=True, timeout=120)
    spec = importlib.util.spec_from_file_location('_release_abi', output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.version().split()[0] != sys.version.split()[0]:
        raise RuntimeError('Bundled headers report a different Python version than the interpreter')
    return dict(python=sys.version.split()[0], include=str(includes[0]), abi_compile='passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-prefix', type=Path, required=True)
    parser.add_argument('--triton', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Do not let an inherited CUDA visibility or a populated user cache mask
    # the missing-header failure that this check is intended to detect.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tempfile.TemporaryDirectory(prefix='spatial-compile-smoke-') as temporary:
        directory = Path(temporary)
        report = compile_abi(args.runtime_prefix, directory)
        if args.triton:
            os.environ['TRITON_CACHE_DIR'] = str(directory / 'triton-cache')
            import triton
            from triton.backends.nvidia.driver import CudaUtils

            CudaUtils()
            compiled = list((directory / 'triton-cache').rglob('*.so'))
            if not compiled:
                raise RuntimeError('Triton did not produce a fresh cuda_utils shared library')
            report.update(triton=triton.__version__, cuda_utils_compile='passed',
                          fresh_cache=True, compiled_modules=[p.name for p in compiled])
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            raise RuntimeError('Compile-only smoke unexpectedly initialized CUDA')
        report['cuda_context_created'] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
