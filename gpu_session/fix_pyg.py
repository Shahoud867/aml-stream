"""Makes the official Multi-GNN code work with torch-geometric 2.7.0 (needed because 2.6.1 breaks to_hetero). Run once per environment:

  .venv-gpu\\Scripts\\python gpu_session\\fix_pyg.py

Why: the official train_util.AddEgoIds subclasses BaseTransform and overrides __call__. In torch-geometric 2.7.0 BaseTransform is an
abstract class whose forward() is abstract, so AddEgoIds() raises "Can't instantiate abstract class AddEgoIds with abstract method forward".
In the version the official code was written for, BaseTransform was not abstract. This makes forward() a plain method again.
Behaviour of every transform that really defines forward() is unchanged, and AddEgoIds keeps using its own __call__.

How: it writes a sitecustomize.py into this environment's site-packages. That file only watches for the import of
torch_geometric.transforms.base_transform and patches that one class after it loads. No library file and no official file is edited.
Nothing here touches src/ or scripts/, so the code hash is unchanged. To undo: delete sitecustomize.py from site-packages.
"""
import sysconfig
from pathlib import Path

HOOK = '''# aml-stream compatibility hook (see gpu_session/fix_pyg.py). Delete this file to undo.
import importlib.abc
import importlib.machinery
import sys

_TARGET = "torch_geometric.transforms.base_transform"


class _PygAbstractForwardFix(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        loader, original = spec.loader, spec.loader.exec_module

        def exec_module(module):
            original(module)
            base = module.BaseTransform

            def forward(self, data):
                raise NotImplementedError(f"{type(self).__name__} defines neither forward() nor __call__()")
            base.forward = forward
            base.__abstractmethods__ = frozenset()
        loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _PygAbstractForwardFix())

# Windows lets a CUDA process spill past the card's memory into system RAM instead of raising out-of-memory, and PyTorch's cache then
# never gives memory back, so a run can crawl at a small fraction of its speed. Setting AML_CUDA_FRACTION (for example 0.9) caps what
# PyTorch may take, so it frees its cache first, or fails loudly if the work truly does not fit. Numbers are unaffected.
# Applied only to the isolated training child process, never to the launcher or other tools.
import os
if os.environ.get("AML_CUDA_FRACTION") and any("child_main" in a for a in getattr(sys, "orig_argv", [])):
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(float(os.environ["AML_CUDA_FRACTION"]), 0)
    except Exception as exc:
        sys.stderr.write(f"AML_CUDA_FRACTION not applied: {exc!r}\\n")
'''


def main():
    target = Path(sysconfig.get_paths()["purelib"]) / "sitecustomize.py"
    if target.exists() and "aml-stream compatibility hook" not in target.read_text(encoding="utf8", errors="ignore"):
        raise SystemExit(f"{target} already exists and is not ours; not overwriting it")
    target.write_text(HOOK, encoding="utf8")
    print("wrote", target)
    # check in a fresh interpreter, as a job would start
    import subprocess
    import sys
    code = ("from torch_geometric.transforms import BaseTransform\n"
            "class A(BaseTransform):\n    def __call__(self, d): return d\n"
            "class B(BaseTransform):\n    def forward(self, d): return d + 1\n"
            "assert A()(1) == 1 and B()(1) == 2\nprint('PASS: subclasses with __call__ or forward both instantiate and run')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-600:])
    raise SystemExit(r.returncode)


if __name__ == "__main__":
    main()
