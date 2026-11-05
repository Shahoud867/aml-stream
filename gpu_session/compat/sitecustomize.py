"""Makes the official Multi-GNN code run properly on this machine, without editing it or the hashed research code.

1. Official train_util.AddEgoIds overrides BaseTransform.__call__ but not forward, which torch-geometric 2.7.0 marks abstract,
   so creating one raises TypeError. Only that flag is cleared: forward is never called for this class, because its own
   __call__ replaces the method that would call it, so the official transform runs exactly as written.

2. Without a limit, PyTorch's allocator kept reserving past the card's memory through fragmentation (measured on Multi-GIN,
   HI-Small, RTX 3080: 8.3 GB actually in use, 16.5 GB reserved). Windows pages the excess to system RAM instead of raising
   an error, so training runs about 5 times slower with no message. With a limit the allocator reuses its cache instead.
   Measured: 2.33 s per step with no limit; 0.98 s with the settings below. 0.85 ran out of memory. No model setting is touched.
   Set AMLH_GPU_MEMORY_FRACTION to change it (0 turns it off). Lower it if another process shares the GPU.

3. mgnn.py's mgnn_replay reads each bucket_*.npz with np.load, then deletes the file while that NpzFile is still open.
   Linux allows deleting an open file; Windows refuses (WinError 32), so every replay job failed deterministically on this
   machine. np.load is replaced with a version that reads an .npz fully into a plain dict and closes it immediately, which
   is all mgnn_replay's dict-style access needs. The only other np.load call in this codebase or the pinned Multi-GNN/FraudGT
   commits is this one, so nothing else is affected.

run_gpu.py puts this folder on PYTHONPATH. It acts only in an isolated job process (amlh.isolate sets AMLH_ISOLATE).
"""
import os

if "AMLH_ISOLATE" in os.environ:
    import numpy as _np

    _orig_np_load = _np.load

    def _closing_np_load(*args, **kwargs):
        result = _orig_np_load(*args, **kwargs)
        if isinstance(result, _np.lib.npyio.NpzFile):
            data = {k: result[k] for k in result.files}
            result.close()
            return data
        return result

    _np.load = _closing_np_load

    # Goes with the limit below. Large blocks are never split, so a free one can always be released whole. Measured on 390 batches
    # with limit 0.95 and the cache release: no failure, 0.98 s per step. Without this, the same limit failed within two epochs.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

    import torch
    import torch.optim.optimizer  # noqa: F401  (torch.optim is not imported by bare `import torch`; the attribute below needs it)
    from torch_geometric.transforms import BaseTransform

    BaseTransform.__abstractmethods__ = frozenset()

    fraction = float(os.environ.get("AMLH_GPU_MEMORY_FRACTION", "0.95"))
    if fraction > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)

        # With the limit alone, fragmentation ran out of memory within two epochs (6.8 GB in use, 2.24 GB reserved and unusable).
        # expandable_segments is not available on Windows, so the unused cache is released after each training step.
        # This frees only memory no tensor is using, so it cannot change any result.
        def _release_cache(optimizer, args, kwargs):
            torch.cuda.empty_cache()

        torch.optim.optimizer.register_optimizer_step_post_hook(_release_cache)
