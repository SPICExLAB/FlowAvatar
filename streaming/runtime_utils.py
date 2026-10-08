"""Runtime utilities for the streaming demo: CUDA memory cleanup."""

import gc
import logging

import torch

logger = logging.getLogger("flowavatar.runtime")


def clean_cuda_memory():
    """
    Clean CUDA memory to prevent memory leaks
    Returns memory usage statistics before and after cleaning
    """
    if not torch.cuda.is_available():
        return None

    # Get memory stats before cleaning
    allocated_before = torch.cuda.memory_allocated() / (1024**2)
    reserved_before = torch.cuda.memory_reserved() / (1024**2)

    # Force garbage collection first
    gc.collect()

    # Clear cache
    torch.cuda.empty_cache()

    # Get memory stats after cleaning
    allocated_after = torch.cuda.memory_allocated() / (1024**2)
    reserved_after = torch.cuda.memory_reserved() / (1024**2)

    memory_freed = reserved_before - reserved_after

    logger.debug(f"CUDA Memory: Allocated {allocated_before:.1f}MB -> {allocated_after:.1f}MB, "
                 f"Reserved {reserved_before:.1f}MB -> {reserved_after:.1f}MB, "
                 f"Freed {memory_freed:.1f}MB")

    return {
        'allocated_before': allocated_before,
        'allocated_after': allocated_after,
        'reserved_before': reserved_before,
        'reserved_after': reserved_after,
        'memory_freed': memory_freed
    }
