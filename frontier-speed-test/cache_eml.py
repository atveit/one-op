import mlx.core as mx
from mlx_lm.models.cache import RotatingKVCache

class TropicalRotatingCache(RotatingKVCache):
    """
    EML-native Rotating KV Cache.
    Natively implements the SLC Windowing (Step B) using MLX primitives.
    """
    def __init__(self, original_cache):
        # We inherit from RotatingKVCache but tune the window size for SLC
        # Gemma 4 Dense window: 1024 tokens = 32MB < 96MB SLC
        super().__init__(max_size=1024)
        self.orig = original_cache

    def __getattr__(self, name):
        return getattr(self.orig, name)

def wrap_cache_rigorous(cache_list):
    if isinstance(cache_list, list):
        return [wrap_cache_rigorous(c) for c in cache_list]
    # No wrapping needed if we use standard RotatingKVCache
    return cache_list
