import mlx.core as mx

class TropicalKVWrapper:
    def __init__(self, original_cache):
        self.orig = original_cache

    def update_and_fetch(self, keys, values):
        k, v = self.orig.update_and_fetch(keys, values)
        # Optimal SLC Fraction: 1024 tokens
        if k.shape[-2] > 1024:
             return k[:, :, -1024:, :], v[:, :, -1024:, :]
        return k, v

    def __getattr__(self, name):
        return getattr(self.orig, name)

def wrap_cache_rigorous(cache_list):
    if isinstance(cache_list, list):
        return [wrap_cache_rigorous(c) for c in cache_list]
    if hasattr(cache_list, "update_and_fetch"):
        return TropicalKVWrapper(cache_list)
    return cache_list
