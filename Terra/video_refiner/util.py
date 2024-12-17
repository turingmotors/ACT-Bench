"""https://github.com/OpenDriveLab/Vista/blob/main/vwm/util.py"""

import functools
import importlib
from inspect import isfunction

import torch
from einops import repeat


def get_obj_from_str(string: str, reload: bool = False, invalidate_cache: bool = True):
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module), cls)


def instantiate_from_config(config: dict):
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


def repeat_as_img_seq(x: list[torch.Tensor] | torch.Tensor | None, num_frames: int):
    if x is not None:
        if isinstance(x, list):
            new_x = list()
            for item_x in x:
                new_x += [item_x] * num_frames
            return new_x
        else:
            x = x.unsqueeze(1)
            x = repeat(x, "b 1 ... -> (b t) ...", t=num_frames)
            return x
    else:
        return None


def append_dims(x, target_dims):
    """
    Appends dimensions to the end of a tensor until it has target_dims dimensions.
    """

    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def autocast(f, enabled=True):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(
                enabled=enabled,
                dtype=torch.get_autocast_gpu_dtype(),
                cache_enabled=torch.is_autocast_cache_enabled()
        ):
            return f(*args, **kwargs)

    return do_autocast


def disabled_train(self, mode=True):
    """
    Overwrite model.train with this function to make sure train/eval mode does not change anymore.
    """

    return self


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params")
    return total_params


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def append_zero(x):
    return torch.cat((x, x.new_zeros([1])))


def isheatmap(x):
    if not isinstance(x, torch.Tensor):
        return False
    else:
        return x.ndim == 2
