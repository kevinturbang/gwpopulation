"""
GWPopulation
============

A collection of code for doing population inference.

All of this code will run on either CPUs or GPUs using cupy for GPU
acceleration.

This includes:
  - commonly used likelihood functions in the Bilby framework.
  - population models for gravitational-wave sources.
  - selection functions for gravitational-wave sources.

The code is hosted at `<www.github.com/ColmTalbot/gwpopulation>`_.
"""
from . import conversions, cupy_utils, hyperpe, models, utils, vt
from .hyperpe import RateLikelihood

__version__ = utils.get_version_information()

__all_with_xp = [
    models.mass,
    models.redshift,
    models.spin,
    cupy_utils,
    hyperpe,
    utils,
    vt,
]


def disable_cupy():
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(module="numpy")


def enable_cupy():
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(module="cupy")


def set_backend(module="numpy"):
    supported = ["numpy", "cupy", "jax.numpy"]
    if module not in supported:
        raise ValueError(
            f"Backed {module} not supported, should be in ', '.join(supported)"
        )
    import importlib

    try:
        xp = importlib.import_module(module)
    except ImportError:
        print(f"Cannot import {module}, falling back to numpy")
        set_backend(module="numpy")
        return
    for module in __all_with_xp:
        module.xp = xp
