"""
Note: Imports are inside the fuction as torch xla installation can be tricky 
and not required for most of the cases.
To install torch xla please visit: https://github.com/pytorch/xla 
"""


def get_TPU():
    # imports the torch_xla package
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    return dev
