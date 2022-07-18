from typing import Tuple

import numpy as np

from chell.core import device


class CPU(device.ComputingDevice):

    def add(self, ctx: device.ComputingContext, x: np.ndarray, y: np.ndarray):
        z = x + y
        ctx.round_ctx.anything[id(x)] = z.shape
        return z

    def jacobian_add(self, ctx: device.ComputingContext, result: np.ndarray, x: np.ndarray, y: np.ndarray,
                     acc: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        z_shape = ctx.round_ctx.anything[id(x)]
        raise NotImplementedError
