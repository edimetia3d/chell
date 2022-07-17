import enum
from typing import Tuple, Union, Dict

import numpy as np

from chell.core import op


class PaddingMode(enum.Enum):
    """Same as OPENGL's definition."""
    REPEAT = 0
    MIRRORED_REPEAT = 1
    CLAMP_TO_EDGE = 2
    CLAMP_TO_BORDER = 3
    CLAMP_TO_CONST = 4


class PaddingConfig:
    def __init__(self,
                 padding_size: Union[Tuple[int, int, int, int], object] = (0, 0, 0, 0),
                 padding_mode: PaddingMode = PaddingMode.CLAMP_TO_BORDER,
                 clamp_value: Union[Tuple[float, float, float, float], float] = None):
        self.padding_size: Tuple[int, int, int, int] = padding_size  # LEFT,RIGHT,UP,DOWN
        self.padding_mode: PaddingMode = padding_mode
        self.clamp_value = clamp_value  # only used when padding_mode is CLAMP_TO_BORDER/CLAMP_TO_CONST, otherwise it is ignored


class Padding2D(op.Operation):

    def __init__(self, x: op.OpArgT, padding_config: PaddingConfig):
        """x is a 3D tensor of shape (C, H, W)"""
        self.padding_config = padding_config
        super().__init__("padding2d", {"x": x})

    def _compute(self) -> np.ndarray:
        x = self.inputs["x"].value
        left_up_i = self.padding_config.padding_size[2]
        left_up_j = self.padding_config.padding_size[0]
        height = x.shape[1] + self.padding_config.padding_size[2] + self.padding_config.padding_size[3]
        width = x.shape[2] + self.padding_config.padding_size[0] + self.padding_config.padding_size[1]

        assert self.padding_config.padding_mode == PaddingMode.CLAMP_TO_CONST

        ret = np.full(shape=(x.shape[0], height, width), fill_value=self.padding_config.clamp_value, dtype=x.dtype)
        ret[:, left_up_i:int(left_up_i + x.shape[1]), left_up_j:int(left_up_j + x.shape[2])] = x
        return ret

    def _jacobian(self) -> Dict[str, np.ndarray]:
        left_up_i = self.padding_config.padding_size[2]
        left_up_j = self.padding_config.padding_size[0]
        x = self.inputs["x"].value
        ret = np.zeros(shape=(self.value.size, x.size))
        for col, ind in enumerate(np.ndindex(*x.shape)):
            raw_ind = [[ind[0]], [ind[1] + left_up_i], [ind[2] + left_up_j]]
            raw_offset = np.ravel_multi_index(raw_ind, self.value.shape)
            ret[raw_offset[0], col] = 1

        return {"x": ret}
