from numbers import Number
from typing import Union, Dict, List

import numpy as np

from chell import common
from chell.core import op


class Tensor(op.Operation):
    """A special node that 1. has no parents, 2. activation in set at beginning"""

    def __init__(self, name=None, value: Union[Number, np.ndarray] = None, requires_grad=False):
        if name is None:
            name = "var"
        op.Operation.__init__(self, name, {})
        if value is not None:
            if np.isscalar(value):
                self.value = np.array([value])
            else:
                self.value = np.array(value)
        self.requires_grad: bool = requires_grad
        self.grad = None

    def set_value(self, value: Union[Number, np.ndarray]):
        self.value = np.array(value)
        self._invalidate_user_value()

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        if shape_var_list is None:
            return 0
        else:
            return None

    def __repr__(self):
        return f"{self.node_name} = {self.__class__.__name__}()"

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        return self.value.shape

    def _compute(self):
        # Tensor has no parents, so no need to compute
        pass

    def _jacobian(self):
        # Tensor has no parents, so no need to update grad, local jacobian is just 1
        pass
