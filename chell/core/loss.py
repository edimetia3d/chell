from typing import List, Union, Dict

from chell import common
from chell.core import op


class Loss(op.Operation):
    """A special node that 1. has no parents, 2. activation in set at beginning"""

    def __init__(self, output: op.Operation, label: op.Operation):
        op.Operation.__init__(self, "loss", {"output": output, "label": label})

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        if shape_var_list is None:
            return 0
        else:
            return {"output": tuple(shape_var_list), "label": tuple(shape_var_list)}

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        return (1,)
