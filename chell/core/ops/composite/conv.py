from typing import List, Tuple

import numpy as np

from chell.core import op
from chell.core import tensor
from chell.core.ops import concat
from chell.core.ops import conv2d
from chell.core.ops import padding2d


def conv(inputs: List[op.OpArgT],
         input_shape: Tuple[int, int, int],
         kw: int, kh: int, oc: int,
         group_num: int = 1,
         add_bias: bool = True,
         padding_config: padding2d.PaddingConfig = None,
         Act: op.OperationClassVar = None) -> op.Operation:
    for v in input_shape:
        assert v >= 1
    assert kw >= 1 and kh >= 1 and oc >= 1
    assert group_num >= 1
    input_group_size = len(inputs) // group_num
    assert input_group_size * group_num == len(inputs), "groups must be divisible to len(inputs)"
    assert oc // group_num * group_num == oc, "groups must be divisible to oc"

    convs = []
    output_channel_id = 0
    for group_id in range(group_num):
        group_member = []
        for i in range(input_group_size):
            group_member.append(inputs[group_id * input_group_size + i])
        if input_group_size != 1:
            real_input = concat.Concat(group_member)
        else:
            real_input = group_member[0]
        real_input_c = input_group_size * input_shape[0]
        for i in range(oc // group_num):
            kernel = tensor.Tensor(f"kernel_oc_{output_channel_id}", np.random.randn(real_input_c, kw, kh),
                                   requires_grad=True)
            conv_op = conv2d.Conv2D(real_input, kernel, padding_config=padding_config)
            if add_bias:
                bias = tensor.Tensor(f"bias_oc_{output_channel_id}", np.random.randn(1), requires_grad=True)
                conv_op = conv_op + bias
            if Act is not None:
                conv_op = Act(conv_op)
            output_channel_id += 1
            convs.append(conv_op)
    if oc == 1:
        return convs[0]
    else:
        return concat.Concat(convs)
