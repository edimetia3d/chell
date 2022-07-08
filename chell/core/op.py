import logging
import weakref
from numbers import Number
from typing import List, Union, Dict, ClassVar, TypeVar

import numpy as np

from chell.core import tensor
from chell.core import uuid
from chell.core.ops import binary

OpArgT = Union[Number, np.ndarray, "Operation"]
OperationClass = TypeVar("OperationClass", bound="Operation")
OperationClassVar = ClassVar[OperationClass]

_loger = logging.getLogger(__name__)


def _binary_template(lhs: "Operation", rhs: OpArgT, result_op: OperationClassVar) -> Union[
    "Operation", None]:
    if not isinstance(lhs, Operation):
        return NotImplemented
    if isinstance(rhs, np.ndarray) or np.isscalar(rhs) or isinstance(lhs, Operation):
        return result_op(lhs, rhs)
    else:
        return NotImplemented


class Operation:
    __used_op_name = set()

    @staticmethod
    def __unique_op_name(name: str):
        if name in Operation.__used_op_name:
            name = f"{name}_{uuid.get()}"
        Operation.__used_op_name.add(name)
        return "%" + name

    def __init__(self, node_name: str, _inputs: Dict[str, OpArgT]):
        self.node_name = Operation.__unique_op_name(node_name)  # node_name will also be the output's name of node
        self.inputs: Dict[str, Operation] = self.__cleanup_op_arg_dict(_inputs)

        # self.value is the output of the node after forward, ONLY one is allowed
        self.value: Union[
            np.ndarray, None] = None
        # abs grad from root to current node, a Jacboian Mat
        # leaf node (i.e. Tensor) will have grad of np.ndarray
        # non-leaf node (i.e. Operation) will have grad of Dict[str, np.ndarray]
        self.grad: Union[Dict[str, np.ndarray], np.ndarray] = {}
        self.on_grad_path = False  # whether this node is on the grad path
        self.users: List[weakref.ref[Operation]] = []  # users of the node's output
        is_all_variable_known = True
        for _, inode in self.inputs.items():
            if inode.value is None:
                is_all_variable_known = False
            inode.users.append(weakref.ref(self))
        if is_all_variable_known:
            self._compute()

    def __cleanup_op_arg_dict(self, _inputs) -> Dict[str, "Operation"]:
        inputs = _inputs.copy()
        for k, v in inputs.items():
            if not isinstance(v, Operation):
                inputs[k] = tensor.Tensor(name="tmp_" + k, value=v)
        return inputs

    def forward(self):
        """Forward may set new inputs values to the node"""
        for _, i in self.inputs.items():
            # get all input's activation ready
            if i.value is None:
                i.forward()
        # compute this node
        if self.value is None:
            self._compute()
            self._invalid_user_value()

    def _invalid_user_value(self):
        new_users = []
        for user in self.users:
            if user() is not None:
                user().value = None  # update this node's value will invalid all user's value
                user()._invalid_user_value()
                new_users.append(user)
        self.users = new_users

    def backward(self):
        # on_grad_path is False by default, and will be cleared during backward
        self.__mark_all_grad_node()
        return self.__backward(0)

    def __mark_all_grad_node(self):
        any_input_on_path = False
        if isinstance(self, tensor.Tensor) and self.requires_grad:
            any_input_on_path = True

        for _, i in self.inputs.items():
            if i.__mark_all_grad_node():
                any_input_on_path = True
        self.on_grad_path = any_input_on_path
        return any_input_on_path

    def __backward(self, depth):
        if not self.on_grad_path:
            if depth == 0:
                _loger.warning(f"Backward ignored, for all tensors need to calculate {self} do not require grad.")
            return
        if depth != 0:
            o_grad = np.zeros(shape=(1, self.value.size))
            for user in self.users:
                if user() is not None and user().on_grad_path:
                    for k, user_i in user().inputs.items():
                        if user_i is self:
                            o_grad += user().grad[k]
        else:
            o_grad = np.ones(shape=(1, self.value.size))
        self._upate_grad_to_jacobian()
        if isinstance(self, tensor.Tensor):
            self.grad = o_grad.reshape(self.value.shape)
        else:
            for k, i in self.inputs.items():
                self.grad[k] = np.matmul(o_grad, self.grad[k])
            for _, i in self.inputs.items():
                i.__backward(depth + 1)
        all_input_graded = True
        for _, i in self.inputs.items():
            if i.on_grad_path and len(i.grad) == 0:
                all_input_graded = False
        if all_input_graded:
            self.on_grad_path = False
            if not isinstance(self, tensor.Tensor):
                self.grad = {}

    def _compute(self):
        raise NotImplementedError

    def _upate_grad_to_jacobian(self):
        raise NotImplementedError

    def dump(self):
        for _, i_op in self.inputs.items():
            i_op.dump()
        print(self)

    def __repr__(self):
        return f"{self.node_name} = {self.__class__.__name__}({' , '.join([f'{k}={i.node_name}' for k, i in self.inputs.items()])})"

    def __eq__(self, other: OpArgT) -> bool:
        if isinstance(other, Operation):
            return np.all(self.value == other.value)
        else:
            return np.all(self.value == other)

    def op_eq(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Eq)

    def all_close(self, other: OpArgT) -> bool:
        if isinstance(other, Operation):
            return np.allclose(self.value, other.value)
        else:
            return np.allclose(self.value, other)

    def all_close_op(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.AllClose)

    def __mul__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Mul)

    def __add__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Add)

    def __sub__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Sub)

    def __truediv__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Div)

    def __matmul__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Matmul)

    def __pow__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Pow)

    def __mod__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.Mod)

    def __divmod__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.DivMod)

    def __lt__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.LT)

    def __le__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, binary.LE)
