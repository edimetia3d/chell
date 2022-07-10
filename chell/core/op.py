import logging
import weakref
from numbers import Number
from typing import List, Union, Dict, ClassVar, TypeVar, Optional, Set

import numpy as np

from chell.core import tensor
from chell.core import uuid
from chell.core.ops import common

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
        # producted jacobian from root to current node
        # leaf node (i.e. Tensor) will have grad of np.ndarray
        # non-leaf node (i.e. non-Tensor operation) will have prod_jacobian of Dict[str, np.ndarray]
        self.prod_jacobian: Union[Dict[str, np.ndarray], np.ndarray] = {}
        self.on_grad_path = False  # whether this node is on the grad path
        self.users: List[weakref.ref[Operation]] = []  # users of the node's output
        is_all_variable_known = True
        for _, inode in self.inputs.items():
            if inode.value is None:
                is_all_variable_known = False
            inode.users.append(weakref.ref(self))
        if is_all_variable_known:
            self.value = self._compute()

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
            self.value = self._compute()
            self._invalidate_user_value()

    def _invalidate_user_value(self):
        new_users = []
        for user in self.users:
            if user() is not None:
                user().value = None  # update this node's value will invalid all user's value
                user()._invalidate_user_value()
                new_users.append(user)
        self.users = new_users

    def backward(self, accumulate_grad: bool = False) -> None:
        """ Calculate the gradient of this node to all tensors that used to compute this node.

        Notes:
             1. Only Tensors that require_grad==True will be calculated.

        Args:
            accumulate_grad: If True, accumulate the newly computed grad to the previous grad.
                If False, replace the previous grad with the newly computed grad. Default is False.

        Returns:
            None

        """
        # on_grad_path is False by default, and will be cleared during backward
        self.__mark_all_grad_node()
        return self.__backward(0, accumulate_grad)

    def __mark_all_grad_node(self):
        any_input_on_path = False
        if isinstance(self, tensor.Tensor) and self.requires_grad:
            any_input_on_path = True

        for _, i in self.inputs.items():
            if i.__mark_all_grad_node():
                any_input_on_path = True
        self.on_grad_path = any_input_on_path
        return any_input_on_path

    def __backward(self, depth: int, accumulate_grad: bool) -> None:
        if not self.on_grad_path:
            if depth == 0:
                _loger.warning(f"Backward ignored, for all tensors needed to calculate {self} do not require grad.")
            return
        if depth == 0 and self.value.size != 1:
            _loger.warning(
                "Calling backward on a node that has non-scalar output,"
                "Chell will update grad like doing backward on `node.sum()`")

        if depth != 0:
            o_grad = None
            for user in self.users:
                if user() is not None and user().on_grad_path:
                    for k, user_i in user().inputs.items():
                        if user_i is self:
                            if o_grad is None:
                                o_grad = user().prod_jacobian[k].copy()
                            else:
                                o_grad += user().prod_jacobian[k]
        else:
            o_grad = np.ones(shape=(1, self.value.size))

        if isinstance(self, tensor.Tensor):
            batch_size = int(o_grad.size / self.value.size)
            final_shape = [batch_size, *self.value.shape]
            final_out = o_grad.reshape(final_shape)
            if self.grad is None or not accumulate_grad:
                self.grad = final_out
            else:
                self.grad += final_out

        else:
            jac = self._jacobian()
            assert jac.keys() == self.inputs.keys()
            for k, i in self.inputs.items():
                self.prod_jacobian[k] = np.matmul(o_grad, jac[k])
            for _, i in self.inputs.items():
                i.__backward(depth + 1, accumulate_grad)
        all_input_graded = True
        for _, i in self.inputs.items():
            if i.on_grad_path and len(i.prod_jacobian) == 0:
                all_input_graded = False
        if all_input_graded:
            self.on_grad_path = False
            if not isinstance(self, tensor.Tensor):
                self.prod_jacobian = {}

    def _compute(self) -> np.ndarray:
        raise NotImplementedError

    def _jacobian(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def dump(self):
        printed = set()
        self.__dump(printed)

    def __dump(self, printed_nodes: Set["Operation"]):
        for _, i_op in self.inputs.items():
            i_op.__dump(printed_nodes)
        if id(self) not in printed_nodes:
            print(self)
            printed_nodes.add(id(self))

    def __str__(self):
        return f"{self.node_name} = {self.__class__.__name__}({' , '.join([f'{k}={i.node_name}' for k, i in self.inputs.items()])})"

    def eq(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Eq)

    def all_close(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.AllClose)

    def sum(self, axis: Optional[int] = None) -> "Operation":
        return common.ReduceSum(self, axis)

    def __eq__(self, other: OpArgT) -> bool:
        if isinstance(other, Operation):
            return np.all(self.value == other.value)
        else:
            return np.all(self.value == other)

    def __mul__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Mul)

    def __add__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Add)

    def __sub__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Sub)

    def __truediv__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Div)

    def __matmul__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Matmul)

    def __pow__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Pow)

    def __mod__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.Mod)

    def __divmod__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.DivMod)

    def __lt__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.LT)

    def __le__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, common.LE)
