import logging
import weakref
from numbers import Number
from typing import List, Union, Dict, ClassVar, TypeVar, Optional, Set

import numpy as np

from chell import common
from chell.core import device
from chell.core import uuid

OpArgT = Union[Number, np.ndarray, "Operation"]
OperationClass = TypeVar("OperationClass", bound="Operation")
OperationClassVar = ClassVar[OperationClass]
_loger = logging.getLogger(__name__)


def _binary_template(lhs: OpArgT, rhs: OpArgT, result_op: OperationClassVar) -> Union["Operation", None]:
    if isinstance(lhs, np.ndarray) or np.isscalar(lhs) or isinstance(lhs, Operation):
        if isinstance(rhs, np.ndarray) or np.isscalar(rhs) or isinstance(rhs, Operation):
            return result_op(lhs, rhs)

    return NotImplemented


class Operation:
    __created_node: Dict[str, weakref.ReferenceType] = {}
    __active_device = device.AvailableDevices()[0]("default", 0)

    @staticmethod
    def __unique_op_name(name: str, new_op: "Operation") -> str:
        mangle_name = "%" + name
        if mangle_name in Operation.__created_node:
            mangle_name = f"{mangle_name}_{uuid.get()}"
        Operation.__created_node[mangle_name] = weakref.ref(new_op)
        return mangle_name

    @staticmethod
    def get_created_op_by_name(name: str) -> Union["Operation", None]:
        if name not in Operation.__created_node:
            return None
        ret = Operation.__created_node[name]()
        if ret is None:
            del Operation.__created_node[name]
            return None
        return ret

    def __init__(self, node_name: str, _inputs: Dict[str, OpArgT]):
        self.node_name = Operation.__unique_op_name(node_name, self)  # node_name will also be the output's name of node
        self.inputs: Dict[str, Operation] = self.__cleanup_op_arg_dict(_inputs)

        # self.value is the output of the node after forward, ONLY one is allowed
        self.value: Union[
            np.ndarray, None] = None
        # producted jacobian from root to current node
        # leaf node (i.e. Tensor) will have grad of np.ndarray
        # non-leaf node (i.e. non-Tensor operation) will have prod_jacobian of Dict[str, np.ndarray]
        self.prod_jacobian: Union[Dict[str, np.ndarray], np.ndarray] = {}
        self.on_grad_path = False  # whether this node is on the grad path
        self.users: List[weakref.ReferenceType[Operation]] = []  # users of the node's output
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

    def get_params(self) -> List["tensor.Tensor"]:
        """A param is a tensor that must meet both 1. used to compute this node 2. tensor.requires_grad==True """
        ret = []
        for i in self.inputs.values():
            if isinstance(i, tensor.Tensor) and i.requires_grad:
                ret.append(i)
            else:
                ret.extend(i.get_params())
        return ret

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

    def input_shape_gen(self, shape_var_list: List[int] = None) -> Union[Dict[str, common.Shape], int]:
        """ Generate input shape from shape_var_list.

        If shape_var_list is None, then return length of shape_var_list, if the length is 0, it means any length is ok.

        This feature is mainly used to generate random input shape for this node, so we can test it without having to
        know the shape rule of the operation.
        ```
        N = node.input_shape_gen(None)
        if N == 0:
           N = randint(1,INT_MAX)
        input_var_list = randint(1,INT_MAX,shape=(N,))
        input_shapes = node.input_shape_gen(input_var_list)
        output_shape = node.shape_infer(input_shapes)
        ```

        Args:
            shape_var_list:

        Returns:

        """
        raise NotImplementedError

    def shape_forward(self) -> Dict[str, common.Shape]:
        """Compute the shape of all node that required to compute this node.

        Note:
            All tensors that required to compute this node should have been set with a value that has a valid shape.

        Returns:
            A map of node-name to shape, all node that will be used to compute this node's shape will be included.

        """
        ret = {}
        self.__shape_forward(ret)
        return ret

    def __shape_forward(self, ret: Dict[str, common.Shape]):
        input_shapes = {}
        for arg_name, i in self.inputs.items():
            if i.node_name not in ret:
                i.__shape_forward(ret)
            input_shapes[arg_name] = ret[i.node_name]
        shape = self.shape_infer(input_shapes)
        ret[self.node_name] = shape

    def shape_infer(self, input_shapes: Dict[str, common.Shape]) -> common.Shape:
        raise NotImplementedError

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
        all_grad_nodes = []
        self.__mark_all_grad_node(all_grad_nodes)
        to_clear_mark = all_grad_nodes.copy()
        if not self.on_grad_path:
            _loger.warning(f"Backward ignored, for all tensors needed to calculate {self} do not require grad.")
            return
        self.__backward(accumulate_grad, True)
        assert all_grad_nodes[-1] is self
        del all_grad_nodes[-1]

        # todo: use topological sort is a better solution
        N = len(all_grad_nodes)
        while N != 0:
            for i in range(N - 1, -1, -1):
                node = all_grad_nodes[i]
                if node.__is_all_user_jacobian_computed():
                    node.__backward(accumulate_grad)
                    del all_grad_nodes[i]
                    break
            N = len(all_grad_nodes)
        for node in to_clear_mark:
            node.on_grad_path = False
            if isinstance(node, tensor.Tensor):
                node.prod_jacobian = {}

    def __mark_all_grad_node(self, ret: List["Operation"]):
        any_input_on_path = False
        if isinstance(self, tensor.Tensor) and self.requires_grad:
            any_input_on_path = True

        for _, i in self.inputs.items():
            if i.__mark_all_grad_node(ret):
                any_input_on_path = True
        self.on_grad_path = any_input_on_path
        if any_input_on_path and self not in ret:
            ret.append(self)
        return any_input_on_path

    def __backward(self, accumulate_grad: bool, at_root: bool = False) -> None:
        o_grad = None
        for user_ref in self.users:
            user = user_ref()
            if user is not None and user.on_grad_path:
                for user_input_k_name, user_input_k in user.inputs.items():
                    if user_input_k is self:
                        assert len(
                            user.prod_jacobian) > 0, "Unknown error, jacobian should have been computed by now"
                        if o_grad is None:
                            o_grad = user.prod_jacobian[user_input_k_name].copy()
                        else:
                            o_grad += user.prod_jacobian[user_input_k_name]
        if o_grad is None:
            assert at_root, "Unknown error, o_grad should have been computed by now"
            if self.value.size != 1:
                _loger.warning(
                    "Calling backward on a node that has non-scalar output,"
                    "Chell will update grad like doing backward on `node.sum()`")
            o_grad = np.ones(shape=(1, self.value.size))

        if isinstance(self, tensor.Tensor):
            final_out = o_grad.reshape(self.value.shape)
            if self.grad is None or not accumulate_grad:
                self.grad = final_out
            else:
                self.grad += final_out
        else:
            self.prod_jacobian = self._producted_jacobian(o_grad)

    def _compute(self) -> np.ndarray:
        raise NotImplementedError

    def _jacobian(self) -> Dict[str, np.ndarray]:
        """Compute the jacobian on every input.

        Note:
            User could override either this method or `_producted_jacobian`, both will be fine.
            1. If only override this one, the jacobian will get producted by _producted_jacobian automatically.
            2. If only _producted_jacobian, the accumulate_grad will be pass to _producted_jacobian, user should product
               it manually.
            3. If override both, only _producted_jacobian will be used.

        """
        raise NotImplementedError

    def _producted_jacobian(self, o_prod: np.ndarray) -> Dict[str, np.ndarray]:
        """ See `_jacobian` for more information. """
        jac = self._jacobian()
        assert set(jac.keys()).issubset(set(self.inputs.keys())), "Jacobian keys must be subset of input keys"
        for input_i_name, input_i_jac in jac.items():
            jac[input_i_name] = np.matmul(o_prod, input_i_jac)
        return jac

    def __is_all_user_jacobian_computed(self) -> bool:
        for userref in self.users:
            user = userref()
            if user is not None and user.on_grad_path:
                if len(user.prod_jacobian) == 0:
                    return False
        return True

    def dump(self):
        printed = set()
        self.__dump(printed)

    def __dump(self, printed_nodes: Set[int]):
        for _, i_op in self.inputs.items():
            i_op.__dump(printed_nodes)
        if id(self) not in printed_nodes:
            print(self)
            printed_nodes.add(id(self))

    def __repr__(self):
        return f"{self.node_name} = {self.__class__.__name__}({' , '.join([f'{k}={i.node_name}' for k, i in self.inputs.items()])})"

    def eq(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Eq)

    def all_close(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.AllClose)

    def sum(self, axis: Optional[int] = None) -> "Operation":
        return math.ReduceSum(self, axis)

    ####
    #    Operator Overloading
    #    1. All binary operator that support commutative property will be overloaded.
    #    e.g. `a * b` is the same as `b * a`, so `__rmul__` and `__mul__` will be overloaded.
    #    2. If the `a op b` could be trickily implemented by `fun(b rop a)`, it will be overloaded.
    #    e.g. `a - b` could be implemented by `(b - a) -1`,so `__rsub__` will be overloaded.
    ####

    def alleq(self, other: OpArgT) -> bool:
        """Warning: operator== returns a bool, not an Operation"""
        if isinstance(other, Operation):
            return np.all(self.value == other.value)
        else:
            return np.all(self.value == other)

    def __mul__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Mul)

    def __rmul__(self, other: OpArgT) -> "Operation":
        return self.__mul__(other)

    def __add__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Add)

    def __radd__(self, other: OpArgT) -> "Operation":
        return self.__add__(other)

    def __sub__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Sub)

    def __rsub__(self, other: OpArgT) -> "Operation":
        try_reverse = self.__sub__(other)
        if try_reverse:
            return math.Neg(try_reverse)
        return NotImplemented

    def __truediv__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Div)

    def __rtruediv__(self, other: OpArgT) -> "Operation":
        try_reverse = self.__truediv__(other)
        if try_reverse:
            return math.Reciprocal(try_reverse)
        return NotImplemented

    def __matmul__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Matmul)

    def __rmatmul__(self, other: OpArgT) -> "Operation":
        if isinstance(other, np.ndarray):
            return math.Transpose(self.__matmul__(other.T))
        elif np.isscalar(other):
            return self.__matmul__(np.array([other]))
        # if other is an Operation, this function will never be called, so we don't need to check it
        return NotImplemented

    def __pow__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Pow)

    def __mod__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.Mod)

    def __divmod__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.DivMod)

    def __lt__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.LT)

    def __le__(self, other: OpArgT) -> "Operation":
        return _binary_template(self, other, math.LE)


from chell.core import tensor  # must be at tail to avoid circular import
from chell.core.ops import math  # must be at tail to avoid circular import
