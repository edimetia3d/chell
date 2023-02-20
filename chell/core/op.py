"""The core class of chell, auto forward/backward computation is implemented in this module.

"""
import logging
import weakref
from numbers import Number
from typing import List, Union, Dict, ClassVar, TypeVar, Optional, Set

import numpy as np


OpArgT = Union[Number, np.ndarray, "Operation"]
OperationClass = TypeVar("OperationClass", bound="Operation")
OperationClassVar = ClassVar[OperationClass]
_loger = logging.getLogger(__name__)


class Operation:
    """ Operation is the core class of computing graph. It is a node in the graph.

    Every operation has a name, and a set of inputs. By connecting inputs, all operations form a graph.

    Operation and Tensor are highly related. Tensor is a special Operation that
    1. has no inputs, thus has no parents.
    2. its value is set by user.
    3. has extra attribute `grad` to store the gradient of the tensor , and `require_grad` to indicate whether the tensor
       need to calculate gradient.

    Tensor is the source of every graph, i.e., every operation will have a path from some Tensor in graph.

    """
    node_name: str
    inputs: Dict[str, "Operation"]
    value: Union[np.ndarray, None]

    def __init__(self, node_name: str, _inputs: Dict[str, OpArgT]):
        self.node_name: str = _unique_op_name(node_name)
        self.inputs: Dict[str, Operation] = _cleanup_op_arg_dict(_inputs)

        # value is the output of the node after forward, an Operation only could have one output
        self.value: Union[np.ndarray, None] = None
        # it is the **accumulated** jacobian
        self._jacobian_v: Dict[str, np.ndarray] = {}

        # Try to compute the value of the node when init
        # and register this node to input's user
        self._users: List[weakref.ReferenceType[Operation]] = []
        is_all_input_known = True
        for _, inode in self.inputs.items():
            if inode.value is None:
                is_all_input_known = False
            inode._users.append(weakref.ref(self))
        if is_all_input_known:
            self.value = self._compute()

    def forward(self):
        """Forward may set new inputs values to the node"""
        for _, i in self.inputs.items():
            # get all input's activation ready
            if i.value is None:
                i.forward()
        # compute this node
        if self.value is None:
            self.value = self._compute()
            self.invalidate_user_value()

    def _compute(self) -> np.ndarray:
        """Return the latest value of this node by computing it with its inputs."""
        raise NotImplementedError

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
        all_grad_nodes = self.__find_all_grad_node()
        if len(all_grad_nodes) == 0:
            _loger.warning(f"Backward ignored, for all tensors needed to calculate {self} do not require grad.")
            return
        if self.value.size != 1:
            _loger.warning(
                "Calling backward on a node that has non-scalar output,"
                "Chell will update grad like doing backward on `node.sum()`")

        # update jacobian_v for this node to break recursion
        assert all_grad_nodes[-1] is self
        done_grad_nodes = set()
        self._jacobian_v = self.__accumulate_jacobian(np.ones(shape=(1, self.value.size)))
        done_grad_nodes.add(self)

        def is_node_on_grad_path(node: Operation) -> bool:
            return node in all_grad_nodes

        while True:
            for i in range(len(all_grad_nodes) - 1, -1, -1):
                node = all_grad_nodes[i]
                if node not in done_grad_nodes and node.__is_all_user_jacobian_computed(is_node_on_grad_path):
                    node.__backward(accumulate_grad, is_node_on_grad_path)
                    done_grad_nodes.add(node)
                    break
            if len(done_grad_nodes) == len(all_grad_nodes):
                break

    def __backward(self, accumulate_grad: bool, is_node_on_grad_path) -> None:
        o_grad = None
        for user_ref in self._users:
            user = user_ref()
            if user is not None and is_node_on_grad_path(user):
                for input_k_name, input_k in user.inputs.items():
                    if input_k is self:
                        assert len(
                            user._jacobian_v) > 0, "Unknown error, jacobian should have been computed by now"
                        if o_grad is None:
                            o_grad = user._jacobian_v[input_k_name].copy()
                        else:
                            o_grad += user._jacobian_v[input_k_name]
        assert o_grad is not None, "Unknown error, o_grad should have been computed by now"
        if isinstance(self, TensorOp):
            final_out = o_grad.reshape(self.value.shape)
            if self.grad is None or not accumulate_grad:
                self.grad = final_out
            else:
                self.grad += final_out
        else:
            self._jacobian_v = self.__accumulate_jacobian(o_grad)

    def _jacobian(self) -> Dict[str, np.ndarray]:
        """Compute the jacobian on every input.

        Note:
            The returned jacobian matrix will be accumulated by framework automatically, and only the accumulated
            jacobian will be stored, this function's return value will be discarded after the accumulation.

        """
        raise NotImplementedError

    def __find_all_grad_node(self) -> List["Operation"]:
        """ Find all tensors that need to calculate gradient, and all nodes
        that has path from these tensors to this node.
        """
        ret = []
        if isinstance(self, TensorOp) and self.requires_grad:
            return [self]

        for _, i in self.inputs.items():
            i_ret = i.__find_all_grad_node()
            ret = i_ret + ret
        if len(ret) > 0 and self not in ret:  # A Node may appear in multiple paths, but only need to be calculated once
            ret.append(self)
        return ret

    def __accumulate_jacobian(self, o_prod: np.ndarray) -> Dict[str, np.ndarray]:
        """ See `_jacobian` for more information. """
        jac = self._jacobian()
        assert set(jac.keys()).issubset(set(self.inputs.keys())), "Jacobian keys must be subset of input keys"
        for input_i_name, input_i_jac in jac.items():
            jac[input_i_name] = np.matmul(o_prod, input_i_jac)
        return jac

    def __is_all_user_jacobian_computed(self, is_node_on_grad_path) -> bool:
        for userref in self._users:
            user = userref()
            if user is not None and is_node_on_grad_path(user):
                if len(user._jacobian_v) == 0:
                    return False
        return True

    ############################ Utils

    def get_params(self) -> List["TensorOp"]:
        """A param is a tensor that must meet both 1. used to compute this node 2. tensor.requires_grad==True """
        ret = []
        for i in self.inputs.values():
            if isinstance(i, TensorOp) and i.requires_grad and i not in ret:
                ret.append(i)
            else:
                ret.extend(i.get_params())
        return ret

    def invalidate_user_value(self):
        """ Invalidate the value of all users of this node recursively."""
        new_users = []
        for user in self._users:
            if user() is not None:
                user().value = None
                user().invalidate_user_value()
                new_users.append(user)
        self._users = new_users

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

    def value_eq(self, other: OpArgT) -> bool:
        """Warning: operator== returns a bool, not an Operation"""
        if isinstance(other, Operation):
            return np.all(self.value == other.value)
        else:
            return np.all(self.value == other)

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


class TensorOp(Operation):
    """A special node that 1. has no parents, 2. activation in set at beginning"""

    def __init__(self, name=None, value: Union[Number, np.ndarray] = None, requires_grad=False):
        if name is None:
            name = "var"
        Operation.__init__(self, name, {})
        if value is not None:
            if np.isscalar(value):
                self.value = np.array([value])
            else:
                self.value = np.array(value)
        self.requires_grad: bool = requires_grad
        self.grad = None

    def set_value(self, value: Union[Number, np.ndarray]):
        self.value = np.array(value)
        self.invalidate_user_value()

    def __repr__(self):
        return f"{self.node_name} = {self.__class__.__name__}()"

    def _compute(self):
        # Tensor has no parents, so no need to compute
        pass

    def _jacobian(self):
        # Tensor has no parents, so no need to calculate jacobian
        pass


def _binary_template(lhs: OpArgT, rhs: OpArgT, result_op: OperationClassVar) -> Union["Operation", None]:
    if isinstance(lhs, np.ndarray) or np.isscalar(lhs) or isinstance(lhs, Operation):
        if isinstance(rhs, np.ndarray) or np.isscalar(rhs) or isinstance(rhs, Operation):
            return result_op(lhs, rhs)

    return NotImplemented


__created_node: Set[str] = set()

__UUID: int = -1


def __uuid_get():
    global __UUID
    __UUID += 1
    return __UUID


def _unique_op_name(name: str) -> str:
    mangle_name = "%" + name
    if mangle_name in __created_node:
        mangle_name = f"{mangle_name}_{__uuid_get()}"
    __created_node.add(mangle_name)
    return mangle_name


def _cleanup_op_arg_dict(inputs_: Dict[str, OpArgT]) -> Dict[str, "Operation"]:
    inputs = inputs_.copy()
    for k, v in inputs.items():
        if isinstance(v, Operation):
            continue
        elif isinstance(v, np.ndarray) or np.isscalar(v):
            inputs[k] = TensorOp(name="tmp_" + k, value=v)
        else:
            raise ValueError(f"Invalid input type {type(v)} for {k}")
    return inputs


from chell.core.ops import math  # must be at tail to avoid circular import
