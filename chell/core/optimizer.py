from typing import Iterable, Dict

from chell.core import tensor


class Optimizer:
    def __init__(self, params: Iterable[tensor.Tensor]):
        self.params: Dict[str, tensor.Tensor] = {}
        for p in params:
            self.params[p.node_name] = p

    def zero_grad(self) -> None:
        """ A helper function to clear the gradients of all parameters, so you don't need to reset them to None manually.

        Notes:
            For chell do not accumulate gradients by default, i.e., everytime we call `Operation.backward`, newly caculated
            gradients will replace the old ones. So this function is generally not needed.
            See `accumulate_grad` argument in `Operation.backward` for more details.

        Returns:
            None

        """
        for _, p in self.params.items():
            p.grad = None

    def step(self) -> None:
        """Should be called after some new gradients had been computed, to update the parameters.

        Returns:

        """
        raise NotImplementedError
