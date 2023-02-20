import numpy as np

from chell.core import loss
from chell.core import op


# todo: support multi batch
class SoftMaxCrossEntropy(loss.Loss):
    """Fused softmax and cross entropy loss."""

    def __init__(self, output: op.Operation, label: op.Operation):
        """ Do a softmax and then compute the cross entropy loss.

        Note that output and label should be M x N .
        1. Every col of label MUST be a one-hot M * 1 vector.
        2. Every col of output MUST be a probability M * 1 vector.
        """
        super().__init__(output, label)

    def _compute(self):
        output_softmax = self.__output_softmax()

        label = self.inputs["label"]
        cross_entropy = -np.sum(label.value * np.log(output_softmax))
        return np.array([cross_entropy])

    def __output_softmax(self) -> np.ndarray:
        output = self.inputs["output"]
        output_softmax = output.value.copy()
        output_softmax[output_softmax > 1e2] = 1e2
        exp_v = np.exp(output_softmax)
        output_softmax = exp_v / (np.sum(exp_v, axis=0, keepdims=True) + 1e-10) + 1e-10  # we will do log later
        return output_softmax

    def _jacobian(self):
        output_softmax = self.__output_softmax()

        return {"output": (output_softmax - self.inputs["label"].value).reshape(1, output_softmax.size),
                "label": -np.log(output_softmax).reshape(1, output_softmax.size)}
