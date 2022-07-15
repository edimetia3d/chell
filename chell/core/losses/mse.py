import numpy as np

from chell.core import loss


class MSE(loss.Loss):

    def _compute(self):
        output = self.inputs["output"]
        label = self.inputs["label"]
        return np.array([np.mean(np.square(output.value - label.value))])

    def _jacobian(self):
        output = self.inputs["output"].value
        label = self.inputs["label"].value
        grad_output = 2 * (output.ravel() - label.ravel()) / output.size
        grad_output = grad_output.reshape((1, grad_output.size))

        return {"output": grad_output, "label": grad_output}
