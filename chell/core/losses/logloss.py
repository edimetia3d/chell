import numpy as np

from chell.core import loss


class LogLoss(loss.Loss):

    def _compute(self):
        output = self.inputs["output"]
        label = self.inputs["label"]
        dot = output.value * label.value
        exp_plus_1 = 1 + np.exp(np.where(dot < -1e2, 1e2, -dot))
        return np.array([np.mean(np.log(exp_plus_1))])

    def _jacobian(self):
        output = self.inputs["output"].value.ravel()
        label = self.inputs["label"].value.ravel()
        dot = output * label
        exp_plus_1 = 1 + np.exp(np.where(dot > 1e2, 1e2, dot))
        grad_output = -label / exp_plus_1
        grad_label = -output / exp_plus_1
        grad_label = grad_label.reshape((1, grad_label.size))
        grad_output = grad_output.reshape((1, grad_output.size))
        return {"output": grad_output, "label": grad_label}
