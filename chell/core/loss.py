from chell.core import op


class Loss(op.Operation):
    """Loss is just an alias of Operation with specific input"""

    def __init__(self, output: op.Operation, label: op.Operation):
        op.Operation.__init__(self, "loss", {"output": output, "label": label})

