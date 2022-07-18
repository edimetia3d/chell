import weakref
from typing import Tuple

import numpy as np


class DeviceContext:
    """A Device context lives within the ComputingDevice, it is used to share some global datas."""

    def __init__(self, device: "ComputingDevice"):
        self.ref_device = weakref.ref(
            device)  # ComputingDevice will own the context, context only weakly reference to device
        self.anything = {}


class TrainContext:
    """A Train context lives in entire training. When calling forward/backward on same op.Operation multiple times, same
    TrainContext will be shared between all backward/forward rounds."""

    def __init__(self):
        self.anything = {}


class RoundContext:
    """A Round context lives only in each forward/backward round."""

    def __init__(self):
        self.anything = {}


class ComputingContext:

    def __init__(self, train_ctx: TrainContext, round_ctx: RoundContext):
        self.train_ctx = train_ctx
        self.round_ctx = round_ctx


class ComputingDevice:
    """A computing device is a virtual device that provide numerical computation APIs on a specific device."""

    __created_device = {}

    @classmethod
    def __must_no_duplicate(cls, id, instance):
        key = (cls, id)
        if key in ComputingDevice.__created_device:
            if ComputingDevice.__created_device[key]() is not None:
                raise RuntimeError(f"Device of {cls.__name__} can not have two instance with same id")
            else:
                del ComputingDevice.__created_device[key]
        ComputingDevice.__created_device[key] = weakref.ref(instance)

    def __init__(self, name: str, id: int):
        """ Device may have multiple instances, each instance has its own device context.

        Notes:
            When there are multiple physical hardware, each device is allowed to run on different hardware.
            When there has only one physical hardware, each device is allowed to run on the same hardware.
        """
        self.name = name
        self.id = id
        self.device_context = DeviceContext(self)
        ComputingDevice.__must_no_duplicate(id, self)

    def add(self, ctx: ComputingContext, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    def backward_add(self,
                     ctx: ComputingContext,
                     result: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     acc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns

        Args:
            ctx:
            result: the result of add
            x: x's final value in forward round
            y: y's final value in forward round
            acc: the jacobian that had been accumulated in backward round, it is an 1 x N array.

        Returns:
            A tuple of (jac_x,jac_y)

        """
        raise NotImplementedError


def AvailableDevices():
    from chell.core import devices  # force import all devices
    unused = devices
    return ComputingDevice.__subclasses__()
