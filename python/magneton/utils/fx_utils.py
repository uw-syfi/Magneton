import enum
from typing import Any, Callable, Union

import torch


class OpType(enum.Enum):
    INPUT = "input"
    FUNCTION = "call_function"
    METHOD = "call_method"
    MODULE = "call_module"
    OUTPUT = "output"



class OpWrapper(torch.nn.Module):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        op_name: str,
        op_type: OpType,
        op_target: Union[str, Callable],
    ):
        super().__init__()
        self.gm = gm
        self.op_name = op_name
        self.op_type = op_type
        self.op_target = op_target

    def original_forward(self, *args, **kwargs) -> Any:
        if self.op_type == OpType.FUNCTION:
            assert isinstance(self.op_target, Callable)
            return self.op_target(*args, **kwargs)
        elif self.op_type == OpType.METHOD:
            assert isinstance(self.op_target, str)
            new_self = args[0]
            return getattr(new_self, self.op_target)(*args[1:], **kwargs)
        elif self.op_type == OpType.MODULE:
            assert isinstance(self.op_target, str)
            return getattr(self.gm, self.op_target)(*args, **kwargs)
        elif self.op_type == OpType.INPUT:
            return args[0]
        elif self.op_type == OpType.OUTPUT:
            return args[0]
        return

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward is not implemented")
