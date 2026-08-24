import torch
from typing import Any, List


def extract_tensors(obj: Any) -> List[torch.Tensor]:
    if isinstance(obj, torch.Tensor) and not torch._C._functorch.is_batchedtensor(obj):
        return [obj]
    elif isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            result.extend(extract_tensors(item))
        return result
    elif isinstance(obj, dict):
        result = []
        for value in obj.values():
            result.extend(extract_tensors(value))
        return result
    else:
        return []
