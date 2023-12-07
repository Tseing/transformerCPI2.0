import numpy as np
import torch
from multimethod import multidispatch
from typing import Union


def check(expected: object, result: object):
    print("Expected: ", expected, sep="\n")
    print("Result: ", result, sep="\n")


@multidispatch
def check_shape_value(
    expected: Union[list, np.ndarray], result: Union[list, np.ndarray], var_name: str
) -> None:
    if isinstance(expected, list):
        expected = np.array(expected)
    if isinstance(result, list):
        result = np.array(result)
    print(f"{var_name}: {result.shape} `{expected.shape}`")
    assert expected.shape == result.shape, f"`{var_name}` shape is wrong."
    assert (expected.ravel() == result.ravel()).all(), f"`{var_name}` value is wrong."


@check_shape_value.register
def _(expected: torch.Tensor, result: torch.Tensor, var_name: str) -> None:
    print(f"{var_name}: {result.shape} `{expected.shape}`")
    assert expected.shape == result.shape, f"`{var_name}` shape is wrong."
    assert (expected.ravel() == result.ravel()).all(), f"`{var_name}` value is wrong."


@check_shape_value.register
def _(expected: np.ndarray, result: torch.Tensor, var_name: str) -> None:
    result = result.cpu().numpy()
    print(f"{var_name}: {result.shape} `{expected.shape}`")
    assert expected.shape == result.shape, f"`{var_name}` shape is wrong."
    assert (expected.ravel() == result.ravel()).all(), f"`{var_name}` value is wrong."


@check_shape_value.register
def _(expected: None, result: None, var_name: str) -> None:
    print(f"`{var_name}` is None, please check:")
    print("expected: ", expected)
    print("result: ", result)
