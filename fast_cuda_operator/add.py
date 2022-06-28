# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for add cuda extension """
import torch
from torch.autograd import Function
from fast_cuda_operator.clib.add_example import add_example


class AddFunction(Function):
    """
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        """
        output = add_example.forward(x, y)
        return output

    def backward(*args):
        raise NotImplementedError


class AddModule(torch.nn.Module):
    """ Wrapper class for calling ngram_repeat_block cuda extension """

    def __init__(self):
        super(AddModule, self).__init__()

    def reset_parameters(self):
        pass

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        对两个形状为[bsz, seq_len]的tensor做加法
        Examples:
        >>> device = "cuda:1"
        >>> x = torch.LongTensor([[0,1,2,3],[4,5,6,7]]).to(device)
        >>> y = torch.LongTensor([[4,5,6,7],[6,5,4,3]]).to(device)
        >>> AddModule()(x,y).cpu()
        tensor([[ 4,  6,  8, 10],
                [10, 10, 10, 10]])
        """
        return AddFunction.apply(x, y)
