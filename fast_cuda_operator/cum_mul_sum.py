# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for add cuda extension """
import torch
from torch.autograd import Function
from fast_cuda_operator.clib.cum_mul_sum import cum_mul_sum


class CumMulSumFunction(Function):
    """
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        d: torch.Tensor,
    ):
        """
        Args:
            x:
            d:
        Returns:
            y: same shape as x
        """
        output = cum_mul_sum.forward(x, d)
        ctx.save_for_backward(output, x, d)
        return output

    @staticmethod
    def backward(ctx, grad_y):
        grad_x, grad_d = cum_mul_sum.backward(
            grad_y.contiguous(), *ctx.saved_tensors)
        return grad_x, grad_d


class CumMulSumModule(torch.nn.Module):
    """ Wrapper class for calling cum_mul_sum cuda extension """

    def __init__(self):
        super(CumMulSumModule, self).__init__()

    def reset_parameters(self):
        pass # todo应该把lambda参数放这里？

    def forward(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
    ):
        """
        Args:
            x: [bsz, seq_len, c]
            d: [c]
        Examples:
        >>> device = "cuda:1"
        >>> x = torch.LongTensor([[0,1,2,3],[4,5,6,7]]).to(device)
        >>> d = torch.LongTensor([1.0, 1.0]).to(device)
        >>> CumMulSumModule()(x,d).cpu()
        tensor([[ 0,  1,  3, 4],
                [4, 9, 15, 22]])
        """
        return CumMulSumFunction.apply(x, d)
