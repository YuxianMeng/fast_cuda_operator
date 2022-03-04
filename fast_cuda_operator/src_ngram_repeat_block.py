# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for ngram_repeat_block cuda extension """
from torch.autograd import Function
import torch
from fast_cuda_operator.clib.src_ngram_repeat import src_ngram_repeat_block


def index2mask(idxs: torch.LongTensor, vocab_size: int, pad: int = 0) -> torch.BoolTensor:
    """
    将index转化为mask tensor
    Args:
        idxs: tensor of shape [bsz, k]
        vocab_size: range of idxs
        pad: idxs中可能有padding元素，从而保证idxs的第二个维度的数量相同，这里指定pad-idx（必须为负数）
    Returns:
        mask: tensor of shape [bsz, vocab_size]. 其中mask[i][j] = True iff j in idxs[i]
    Examples:
        >>> idxs = torch.LongTensor([[0, -1], [1, -1], [2, 3]])
        >>> SrcNGramRepeatBlockFunction.index2mask(idxs, vocab_size=4, pad=-1).long()
        tensor([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1]])
    """
    assert pad <= 0
    bsz = idxs.shape[0]
    mask = torch.zeros([bsz, vocab_size - pad], dtype=torch.bool, device=idxs.device)
    if pad:
        idxs = torch.where(idxs >= 0, idxs, vocab_size)
    mask.scatter_(dim=1, index=idxs, value=True)
    if not pad:
        return mask
    return mask[:, :pad]


class SrcNGramRepeatBlockFunction(Function):
    """
    forward inputs to ngram_repeat_block cuda extension
    backward method not needed.

    """
    def forward(
        self,
        orig_tokens: torch.Tensor,
        prev_tokens: torch.Tensor,
        n: int,
        vocab_size: int,
        mask: torch.BoolTensor,
        pad: int = -1
    ):
        """
        Args:
            orig_tokens: 降重前的原文, [bsz, src_len]
            prev_tokens: 目前已经预测的前文, [bsz, step]
            n: 需要计算的ngram重复
            vocab_size: int, 词表大小
            mask: orig_tokens的保护， [bsz, src_len]， True代表这个词不可以被block，False则可以
            pad: 为了保证返回为tensor，允许pad
        Returns:
            # ngram_block_tokens: LongTensor, 需要block的ngram，是对词表的mask.  [bsz, V]
            ngram_block_tokens: LongTensor, 需要block的ngram，[bsz, src_len]
               block[i]中包含的是第i个样本中需要block的token idx.
        """
        output = src_ngram_repeat_block.forward(orig_tokens, prev_tokens, mask, vocab_size, n, pad)
        return output

    def backward(*args):
        raise NotImplementedError


class SrcNGramRepeatBlock(torch.nn.Module):
    """ Wrapper class for calling ngram_repeat_block cuda extension """
    def __init__(self):
        super(SrcNGramRepeatBlock, self).__init__()

    def reset_parameters(self):
        pass

    def forward(
        self,
        orig_tokens: torch.Tensor,
        prev_tokens: torch.Tensor,
        n: int,
        vocab_size: int,
        mask: torch.BoolTensor = None,
        pad: int = -1
    ):
        """
        Args:
            orig_tokens: 降重前的原文, [bsz, src_len]
            prev_tokens: 目前已经预测的前文, [bsz, step]
            n: 需要计算的ngram重复
            vocab_size: int, 词表大小
            mask: orig_tokens的保护， [bsz, src_len]， True代表这个词不可以被block，False则可以
            pad: 为了保证返回为tensor，允许pad
        Returns:
            # ngram_block_tokens: BoolTensor, 需要block的ngram，是对词表的mask.  [bsz, V]
            ngram_block_tokens: LongTensor, 需要block的ngram，[bsz, src_len]
               block[i]中包含的是第i个样本中需要block的token idx. -1代表pad
        Examples:
        >>> orig_tokens = torch.LongTensor([[0,1,2,3],[4,5,6,7]]).cuda()
        >>> prev_tokens = torch.LongTensor([[0,1,2],[4,5,6]]).cuda()
        >>> mask = torch.zeros_like(orig_tokens).bool().cuda()
        >>> v=8
        >>> index2mask(SrcNGramRepeatBlock()(orig_tokens, prev_tokens,4, vocab_size=v, mask=mask), vocab_size=v, pad=-1).long().cpu()
        tensor([[0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])
        """
        if mask is None:
            mask = torch.zeros_like(orig_tokens).bool()
        return SrcNGramRepeatBlockFunction.apply(
            orig_tokens,
            prev_tokens,
            n,
            vocab_size,
            mask,
            pad
        )
