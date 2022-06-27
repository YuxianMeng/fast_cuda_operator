# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Unit test for Ngram repeat block cuda op """

from time import time

import torch

from fast_cuda_operator import SrcNGramRepeatBlock
from fast_cuda_operator.src_ngram_repeat_block import index2mask


def get_ngram_block_tokens(
    orig_tokens: torch.Tensor,
    prev_tokens: torch.Tensor,
    n: int,
    vocab_size: int,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    根据orig_tokens和前t步的tokens确定1,2,...n gram需要block的token.
    Args:
        orig_tokens: 降重前的原文, [bsz, src_len]
        prev_tokens: 目前已经预测的前文, [bsz, step]
        n: 需要计算的ngram重复
        vocab_size: int, 词表大小
        mask: orig_tokens的保护， [bsz, src_len]， True代表这个词不可以被block，False则可以
    Returns:
        ngram_block_tokens: BoolTensor, 需要block的ngram，是对词表的mask.  [bsz, V]
    Examples:
        >>> orig_tokens = torch.LongTensor([[0,1,2,3],[4,5,6,7]])
        >>> prev_tokens = torch.LongTensor([[0,1,2],[4,5,6]])
        >>> mask = torch.zeros_like(orig_tokens).bool()
        >>> get_ngram_block_tokens(orig_tokens, prev_tokens,4, vocab_size=8, mask=mask).long()
        tensor([[0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]])
    """
    bsz, src_len = orig_tokens.size()
    step = prev_tokens.shape[1]
    output = torch.zeros([bsz, vocab_size], dtype=torch.bool, device=orig_tokens.device)

    for batch_idx in range(bsz):
        orig_token = orig_tokens[batch_idx]
        prev_token = prev_tokens[batch_idx]
        sample_mask = mask[batch_idx]
        tgt_start = step - n + 1  # 我们要block的是prev_tokens[step-n+1: ]的这n-1个gram是否match到了orig_tokens中的n-1个gram
        if tgt_start < 0:
            continue
        for orig_start in range(src_len - n + 1):
            if sample_mask[orig_start + n - 1]:
                continue
            block = True
            for offset in range(n - 1):
                if orig_token[orig_start + offset] != prev_token[tgt_start + offset] or sample_mask[
                    orig_start + offset]:
                    block = False
                    break
            if block:
                output[batch_idx][orig_token[orig_start + n - 1]] = True

    return output


def test_ngram_repeat_block_kernel(device="cuda:0"):
    """"""
    orig_tokens = torch.LongTensor([[0, 1, 2, 3], [4, 5, 6, 7]]).to(device)
    prev_tokens = torch.LongTensor([[0, 1, 2], [4, 5, 6]]).to(device)
    vocab_size = 8
    mask = torch.zeros_like(orig_tokens).bool().to(device)

    orig_tokens = torch.cat([orig_tokens] * 100, dim=0)
    prev_tokens = torch.cat([prev_tokens] * 100, dim=0)
    mask = torch.cat([mask] * 100, dim=0)

    module = SrcNGramRepeatBlock()

    # Cuda opt implementation
    t = time()
    output_fast = index2mask(module(orig_tokens, prev_tokens, 4, vocab_size, mask), vocab_size=vocab_size)
    print("optimized t:", time() - t)
    # Original implementation
    t = time()
    output_slow = get_ngram_block_tokens(orig_tokens, prev_tokens, 4, vocab_size=vocab_size, mask=mask)
    print("naive t:", time() - t)
    assert torch.all(output_slow == output_fast)
