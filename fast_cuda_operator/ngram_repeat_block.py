# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Wrapper for ngram_repeat_block cuda extension """
from torch.autograd import Function
import torch
from fast_cuda_operator.clib.ngram_repeat import ngram_repeat_block


class NGramRepeatBlockFunction(Function):
    """
    forward inputs to ngram_repeat_block cuda extension
    backward method not needed.

    """
    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        """
        Args:
            tokens(Tensor): Input tokens(Bsz*beam, seq_len)
            lprobs(Tensor): likelihood probability of current step
                Expected to be updated in place.(Bsz*beam, vocab_size)
            bsz(int): batch size
            step(int): current step: 我们默认tokens[:, 0]是<eos>， tokens[:, :step+1]是previous decoded tokens
            beam_size(int): beam size
            no_repeat_ngram_size(int): Ngram size
        """
        outputs = ngram_repeat_block.forward(tokens,
        lprobs, bsz, step, beam_size, no_repeat_ngram_size)
        return outputs

    def backward(*args):
        raise NotImplementedError


class NGramRepeatBlock(torch.nn.Module):
    """ Wrapper class for calling ngram_repeat_block cuda extension """
    def __init__(self):
        super(NGramRepeatBlock, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, tokens, lprobs, bsz,
        step, beam_size, no_repeat_ngram_size):
        """
        Args:
            tokens(Tensor): Input tokens(Bsz*beam, seq_len)
            lprobs(Tensor): likelihood probability,
            Expected to be updated in place.(Bsz*beam, vocab_size)
            bsz(int): batch size
            step(int): current step
            beam_size(int): beam size
            no_repeat_ngram_size(int): Ngram size
        """
        assert tokens.size(0)== bsz*beam_size
        assert lprobs.size(0)== bsz*beam_size

        return NGramRepeatBlockFunction.apply(tokens, lprobs,
               bsz, step, beam_size, no_repeat_ngram_size)
