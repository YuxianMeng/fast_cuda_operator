# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Unit test for Ngram repeat block cuda op """

import math

import pytest
import torch
from time import time
import math

from fast_cuda_operator import NGramRepeatBlock


def apply_no_repeat_ngram(tokens, lprobs, bsz, step,
                          beam_size, no_repeat_ngram_size):
    """ Fairseq implementation of blocking
        repeated ngrams
    """
    banned_list = [[] for bbsz_idx in range(bsz * beam_size)]
    cpu_tokens = tokens.cpu()[:, :step + 1].numpy()
    check_start_pos = step + 2 - no_repeat_ngram_size
    for bbsz_idx in range(bsz * beam_size):
        for i in range(check_start_pos):
            is_banned = True
            for k in range(no_repeat_ngram_size - 1):
                if cpu_tokens[bbsz_idx, i + k] != cpu_tokens[
                    bbsz_idx, check_start_pos + k]:
                    is_banned = False
                    break
            if is_banned:
                banned_list[bbsz_idx].append(
                    cpu_tokens[bbsz_idx,
                               i + no_repeat_ngram_size - 1])

    def calculate_banned_tokens(bbsz_idx):
        """before decoding the next token, prevent decoding
        of ngrams that have already appeared
        """
        banned_tokens_per_sample = [
            (bbsz_idx, t) for t in banned_list[bbsz_idx]
        ]
        return banned_tokens_per_sample

    banned_tokens = []
    if step + 2 - no_repeat_ngram_size >= 0:
        for bbsz_idx in range(bsz * beam_size):
            banned_tokens.extend(calculate_banned_tokens(bbsz_idx))

    if banned_tokens:
        banned_tokens = torch.LongTensor(banned_tokens)
        lprobs.index_put_(
            tuple(banned_tokens.t()),
            lprobs.new_tensor([-math.inf] * len(banned_tokens)))

    return lprobs


FIELDS = [
    'vocab_size',
    'bsz',
    'beam_size',
    'step',
    'ngram_repeat_block_size',
    'sequence_length',
    'pos1'
]
FIXTURES = [
    {
        # 'testcase_name': 'overlapping_ngrams',
        'vocab_size': 10,
        'bsz': 256,
        'beam_size': 1,
        'step': 4,
        'ngram_repeat_block_size': 3,
        'sequence_length': 2048,
        'pos1': 0,
    },
    {
        # 'testcase_name': 'min_step',
        'vocab_size': 10,
        'bsz': 256,
        'beam_size': 1,
        'step': 3,
        'ngram_repeat_block_size': 3,
        'sequence_length': 2048,
        'pos1': 0,
    },
    {
        # 'testcase_name': 'higher_beam_size',
        'vocab_size': 10,
        'bsz': 256,
        'beam_size': 2,
        'step': 6,
        'ngram_repeat_block_size': 3,
        'sequence_length': 2048,
        'pos1': 0,
    },
    {
        # 'testcase_name': 'higher_ngram_size',
        'vocab_size': 10,
        'bsz': 256,
        'beam_size': 1,
        'step': 12,
        'ngram_repeat_block_size': 5,
        'sequence_length': 2048,
        'pos1': 0,
    },
    {
        # 'testcase_name': 'higher_vocab_size',
        'vocab_size': 30000,
        'bsz': 256,
        'beam_size': 1,
        'step': 6,
        'ngram_repeat_block_size': 3,
        'sequence_length': 2048,
        'pos1': 0,
    }
]


@pytest.mark.parametrize(",".join(FIELDS), [tuple(case[k] for k in FIELDS) for case in FIXTURES])
def test_ngram_repeat_block_kernel(bsz, beam_size, vocab_size,
                                   step, ngram_repeat_block_size, sequence_length, pos1):
    """ Use random input with repeated ngram to check
        whether corresponding token in vocabulary is blocked (-Inf score)

    Args:
    bsz (int): batch size
    beam_size (int): beam size
    vocab_size (int): vocab size
    step (int): current decoding step
    ngram_repeat_block_size (int): size of ngram
    sequence_length (int): sequence length
    pos1 (int) first position where repeated ngram occurs
                within a sentence.
    """

    lprobs_fairseq = torch.zeros(bsz * beam_size,
                                 vocab_size).type(torch.FloatTensor)
    lprobs_fastseq = lprobs_fairseq.clone()
    repeated_ngram = torch.randint(0, 10, (1, 2))
    # second place where ngram is repeated
    pos2 = step - ngram_repeat_block_size + 2
    # Dummy input with repeated ngram
    inp = torch.cat((torch.randint(0, 10, (1, pos1)),
                     repeated_ngram, torch.randint(0, 10,
                                                   (1, pos2 - pos1 - ngram_repeat_block_size + 1)),
                     repeated_ngram, torch.randint(0, 10,
                                                   (1, sequence_length -
                                                    pos2 - ngram_repeat_block_size + 1))), 1)
    tokens = inp.repeat((bsz * beam_size, 1))
    # CUDA kernel initialization
    module = NGramRepeatBlock()
    lprobs_fastseq = lprobs_fastseq.cuda()
    lprobs_fairseq = lprobs_fairseq.cuda()
    tokens = tokens.cuda()
    # Cuda opt implementation
    t = time()
    lprobs_fastseq = module(tokens, lprobs_fastseq, bsz, step, beam_size,
                            ngram_repeat_block_size)
    print("optimized t:", time()-t)
    # Original implementation
    t = time()
    lprobs_fairseq = apply_no_repeat_ngram(tokens, lprobs_fairseq, bsz,
                                           step, beam_size, ngram_repeat_block_size)
    print("naive t:", time() - t)
    assert torch.allclose(lprobs_fairseq, lprobs_fastseq)
