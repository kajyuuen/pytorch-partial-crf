import itertools

import random
import math

import pytest
from pytest import approx
import torch

from pytorch_partial_crf import CRF

SEED = 4738
random.seed(SEED)
torch.manual_seed(SEED)

def manually_score(transitions_from_start, transitions_to_end, transitions, emissions, tags):
    # Add start and end scores
    total = transitions_from_start[tags[0]] + transitions_to_end[tags[-1]]
    # Add transition scores
    for tag, next_tag in zip(tags, tags[1:]):
        total += transitions[tag, next_tag]
    # Add emission scores
    for emission, tag in zip(emissions, tags):
        total += emission[tag]
    return total

class TestCRF:
    def setup(self):
        self.emissions = torch.Tensor([
            [[1, 0, 0, .1, .4, .6], [0, .5, .7, 0, .1, .3], [.1, .5, 2, .7, 1, 0], [.4, 1, .9, .2, .9, 0]],
            [[0, 0, 1, .7, .4, .3], [0, .1, .4, .8, 0, .2], [0, .5, .4, .9, 0, 1], [.3, .1, .7, 0, 0, .6]]
        ])
        self.tags = torch.LongTensor([
                [1, 3, 4, 5],
                [3, 0, 2, 5]
        ])

        self.transitions = torch.Tensor([
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                [0.5, 0.2, 0.9, 0.7, 0.4, 0.1],
                [-0.8, 7.3, -0.2, 3.7, 0.3, 1.0],
                [0.2, 0.4, 0.6, -0.8, 1.0, -1.2],
                [-1.0, 0, -1.0, 0.1, 1.0, 0.1],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4, -0.5])

        self.num_tags = 6
        self.crf = CRF(self.num_tags)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def test_forward_without_mask(self):
        log_likelihood = self.crf(self.emissions, self.tags)

        manual_log_likelihood = 0.0
        for emission, tag in zip(self.emissions, self.tags):
            gold_score = manually_score(self.transitions_from_start, self.transitions_to_end, self.transitions, emission.detach(), tag.detach())
            forward_scores = [manually_score(self.transitions_from_start, self.transitions_to_end, self.transitions, emission.detach(), tags_j)
                          for tags_j in itertools.product(range(self.num_tags), repeat=4)]
            denominator = math.log(sum(math.exp(forward_score) for forward_score in forward_scores))
            manual_log_likelihood += denominator - gold_score

        assert manual_log_likelihood.item() == approx(log_likelihood.item())

    def test_forward_with_mask(self):
        mask = torch.ByteTensor([
                [1, 1, 1, 1],
                [1, 1, 0, 0]
        ])

        log_likelihood = self.crf(self.emissions, self.tags, mask)

        manual_log_likelihood = 0.0
        for emission, tag, mask_i in zip(self.emissions, self.tags, mask):
            sequence_length = torch.sum(mask_i.detach())
            emission = emission.data[:sequence_length]
            tag = tag.data[:sequence_length]

            gold_score = manually_score(self.transitions_from_start, self.transitions_to_end, self.transitions, emission.detach(), tag.detach())
            forward_scores = [manually_score(self.transitions_from_start, self.transitions_to_end, self.transitions, emission.detach(), tags_j)
                          for tags_j in itertools.product(range(self.num_tags), repeat=sequence_length)]
            denominator = math.log(sum(math.exp(forward_score) for forward_score in forward_scores))
            manual_log_likelihood += denominator - gold_score

        assert manual_log_likelihood.item() == approx(log_likelihood.item())

    def test_decode_without_mask(self):
        mask = torch.ByteTensor([
                [1, 1, 1, 1],
                [1, 1, 0, 0]
        ])

        viterbi_path = self.crf.viterbi_decode(self.emissions, mask)
        assert viterbi_path == [[2, 1, 2, 1], [2, 1]]

    def test_marginal_probabilities(self):
        marginal_probabilities = self.crf.marginal_probabilities(self.emissions)
        # TODO: Add test
        assert torch.allclose(marginal_probabilities.sum(dim=2), torch.ones_like(marginal_probabilities.sum(dim=2)))
