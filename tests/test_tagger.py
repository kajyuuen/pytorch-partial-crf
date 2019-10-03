import itertools

import random
import math

import pytest
from pytest import approx
import torch

from pytorch_partial_crf.tagger import Tagger

class TestCRF:
    def setup(self):
        self.emissions = torch.Tensor([
            [[1, 0, 0, .1, .4, .6], [0, .5, .7, 0, .1, .3], [.1, .5, 2, .7, 1, 0], [.4, 1, .9, .2, .9, 0]],
            [[0, 0, 1, .7, .4, .3], [0, .1, .4, .8, 0, .2], [0, .5, .4, .9, 0, 1], [.3, .1, .7, 0, 0, .6]]
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
        self.tagger = Tagger(self.num_tags)

    def test_decode_without_mask(self):
        viterbi_path = self.tagger.viterbi_decode(self.emissions,
                                                  self.transitions,
                                                  self.transitions_from_start,
                                                  self.transitions_to_end)
        assert viterbi_path == [[2, 1, 2, 1], [2, 1, 2, 1]]

    def test_decode_with_mask(self):
        mask = torch.ByteTensor([
                [1, 1, 1, 1],
                [1, 1, 0, 0]
        ])

        viterbi_path = self.tagger.viterbi_decode(self.emissions,
                                                  self.transitions,
                                                  self.transitions_from_start,
                                                  self.transitions_to_end,
                                                  mask)
        assert viterbi_path == [[2, 1, 2, 1], [2, 1]]

    def test_marginal_probabilities(self):
        marginal_probabilities = self.tagger.marginal_probabilities(self.emissions,
                                                                    self.transitions,
                                                                    self.transitions_from_start,
                                                                    self.transitions_to_end)
        # TODO: Add test
        assert torch.allclose(marginal_probabilities.sum(dim=2), torch.ones_like(marginal_probabilities.sum(dim=2)))
