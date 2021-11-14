import random

import torch

from pytorch_partial_crf.base_crf import BaseCRF

SEED = 4738
random.seed(SEED)
torch.manual_seed(SEED)

class TestBaseCRF:
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
        self.base_crf = BaseCRF(self.num_tags)
        self.base_crf.transitions = torch.nn.Parameter(self.transitions)
        self.base_crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.base_crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def test_decode_without_mask(self):
        viterbi_path = self.base_crf.viterbi_decode(self.emissions)
        assert viterbi_path == [[2, 1, 2, 1], [2, 1, 2, 1]]

    def test_decode_with_mask(self):
        mask = torch.ByteTensor([
                [1, 1, 1, 1],
                [1, 1, 0, 0]
        ])

        viterbi_path = self.base_crf.viterbi_decode(self.emissions, mask)
        assert viterbi_path == [[2, 1, 2, 1], [2, 1]]

    def test_restricted_viterbi_decode_without_mask(self):
        possible_tags = torch.ByteTensor([
                [
                    [1, 1, 0, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                ]
        ])

        viterbi_path = self.base_crf.viterbi_decode(self.emissions)
        restricted_viterbi_path = self.base_crf.restricted_viterbi_decode(self.emissions, possible_tags)
        assert restricted_viterbi_path != viterbi_path
        assert restricted_viterbi_path == [[5, 2, 3, 2], [5, 5, 5, 5]]

    def test_restricted_viterbi_decode_with_mask(self):
        possible_tags = torch.ByteTensor([
                [
                    [1, 1, 0, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1],
                    [0, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 1],
                ]
        ])
        mask = torch.ByteTensor([
                [1, 1, 1, 0],
                [1, 1, 0, 0]
        ])

        viterbi_path = self.base_crf.viterbi_decode(self.emissions, mask)
        restricted_viterbi_path = self.base_crf.restricted_viterbi_decode(self.emissions, possible_tags, mask)
        assert restricted_viterbi_path != viterbi_path
        assert restricted_viterbi_path == [[5, 2, 3], [5, 5]]

    def test_marginal_probabilities(self):
        marginal_probabilities = self.base_crf.marginal_probabilities(self.emissions)
        # TODO: Add test
        assert torch.allclose(marginal_probabilities.sum(dim=2), torch.ones_like(marginal_probabilities.sum(dim=2)))
