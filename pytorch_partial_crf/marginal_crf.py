from typing import Optional

import torch
import torch.nn as nn

from pytorch_partial_crf.base_crf import BaseCRF
from pytorch_partial_crf.utils import log_sum_exp

from pytorch_partial_crf.utils import IMPOSSIBLE_SCORE

class MarginalCRF(BaseCRF):
    """Marginal Conditional random field.
    """
    def __init__(self, num_tags: int, padding_idx: int = None) -> None:
        super().__init__(num_tags, padding_idx)

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self,
                emissions: torch.Tensor,
                marginal_tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None) -> torch.Tensor:
        batch_size, sequence_length, _ = emissions.shape
        if mask is None:
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8)

        gold_score = self._numerator_score(emissions, marginal_tags, mask)
        forward_score = self._denominator_score(emissions, mask)
        return torch.sum(forward_score - gold_score)

    def _denominator_score(self,
                     emissions: torch.Tensor,
                     mask: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            mask: Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emissions score
        alpha = self.start_transitions.view(1, num_tags) + emissions[0]
        for i in range(1, sequence_length):

            emissions_score = emissions[i].view(batch_size, 1, num_tags)      # (batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)  # (1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)             # (batch_size, num_tags, 1)

            inner = broadcast_alpha + emissions_score + transition_scores     # (batch_size, num_tags, num_tags)

            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)
        return log_sum_exp(stops) # (batch_size,)

    def _numerator_score(self,
                        emissions: torch.Tensor,
                        marginal_tags: torch.LongTensor,
                        mask: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            marginal_tags:  (batch_size, sequence_length, num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """

        batch_size, sequence_length, num_tags = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        marginal_tags = marginal_tags.float().transpose(0, 1)
        log_marginal_tags = torch.log(marginal_tags)
        log_marginal_tags[log_marginal_tags == -float('inf')] = IMPOSSIBLE_SCORE

        # Start transition score and first emission
        alpha = self.start_transitions + emissions[0] + log_marginal_tags[0]

        for i in range(1, sequence_length):
            log_next_marginal_tags = log_marginal_tags[i]      # (batch_size, num_tags)

            # Emissions scores
            emissions_score = emissions[i].view(batch_size, 1, num_tags)
            # Transition scores
            transition_scores = self.transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags).clone()
            # Broadcast alpha
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores # (batch_size, num_tags, num_tags)
            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))
            alpha += log_next_marginal_tags * mask[i].view(batch_size, 1)

        # Add end transition score
        last_tag_indexes = mask.sum(0).long() - 1
        end_transitions = self.end_transitions.expand(batch_size, num_tags)
        stops = alpha + end_transitions
        return log_sum_exp(stops) # (batch_size,)