from typing import List, Optional

import torch
import torch.nn as nn

from pytorch_partial_crf.base_crf import BaseCRF
from pytorch_partial_crf.utils import create_possible_tag_masks
from pytorch_partial_crf.utils import log_sum_exp

class CRF(BaseCRF):
    """Conditional random field.
    """
    def __init__(self, num_tags: int, padding_idx: int = None) -> None:
        super().__init__(num_tags, padding_idx)

    def forward(self,
                emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        gold_score = self._numerator_score(emissions, tags, mask)
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
            # Emissions scores
            emissions_score = emissions[i].view(batch_size, 1, num_tags)     # (batch_size, 1, num_tags)
            # Transition scores
            transition_scores = self.transitions.view(1, num_tags, num_tags) # (1, num_tags, num_tags)
            # Broadcast alpha
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)            # (batch_size, num_tags, 1)

            # Add all scores
            inner = broadcast_alpha + emissions_score + transition_scores    # (batch_size, num_tags, num_tags)
            alpha = (log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Add end transition score
        stops = alpha + self.end_transitions.view(1, num_tags)

        return log_sum_exp(stops) # (batch_size,)

    def _numerator_score(self,
                        emissions: torch.Tensor,
                        tags: torch.LongTensor,
                        mask: torch.ByteTensor) -> torch.Tensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            tags:  (batch_size, sequence_length)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            scores: (batch_size)
        """

        batch_size, sequence_length, _ = emissions.data.shape

        emissions = emissions.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()

        # Start transition score and first emission
        score = self.start_transitions.index_select(0, tags[0])
        
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags[i], tags[i+1]
            # Emissions score for next tag
            emissions_score = emissions[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            # Transition score from current_tag to next_tag
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]

            # Add all score
            score += transition_score * mask[i + 1] + emissions_score * mask[i]

        # Add end transition score
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)

        # Compute score of transitioning to STOP_TAG from each LAST_TAG
        last_transition_score = self.end_transitions.index_select(0, last_tags)

        last_inputs = emissions[-1]                                      # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score
