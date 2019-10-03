from typing import Optional

import torch

from pytorch_partial_crf.utils import create_possible_tag_masks
from pytorch_partial_crf.utils import log_sum_exp

from pytorch_partial_crf.utils import IMPOSSIBLE_SCORE

class Tagger:
    """Tagger
    """
    def __init__(self, num_tags: int) -> None:
        self.num_tags = num_tags

    def viterbi_decode(self,
                       emissions: torch.Tensor,
                       transitions: torch.Tensor,
                       start_transitions: torch.Tensor,
                       end_transitions: torch.Tensor,
                       mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            transitions: (num_tags, num_tags)
            start_transitions: (num_tags)
            end_transitions: (num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            tags: (batch_size)
        """
        batch_size, sequence_length, _ = emissions.shape
        if mask is None:
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8)

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()

        # Start transition and first emission score
        score = start_transitions + emissions[0]
        history = []

        for i in range(1, sequence_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + transitions + broadcast_emissions
            next_score, indices = next_score.max(dim = 1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition score
        score += end_transitions

        # Compute the best path
        seq_ends = mask.long().sum(dim = 0) - 1

        best_tags_list = []
        for i in range(batch_size):
            _, best_last_tag = score[i].max(dim = 0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[i]]):
                best_last_tag = hist[i][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list

    def marginal_probabilities(self,
                               emissions: torch.Tensor,
                               transitions: torch.Tensor,
                               start_transitions: torch.Tensor,
                               end_transitions: torch.Tensor,
                               mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            transitions: (num_tags, num_tags)
            start_transitions: (num_tags)
            end_transitions: (num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            marginal_probabilities: (sequence_length, sequence_length, num_tags)
        """
        if mask is None:
            batch_size, sequence_length, _ = emissions.data.shape
            mask = torch.ones([batch_size, sequence_length], dtype=torch.uint8)

        alpha = self._forward_algorithm(emissions, 
                                        transitions,
                                        start_transitions,
                                        end_transitions,
                                        mask, 
                                        reverse_direction = False)
        beta = self._forward_algorithm(emissions, 
                                        transitions,
                                        start_transitions,
                                        end_transitions,
                                        mask, 
                                        reverse_direction = True)
        z = log_sum_exp(alpha[alpha.size(0) - 1] + end_transitions, dim = 1)

        proba = alpha + beta - z.view(1, -1, 1)
        return torch.exp(proba)

    def _forward_algorithm(self,
                           emissions: torch.Tensor,
                           transitions: torch.Tensor,
                           start_transitions: torch.Tensor,
                           end_transitions: torch.Tensor,
                           mask: torch.ByteTensor,
                           reverse_direction: bool = False) -> torch.FloatTensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            transitions: (num_tags, num_tags)
            start_transitions: (num_tags)
            end_transitions: (num_tags)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
            reverse: This parameter decide algorithm direction.
        Returns:
            log_probabilities: (sequence_length, batch_size, num_tags)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape

        broadcast_emissions = emissions.transpose(0, 1).unsqueeze(2).contiguous() # (sequence_length, batch_size, 1, num_tags)
        mask = mask.float().transpose(0, 1).contiguous()                          # (sequence_length, batch_size)
        broadcast_transitions = transitions.unsqueeze(0)                     # (1, num_tags, num_tags)
        sequence_iter = range(1, sequence_length)

        # backward algorithm
        if reverse_direction:
            # Transpose transitions matrix and emissions
            broadcast_transitions = broadcast_transitions.transpose(1, 2)         # (1, num_tags, num_tags)
            broadcast_emissions = broadcast_emissions.transpose(2, 3)             # (sequence_length, batch_size, num_tags, 1)
            sequence_iter = reversed(sequence_iter)

            # It is beta
            log_proba = [end_transitions.expand(batch_size, num_tags)]
        # forward algorithm
        else:
            # It is alpha
            log_proba = [emissions.transpose(0, 1)[0] + start_transitions.view(1, -1)]

        for i in sequence_iter:
            # Broadcast log probability
            broadcast_log_proba = log_proba[-1].unsqueeze(2) # (batch_size, num_tags, 1)

            # Add all scores
            # inner: (batch_size, num_tags, num_tags)
            # broadcast_log_proba:   (batch_size, num_tags, 1)
            # broadcast_transitions: (1, num_tags, num_tags)
            # broadcast_emissions:   (batch_size, 1, num_tags)
            inner = broadcast_log_proba \
                    + broadcast_transitions \
                    + broadcast_emissions[i]

            # Append log proba
            log_proba.append((log_sum_exp(inner, 1) * mask[i].view(batch_size, 1) +
                     log_proba[-1] * (1 - mask[i]).view(batch_size, 1)))

        if reverse_direction:
            log_proba.reverse()

        return torch.stack(log_proba)

    def restricted_viterbi_decode(self,
                                  emissions: torch.Tensor,
                                  transitions: torch.Tensor,
                                  start_transitions: torch.Tensor,
                                  end_transitions: torch.Tensor,
                                  incomplete_tags: torch.ByteTensor,
                                  mask: Optional[torch.ByteTensor] = None) -> torch.FloatTensor:
        """
        Parameters:
            emissions: (batch_size, sequence_length, num_tags)
            transitions: (num_tags, num_tags)
            start_transitions: (num_tags)
            end_transitions: (num_tags)
            incomplete_tags: (batch_size, sequence_length)
            mask:  Show padding tags. 0 don't calculate score. (batch_size, sequence_length)
        Returns:
            tags: (batch_size)
        """
        batch_size, sequence_length, num_tags = emissions.data.shape
        if mask is None:
            mask = torch.ones_like(incomplete_tags, dtype=torch.uint8)

        emissions = emissions.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        possible_tags = create_possible_tag_masks(self.num_tags, incomplete_tags)

        # Start transition score and first emission
        first_possible_tag = possible_tags[0]

        score = start_transitions + emissions[0]      # (batch_size, num_tags)
        score[(first_possible_tag == 0)] = IMPOSSIBLE_SCORE

        history = []

        for i in range(1, sequence_length):
            current_possible_tags = possible_tags[i-1]
            next_possible_tags = possible_tags[i]
            
            # Feature score
            emissions_score = emissions[i]
            emissions_score[(next_possible_tags == 0)] = IMPOSSIBLE_SCORE
            emissions_score = emissions_score.view(batch_size, 1, num_tags)

            # Transition score
            transition_scores = transitions.view(1, num_tags, num_tags).expand(batch_size, num_tags, num_tags).clone()
            transition_scores[(current_possible_tags == 0)] = IMPOSSIBLE_SCORE
            transition_scores.transpose(1, 2)[(next_possible_tags == 0)] = IMPOSSIBLE_SCORE

            broadcast_score = score.unsqueeze(2)
            next_score = broadcast_score + transitions + emissions_score
            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transition score
        score += end_transitions

        # Compute the best path for each sample
        seq_ends = mask.long().sum(dim=0) - 1
        max_len = int(seq_ends[0])
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags.extend([self.PAD_INDEX] * ( max_len - len(best_tags) + 1) )
            best_tags_list.append(best_tags)

        return best_tags_list