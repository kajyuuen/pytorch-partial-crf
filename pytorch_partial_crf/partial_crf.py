import torch
import torch.nn as nn

IMPOSSIBLE_SCORE = -10000.0

def possible_tag_masks(num_tags, tags, unlabeled_index):
    no_annotation_idx = (tags == unlabeled_index)
    tags[tags == unlabeled_index] = 0

    tags_ = torch.unsqueeze(tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks

class PartialCRF(nn.Module):
    """Partial/Fuzzy Conditional random field.
    """
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError('invalid number of tags: {}'.format(num_tags))
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)


