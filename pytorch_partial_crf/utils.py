import torch

UNLABELED_INDEX = -1

def create_possible_tag_masks(num_tags: int, tags: torch.FloatTensor):
    no_annotation_idx = (tags == UNLABELED_INDEX)
    tags[tags == UNLABELED_INDEX] = 0

    tags_ = torch.unsqueeze(tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks
