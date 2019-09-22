import random

import pytest
import torch

from pytorch_partial_crf.utils import create_possible_tag_masks

SEED = 4738
random.seed(SEED)
torch.manual_seed(SEED)

def simple_create_possible_tag_masks(num_tags, tags_list):
    results = []
    for tags in tags_list:
        result = []
        for tag in tags:
            if tag == -1:
                possible_tag_mask = [1] * num_tags
            else:
                possible_tag_mask = [0] * num_tags
                possible_tag_mask[tag] = 1
            result.append(possible_tag_mask)
        results.append(result)
    return results

def test_possible_tag_masks():
    num_tags = 5
    tags_list = [
        [0, 1, 2, 3, 4, -1, -1],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    tags = torch.tensor(tags_list, dtype=torch.long)
    assert simple_create_possible_tag_masks(num_tags, tags_list) == create_possible_tag_masks(num_tags, tags).tolist()


