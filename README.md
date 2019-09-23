# pytorch-partial-crf

Partial/Fuzzy conditional random field in PyTorch.

## How to use

### Install

```
pip install pytorch-partial-crf
```

### Use partial CRF

```
import torch
from pytorch_partial_crf import PartialCRF

# Create 
num_tags = 6
model = PartialCRF(num_tags)

# Computing log likelihood
batch_size, sequence_length = 3, 5
emissions = torch.randn(batch_size, sequence_length, num_tags)

# Set unknown tag to -1
tags = torch.LongTensor([
    [1, 2, 3, 3, 5],
    [-1, 3, 3, 2, -1],
    [-1, 0, -1, -1, 4],
])
model(emissions, tags)
```

## License

MIT

### References

The implementation is based on AllenNLP CRF module and pytorch-crf.
