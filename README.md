# pytorch-partial-crf

Partial/Fuzzy conditional random field in PyTorch.

## How to use

### Install

```
pip install pytorch-partial-crf
```

## Use CRF

```
import torch
from pytorch_partial_crf import CRF

# Create 
num_tags = 6
model = CRF(num_tags)

batch_size, sequence_length = 3, 5
emissions = torch.randn(batch_size, sequence_length, num_tags)

tags = torch.LongTensor([
    [1, 2, 3, 3, 5],
    [1, 3, 4, 2, 1],
    [1, 0, 2, 4, 4],
])

# Computing log likelihood
model(emissions, tags)
```

### Use partial CRF

```
import torch
from pytorch_partial_crf import PartialCRF

# Create 
num_tags = 6
model = PartialCRF(num_tags)

batch_size, sequence_length = 3, 5
emissions = torch.randn(batch_size, sequence_length, num_tags)

# Set unknown tag to -1
tags = torch.LongTensor([
    [1, 2, 3, 3, 5],
    [-1, 3, 3, 2, -1],
    [-1, 0, -1, -1, 4],
])

# Computing log likelihood
model(emissions, tags)
```

### Use viterbi decode

```
import torch
from pytorch_partial_crf import CRF
from pytorch_partial_crf import PartialCRF


num_tags = 6
model = CRF(num_tags) # or FuzzyCRF

batch_size, sequence_length = 3, 5
emissions = torch.randn(batch_size, sequence_length, num_tags)

model.viterbi_decode(emissions)

# restricted viterbi decode
possible_tags = torch.randn(batch_size, sequence_length, num_tags)
possible_tags[possible_tags<=0] = 0
possible_tags[possible_tags>0] = 1
possible_tags = possible_tags.byte()  

model.restricted_viterbi_decode(emissions, possible_tags)
```

## License

MIT

### References

The implementation is based on AllenNLP CRF module and pytorch-crf.
