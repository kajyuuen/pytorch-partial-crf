# pytorch-partial-crf

Partial/Fuzzy conditional random field in PyTorch.

Document: https://pytorch-partial-crf.readthedocs.io/

# How to use

## Install

```sh
pip install pytorch-partial-crf
```

### Use CRF

```python
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

# Computing negative log likelihood
model(emissions, tags)
```

### Use partial CRF

```python
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

# Computing negative log likelihood
model(emissions, tags)
```


### Use Marginal CRF

```python
import torch
from pytorch_partial_crf import MarginalCRF

# Create 
num_tags = 6
model = MarginalCRF(num_tags)

batch_size, sequence_length = 3, 5
emissions = torch.randn(batch_size, sequence_length, num_tags)

# Set probability tags
marginal_tags = torch.Tensor([
        [
            [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
            [0.8, 0.0, 0.0, 0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.3, 0.0, 0.0, 0.1, 0.6, 0.0],
        ],
        [
            [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
            [0.8, 0.0, 0.0, 0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.3, 0.0, 0.0, 0.1, 0.6, 0.0],
        ],
        [
            [0.2, 0.2, 0.2, 0.1, 0.1, 0.2],
            [0.8, 0.0, 0.0, 0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.3, 0.0, 0.0, 0.1, 0.6, 0.0],
        ],
])
# Computing negative log likelihood
model(emissions, marginal_tags)
```

### Decoding

Viterbi decode

```
model.viterbi_decode(emissions)
```

Restricted viterbi decode

```python
possible_tags = torch.randn(batch_size, sequence_length, num_tags)
possible_tags[possible_tags <= 0] = 0 # `0` express that can not pass.
possible_tags[possible_tags > 0] = 1  # `1` express that can pass.
possible_tags = possible_tags.byte()
model.restricted_viterbi_decode(emissions, possible_tags)
```

Marginal probabilities

```python
model.marginal_probabilities(emissions)
```

### Contributing

We welcome contributions! Please post your requests and comments on Issue.


### License

MIT

### References

The implementation is based on AllenNLP CRF module and pytorch-crf.
