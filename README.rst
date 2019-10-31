pytorch-partial-crf
===================

Partial/Fuzzy conditional random field in PyTorch.

How to use
============

Install
------------------------

.. code-block:: shell

    pip install pytorch-partial-crf


Use CRF
--------

.. code-block:: python

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

Use partial CRF
---------------

.. code-block:: python

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

Decoding
--------

Viterbi decode

.. code-block:: python

   model.viterbi_decode(emissions)

Restricted viterbi decode

.. code-block:: python

    possible_tags = torch.randn(batch_size, sequence_length, num_tags)
    possible_tags[possible_tags <= 0] = 0 # `0` express that can not pass.
    possible_tags[possible_tags > 0] = 1 # `1` express that can pass.
    possible_tags = possible_tags.byte()
    model.restricted_viterbi_decode(emissions, possible_tags)

Marginal probabilities

.. code-block:: python

   model.marginal_probabilities(emissions)

License
-------

MIT

References
----------

The implementation is based on AllenNLP CRF module and pytorch-crf.
