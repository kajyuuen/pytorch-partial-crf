.. pytorch-partial-crf documentation master file, created by
   sphinx-quickstart on Sun Oct 20 15:07:21 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pytorch-partial-crf's documentation!
===============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Installation
============

Install with pip::

    pip install pytorch_partial_crf


Getting started
===============

.. currentmodule:: pytorch_partial_crf

This package provides an implementation of a Partial/Fuzzy CRF layer for learning incompleted tag sequences, and a linear-chain CRF layer for learning tag sequences.

.. code-block:: python

   import torch
   from pytorch_partial_crf import PartialCRF
   num_tags = 6  # number of tags is 6
   model = PartialCRF(num_tags)

Computing log likelihood
------------------------

.. code-block:: python

   batch_size = 3
   sequence_length = 5
   emissions = torch.randn(batch_size, sequence_length, num_tags)
   # Set to -1 if it is unknown tag
   tags = torch.LongTensor([
         [1, 2, 3, 3, 5],
         [-1, 3, -1, 2, 1],
         [1, 0, -1, 4, -1],
   ])  # (seq_length, batch_size)
   model(emissions, tags) # Computing log likelihood

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
