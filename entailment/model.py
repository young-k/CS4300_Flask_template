"""
:py:class:`Model` is an abstract class representing
an AllenNLP model.
"""

import os
import torch

from time_distributed import TimeDistributed
from matrix_attention import MatrixAttention
from nn_util import get_text_field_mask, last_dim_softmax, weighted_sum
import nn_util

# When training a model, many sets of weights are saved. By default we want to
# save/load this set of weights.
_DEFAULT_WEIGHTS = "best.th"


class DecomposableAttention(torch.nn.Module):
    """
    This abstract class represents a model to be trained. Rather than relying completely
    on the Pytorch Module, we modify the output spec of ``forward`` to be a dictionary.
    Models built using this API are still compatible with other pytorch models and can
    be used naturally as modules within other models - outputs are dictionaries, which
    can be unpacked and passed into other layers. One caveat to this is that if you
    wish to use an AllenNLP model inside a Container (such as nn.Sequential), you must
    interleave the models with a wrapper module which unpacks the dictionary into
    a list of tensors.
    In order for your model to be trained using the :class:`~allennlp.training.Trainer`
    api, the output dictionary of your Model must include a "loss" key, which will be
    optimised during the training process.
    Finally, you can optionally implement :func:`Model.get_metrics` in order to make use
    of early stopping and best-model serialization based on a validation metric in
    :class:`~allennlp.training.Trainer`.
    """
    def __init__(self, vocab, text_field_embedder, attend_feedforward, similarity_function, compare_feedforward,
                 aggregate_feedforward):
        super(DecomposableAttention, self).__init__()
        self.vocab = vocab

        self._text_field_embedder = text_field_embedder
        self._attend_feedforward = TimeDistributed(attend_feedforward)
        self._matrix_attention = MatrixAttention(similarity_function)
        self._compare_feedforward = TimeDistributed(compare_feedforward)
        self._aggregate_feedforward = aggregate_feedforward

        self._num_labels = len(vocab)
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, premise, hypothesis, label=None):
        """
        Parameters
        ----------
        premise : Dict[str, torch.LongTensor]
            From a ``TextField``
        hypothesis : Dict[str, torch.LongTensor]
            From a ``TextField``
        label : torch.IntTensor, optional (default = None)
            From a ``LabelField``
        Returns
        -------
        An output dictionary consisting of:
        label_logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalised log
            probabilities of the entailment label.
        label_probs : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing probabilities of the
            entailment label.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_premise = self._text_field_embedder(premise)
        embedded_hypothesis = self._text_field_embedder(hypothesis)
        premise_mask = get_text_field_mask(premise).float()
        hypothesis_mask = get_text_field_mask(hypothesis).float()

        projected_premise = self._attend_feedforward(embedded_premise)
        projected_hypothesis = self._attend_feedforward(embedded_hypothesis)
        # Shape: (batch_size, premise_length, hypothesis_length)
        similarity_matrix = self._matrix_attention(projected_premise, projected_hypothesis)

        # Shape: (batch_size, premise_length, hypothesis_length)
        p2h_attention = last_dim_softmax(similarity_matrix, hypothesis_mask)
        # Shape: (batch_size, premise_length, embedding_dim)
        attended_hypothesis = weighted_sum(embedded_hypothesis, p2h_attention)

        # Shape: (batch_size, hypothesis_length, premise_length)
        h2p_attention = last_dim_softmax(similarity_matrix.transpose(1, 2).contiguous(), premise_mask)
        # Shape: (batch_size, hypothesis_length, embedding_dim)
        attended_premise = weighted_sum(embedded_premise, h2p_attention)

        premise_compare_input = torch.cat([embedded_premise, attended_hypothesis], dim=-1)
        hypothesis_compare_input = torch.cat([embedded_hypothesis, attended_premise], dim=-1)

        compared_premise = self._compare_feedforward(premise_compare_input)
        compared_premise = compared_premise * premise_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_premise = compared_premise.sum(dim=1)

        compared_hypothesis = self._compare_feedforward(hypothesis_compare_input)
        compared_hypothesis = compared_hypothesis * hypothesis_mask.unsqueeze(-1)
        # Shape: (batch_size, compare_dim)
        compared_hypothesis = compared_hypothesis.sum(dim=1)

        aggregate_input = torch.cat([compared_premise, compared_hypothesis], dim=-1)
        label_logits = self._aggregate_feedforward(aggregate_input)
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)

        output_dict = {"label_logits": label_logits, "label_probs": label_probs}

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label.squeeze(-1))
            output_dict["loss"] = loss

        return output_dict

    def load(self, serialization_dir, weights_file=None, cuda_device=-1):
        """
        Instantiates an already-trained model, based on the experiment
        configuration and some optional overrides.
        """
        weights_file = weights_file or os.path.join(serialization_dir, _DEFAULT_WEIGHTS)

        # model_params = config.get('model')
        # model = DecomposableAttention(vocab=vocab, **model_params)
        model_state = torch.load(weights_file, map_location=nn_util.device_mapping(cuda_device))
        self.load_state_dict(model_state)
