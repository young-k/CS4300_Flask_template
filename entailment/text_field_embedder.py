import torch

from time_distributed import TimeDistributed


class TextFieldEmbedder(torch.nn.Module):
    """
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary order.
    """
    def __init__(self, token_embedders):
        super(TextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)

    def get_output_dim(self):
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input, num_wrapping_dims=0):
        embedded_representations = []
        keys = sorted(text_field_input.keys())
        for key in keys:
            tensor = text_field_input[key]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(tensor)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)
