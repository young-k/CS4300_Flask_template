import torch

from elmo import Elmo
from time_distributed import TimeDistributed


class ElmoTokenEmbedder(torch.nn.Module):
    """
    Compute a single layer of ELMo representations.
    This class serves as a convenience when you only want to use one layer of ELMo representations at the input of
    your network.  It's essentially a wrapper around Elmo(num_output_representations=1, ...)

    Parameters:
    ----------
    options_file : ``str``, required.
        An ELMo JSON options file.
    weight_file : ``str``, required.
        An ELMo hdf5 weight file.
    do_layer_norm : ``bool``, optional.
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional.
        The dropout value to be applied to the ELMo representations.
    requires_grad : ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    projection_dim : ``int``, optional
        If given, we will project the ELMo embedding down to this dimension.  We recommend that you
        try using ELMo with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particularly where there is very limited training data).
    """
    def __init__(self, options_file, weight_file, do_layer_norm=False, dropout=0.0, requires_grad=False):
        super(ElmoTokenEmbedder, self).__init__()
        self._elmo = Elmo(options_file, weight_file, 1, do_layer_norm=do_layer_norm, dropout=dropout,
                          requires_grad=requires_grad)
        self._projection = None

    def get_output_dim(self):
        return self._elmo.get_output_dim()

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: ``torch.autograd.Variable``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        elmo_output = self._elmo(inputs)
        elmo_representations = elmo_output['elmo_representations'][0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations
