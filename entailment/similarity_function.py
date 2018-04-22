import math
import torch


class SimilarityFunction(torch.nn.Module):
    """
    This similarity function simply computes the dot product between each pair of vectors, with an
    optional scaling to reduce the variance of the output elements.
    Parameters
    ----------
    scale_output : ``bool``, optional
        If ``True``, we will scale the output by ``math.sqrt(tensor.size(-1))``, to reduce the
        variance in the result.
    """
    def __init__(self, scale_output=False):
        super(SimilarityFunction, self).__init__()
        self._scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self._scale_output:
            result *= math.sqrt(tensor_1.size(-1))
        return result

    @classmethod
    def from_params(cls, params):
        scale_output = params.pop_bool('scale_output', False)
        params.assert_empty(cls.__name__)
        return cls(scale_output=scale_output)
