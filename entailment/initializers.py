import itertools
import torch
from torch.autograd import Variable
import torch.nn.init
import re


def block_orthogonal(tensor, split_sizes, gain=1.0):
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """

    if isinstance(tensor, Variable):
        block_orthogonal(tensor.data, split_sizes, gain)
    else:
        sizes = list(tensor.size())
        indexes = [list(range(0, max_size, split))
                   for max_size, split in zip(sizes, split_sizes)]
        # Iterate over all possible blocks within the tensor.
        for block_start_indices in itertools.product(*indexes):
            # A list of tuples containing the index to start at for this block
            # and the appropriate step size (i.e split_size[i] for dimension i).
            index_and_step_tuples = zip(block_start_indices, split_sizes)
            # This is a tuple of slices corresponding to:
            # tensor[index: index + step_size, ...]. This is
            # required because we could have an arbitrary number
            # of dimensions. The actual slices we need are the
            # start_index: start_index + step for each dimension in the tensor.
            block_slice = tuple([slice(start_index, start_index + step)
                                 for start_index, step in index_and_step_tuples])
            tensor[block_slice] = torch.nn.init.orthogonal(tensor[block_slice].contiguous(), gain=gain)


class InitializerApplicator:
    """
    Applies initializers to the parameters of a Module based on regex matches.  Any parameter not
    explicitly matching a regex will not be initialized, instead using whatever the default
    initialization was in the module's code.
    """
    def __init__(self, initializers=None):
        """
        Parameters
        ----------
        initializers : ``List[Tuple[str, Initializer]]``, optional (default = [])
            A list mapping parameter regexes to initializers.  We will check each parameter against
            each regex in turn, and apply the initializer paired with the first matching regex, if
            any.
        """
        self._initializers = initializers or []

    def __call__(self, module):
        """
        Applies an initializer to all parameters in a module that match one of the regexes we were
        given in this object's constructor.  Does nothing to parameters that do not match.
        Parameters
        ----------
        module : torch.nn.Module, required.
            The Pytorch module to apply the initializers to.
        """
        unused_regexes = set([initializer[0] for initializer in self._initializers])
        uninitialized_parameters = set()
        # Store which initialisers were applied to which parameters.
        for name, parameter in module.named_parameters():
            for initializer_regex, initializer in self._initializers:
                if re.search(initializer_regex, name):
                    initializer(parameter)
                    unused_regexes.discard(initializer_regex)
                    break
            else:  # no break
                uninitialized_parameters.add(name)
        uninitialized_parameter_list = list(uninitialized_parameters)
        uninitialized_parameter_list.sort()
