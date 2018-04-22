import torch
from torch.autograd import Variable


def get_lengths_from_binary_sequence_mask(mask):
    """
    Compute sequence lengths for each batch element in a tensor using a
    binary mask.
    Parameters
    ----------
    mask : torch.Tensor, required.
        A 2D binary mask of shape (batch_size, sequence_length) to
        calculate the per-batch sequence lengths from.
    Returns
    -------
    A torch.LongTensor of shape (batch_size,) representing the lengths
    of the sequences in the batch.
    """
    return mask.long().sum(-1)


def sort_batch_by_length(tensor, sequence_lengths):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : Variable(torch.LongTensor)
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index


def get_dropout_mask(dropout_probability, tensor_for_masking):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Variable, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.clone()
    binary_mask.data.copy_(torch.rand(tensor_for_masking.size()) > dropout_probability)
    # Scale mask by 1/keep_prob to preserve output statistics.
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


def remove_sentence_boundaries(tensor, mask):
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps, dim)``
    this returns a tensor of shape ``(batch_size, timesteps - 2, dim)`` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
    to be 1).
    Returns both the new tensor and updated mask.
    This function is the inverse of ``add_sentence_boundary_token_ids``.
    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    Returns
    -------
    tensor_without_boundary_tokens : ``torch.Tensor``
        The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 2, dim)``
    new_mask : ``torch.Tensor``
        The new mask for the tensor of shape ``(batch_size, timesteps - 2)``.
    """
    sequence_lengths = mask.sum(dim=1).data.cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = Variable(tensor.data.new(*new_shape).fill_(0))
    new_mask = Variable(tensor.data.new(new_shape[0], new_shape[1]).fill_(0)).long()
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, :(j - 2), :] = tensor[i, 1:(j - 1), :]
            new_mask[i, :(j - 2)] = 1

    return tensor_without_boundary_tokens, new_mask


def add_sentence_boundary_token_ids(tensor, mask, sentence_begin_token, sentence_end_token):
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape
    ``(batch_size, timesteps + 2)`` or ``(batch_size, timesteps + 2, dim)`` respectively.
    Returns both the new tensor and updated mask.
    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.
    Returns
    -------
    tensor_with_boundary_tokens : ``torch.Tensor``
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : ``torch.Tensor``
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    sequence_lengths = mask.sum(dim=1).data.cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = Variable(tensor.data.new(*new_shape).fill_(0))
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens != 0).long()
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = ((tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask


def get_text_field_mask(text_field_tensors, num_wrapping_dims=0):
    """
    Takes the dictionary of tensors produced by a ``TextField`` and returns a mask
    with 0 where the tokens are padding, and 1 otherwise.  We also handle ``TextFields``
    wrapped by an arbitrary number of ``ListFields``, where the number of wrapping ``ListFields``
    is given by ``num_wrapping_dims``.
    If ``num_wrapping_dims == 0``, the returned mask has shape ``(batch_size, num_tokens)``.
    If ``num_wrapping_dims > 0`` then the returned mask has ``num_wrapping_dims`` extra
    dimensions, so the shape will be ``(batch_size, ..., num_tokens)``.
    There could be several entries in the tensor dictionary with different shapes (e.g., one for
    word ids, one for character ids).  In order to get a token mask, we use the tensor in
    the dictionary with the lowest number of dimensions.  After subtracting ``num_wrapping_dims``,
    if this tensor has two dimensions we assume it has shape ``(batch_size, ..., num_tokens)``,
    and use it for the mask.  If instead it has three dimensions, we assume it has shape
    ``(batch_size, ..., num_tokens, num_features)``, and sum over the last dimension to produce
    the mask.  Most frequently this will be a character id tensor, but it could also be a
    featurized representation of each token, etc.
    NOTE: Our functions for generating masks create torch.LongTensors, because using
    torch.ByteTensors inside Variables makes it easy to run into overflow errors
    when doing mask manipulation, such as summing to get the lengths of sequences - see below.
    >>> mask = torch.ones([260]).byte()
    >>> mask.sum() # equals 260.
    >>> var_mask = torch.autograd.Variable(mask)
    >>> var_mask.sum() # equals 4, due to 8 bit precision - the sum overflows.
    """
    tensor_dims = [(tensor.dim(), tensor) for tensor in text_field_tensors.values()]
    tensor_dims.sort(key=lambda x: x[0])

    smallest_dim = tensor_dims[0][0] - num_wrapping_dims
    if smallest_dim == 2:
        token_tensor = tensor_dims[0][1]
        return (token_tensor != 0).long()
    elif smallest_dim == 3:
        character_tensor = tensor_dims[0][1]
        return ((character_tensor > 0).long().sum(dim=-1) > 0).long()
    else:
        raise ValueError("Expected a tensor with dimension 2 or 3, found {}".format(smallest_dim))


def _last_dimension_applicator(function_to_apply, tensor, mask=None):
    """
    Takes a tensor with 3 or more dimensions and applies a function over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.  We first unsqueeze and expand the mask so that it
    has the same shape as the tensor, then flatten them both to be 2D, pass them through
    the function and put the tensor back in its original shape.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor.size()[-1])
    if mask is not None:
        while mask.dim() < tensor.dim():
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(tensor).contiguous().float()
        mask = mask.view(-1, mask.size()[-1])
    reshaped_result = function_to_apply(reshaped_tensor, mask)
    return reshaped_result.view(*tensor_shape)


def masked_softmax(vector, mask):
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
    We assume that both ``vector`` and ``mask`` (if given) have shape ``(batch_size, vector_dim)``.
    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorical cross-entropy loss.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=-1)
    else:
        # To limit numerical errors from large vector elements outside the mask, we zero these out.
        result = torch.nn.functional.softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
    return result


def last_dim_softmax(tensor, mask=None):
    """
    Takes a tensor with 3 or more dimensions and does a masked softmax over the last dimension.  We
    assume the tensor has shape ``(batch_size, ..., sequence_length)`` and that the mask (if given)
    has shape ``(batch_size, sequence_length)``.
    """
    return _last_dimension_applicator(masked_softmax, tensor, mask)


def weighted_sum(matrix, attention):
    """
    Takes a matrix of vectors and a set of weights over the rows in the matrix (which we call an
    "attention" vector), and returns a weighted sum of the rows in the matrix.  This is the typical
    computation performed after an attention mechanism.
    Note that while we call this a "matrix" of vectors and an attention "vector", we also handle
    higher-order tensors.  We always sum over the second-to-last dimension of the "matrix", and we
    assume that all dimensions in the "matrix" prior to the last dimension are matched in the
    "vector".  Non-matched dimensions in the "vector" must be `directly after the batch dimension`.
    For example, say I have a "matrix" with dimensions ``(batch_size, num_queries, num_words,
    embedding_dim)``.  The attention "vector" then must have at least those dimensions, and could
    have more. Both:
        - ``(batch_size, num_queries, num_words)`` (distribution over words for each query)
        - ``(batch_size, num_documents, num_queries, num_words)`` (distribution over words in a
          query for each document)
    are valid input "vectors", producing tensors of shape:
    ``(batch_size, num_queries, embedding_dim)`` and
    ``(batch_size, num_documents, num_queries, embedding_dim)`` respectively.
    """
    # We'll special-case a few settings here, where there are efficient (but poorly-named)
    # operations in pytorch that already do the computation we need.
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


def device_mapping(cuda_device):
    """
    In order to `torch.load()` a GPU-trained model onto a CPU (or specific GPU),
    you have to supply a `map_location` function. Call this with
    the desired `cuda_device` to get the function that `torch.load()` needs.
    """
    def inner_device_mapping(storage, location):
        if cuda_device >= 0:
            return storage.cuda(cuda_device)
        else:
            return storage
    return inner_device_mapping
