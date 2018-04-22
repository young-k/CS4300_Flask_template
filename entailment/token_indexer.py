from common_util import pad_sequence_to_length


class TokenIndexer(object):
    """
    This :class:`TokenIndexer` represents tokens as single integers.
    Parameters
    ----------
    namespace : ``str``, optional (default=``tokens``)
        We will use this namespace in the :class:`Vocabulary` to map strings to indices.
    lowercase_tokens : ``bool``, optional (default=``False``)
        If ``True``, we will call ``token.lower()`` before getting an index for the token from the
        vocabulary.
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace='tokens', lowercase_tokens=False):
        self.namespace = namespace
        self.lowercase_tokens = lowercase_tokens
    
    def count_vocab_items(self, token, counter):
        # If `text_id` is set on the token (e.g., if we're using some kind of hash-based word
        # encoding), we will not be using the vocab for this token.
        if getattr(token, 'text_id', None) is None:
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            counter[self.namespace][text] += 1
    
    def token_to_indices(self, token, vocabulary):
        if getattr(token, 'text_id', None) is not None:
            # `text_id` being set on the token means that we aren't using the vocab, we just use
            # this id instead.
            index = token.text_id
        else:
            text = token.text
            if self.lowercase_tokens:
                text = text.lower()
            index = vocabulary.get_token_index(text, self.namespace)
        return index

    def get_padding_token(self):
        return 0
    
    def get_padding_lengths(self, token):
        return {}

    def pad_token_sequence(self, tokens, desired_num_tokens, padding_lengths):
        return pad_sequence_to_length(tokens, desired_num_tokens)

    @classmethod
    def from_params(cls, params):
        namespace = params.pop('namespace', 'tokens')
        lowercase_tokens = params.pop_bool('lowercase_tokens', False)
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace, lowercase_tokens=lowercase_tokens)