from token_indexer import TokenIndexer
from common_util import pad_sequence_to_length


def _make_bos_eos(character, padding_character, beginning_of_word_character, end_of_word_character,
                  max_word_length):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.
    """
    max_word_length = 50

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260 # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
            beginning_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )
    end_of_sentence_characters = _make_bos_eos(
            end_of_sentence_character,
            padding_character,
            beginning_of_word_character,
            end_of_word_character,
            max_word_length
    )

    bos_token = '<S>'
    eos_token = '</S>'

    @staticmethod
    def convert_word_to_char_ids(word):
        if word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[:(ELMoCharacterMapper.max_word_length-2)]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = ord(chr_id)
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]


class ELMoTokenCharactersIndexer(TokenIndexer):
    """
    Convert a token to an array of character ids to compute ELMo representations.
    Parameters
    ----------
    namespace : ``str``, optional (default=``elmo_characters``)
    """
    # pylint: disable=no-self-use
    def __init__(self, namespace='elmo_characters'):
        self._namespace = namespace
    
    def count_vocab_items(self, token, counter):
        pass
    
    def token_to_indices(self, token, vocabulary):
        return ELMoCharacterMapper.convert_word_to_char_ids(token.text)

    def get_padding_lengths(self, token):
        return {}
    
    def get_padding_token(self):
        return []

    @staticmethod
    def _default_value_for_padding():
        return [0] * ELMoCharacterMapper.max_word_length

    def pad_token_sequence(self, tokens, desired_num_tokens, padding_lengths):
        return pad_sequence_to_length(tokens, desired_num_tokens, default_value=self._default_value_for_padding)

    @classmethod
    def from_params(cls, params):
        """
        Parameters
        ----------
        namespace : ``str``, optional (default=``elmo_characters``)
        """
        namespace = params.pop('namespace', 'elmo_characters')
        params.assert_empty(cls.__name__)
        return cls(namespace=namespace)

