def pad_sequence_to_length(sequence, desired_length, default_value=lambda: 0, padding_on_right=True):
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.
    Parameters
    ----------
    sequence : List
        A list of objects to be padded.
    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.
    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.
    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?
    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence
