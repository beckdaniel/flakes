import numpy as np


def build_one_hot(alphabet):
    """
    Build one-hot encodings for a given alphabet.
    """
    embs = {}
    dim = len(alphabet)
    for i, symbol in enumerate(alphabet):
        embs[symbol] = np.zeros(dim)
        embs[symbol][i] = 1.0
    return embs


def build_input_matrix(string, embs, length=None, dim=None):
    """
    Transform an input (string or list) into a
    numpy matrix. Notice that we use an implicit
    zero padding here when length > len(s).
    """
    # Get length. If length is given we assume it is for
    # padding purposes.
    if length is None:
        length = len(string)
    # Get dimensionality
    if dim is None:
        dim = embs[embs.keys()[0]].shape[0]
    result = np.zeros((length, dim))    
    for i, symbol in enumerate(string):
        try: 
            result[i] = embs[symbol] 
        except KeyError:
            # Unknown word, we assume a 0 vector
            result[i] = np.zeros(dim)
    return result
