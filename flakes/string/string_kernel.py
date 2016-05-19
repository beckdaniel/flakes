import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
from sk_tf import TFStringKernel
from sk_tf_batch import TFBatchStringKernel
from sk_tf_gram import TFGramStringKernel
from sk_numpy import NumpyStringKernel
from sk_naive import NaiveStringKernel
from sk_util import build_one_hot
import line_profiler


class StringKernel(object):
    """
    A general class for String Kernels. Default mode is
    TensorFlow mode but numpy and non-vectorized implementations
    are also available. The last two should be used only for
    testing and debugging, TF is generally faster, even in
    CPU-only environments.
    
    The parameterization is based on:
    
    Cancedda et. al (2003) "Word-Sequence Kernels" JMLR

    with two different decay parameters: one for gaps
    and another one for symbol matchings. There is 
    also a list of order coeficients which weights
    different n-grams orders. The overall order
    is implicitly obtained by the size of this list.
    This is *not* the symbol-dependent version.
    
    :param gap_decay: decay for symbols gaps, defaults to 1.0
    :param match_decay: decay for symbols matches, defaults to 1.0
    :param order_coefs: list of coefficients for different ngram
    orders, defaults to [1.0]
    :param mode: inner kernel implementation, defaults to TF
    :param device: where to run the inner kernel calculation,
    in TF nomenclature (only used if in TF mode).
    """

    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], mode='tf', 
                 embs=None, alphabet=None, device='/cpu:0',
                 trace=None):
        if (embs is None) and (alphabet is None):
            raise ValueError("You need to provide either an embedding" + 
                             " dictionary through \"embs\" or a list" +
                             " of symbols through \"alphabet\".")
        # In case we have an alphabet we build
        # one-hot encodings as the embeddings, i.e.,
        # we assume hard match between symbols.
        if embs is None:
            embs = build_one_hot(alphabet)
        self.gap_decay = gap_decay
        self.match_decay = match_decay
        self.order_coefs = order_coefs
        if mode == 'tf':
            self._implementation = TFStringKernel(embs, device)
        elif mode == 'tf-gram':
            self._implementation = TFGramStringKernel(embs, device=device, trace=trace)
        elif mode == 'tf-batch':
            self._implementation = TFBatchStringKernel(embs, device)
        elif mode == 'numpy':
            self._implementation = NumpyStringKernel(embs)
        elif mode == 'naive':
            self._implementation = NaiveStringKernel(embs)

    @property
    def order(self):
        """
        Kernel ngram order, defined implicitly.
        """
        return len(self.order_coefs)

    def _get_params(self):
        return [self.gap_decay, self.match_decay, self.order_coefs]

    def K(self, X, X2=None):
        """
        Calculate the Gram matrix over two lists of strings. The
        underlying method used for kernel calculation depends
        on self.mode (slow, numpy or TF). 
        """
        # Symmetry check to ensure that we only calculate
        # the lower diagonal.
        if X2 is None:
            X2 = X
            gram = True
        else:
            gram = False

        # This can also be calculated for single elements but
        # we need to explicitly convert to lists before any
        # processing
        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            X = [[X]]
        if not (isinstance(X2, list) or isinstance(X2, np.ndarray)):
            X2 = [[X2]]

        params = self._get_params()
        result = self._implementation.K(X, X2, gram, params)
        k_result = result[0]
        self.gap_grads = result[1]
        self.match_grads = result[2]
        self.coef_grads = result[3]

        return k_result

