import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops as cfops
from tensorflow.python.ops import tensor_array_ops as taops
from sk_tf import TFStringKernel
from sk_tf_batch import TFBatchStringKernel
from sk_numpy import NumpyStringKernel
from sk_naive import NaiveStringKernel
from sk_util import build_one_hot


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
                 embs=None, alphabet=None, device='/cpu:0'):
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

        ####################################

        # If we are in TF mode we need to do some preparations...
        if self.mode == 'tf' or self.mode == 'tf-row':
            # Instead of using one graph per string pair, we
            # build a single one using the maximum length in
            # the data. Smaller strings are padded with zeroes
            # when converted to the matrix input.
            maxlen = max([len(x[0]) for x in np.concatenate((X, X2))])

            # We have to build a new graph if 1) there is no graph or
            # 2) current graph maxlen is not large enough for these inputs
            if self.graph is None:
                sys.stderr.write("No graph found. Building one.\n")
                if self.mode == 'tf':
                    self._build_graph(maxlen)
                elif self.mode == 'tf-row':
                    self._build_graph_row(maxlen)
                self.maxlen = maxlen
            elif maxlen > self.maxlen:
                sys.stderr.write("Current graph does not have space for" +
                                 "these string sizes. Building a new one.\n")
                if self.mode == 'tf':
                    self._build_graph(maxlen)
                elif self.mode == 'tf-row':
                    self._build_graph_row(maxlen)
                self.maxlen = maxlen

            # Finally, we also start a TF session
            if 'gpu' in self.device:
                self.sess = tf.Session(graph=self.graph, config=self.gpu_config)
            else:
                self.sess = tf.Session(graph=self.graph)
            
        # All set up. Proceed with Gram matrix calculation.
        result = np.zeros(shape=(len(X), len(X2)))
        self.gap_grads = np.zeros(shape=(len(X), len(X2)))
        self.match_grads = np.zeros(shape=(len(X), len(X2)))
        self.coef_grads = np.zeros(shape=(len(X), len(X2), self.order))
        
        if self.mode == 'tf-row':
            #result = self._k_tf_mat(X, X2)
            for i, x1 in enumerate(X):
                print i
                if gram:
                    row = X2[:i+1]
                else:
                    row = X2
                row_result = np.zeros((1,0))
                for j in xrange((i / BATCH_SIZE) + 1):
                    partial_result = self._k_tf_row(x1, row[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
                    row_result = np.concatenate((row_result, partial_result), axis=1)
                if gram:
                    result[i, :i+1] = row_result
                    if i > 0:
                        result[:i+1, i] = row_result
                else:
                    result[i] = row_result              

        return result
