import flakes
import unittest
import numpy as np
import GPy
import datetime
from copy import deepcopy


@unittest.skip('')
class StringKernelBasicTests(unittest.TestCase):
    
    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        alphabet = 'acgt'
        self.k_slow = flakes.string.StringKernel(mode='naive', alphabet=alphabet)
        self.k_np = flakes.string.StringKernel(mode='numpy', alphabet=alphabet, sim='dot')
        self.k_tf = flakes.string.StringKernel(mode='tf', alphabet=alphabet)
        self.k_tf_batch = flakes.string.StringKernel(mode='tf-batch', alphabet=alphabet)
        
    def test_sk_slow_1(self):
        self.k_slow.order_coefs = [1.] * 5
        self.k_slow.gap_decay = 2.0
        self.k_slow.match_decay = 2.0
        expected = 504.0
        result = self.k_slow.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_numpy_1(self):
        #self.k_np.order = 5
        self.k_np.order_coefs = [1.] * 5
        self.k_np.gap_decay = 2.0
        self.k_np.match_decay = 2.0
        expected = 504.0
        result = self.k_np.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_1(self):
        self.k_tf.order_coefs = [1.] * 1
        self.k_tf.gap_decay = 2.0
        self.k_tf.match_decay = 2.0
        expected = 24.0
        result = self.k_tf.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_2(self):
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        expected = 5.943705
        result = self.k_tf.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_sk_tf_batch_1(self):
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 2.0
        self.k_tf.match_decay = 2.0
        expected = 504.0
        result = self.k_tf.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_batch_2(self):
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        expected = 5.943705
        result = self.k_tf.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected, places=4)

    def test_sk_numpy_gram_non_gram(self):
        self.k_np.order_coefs = [1.] * 5
        self.k_np.gap_decay = 2.0
        self.k_np.match_decay = 2.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        X2 = deepcopy(X)
        result1 = self.k_np.K(X)
        result2 = self.k_np.K(X, X2)
        self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)

    def test_sk_tf_gram_non_gram(self):
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 2.0
        self.k_tf.match_decay = 2.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        X2 = deepcopy(X)
        result1 = self.k_tf.K(X)
        result2 = self.k_tf.K(X, X2)
        self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)

    def test_sk_tf_batch_gram_non_gram(self):
        self.k_tf_batch.order_coefs = [1.] * 5
        self.k_tf_batch.gap_decay = 2.0
        self.k_tf_batch.match_decay = 2.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        X2 = deepcopy(X)
        result1 = self.k_tf_batch.K(X)
        result2 = self.k_tf_batch.K(X, X2)
        self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)

    def test_sk_numpy_diag_non_diag(self):
        self.k_np.order_coefs = [1.] * 5
        self.k_np.gap_decay = 2.0
        self.k_np.match_decay = 2.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        result1 = np.diag(self.k_np.K(X))
        result2 = self.k_np.K(X, diag=True)
        self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)

    def test_sk_tf_diag_non_diag(self):
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 2.0
        self.k_tf.match_decay = 2.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        result1 = np.diag(self.k_tf.K(X))
        result2 = self.k_tf.K(X, diag=True)
        self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)

    def test_sk_tf_batch_diag_non_diag(self):
        self.k_tf_batch.order_coefs = [1.] * 5
        self.k_tf_batch.gap_decay = 2.0
        self.k_tf_batch.match_decay = 2.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        result1 = np.diag(self.k_tf_batch.K(X))
        result2 = self.k_tf_batch.K(X, diag=True)
        self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)


class StringKernelComparisonTests(unittest.TestCase):

    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        alphabet = 'acgt'
        self.k_slow = flakes.string.StringKernel(mode='naive', alphabet=alphabet)
        self.k_np = flakes.string.StringKernel(mode='numpy', alphabet=alphabet)
        self.k_tf = flakes.string.StringKernel(mode='tf',alphabet=alphabet)
        self.k_np_acos = flakes.string.StringKernel(mode='numpy', alphabet=alphabet, sim='arccosine')
        self.k_tf_acos = flakes.string.StringKernel(mode='tf',alphabet=alphabet, sim='arccosine')
        self.k_tf_preload = flakes.string.StringKernel(mode='tf-batch-preload', alphabet=alphabet)  
        self.k_tf_batch = flakes.string.StringKernel(mode='tf-batch', alphabet=alphabet, wrapper='none', batch_size=10)

    def test_compare_1(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result1 = self.k_tf.K(self.s1, self.s2)

        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.gap_decay = 0.8
        self.k_np.match_decay = 0.8
        result2 = self.k_np.K(self.s1, self.s2)
        self.assertAlmostEqual(result1, result2, places=4)

    def test_compare_2(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result1 = self.k_tf.K(self.s3, self.s4)

        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.gap_decay = 0.8
        self.k_np.match_decay = 0.8
        result2 = self.k_np.K(self.s3, self.s4)
        self.assertAlmostEqual(result1, result2, places=2)

    def test_compare_3(self):
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result1 = self.k_tf.K(self.s1, self.s4)

        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.gap_decay = 0.8
        self.k_np.match_decay = 0.8
        result2 = self.k_np.K(self.s1, self.s4)
        self.assertAlmostEqual(result1, result2, places=2)

    def test_compare_preload_based(self):
        #X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        X = [[self.s1], [self.s2]]#, [self.s3]]
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result1 = self.k_tf.K(X)
        self.k_tf_preload.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_preload.gap_decay = 0.8
        self.k_tf_preload.match_decay = 0.8
        result2 = self.k_tf_preload.K(X)
        np.set_printoptions(suppress=True)

        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=2)
        self.assertAlmostEqual(np.sum(self.k_tf.gap_grads), np.sum(self.k_tf_preload.gap_grads), places=2)
        self.assertAlmostEqual(np.sum(self.k_tf.match_grads)/1000, np.sum(self.k_tf_preload.match_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf.coef_grads)/1000, np.sum(self.k_tf_preload.coef_grads)/1000, places=2)

    def test_compare_tf_and_numpy_based(self):
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        #X = [[self.s1], [self.s2], [self.s3]]
        #X = [[self.s1], [self.s2]]#, [self.s3]]
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5]#, 0.7]
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result1 = self.k_tf.K(X)
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5]#, 0.7]
        self.k_np.gap_decay = 0.8
        self.k_np.match_decay = 0.8
        result2 = self.k_np.K(X)
        np.set_printoptions(suppress=True)

        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=2)
        self.assertAlmostEqual(np.sum(self.k_tf.gap_grads)/1000, np.sum(self.k_np.gap_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf.match_grads)/1000, np.sum(self.k_np.match_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf.coef_grads)/1000, np.sum(self.k_np.coef_grads)/1000, places=2)

    def test_compare_tf_and_numpy_based_acos(self):
        #X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        #X = [[self.s1], [self.s2], [self.s3]]
        X = [[self.s1], [self.s2]]#, [self.s3]]
        self.k_tf_acos.order_coefs = [0.1, 0.2, 0.4, 0.5]#, 0.7]
        self.k_tf_acos.gap_decay = 0.8
        self.k_tf_acos.match_decay = 0.8
        result1 = self.k_tf_acos.K(X)
        self.k_np_acos.order_coefs = [0.1, 0.2, 0.4, 0.5]#, 0.7]
        self.k_np_acos.gap_decay = 0.8
        self.k_np_acos.match_decay = 0.8
        result2 = self.k_np_acos.K(X)
        np.set_printoptions(suppress=True)

        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=2)
        self.assertAlmostEqual(np.sum(self.k_tf_acos.gap_grads)/1000, np.sum(self.k_np_acos.gap_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf_acos.match_grads)/1000, np.sum(self.k_np_acos.match_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf_acos.coef_grads)/1000, np.sum(self.k_np_acos.coef_grads)/1000, places=2)

    #@unittest.skip('')
    def test_compare_preload_and_batch(self):
        #X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        X = [[self.s1], [self.s2], [self.s3]]
        #X = [[self.s1], [self.s2]]
        self.k_tf_batch.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_batch.gap_decay = 0.8
        self.k_tf_batch.match_decay = 0.8
        result1 = self.k_tf_batch.K(X)
        self.k_tf_preload.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_preload.gap_decay = 0.8
        self.k_tf_preload.match_decay = 0.8
        result2 = self.k_tf_preload.K(X)
        np.set_printoptions(suppress=True)

        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=2)
        self.assertAlmostEqual(np.sum(self.k_tf_batch.gap_grads)/1000, np.sum(self.k_tf_preload.gap_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf_batch.match_grads)/1000, np.sum(self.k_tf_preload.match_grads)/1000, places=2)
        self.assertAlmostEqual(np.sum(self.k_tf_batch.coef_grads)/1000, np.sum(self.k_tf_preload.coef_grads)/1000, places=2)


class StringKernelGradientTests(unittest.TestCase):
    """
    We use the AD capabilities of TensorFlow to check if
    our hand-made gradient implementations are correct.
    """

    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        alphabet = 'acgt'
        self.k_slow = flakes.string.StringKernel(mode='naive', alphabet=alphabet)
        self.k_np = flakes.string.StringKernel(mode='numpy', alphabet=alphabet, sim='dot')
        self.k_tf = flakes.string.StringKernel(mode='tf',alphabet=alphabet)
        self.k_tf_preload = flakes.string.StringKernel(mode='tf-batch-preload', alphabet=alphabet)  
        self.k_tf_batch = flakes.string.StringKernel(mode='tf-batch', alphabet=alphabet, wrapper='none', batch_size=10)
        self.k_tf_batch_norm = flakes.string.StringKernel(mode='tf-batch', alphabet=alphabet, wrapper='norm', batch_size=10)
        self.k_np_pos = flakes.string.StringKernel(mode='numpy', alphabet=alphabet, sim='pos_dot')
        self.k_tf_pos = flakes.string.StringKernel(mode='tf', alphabet=alphabet, sim='pos_dot')

    def test_gradient_gap_1(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.order_coefs = [0.1] * 2
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result = self.k_tf.K(self.s1, self.s2)
        true_grads = self.k_tf.gap_grads

        E = 1e-4
        self.k_tf.gap_decay = 0.8 + E
        g_result1 = self.k_tf.K(self.s1, self.s2)
        self.k_tf.gap_decay = 0.8 - E
        g_result2 = self.k_tf.K(self.s1, self.s2)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(true_grads, g_result, places=2)

    def test_gradient_gap_1_numpy(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.order_coefs = [0.1] * 2
        self.k_np.gap_decay = 0.8
        self.k_np.match_decay = 0.8
        result = self.k_np.K(self.s1, self.s2)
        true_grads = self.k_np.gap_grads

        E = 1e-4
        self.k_np.gap_decay = 0.8 + E
        g_result1 = self.k_np.K(self.s1, self.s2)
        self.k_np.gap_decay = 0.8 - E
        g_result2 = self.k_np.K(self.s1, self.s2)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(true_grads, g_result, places=2)

    def test_gradient_gap_1_preload(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_preload.order_coefs = [0.1] * 2
        self.k_tf_preload.gap_decay = 0.8
        self.k_tf_preload.match_decay = 0.8
        result = self.k_tf_preload.K(self.s1, self.s2)
        true_grads = self.k_tf_preload.gap_grads

        E = 1e-4
        self.k_tf_preload.gap_decay = 0.8 + E
        g_result1 = self.k_tf_preload.K(self.s1, self.s2)
        self.k_tf_preload.gap_decay = 0.8 - E
        g_result2 = self.k_tf_preload.K(self.s1, self.s2)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(true_grads, g_result, places=2)

    def test_gradient_match_1(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.order_coefs = [0.1] * 2
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result = self.k_tf.K(self.s1, self.s2)
        true_grads = self.k_tf.match_grads

        E = 1e-4
        self.k_tf.match_decay = 0.8 + E
        g_result1 = self.k_tf.K(self.s1, self.s2)
        self.k_tf.match_decay = 0.8 - E
        g_result2 = self.k_tf.K(self.s1, self.s2)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(true_grads, g_result, places=2)

    def test_gradient_coefs_1(self):
        self.k_tf.order_coefs = [0.1] * 2
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result = self.k_tf.K(self.s1, self.s2)
        true_grads = self.k_tf.coef_grads

        E = 1e-4
        self.k_tf.order_coefs = [0.1 + E, 0.1]
        g_result1 = self.k_tf.K(self.s1, self.s2)
        self.k_tf.order_coefs = [0.1 - E, 0.1]
        g_result2 = self.k_tf.K(self.s1, self.s2)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(true_grads[0][0][0], g_result[0][0], places=2)

    def test_gradient_coefs_2(self):
        self.k_tf.order_coefs = [0.1] * 2
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        result = self.k_tf.K(self.s1, self.s2)
        true_grads = self.k_tf.coef_grads

        E = 1e-4
        self.k_tf.order_coefs = [0.1, 0.1 + E]
        g_result1 = self.k_tf.K(self.s1, self.s2)
        self.k_tf.order_coefs = [0.1, 0.1 - E]
        g_result2 = self.k_tf.K(self.s1, self.s2)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(true_grads[0][0][1], g_result[0][0], places=2)

    def test_gradient_gap_2(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.order_coefs = [0.1] * 2
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        result = self.k_tf.K(X)
        true_grads = self.k_tf.gap_grads

        E = 1e-4
        self.k_tf.gap_decay = 0.8 + E
        g_result1 = self.k_tf.K(X)
        self.k_tf.gap_decay = 0.8 - E
        g_result2 = self.k_tf.K(X)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(np.sum(true_grads)/100, np.sum(g_result)/100, places=2)

    def test_gradient_match_2(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.order_coefs = [0.1] * 2
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        result = self.k_tf.K(X)
        true_grads = self.k_tf.match_grads

        E = 1e-4
        self.k_tf.match_decay = 0.8 + E
        g_result1 = self.k_tf.K(X)
        self.k_tf.match_decay = 0.8 - E
        g_result2 = self.k_tf.K(X)
        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(np.sum(true_grads)/100, np.sum(g_result)/100, places=2)

    def test_gradient_batch_norm_gap_1(self):
        #self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_batch_norm.order_coefs = [1.0] * 2
        self.k_tf_batch_norm.gap_decay = 1.0
        self.k_tf_batch_norm.match_decay = 0.8
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        result = self.k_tf_batch_norm.K(X) 
        true_grads = self.k_tf_batch_norm.gap_grads

        E = 1e-2
        self.k_tf_batch_norm.gap_decay = 1.0 + E
        g_result1 = self.k_tf_batch_norm.K(X)
        self.k_tf_batch_norm.gap_decay = 1.0 - E
        g_result2 = self.k_tf_batch_norm.K(X)
        g_result = (g_result1 - g_result2) / (2 * E)

        self.assertAlmostEqual(np.sum(true_grads), np.sum(g_result), places=2)

    def test_gradient_positional_1(self):
        self.k_np_pos.order_coefs = [1.0] * 2
        self.k_np_pos.lengthscale = 1.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        #X = np.array([np.array(s) for s in X])
        result = self.k_np_pos.K(X) 
        true_grads = self.k_np_pos.ls_grads

        E = 1e-2
        self.k_np_pos.lengthscale = 1.0 + E
        g_result1 = self.k_np_pos.K(X)
        self.k_np_pos.lengthscale = 1.0 - E
        g_result2 = self.k_np_pos.K(X)
        g_result = (g_result1 - g_result2) / (2 * E)

        #print g_result1
        #print g_result2
        #print g_result
        #print true_grads
        self.assertAlmostEqual(np.sum(true_grads), np.sum(g_result), places=2)

    def test_gradient_positional_sim_1(self):
        enc1 = flakes.string.sk_util.encode_string(self.s1, self.k_np_pos.index)
        enc2 = flakes.string.sk_util.encode_string(self.s2, self.k_np_pos.index)
        ls = 1.0
        result, true_grads = self.k_np_pos._implementation._pos_dot(enc1, enc2, ls)

        E = 1e-2
        ls = 1.0 + E
        g_result1, _ = self.k_np_pos._implementation._pos_dot(enc1, enc2, ls)
        ls = 1.0 - E
        g_result2, _ = self.k_np_pos._implementation._pos_dot(enc1, enc2, ls)
        g_result = (g_result1 - g_result2) / (2 * E)

        #print g_result1
        #print g_result2
        #print g_result
        #print true_grads
        self.assertAlmostEqual(np.sum(true_grads), np.sum(g_result), places=2)

    def test_gradient_positional_tf_vs_np(self):
        self.k_np_pos.order_coefs = [1.0] * 2
        self.k_np_pos.lengthscale = 1.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        #X = np.array([np.array(s) for s in X])
        result = self.k_np_pos.K(X) 
        true_grads = self.k_np_pos.ls_grads

        self.k_tf_pos.order_coefs = [1.0] * 2
        self.k_tf_pos.lengthscale = 1.0
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        #X = np.array([np.array(s) for s in X])
        result_tf = self.k_tf_pos.K(X) 
        true_grads_tf = self.k_tf_pos.ls_grads

        self.assertAlmostEqual(np.sum(result), np.sum(result_tf), places=2)
        self.assertAlmostEqual(np.sum(true_grads), np.sum(true_grads_tf), places=2)
        
if __name__ == "__main__":
    unittest.main()
