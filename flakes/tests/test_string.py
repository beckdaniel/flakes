import flakes
import unittest
import numpy as np
import GPy
import datetime


class StringKernelTests(unittest.TestCase):
    
    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        alphabet = 'acgt'
        self.k_slow = flakes.string.StringKernel(mode='naive', alphabet=alphabet)
        self.k_np = flakes.string.StringKernel(mode='numpy', alphabet=alphabet)
        self.k_tf = flakes.string.StringKernel(alphabet=alphabet)
        self.k_tf_row = flakes.string.StringKernel(mode='tf-batch', alphabet=alphabet)
        self.k_tf_gram = flakes.string.StringKernel(mode='tf-gram', alphabet=alphabet)
        
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
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 2.0
        self.k_tf.match_decay = 2.0
        expected = 504.0
        result = self.k_tf.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_2(self):
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.gap_decay = 0.8
        self.k_tf.match_decay = 0.8
        expected = 5.943705
        result = self.k_tf.K(self.s1, self.s2)
        self.assertAlmostEqual(result, expected, places=4)

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
        print g_result1
        print g_result2
        print true_grads

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
        print g_result1
        print g_result2
        print true_grads

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
        print g_result1
        print g_result2
        print true_grads

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
        print g_result1
        print g_result2
        print true_grads

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
        print g_result1
        print g_result2
        print true_grads

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
        print g_result1
        print g_result2
        print true_grads

        g_result = (g_result1 - g_result2) / (2 * E)
        self.assertAlmostEqual(np.sum(true_grads)/100, np.sum(g_result)/100, places=2)

    def test_compare_gram_based(self):
        #X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        X = [[self.s1], [self.s2], [self.s3]]
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.K(X)
        self.k_tf_gram.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_gram.decay = 0.8
        result2 = self.k_tf_gram.K(X)
        print "NORMAL"
        print result1
        print "GRAM"
        print result2
        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=7)

    @unittest.skip('batch-based is not 100% done')
    def test_compare_row_based(self):
        X = [[self.s1], [self.s2], [self.s3], [self.s4]]
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.K(X)
        self.k_tf_row.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf_row.decay = 0.8
        result2 = self.k_tf_row.K(X)
        print result1
        print result2
        self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=7)


if __name__ == "__main__":
    unittest.main()
