import flakes
import unittest


class StringKernelTests(unittest.TestCase):
    
    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.s3 = 'cgtagctagcgacgcagccaatcgatcg'
        self.s4 = 'cgagatgccaatagagagagcgctgta'
        self.k_slow = flakes.string.StringKernel(mode='slow')
        self.k_np = flakes.string.StringKernel(mode='numpy')
        self.k_tf = flakes.string.StringKernel()
        self.k_tf.alphabet = {'a': 0, 'c': 1, 'g': 2, 't': 3}
        
    def test_sk_slow_1(self):
        self.k_slow.order = 5
        self.k_slow.order_coefs = [1.] * 5
        self.k_slow.decay = 2.0
        expected = 504.0
        result = self.k_slow.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_numpy_1(self):
        self.k_np.order = 5
        self.k_np.order_coefs = [1.] * 5
        self.k_np.decay = 2.0
        expected = 504.0
        result = self.k_np.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_1(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.decay = 2.0
        expected = 504.0
        result = self.k_tf.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_sk_tf_2(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [1.] * 5
        self.k_tf.decay = 0.8
        expected = 5.943705
        result = self.k_tf.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)

    def test_compare_1(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s1, self.s2)

        self.k_np.order = 5
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s1, self.s2)
        self.assertAlmostEqual(result1, result2)

    def test_compare_2(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s3, self.s4)

        self.k_np.order = 5
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s3, self.s4)
        self.assertAlmostEqual(result1, result2, places=2)

    def test_compare_3(self):
        self.k_tf.order = 5
        self.k_tf.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_tf.decay = 0.8
        result1 = self.k_tf.k(self.s1, self.s4)

        self.k_np.order = 5
        self.k_np.order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7]
        self.k_np.decay = 0.8
        result2 = self.k_np.k(self.s1, self.s4)
        self.assertAlmostEqual(result1, result2, places=2)
        

if __name__ == "__main__":
    unittest.main()
