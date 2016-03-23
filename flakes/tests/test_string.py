import flakes
import unittest


class StringKernelTests(unittest.TestCase):
    
    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.k_slow = flakes.string.StringKernel(mode='slow')
        self.k_np = flakes.string.StringKernel(mode='numpy')
        
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
        

if __name__ == "__main__":
    unittest.main()
