import flakes
import unittest


class StringKernelTests(unittest.TestCase):
    
    def setUp(self):
        self.s1 = 'cata'
        self.s2 = 'gatta'
        self.k_slow = flakes.string.StringKernel(slow=True)
        self.k_fast = flakes.string.StringKernel()
        
    def test_sk_1(self):
        self.k_slow.order = 5
        self.k_slow.order_coefs = [1.] * 5
        self.k_slow.decay = 2.0
        expected = 504.0
        result = self.k_slow.k(self.s1, self.s2)
        self.assertAlmostEqual(result, expected)
        

if __name__ == "__main__":
    unittest.main()
