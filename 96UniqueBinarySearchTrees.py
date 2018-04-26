class Solution:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        from math import factorial
        return factorial(2 * n) // (factorial(n)) ** 2 // (n + 1)  # Three times faster than comb(N, k)


class SolutionB:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        from scipy.special import comb
        eps = 0.01
        return int(comb(2 * n, n) + eps) // (n + 1)  # Note: comb(N, k) returns float64
