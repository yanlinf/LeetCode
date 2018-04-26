class Solution:
    def climbStairs(self, n):  # Complexity: O(n)
        """
        :type n: int
        :rtype: int
        """
        a, b = 1, 2
        for i in range(0, n - 2):
            a, b = b, a + b
        return b if n > 1 else 1


class SolutionB:
    def climbStairs(self, n):  # Complexity: O(n^2)
        """
        :type n: int
        :rtype: int
        """
        from math import factorial
        def comb(N, k):
            return factorial(N) // factorial(k) // factorial(N - k)

        res = 0
        for x in range(n // 2 + 1):
            res += int(comb(n - x, x))
        return res
