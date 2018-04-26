class Solution:
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        from scipy.special import comb
        return int(comb(m + n - 2, n - 1))  # Feature: comb, perm
