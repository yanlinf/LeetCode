class Solution:
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        ans = x ^ y
        res = 0
        while ans > 0:
            ans -= ans & (-ans)  # Equivilant to: ans &= ans - 1
            res += 1
        return res
        
        # Another solution: return bin(x ^ y).count('1')