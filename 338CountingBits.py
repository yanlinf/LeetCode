class Solution:
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        dp = [0]
        for i in range(1, num + 1):
            dp.append(1 + dp[i & (i - 1)])
        return dp
