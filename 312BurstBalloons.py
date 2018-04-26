class Solution:
    ans = 0
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        nums = [1] + nums + [1]
        dp = [[0] * (n + 2) for _ in range(n + 2)]  # Algorithm: DP (key: treat k as the last balloon burst)
        for l in range(1, n + 1):
            for i in range(1, n + 2 - l):
                j = i + l
                for k in range(i, j):
                    ans = dp[i][k] + dp[k + 1][j] + nums[k] * nums[i - 1] * nums[j]
                    if ans > dp[i][j]:
                        dp[i][j] = ans
        return dp[1][n + 1]
