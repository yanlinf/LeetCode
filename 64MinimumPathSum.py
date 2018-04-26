class Solution:
    def minPathSum(self, grid):  # Space Complexity: O(n)
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        h, w = len(grid), len(grid[0])
        dp = [float('inf')] * (w + 1)  # Feature: float('inf')
        dp[w - 1] = 0
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                dp[j] = min(dp[j], dp[j + 1]) + grid[i][j]
        return dp[0]
