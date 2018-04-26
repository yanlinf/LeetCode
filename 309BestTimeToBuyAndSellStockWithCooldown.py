class Solution:
    def maxProfit(self, prices):  # DP, Complexity: O(n)
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n < 2:
            return 0
        profit, profit_with_stock = [0] * (n + 2), [0] * (n + 2)
        profit_with_stock[0] = -prices[0]
        for i in range(1, n):
            profit[i] = max(profit[i - 1], profit_with_stock[i - 1] + prices[i])
            profit_with_stock[i] = max(profit_with_stock[i - 1], profit[i - 2] - prices[i])
        return profit[n - 1]


class SolutionB:
    def maxProfit(self, prices):  # DP, Complexity: O(n ^ 2)
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        prices.insert(0, 0)
        dp = [0] * (n + 2)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            for k in range(1, i):
                ans = dp[k - 2] + prices[i] - prices[k]
                if ans > dp[i]:
                    dp[i] = ans
        return dp[n]
