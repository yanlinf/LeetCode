class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        curr_min = prices[0]
        res = 0
        for price in prices:
            res = price - curr_min if price - curr_min > res else res
            # Note: faster than max(res, ...)
            # since no function is called
            curr_min = price if price < curr_min else curr_min
        return res
