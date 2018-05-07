class Solution:
    def maxProduct(self, nums):  # Algorithm: DP (O(n) Time, O(1) Space)
        """
        :type nums: List[int]
        :rtype: int
        """
        dp, res = (nums[0], nums[0]), nums[0]
        for n in nums[1:]:
            tmp = [dp[0] * n, dp[1] * n, n]
            dp = (max(tmp), min(tmp))
            res = max(res, dp[0])
        return res
