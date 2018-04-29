class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0] * len(nums)
        res = 0
        for i in range(len(nums)):
            dp[i] = 1 + max([0] + [dp[j] for j in range(i) if nums[j] < nums[i]])
            res = max(res, dp[i])
        return res
