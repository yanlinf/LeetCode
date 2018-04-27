class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans, res = nums[0], nums[0]
        for n in nums[1:]:
            ans = ans + n if ans > 0 else n
            res = max(ans, res)
        return res