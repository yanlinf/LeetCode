class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = 0
        for n in nums:
            ans = ans ^ n
        return ans  # Equivilant to: return reduce(lambda x, y: x ^ y, nums)