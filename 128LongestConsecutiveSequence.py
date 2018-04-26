class Solution:
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = set(nums)  # Feature: lookup in O(1)
        res = 1
        for n in nums:
            if n - 1 not in nums:
                k = n + 1
                while k in nums:
                    k += 1
                res = max(res, k - n)
        return res if nums else 0
