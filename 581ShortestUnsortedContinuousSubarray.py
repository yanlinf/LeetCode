class Solution:
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        low = min([i for i in range(1, n) if nums[i - 1] > nums[i]] + [n])
        if low == n:
            return 0
        up = max([i for i in range(1, n) if nums[i - 1] > nums[i]] + [0])
        nmin, nmax = min(nums[low - 1:up + 1]), max(nums[low - 1:up + 1])
        start = max([i for i in range(low) if nums[i] <= nmin] + [-1])
        end = min([i for i in range(up, n) if nums[i] >= nmax] + [n])
        return end - start - 1
