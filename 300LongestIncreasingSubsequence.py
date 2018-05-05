class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        size, dp = 0, [0] * len(nums)
        for n in nums:
            left, right = 0, size
            while left < right:  # Algorithm: Binary Search
                mid = (left + right) // 2
                if dp[mid] >= n:
                    right = mid
                else:
                    left = mid + 1
            dp[left] = n
            size = max(left + 1, size)
        return size


class SolutionB:
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
