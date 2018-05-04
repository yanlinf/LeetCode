class Solution:
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums:
            return [-1, -1]
        left, right = 0, len(nums)
        while left + 1 < right:
            mid = (left + right - 1) // 2
            if nums[mid] >= target:
                right = mid + 1
            else:
                left = mid + 1
        start = left if nums[left] == target else -1
        left, right = start, len(nums)
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid
        end = left if nums[left] == target else -1
        return [start, end]