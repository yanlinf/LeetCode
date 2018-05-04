class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1
        left, right = 0, len(nums)
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] > nums[left] and \
                    nums[left] <= target < nums[mid] or \
                    nums[mid] < nums[left] and \
                    (target >= nums[left] or target < nums[mid]):
                    # Feature: a < b < c
                right = mid
            else:
                left = mid
        return left if nums[left] == target else -1


class SolutionB:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def myBinarySearch(i, j):
            if i + 1 == j:
                return i if nums[i] == target else -1
            mid = (i + j) // 2
            if nums[mid] > nums[i] and \
                    nums[i] <= target < nums[mid] or \
                    nums[mid] < nums[i] and \
                    (target >= nums[i] or target < nums[mid]):
                return myBinarySearch(i, mid)
            else:
                return myBinarySearch(mid, j)
        return myBinarySearch(0, len(nums)) if nums else -1
