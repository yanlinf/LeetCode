class Solution:
    def findMedianSortedArrays(self, nums1, nums2):  # *there's an O(log(min(m, n))) solution
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = sorted(nums1 + nums2)
        size = len(nums)
        if size % 2 == 1:
            return float(nums[size // 2])
        else:
            return (nums[size // 2 - 1] + nums[size // 2]) / 2