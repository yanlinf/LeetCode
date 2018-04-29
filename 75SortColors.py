class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        p0, p1, p2 = 0, 0, 0  # Algorithm: in-place partition using ptrs
        while p2 < len(nums):
            print(p0, p1, p2, nums)
            if nums[p2] == 0:
                tmp, nums[p2] = nums[p2], nums[p1]
                nums[p1] = nums[p0]
                nums[p0] = tmp
                p0, p1, p2 = p0 + 1, p1 + 1, p2 + 1
            elif nums[p2] == 1:
                nums[p1], nums[p2] = nums[p2], nums[p1]
                p1, p2 = p1 + 1, p2 + 1
            else:
                p2 += 1
