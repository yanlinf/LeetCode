class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        size = len(nums)
        for i in range(nums.count(0)):
            nums.remove(0)
        nums += [0] * (size - len(nums))