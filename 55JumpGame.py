class Solution:
    def canJump(self, nums):  # Algorithm: Greedy / O(n) DP
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        smallest = n - 1
        for pos in range(n - 2, -1, -1):
            if pos + nums[pos] >= smallest:
                smallest = pos
        return nums[0] >= smallest


class SolutionB:
    def canJump(self, nums):  # Algorithm: Greedy
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) == 1:
            return True
        pos = 0
        while nums[pos] > 0:
            if pos + nums[pos] >= len(nums) - 1:
                return True
            ans = 1
            for step in range(1, nums[pos] + 1):
                if nums[pos + step] + step > nums[pos + ans] + ans:
                    ans = step
            pos = pos + ans
        return False
