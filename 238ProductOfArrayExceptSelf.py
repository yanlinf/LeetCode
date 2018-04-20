class Solution:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        ans1, ans2, size = 1, 1, len(nums)
        res = [1] * size
        for i in range(size):
            res[i] *= ans1
            res[size - i - 1] *= ans2
            ans1 *= nums[i]
            ans2 *= nums[size - i - 1]
        return res


class SolutionB:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        prod, zero = 1, 0
        for n in nums:
            if n != 0:
                prod *= n
            else:
                zero += 1
        if zero == 0:
            return [prod // n for n in nums]
        elif zero == 1:
            return [prod if n == 0 else 0 for n in nums]
        else:
            return [0] * len(nums)
