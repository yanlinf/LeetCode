class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        norob, rob = 0, 0
        for n in nums:
            norob, rob = rob, max(norob + n, rob)
        return rob