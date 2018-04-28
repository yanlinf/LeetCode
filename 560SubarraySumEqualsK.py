class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        from collections import defaultdict
        cache = defaultdict(int)  # Feature: defaultdict(<callable object>)
        cache[0] = 1
        ans, res = 0, 0
        for n in nums:
            ans += n
            res += cache[ans - k]
            cache[ans] += 1
        return res
