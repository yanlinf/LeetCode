from collections import Counter

class Solution:
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        counter = Counter(num for num in nums)
        return [x for x in counter if counter[x] == 2]