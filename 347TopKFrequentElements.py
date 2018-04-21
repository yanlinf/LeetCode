from collections import Counter

class Solution:
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        counter = Counter(nums)
        res = sorted(counter.keys(), key=lambda x: -counter[x])[0: k]
                # Equivilant to: res = [x[0] for x in counter.most_common(k)]
                # Feature: counter.most_common(n) counter[x]
        return res

