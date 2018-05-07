class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, res, last = 0, 0, {}  # Algorithm: optimized sliding window
        for right, c in enumerate(s):  # Feature: enumerate, cleaner and
                                       #          faster than frequent s[i]
            if c in last and left <= last[c]:
                left = last[c] + 1
            last[c] = right
            res = max(res, right - left + 1)
        return res


class SolutionB:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import defaultdict
        counts = defaultdict(int)
        left, res = 0, 0
        for right in range(len(s)):  # Algorithm: Sliding Window
            counts[s[right]] += 1
            while counts[s[right]] > 1:
                counts[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res
