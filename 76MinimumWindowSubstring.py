class Solution:
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import defaultdict
        left = 0
        counts = defaultdict(int)
        total_match = 0
        for c in t:
            counts[c] += 1  # Algorithm: using dict and total_match to track 
                            #            the letters matched
                            #            (all counts <= zero means already matched)
        res = ''
        for right in range(len(s)):  # Algorithm: sliding window using two pointers
            counts[s[right]] -= 1
            if counts[s[right]] >= 0:
                total_match += 1

            if total_match == len(t):
                while counts[s[left]] < 0:
                    counts[s[left]] += 1
                    left += 1
                if res == '' or right - left + 1 < len(res):
                    res = s[left:right + 1]

                counts[s[left]] += 1
                total_match -= 1
                left += 1
        return res
