class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        left, right = 0, 0
        res = ''
        while left < len(s):  # Algorithm: traversing all palindromic substrings
            while right < len(s) and s[right] == s[left]:
                right += 1
            lp, rp = left - 1, right
            while lp >= 0 and rp < len(s) and s[lp] == s[rp]:
                lp -= 1
                rp += 1
            if rp - lp - 1 > len(res):
                res = s[lp + 1:rp]
            left = right
        return res
