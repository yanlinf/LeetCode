class Solution:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, right = 0, 0
        res = 0
        while left < len(s):
            while right < len(s) and s[right] == s[left]:
                right += 1
            res += int((right - left) * (right - left + 1) / 2)
            lp, rp = left - 1, right
            while lp >= 0 and rp < len(s) and s[lp] == s[rp]:
                res += 1
                lp -= 1
                rp += 1
            left = rp
        return res


# Complexity: O(n^2)
class SolutionB:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = [[False] * (len(s) + 1) for i in range(len(s))]
        res = len(s)
        for i in range(len(s)):
            dp[i][i] = True
            dp[i][i + 1] = True
        for length in range(2, len(s) + 1):
            for offset in range(0, len(s) - length + 1):
                dp[offset][offset + length] = s[offset] == s[offset + length - 1] \
                                              and dp[offset + 1][offset + length - 1]
                res += int(dp[offset][offset + length])
        return res
