class Solution:
    def longestValidParentheses(self, s):  # Algorithm: DP, O(n) time
                                           # * there's another solution using stack
        """
        :type s: str
        :rtype: int
        """
        dp, res = [0] * len(s), 0
        for i, char in enumerate(s):
            p = i - dp[i - 1] - 1
            if char == ')' and p >= 0 and s[p] == '(':
                dp[i] = dp[i - 1] + 2 + (dp[p - 1] if p > 0 else 0)
            else:
                dp[i] = 0
            res = max(res, dp[i])
        return res
