class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [True] + [False] * len(s)
        for i in range(1, len(s) + 1):
            for k in range(0, i):
                if dp[k] and s[k:i] in wordDict:
                    dp[i] = True
        return dp[len(s)]
