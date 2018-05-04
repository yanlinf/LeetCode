class Solution:
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if len(s) < len(p):
            return []
        window, pcount, res = [0] * 26, [0] * 26, []
        for c in p:
            pcount[ord(c) - 97] += 1
        for c in s[:len(p)]:
            window[ord(c) - 97] += 1
        for i in range(len(p), len(s)):
            if window == pcount:
                res.append(i - len(p))
            window[ord(s[i]) - 97] += 1
            window[ord(s[i - len(p)]) - 97] -= 1
        if window == pcount:
            res.append(len(s) - len(p))
        return res

