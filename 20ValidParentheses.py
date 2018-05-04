class Solution:
    def isValid(self, s):  # Algorithm(data structure): stack
                           # Note: expr is valid <=> valid stack operations
        """
        :type s: str
        :rtype: bool
        """
        dic = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for c in s:
            if c in dic:
                stack.append(c)
            else:
                if not stack or dic[stack.pop()] != c:
                    return False
        return not stack
