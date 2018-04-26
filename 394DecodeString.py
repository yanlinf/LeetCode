class Solution:
    def decodeString(self, s):  # Algorithm: using stack
        """
        :type s: str
        :rtype: str
        """
        stack, curr_num, curr_str = [], 0, ''
        for c in s:
            if c == '[':
                stack.append(curr_str)
                stack.append(curr_num)
                curr_num, curr_str = 0, ''
            elif c == ']':
                num = stack.pop()
                curr_str = stack.pop() + curr_str * num
            elif c.isdigit():
                curr_num = curr_num * 10 + int(c)
            else:
                curr_str += c
        return curr_str


class SolutionB:
    def decodeString(self, s):  # Recursive version
        """
        :type s: str
        :rtype: str
        """
        l = s.find('[')
        if l == -1:
            return s
        p = l - 1
        while p > 0 and s[p - 1].isdigit():
            p -= 1
        r = l
        lcnt, rcnt = 1, 0
        while lcnt > rcnt:
            r += 1
            if s[r] == '[':
                lcnt += 1
            elif s[r] == ']':
                rcnt += 1
        return s[0: p] + self.decodeString(s[l + 1: r]) * int(s[p: l]) + self.decodeString(s[r + 1:])
