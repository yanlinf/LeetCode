class Solution:
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def revise(k, expr, pars):  # Note: expr is valid 
                                    # <=> (lcount == rcount) and (lcount >= rcount during forward pass)
                                    # <=> (lcount >= rcount during forward pass) and (rcount >= lcount during backward pass)
            cnt = 0
            while k < len(expr):
                cnt += int(expr[k] == pars[0])
                cnt -= int(expr[k] == pars[1])
                if cnt < 0:
                    break
                k += 1
            else:
                if pars[0] == '(':
                    revise(0, expr[::-1], (')', '('))
                else:
                    res.append(expr[::-1])
                return

            for i in range(0, k + 1):
                if expr[i] == pars[1] and (i == 0 or expr[i - 1] != pars[1]):
                    revise(k, expr[:i] + expr[i + 1:], pars)

        res = []
        revise(0, s, ('(', ')'))
        return list(set(res))
