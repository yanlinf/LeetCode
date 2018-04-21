from collections import deque


class Solution:
    def generateParenthesis(self, n):
        # Note: S is valid when and only when '('.count >= ')'.count
        #       for any prefix of S
        """
        :type n: int
        :rtype: List[str]
        """
        queue = deque([' ' * (2 * n)])
        while queue[0].count('(') < n:
            curr = queue.popleft()
            lp = curr.find(' ')
            curr = curr[:lp] + '(' + curr[lp + 1:]
            rp = lp
            while rp < 2 * n - 1 and curr[rp + 1] == ' ':
                rp += 1
            for p in range(rp, lp, -2):
                queue.append(curr[:p] + ')' + curr[p + 1:])
        return list(queue)
