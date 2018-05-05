class Solution:
    def maximalRectangle(self, matrix):  # Complexity: O(n*m)
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        def largestRectangleArea(heights):
            stack, res = [(0, -1, 0)], 0
            heights.append(0)
            for i in range(len(heights)):
                while stack and stack[-1][0] > heights[i]:
                    h, pos, l = stack.pop()
                    res = max(res, h * (i - pos + l))
                stack.append((heights[i], i, i - stack[-1][1] - 1))
            return res

        if len(matrix) == 0:
            return 0
        h, w, res = len(matrix), len(matrix[0]), 0
        heights = [0] * w
        for i in range(h):
            heights = [heights[j] + 1 if matrix[i][j] == '1' else 0 for j in range(w)]
            res = max(res, largestRectangleArea(heights))
        return res


class SolutionB:
    def maximalRectangle(self, matrix):  # Complexity: O(n*m*m)
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix) == 0:
            return 0
        h, w, res = len(matrix), len(matrix[0]), 0
        dp = [[0] * w for _ in range(w)]
        for l in range(h):
            for i in range(w):
                dp[i][i] = dp[i][i] + 1 if matrix[l][i] == '1' else 0
                res = max(res, dp[i][i])
                for j in range(i + 1, w):
                    dp[i][j] = dp[i][j] + \
                        1 if (matrix[l][j] == '1' and dp[i][j - 1]) else 0
                    res = max(res, dp[i][j] * (j - i + 1))
        return res
