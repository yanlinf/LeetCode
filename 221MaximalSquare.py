class Solution:

    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix) == 0:
            return 0
        ans = 0
        matrix = [[int(x) for x in row] for row in matrix]
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if row > 0 and col > 0 and matrix[row][col] == 1:
                    matrix[row][col] = 1 + min(matrix[row - 1][col - 1],
                                               matrix[row][col - 1], matrix[row - 1][col])
                ans = max(ans, matrix[row][col])
        return ans * ans
