class Solution:
    def rotate(self, matrix):  # Space Complexity: O(1)
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        h = n // 2
        w = n - h
        for i in range(h):
            for j in range(w):
                tmp = matrix[i][j]
                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = tmp


class SolutionB:
    def rotate(self, matrix):  # Space Complexity: O(n^2)
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        rotated = list(zip(*matrix))
        for i in range(len(rotated)):
            matrix[i] = rotated[i][::-1]
