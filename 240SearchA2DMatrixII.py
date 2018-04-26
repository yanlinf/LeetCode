class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        h, w = len(matrix), len(matrix[0])
        row, col = h - 1, 0
        while row >= 0 and col < w:
            if matrix[row][col] == target:  # Algorithm
                return True
            elif matrix[row][col] < target:
                col += 1
            else:
                row -= 1
        return False
