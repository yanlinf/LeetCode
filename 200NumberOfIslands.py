class Solution:
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        def dfs(row, col):
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
                return
            if grid[row][col] == '1':
                grid[row][col] = '0'
                for i in range(4):
                    dfs(row + dx[i], col + dy[i])

        dx, dy = [-1, 1, 0, 0], [0, 0, 1, -1]
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1
        return res
