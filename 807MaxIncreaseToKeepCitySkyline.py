class Solution:
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        left_skyline = list(map(max, grid))
        up_skyline = list(map(max, *grid)) # Python Feature: map(max, *grid)
                                           # equivilant to: up_skyline = [max(col) for col in zip(*grid)]
                                           # can be used to transpose a matrix
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                result += min(up_skyline[j], left_skyline[i]) - grid[i][j]
        return result

if __name__ == '__main__':
    grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
    print(Solution().maxIncreaseKeepingSkyline(grid))