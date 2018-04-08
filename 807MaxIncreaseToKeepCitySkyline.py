class Solution:
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        up_skyline = list(map(max, grid))
        left_skyline = list(map(max, *grid)) # Note: map(max, *grid)
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                result += min(up_skyline[j], left_skyline[i]) - grid[i][j]
        return result

if __name__ == '__main__':
    grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
    print(Solution().maxIncreaseKeepingSkyline(grid))