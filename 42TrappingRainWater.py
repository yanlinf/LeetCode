class Solution:
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0

        n = len(height)
        outline, outline2 = [height[0]] * n, [height[n - 1]] * n
        for i in range(1, n):
            outline[i] = max(height[i], outline[i - 1])
        for i in range(n - 2, -1, -1):
            outline2[i] = max(height[i], outline2[i + 1])
        outline = [min(outline[i], outline2[i]) for i in range(n)]

        res = 0
        for i in range(n):
            res += outline[i] - height[i]
        return res
