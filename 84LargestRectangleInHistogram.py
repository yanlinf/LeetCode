class Solution:
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack, res = [(0, -1, 0)], 0  # Algorithm: Ordered Stack(Deque)
        heights.append(0)
        for i in range(len(heights)):
            while stack and stack[-1][0] > heights[i]:
                h, pos, l = stack.pop()
                res = max(res, h * (i - pos + l))
            stack.append((heights[i], i, i - stack[-1][1] - 1))
        return res
