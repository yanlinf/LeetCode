class Solution:  # BFS version
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        from collections import deque
        queue = deque([([], 0)])
        while queue[0][1] < len(nums):
            ans, n = queue.popleft()
            queue.append((ans, n + 1))
            queue.append((ans + [nums[n]], n + 1))
        return [x[0] for x in queue]


class SolutionB:  # DFS version
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def gen(ans, n):
            if n == len(nums):
                res.append(ans)
            else:
                gen(ans.copy(), n + 1)
                gen(ans + [nums[n]], n + 1)

        res = []
        gen([], 0)
        return list(res)
