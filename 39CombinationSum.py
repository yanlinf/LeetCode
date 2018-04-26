class Solution:
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        def dfs(k, s, curr):
            if k == len(candidates):
                if s == target:
                    res.append(curr.copy())
                return
            tmp = curr.copy()
            for i in range((target - s) // candidates[k] + 1):
                dfs(k + 1, s, tmp)
                s += candidates[k]
                tmp.append(candidates[k])

        res = []
        dfs(0, 0, [])
        return res
