class Solution:
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        target, mod = divmod(sum(nums), 2)
        if mod:
            return False
        dp = [False] * (target + 1)
        dp[0] = True
        for n in nums:
            for i in range(target, n - 1, -1):
                dp[i] = dp[i] or dp[i - n]
        return dp[target]


class Solution:
    def canPartition(self, nums):  # Algorithm: DFS with pruning
        """
        :type nums: List[int]
        :rtype: bool
        """
        nums.sort(reverse=True)  # Sorting before DFS, important!
        target, mod = divmod(sum(nums), 2)  # Feature: divmod
        if mod:
            return False
        ans = 0
        rsum = [0] * len(nums)
        for i in range(len(nums) - 1, -1, -1):
            ans += nums[i]
            rsum[i] = ans

        def dfs(k, sofar):
            if k == len(nums) or sofar >= target:  # Pruning 1
                return sofar == target
            elif sofar + rsum[k] <= target:  # Prunning 2
                return sofar + rsum[k] == target
            return dfs(k + 1, sofar) or dfs(k + 1, sofar + nums[k])
        return dfs(0, 0)
