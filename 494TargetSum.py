class Solution:
    def findTargetSumWays(self, nums, S):   # Algorithm: DP
                                            # Time: O(n*l), Space: O(l)
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        maxsum = sum(nums)
        S += maxsum
        if S < 0 or S > 2 * maxsum:
            return 0
        dp = [0] * (2 * maxsum + 1)
        dp[nums[0] + maxsum] += 1
        dp[-nums[0] + maxsum] += 1
        for i in range(1, len(nums)):
            next = [0] * (2 * maxsum + 1)
            for j in range(2 * maxsum + 1):
                if j - nums[i] >= 0:
                    next[j] += dp[j - nums[i]]
                if j + nums[i] <= 2 * maxsum:
                    next[j] += dp[j + nums[i]]
            dp = next
        return dp[S]

class SolutionB:
    def findTargetSumWays(self, nums, S):  # Time Complexity: O(2^n) (Bad!)
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        from collections import deque
        
        nums.sort(reverse=True)
        maxsum = [0] * (len(nums) + 1)
        sum = 0
        for i in range(len(nums) - 1, -1, -1):
            sum += nums[i]
            maxsum[i] = sum
        
        queue = deque([(0, 0)])
        res = 0
        while queue:
            n, k = queue.popleft()
            if k == len(nums):
                if n == S:
                    res += 1
            else:
                if n - maxsum[k] <= S and n + maxsum[k] >= S:
                    queue.append((n + nums[k], k + 1))
                    queue.append((n - nums[k], k + 1))
        return res
