class Solution:
    ans = float('inf')
    def coinChange(self, coins, amount):  # Algorithm: DFS with pruning
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        def dfs(index, target, used):
            if index == len(coins) or used + target // coins[index] >= self.ans:
                if target == 0 and used < self.ans:
                    self.ans = used
                return
            for k in range(target // coins[index], -1, -1):
                dfs(index + 1, target - coins[index] * k, used + k)

        coins.sort(reverse=True)
        dfs(0, amount, 0)
        return self.ans if self.ans != float('inf') else -1
