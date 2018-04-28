# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    ans = 0
    cache = {0: 1}

    def pathSum(self, root, sum): # Algorithm: PATH(a, b) = PATH(root, b) - PATH(root, a) -> DFS
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """

        def dfs(t, sofar):  # Algorithm: DFS
            if not t:
                return
            sofar += t.val
            self.ans += self.cache.get(sofar - sum, 0)
            self.cache[sofar] = self.cache.get(sofar, 0) + 1
            dfs(t.left, sofar)
            dfs(t.right, sofar)

            self.cache[sofar] -= 1






        dfs(root, 0)
        return self.ans
