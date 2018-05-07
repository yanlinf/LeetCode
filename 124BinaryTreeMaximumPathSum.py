# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    ans = -float('inf')
    def maxPathSum(self, root):  # Algorithm: Divide-and-Conquer / DFS
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(root):
            if not root:
                return 0
            left, right = dfs(root.left), dfs(root.right)
            self.ans = max(self.ans, root.val, left + root.val,
                           right + root.val, left + right + root.val)
            return max(root.val, root.val + left, root.val + right)

        dfs(root)
        return self.ans
