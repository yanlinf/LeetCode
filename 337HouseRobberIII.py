# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def solve(node):
            if node:
                lnmax, lmax = solve(node.left)
                rnmax, rmax = solve(node.right)
                return lmax + rmax, max(lmax + rmax, node.val + lnmax + rnmax)
            else:
                return 0, 0

        return solve(root)[1]
