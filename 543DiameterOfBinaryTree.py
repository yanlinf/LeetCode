# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def solve(node):
            if node:
                ldepth = solve(node.left)
                rdepth = solve(node.right)
                self.ans = max(self.ans, ldepth + rdepth)
                return max(ldepth, rdepth) + 1
            else:
                return 0

        self.ans = 0
        solve(root)
        return self.ans
