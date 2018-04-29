# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        def myflaten(root):
            if not root:
                return
            tmp = root.right
            root.right = myflaten(root.left)
            root.left = None
            s = root
            while s.right:
                s = s.right
            s.right = myflaten(tmp)
            return root

        myflaten(root)
