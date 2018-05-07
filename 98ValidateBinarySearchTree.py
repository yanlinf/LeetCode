# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def valid(root, lower_bound, upper_bound):
            if not root:
                return True
            if root.val <= lower_bound or root.val >= upper_bound:
                return False
            return valid(root.left, lower_bound, root.val) and\
                valid(root.right, root.val, upper_bound)
        return valid(root, -float('inf'), float('inf'))


class SolutionB:
    ans = True
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def dfs(root):
            if not root:
                return (-float('inf'), float('inf'))
            left = dfs(root.left)
            right = dfs(root.right)
            if root.val <= left[0] or root.val >= right[1]:
                self.ans = False
            return (max(root.val, right[0]), min(root.val, left[1]))

        dfs(root)
        return self.ans
