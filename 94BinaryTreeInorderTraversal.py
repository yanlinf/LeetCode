# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stk = []
        ptr = root
        while stk or ptr:
            if ptr:
                ptr = stk.pop()
                res.append(ptr.val)
                ptr = ptr.right
            else:
                stk.append(ptr)
                ptr = ptr.left
        return res

class SolutionB:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        def visit(node):
            if node:
                visit(node.left)
                res.append(node.val)
                visit(node.right)

        res = []
        visit(root)
        return res

