# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    ans = 0
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def traverse(node):  # In-order traversal
            if node:
                traverse(node.right)
                node.val += self.ans
                self.ans = node.val
                traverse(node.left)
        traverse(root)
        return root

class SolutionB:
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def convertNode(node, sum):  # Divide-and-Conquer
            if node:
                ans = sum + convertNode(node.right, sum) + node.val
                node.val = ans
                return ans + convertNode(node.left, ans) - sum
            else:
                return 0
        convertNode(root, 0)
        return root