# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or root is p or root is q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right


class SolutionB(object):
    ans = []
    paths = []
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def dfs(root):
            if not root:
                return
            self.ans.append(root)
            if root is p or root is q:
                self.paths.append(self.ans[:])
            dfs(root.left)
            dfs(root.right)
            self.ans.pop()

        if p is q:
            return p
        self.paths = []
        dfs(root)
        minlen = min(len(self.paths[0]), len(self.paths[1]))
        for i in range(minlen - 1, -1, -1):
            if self.paths[0][i] is self.paths[1][i]:
                return self.paths[0][i]
