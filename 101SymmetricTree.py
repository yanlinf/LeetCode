# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def isSym(a, b):
            if not a and not b:
                return True
            elif a or b:
                return False
            else:
                return a.val == b.val and isSym(a.left, b.right) and isSym(a.right, b.left)

        return isSym(root.left, root.right) if root else True


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class SolutionB:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        queue = [root]
        while any(queue):
            vals = [x.val if x else None for x in queue]
            if vals != vals[::-1]:
                return False
            tmp = []
            for node in queue:
                tmp += [node.left, node.right] if node else [None, None]
            queue = tmp
        return True


if __name__ == '__main__':
    root = TreeNode(1)
    root.left, root.right = TreeNode(2), TreeNode(2)
    root.left.left, root.left.right, root.right.left, root.left.right = TreeNode(3), TreeNode(4), TreeNode(4), TreeNode(
        3)
    print(Solution().isSymmetric(root))
