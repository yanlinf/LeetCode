class Solution:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """

        # Algorithm: mapping binary trees to strings using pre-order traversal
        # -> mapping recursive structures to CFG strings for comparison
        def preorder(t):
            if not t:
                return '^$'
            else:
                return '^' + str(t.val) + preorder(t.left) + preorder(t.right) + '$'

        return preorder(t) in preorder(s)


class SolutionB:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """

        def isSame(s, t):
            if not s and not t:
                return True
            elif s and t:
                return s.val == t.val and isSame(s.left, t.left) and isSame(s.right, t.right)
            else:
                return False

        if not s and not t:
            return True
        elif s and t:
            return self.isSubtree(s.left, t) or self.isSubtree(s.right, t) or isSame(s, t)
        else:
            return False
