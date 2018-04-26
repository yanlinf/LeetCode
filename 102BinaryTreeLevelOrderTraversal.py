# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root):  # Algorigthm: Level-BST
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue, res = [root], []
        while queue:
            tmp = []
            for i in range(len(queue)):
                elem = queue.pop(0)
                tmp.append(elem.val)
                if elem.left:
                    queue.append(elem.left)
                if elem.right:
                    queue.append(elem.right)
            res.append(tmp)
        return res


class SolutionB:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        from collections import deque
        if not root:
            return []
        queue, res = deque([(1, root)]), []
        depth = 0
        while queue:
            d, node = queue.popleft()
            if d == depth:
                res[-1].append(node.val)
            else:
                res.append([node.val])
                depth += 1
            if node.left:
                queue.append((d + 1, node.left))
            if node.right:
                queue.append((d + 1, node.right))
        return res
