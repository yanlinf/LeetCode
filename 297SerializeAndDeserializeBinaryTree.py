# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return '^$'
        else:
            return '^' + str(root.val) + self.serialize(root.left) + \
                self.serialize(root.right) + '$'

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if data == '^$':
            return None
        else:
            p1 = data[1:].index('^') + 1
            cnt, p2 = 1, p1 + 1
            while cnt > 0:
                if data[p2] == '^':
                    cnt += 1
                elif data[p2] == '$':
                    cnt -= 1
                p2 += 1
            root = TreeNode(int(data[1:p1]))
            root.left = self.deserialize(data[p1:p2])
            root.right = self.deserialize(data[p2:-1])
            return root

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))
