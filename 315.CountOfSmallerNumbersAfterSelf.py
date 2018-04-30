class BinaryIndexedTree:
    @staticmethod
    def lowbit(x):
        return x & (-x)

    def __init__(self, n):
        self._size = n
        self._array = [0] * (n + 1)

    def __str__(self):
        return str(self._array[1:])

    def __repr__(self):
        return str(self._array[1:])

    def __len__(self):
        return self._size

    def update(self, index, delta):
        while index <= self._size:
            self._array[index] += delta
            index += BinaryIndexedTree.lowbit(index)

    def getsum(self, index):  # index begins from 1
        res = 0
        while index >= 1:
            res += self._array[index]
            index -= BinaryIndexedTree.lowbit(index)
        return res


class Solution(object):
    def countSmaller(self, nums):  # Algorithm(data structure): Binary Indexed Tree
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def update(index):
            while index <= size:
                BITree[index] += 1
                index += index & -index

        def getsum(index):
            res = 0
            while index >= 1:
                res += BITree[index]
                index -= index & -index
            return res

        size = len(nums)
        numpos = sorted((zip(nums, range(size))))
        BITree = [0] * (size + 1)
        res = [0] * size
        for n, pos in numpos:
            res[pos] = getsum(size - pos)
            update(size - pos)
        return res
