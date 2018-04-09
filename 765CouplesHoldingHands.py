class Solution:
    def minSwapsCouples(self, row):
        """
        :type row: List[int]
        :rtype: int
        """
        pos = {}
        for i in range(len(row)):
            pos[row[i]] = i
        swap_count = 0
        for i in range(0, len(row), 2):
            if row[i + 1] == (row[i] - 1 if row[i] % 2 else row[i] + 1):
                continue
            swap_count += 1
            p = row[i + 1]
            q = row[i] - 1 if row[i] % 2 else row[i] + 1
            row[i + 1], row[pos[q]] = q, p
            pos[p], pos[q] = pos[q], pos[p]
        return swap_count