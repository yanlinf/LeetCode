class Solution:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import Counter  # Algorithm: Math, O(1) Time (Without counting the time of reading inputs)
        counts = list(Counter(tasks).values())
        nmax = max(counts)
        ans = counts.count(nmax)
        return max((nmax - 1) * (n + 1) + ans, len(tasks))


class SolutionB:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import Counter  # Algorithm: Greedy
        counts = Counter(tasks)
        ntask, res = len(counts), 0
        while ntask:
            if n + 1 >= ntask:
                iter = counts.most_common()[-1][1]
                for task in counts:
                    counts[task] -= iter
            else:
                elems = counts.most_common(n + 2)
                iter = elems[n][1] - elems[n + 1][1] + 1
                for task, _ in elems[0: -1]:
                    counts[task] -= iter
            for x in list(counts):
                if counts[x] == 0:
                    counts.pop(x)
            res += (n + 1) * iter if len(counts) > 0 else (n + 1) * (iter - 1) + ntask
            ntask = len(counts)
        return res
