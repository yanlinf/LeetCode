# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e


class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        res = []
        for interval in sorted(intervals, key=lambda x: x.start):
            if res and interval.start <= res[-1].end:
                res[-1].end = max(res[-1].end, interval.end)
            else:
                res.append(interval)
        return res


class SolutionB:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return []

        from collections import Counter
        start_counts = Counter([x.start for x in intervals])
        end_counts = Counter([x.end for x in intervals])
        timeline = sorted({x.start for x in intervals} | {x.end for x in intervals})
        res = []
        curr_start = timeline[0]
        curr_counts = 0
        for t in timeline:
            if curr_counts == 0:
                curr_start = t
            curr_counts = curr_counts + start_counts[t] - end_counts[t]
            if curr_counts == 0:
                res.append(Interval(curr_start, t))
        return res