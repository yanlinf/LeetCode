class Solution:
    def getSkyline(self, buildings):  # Algorithm(data structure): PriorityQueue
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        import heapq

        points = set([b[0] for b in buildings] + [b[1] for b in buildings])
        p, active, res = 0, [], []

        for x in sorted((points)):
            while p < len(buildings) and buildings[p][0] <= x:
                heapq.heappush(active, (-buildings[p][2], buildings[p][1]))
                p += 1

            while active and active[0][1] <= x:
                heapq.heappop(active)

            curr_height = -active[0][0] if active else 0
            if not res or curr_height != res[-1][1]:
                res.append([x, curr_height])

        return res


class SolutionB:
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        import heapq

        if not buildings:
            return []

        heap = [(0, -float('inf'))]
        rightmost = max(list(zip(*buildings))[1])
        curr_height, res = 0, []

        buildings.append([rightmost, rightmost, 0])

        for li, ri, hi in buildings:
            while -heap[0][1] <= li:
                h, r = heapq.heappop(heap)
                h, r = -h, -r
                while -heap[0][1] <= r:
                    heapq.heappop(heap)
                if r == li and curr_height == hi:
                    continue
                curr_height = -heap[0][0] if r < li else hi
                res.append([r, curr_height])

            if hi > curr_height:
                if res and li == res[-1][0]:
                    res[-1][1] = hi
                else:
                    res.append([li, hi])
                curr_height = hi
            heapq.heappush(heap, (-hi, -ri))

        return res
