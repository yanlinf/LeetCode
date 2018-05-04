class Solution:
    def canFinish(self, numCourses, prerequisites):  # Algorithm: BFS TopSort
                                                     # Complexity: O(|V| + |E|)
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for t, s in prerequisites:
            graph[s].append(t)
            indegree[t] += 1
        queue = [t for t in range(numCourses) if indegree[t] == 0]
        visited = 0
        while queue:
            visited += 1
            for t in graph[queue[0]]:
                indegree[t] -= 1
                if indegree[t] == 0:
                    queue.append(t)
            queue.pop(0)
        return visited == numCourses

