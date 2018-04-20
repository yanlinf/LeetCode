class Solution:
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda x: (-x[0], x[1]))
        queue = []
        for person in people:
            queue.insert(person[1], person)
        return queue
        
class SolutionB:
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        n = len(people)
        people.sort(key=lambda x: (x[0], -x[1]))
        queue = [None] * n
        for person in people:
            pos = [i for i in range(n) if queue[i] is None][person[1]]
            queue[pos] = person
        return queue
        
        # Complexity: O(n^2)