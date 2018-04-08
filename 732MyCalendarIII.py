from collections import Counter

class MyCalendarThree:

    def __init__(self):
        self.counter = Counter()

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: int
        """
        self.counter[start] += 1
        self.counter[end] -= 1
        active, ans = 0, 0
        for i in sorted(self.counter):
            active += self.counter[i]
            ans = max(ans, active)
        return ans


# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)