class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.array = []
        self.mins = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.array.append(x)
        if not mins or x <= mins[-1]:
            self.mins.append(x)

    def pop(self):
        """
        :rtype: void
        """
        x = self.array.pop()
        if x == self.mins[-1]:
            self.mins.pop()

    def top(self):
        """
        :rtype: int
        """

    def getMin(self):
        """
        :rtype: int
        """


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
