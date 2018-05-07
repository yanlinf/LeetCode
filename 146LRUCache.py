from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = OrderedDict()  # Feature: OrderedDict to maintain insertion order
        self.slots = capacity
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.cache:
            return -1
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.cache:
            self.cache.pop(key)
            self.slots += 1
        if self.slots == 0:
            self.cache.popitem(last=False)
        else:
            self.slots -=1
        self.cache[key] = value

        


class Node:  # Note: can be replaced by a list, which is simpler and faster
    def __init__(self, key, val, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev, self.next = prev, next

    def pop(self):
        self.next.prev = self.prev
        self.prev.next = self.next

    def insert(self, node):
        node.prev.next = self
        self.prev = node.prev
        self.next = node
        node.prev = self


class LRUCache:  # Algorithm(data structure): doubly linked list
                 # (which is exactly the way OrderedDict was implemented)

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.dic = {}
        self.empty = capacity
        self.head = Node(0, 0)
        self.tail = Node(0, 0, self.head, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.dic:
            return -1
        self.dic[key].pop()
        self.dic[key].insert(self.tail)
        return self.dic[key].val

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.dic:
            self.dic[key].val = value
            self.dic[key].pop()
            self.dic[key].insert(self.tail)
            return
        self.dic[key] = Node(key, value)
        if self.empty > 0:
            self.empty -= 1
        else:
            self.dic.pop(self.head.next.key)
            self.head.next.pop()
        self.dic[key].insert(self.tail)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
