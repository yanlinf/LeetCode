# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def sortList(self, head):  # Algorithm: Merge Sort
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        prev = slow
        slow = slow.next
        prev.next = None
        h1, h2 = self.sortList(head), self.sortList(slow)
        head = ListNode(0)
        tail = head
        while h1 and h2:
            if h1.val < h2.val:
                tail.next = h1
                h1, tail = h1.next, tail.next
            else:
                tail.next = h2
                h2, tail = h2.next, tail.next
        if h1:
            while h1:
                tail.next = h1
                h1, tail = h1.next, tail.next
        elif h2:
            while h2:
                tail.next = h2
                h2, tail = h2.next, tail.next
        tail.next = None
        return head.next
