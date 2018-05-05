# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head, zero = ListNode(0), ListNode(0)
        ptr, carryout = head, 0
        while l1 or l2:
            l1 = zero if not l1 else l1
            l2 = zero if not l2 else l2
            carryout, num = divmod(l1.val + l2.val + carryout, 10)
            ptr.next = ListNode(num)
            ptr, l1, l2 = ptr.next, l1.next, l2.next
        if carryout:
            ptr.next = ListNode(carryout)
            ptr = ptr.next
        ptr.next = None
        return head.next
