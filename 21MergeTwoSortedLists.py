# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        res = []
        p1, p2 = l1, l2
        while p1 and p2:
            if p1.val < p2.val:
                res.append(p1.val)
                p1 = p1.next
            else:
                res.append(p2.val)
                p2 = p2.next
        while p1:
            res.append(p1.val)
            p1 = p1.next
        while p2:
            res.append(p2.val)
            p2 = p2.next

        return res
