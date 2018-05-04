# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution(object):
    def getIntersectionNode(self, headA, headB):  # Algorithm: two pointers
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        pa, pb = headA, headB
        while pa is not pb:
            pa = pa.next if pa else headB
            pb = pb.next if pb else headA
        return pa


class SolutionB(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        def linkedListLen(head):
            res = 0
            while head:
                res += 1
                head = head.next
            return res

        lenA, lenB = linkedListLen(headA), linkedListLen(headB)
        pa, pb = (headA, headB) if lenA < lenB else (headB, headA)
        for i in range(abs(lenA - lenB)):
            pb = pb.next
        while pa and pa != pb:
            pa, pb = pa.next, pb.next
        return pa if pa else None
