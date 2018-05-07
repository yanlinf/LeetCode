# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists):  # Algorithm: merge k sorted lists using PriorityQueue
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        import heapq

        heap = [(lists[i].val, i) for i in range(len(lists)) if lists[i]]
        heapq.heapify(heap)

        p = head = ListNode(0)  # Feature: a = b = c
        while heap:
            val, i = heapq.heappop(heap)
            p.next = ListNode(val)
            p = p.next
            if lists[i] and lists[i].next:
                lists[i] = lists[i].next
                heapq.heappush(heap, (lists[i].val, i))
        return head.next

