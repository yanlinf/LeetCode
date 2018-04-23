class Solution:  # Space complexity: O(1)
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = fast = nums[0]   # Algorithm: Two Pointers
                                # used to detect cyclic linked list
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        ptr1, ptr2 = nums[0], fast
        while ptr1 != ptr2:
            ptr1 = nums[ptr1]
            ptr2 = nums[ptr2]
        return ptr1
        

class SolutionB:  # Space complexity: O(n)
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        from collections import Counter
        return Counter(nums).most_common(1)[0][0]