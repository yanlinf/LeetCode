class Solution:
    def maxSlidingWindow(self, nums, k):  # Complexity: O(n)
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums:  # Algorithm: maintaining a deque
            return []
        window, res = [], []
        for i in range(len(nums)):
            if i >= k:
                res.append(window[0])
                if window[0] == nums[i - k]:
                    window.pop(0)
            for j in range(len(window) - 1, -1, -1):
                if window[j] >= nums[i]:
                    window = window[:j + 1] + [nums[i]]
                    break
            else:
                window = [nums[i]]
        return res + [window[0]]
