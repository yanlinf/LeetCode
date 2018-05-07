class Solution:
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = {(0, 0, 0)} if nums.count(0) >= 3 else set()
        nums.sort()
        nums = [nums[i] for i in range(len(nums)) if i < 2 or nums[i] != nums[i - 1]
                or nums[i] != nums[i - 2]]
        for i in range(len(nums)):
            seen = set()
            for j in range(i + 1, len(nums)):
                if -nums[i] - nums[j] in seen:
                    res.add((nums[i], -nums[i] - nums[j], nums[j]))
                seen.add(nums[j])
        return list(res)
