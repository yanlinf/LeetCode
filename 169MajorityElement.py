class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        from collections import Counter
        counts = Counter(nums)  # Feature: Counter counts elements in O(n)! (Using Hashmap)
        return counts.most_common(1)[0][0]


class SolutionB:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        def find(k, left, right):  # Quick Sort: complexity could be O(n^2)
            if left == right:
                return nums[left]
            l, r = left, right
            temp = nums[left]
            flag = True
            while left < right:
                if flag:
                    if nums[right] < temp:
                        nums[left] = nums[right]
                        left += 1
                        flag = False
                    else:
                        right -= 1
                else:
                    if nums[left] > temp:
                        nums[right] = nums[left]
                        right -= 1
                        flag = True
                    else:
                        left += 1
            nums[left] = temp
            if left - l == k:
                return temp
            elif left - l < k:
                return find(k - (left - l) - 1, left + 1, r)
            else:
                return find(k, l, left - 1)

        return find(len(nums) // 2, 0, len(nums) - 1)

