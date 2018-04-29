class Solution:
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        from collections import defaultdict
        dic = defaultdict(list)
        for s in strs:
            dic[str(sorted(s))].append(s)
        return list(dic.values())
