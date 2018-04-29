class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        def gen(k, curr_string):
            if k == len(digits):
                res.append(curr_string)
                return
            for c in dic[digits[k]]:
                gen(k + 1, curr_string + c)

        if not digits:
            return []
        res = []
        dic = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
            '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz',
        }
        gen(0, '')
        return res
