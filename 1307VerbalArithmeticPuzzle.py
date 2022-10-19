from typing import List


class Solution:
    def isSolvable(self, words: List[str], result: str) -> bool:
        if len(result) < max(len(w) for w in words):
            return False

        def dfs(pos, wid, carry):
            if pos > len(result):
                if carry == 0 and (mapping[result[0]] > 0 or len(result) == 1):
                    print(mapping)
                    return True
                else:
                    return False
            if wid == len(words):
                r = sum([mapping[w[-pos]] if pos <= len(w) else 0 for w in words])
                r += carry
                if result[-pos] in mapping:
                    if mapping[result[-pos]] == r % 10:
                        return dfs(pos + 1, 0, r // 10)
                    else:
                        return False
                elif (r % 10) in used:
                    return False
                else:
                    mapping[result[-pos]] = r % 10
                    used.add(r % 10)
                    if dfs(pos + 1, 0, r // 10):
                        return True
                    mapping.pop(result[-pos])
                    used.remove(r % 10)
                    return False
            w = words[wid]
            if len(w) > 1 and pos == len(w) and mapping.get(w[-pos]) == 0:
                return False
            if pos > len(w) or w[-pos] in mapping:
                return dfs(pos, wid + 1, carry)
            else:
                for n in range(10):
                    if n not in used and (pos < len(w) or n > 0 or len(w) == 1):
                        mapping[w[-pos]] = n
                        used.add(n)
                        if dfs(pos, wid + 1, carry):
                            return True
                        used.remove(n)
                        mapping.pop(w[-pos])
                return False

        mapping = {}
        used = set()
        return dfs(1, 0, 0)


if __name__ == '__main__':
    words = ["SEND", "MORE"]
    result = "MONEY"
    print(Solution().isSolvable(words, result))
    words = ["SIX", "SEVEN", "SEVEN"]
    result = "TWENTY"
    print(Solution().isSolvable(words, result))
