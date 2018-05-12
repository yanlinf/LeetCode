# 101SymmetricTree.py

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def isSym(a, b):
            if not a and not b:
                return True
            elif a or b:
                return False
            else:
                return a.val == b.val and isSym(a.left, b.right) and isSym(a.right, b.left)

        return isSym(root.left, root.right) if root else True


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class SolutionB:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        queue = [root]
        while any(queue):
            vals = [x.val if x else None for x in queue]
            if vals != vals[::-1]:
                return False
            tmp = []
            for node in queue:
                tmp += [node.left, node.right] if node else [None, None]
            queue = tmp
        return True


if __name__ == '__main__':
    root = TreeNode(1)
    root.left, root.right = TreeNode(2), TreeNode(2)
    root.left.left, root.left.right, root.right.left, root.left.right = TreeNode(3), TreeNode(4), TreeNode(4), TreeNode(
        3)
    print(Solution().isSymmetric(root))




# 102BinaryTreeLevelOrderTraversal.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def levelOrder(self, root):  # Algorigthm: Level-BST
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        queue, res = [root], []
        while queue:
            tmp = []
            for i in range(len(queue)):
                elem = queue.pop(0)
                tmp.append(elem.val)
                if elem.left:
                    queue.append(elem.left)
                if elem.right:
                    queue.append(elem.right)
            res.append(tmp)
        return res


class SolutionB:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        from collections import deque
        if not root:
            return []
        queue, res = deque([(1, root)]), []
        depth = 0
        while queue:
            d, node = queue.popleft()
            if d == depth:
                res[-1].append(node.val)
            else:
                res.append([node.val])
                depth += 1
            if node.left:
                queue.append((d + 1, node.left))
            if node.right:
                queue.append((d + 1, node.right))
        return res




# 104MaximumDepthOfBinaryTree.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))




# 105.ConstructBinaryTreeFromPreorderAndInorderTraversal.py

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder:
            return None
        k = inorder.index(preorder[0])
        root = TreeNode(preorder[0])
        root.left = self.buildTree(preorder[1:k + 1], inorder[:k])
        root.right = self.buildTree(preorder[k + 1:], inorder[k + 1:])
        return root




# 114FlattenBinaryTreeToLinkedList.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        def myflaten(root):
            if not root:
                return
            tmp = root.right
            root.right = myflaten(root.left)
            root.left = None
            s = root
            while s.right:
                s = s.right
            s.right = myflaten(tmp)
            return root

        myflaten(root)




# 11ContainerWithMostWater.py

class Solution:
    def maxArea(self, height):  # Algorithm: Greedy, Two pointers
        """
        :type height: List[int]
        :rtype: int
        """
        left, right = 0, len(height) - 1
        res = 0
        while left < right:
            res = max(res, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return res




# 121BestTimeToBuyAndSellStock.py

class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        curr_min = prices[0]
        res = 0
        for price in prices:
            res = price - curr_min if price - curr_min > res else res
            # Note: faster than max(res, ...)
            # since no function is called
            curr_min = price if price < curr_min else curr_min
        return res




# 124BinaryTreeMaximumPathSum.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    ans = -float('inf')
    def maxPathSum(self, root):  # Algorithm: Divide-and-Conquer / DFS
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(root):
            if not root:
                return 0
            left, right = dfs(root.left), dfs(root.right)
            self.ans = max(self.ans, root.val, left + root.val,
                           right + root.val, left + right + root.val)
            return max(root.val, root.val + left, root.val + right)

        dfs(root)
        return self.ans




# 128LongestConsecutiveSequence.py

class Solution:
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums = set(nums)  # Feature: lookup in O(1)
        res = 1
        for n in nums:
            if n - 1 not in nums:
                k = n + 1
                while k in nums:
                    k += 1
                res = max(res, k - n)
        return res if nums else 0




# 136SingleNumber.py

class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans = 0
        for n in nums:
            ans = ans ^ n
        return ans  # Equivilant to: return reduce(lambda x, y: x ^ y, nums)



# 139WordBreak.py

class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        dp = [True] + [False] * len(s)
        for i in range(1, len(s) + 1):
            for k in range(0, i):
                if dp[k] and s[k:i] in wordDict:
                    dp[i] = True
        return dp[len(s)]




# 141LinkedListCycle.py

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def hasCycle(self, head):  # Algorithm: two pointers
        """
        :type head: ListNode
        :rtype: bool
        """
        if not head:
            return False
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        else:
            return False




# 142LinkedListCycleII.py

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):  # Algorithm: two pointers
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow, fast = head, head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:
                break
        else:
            return None
        slow = head
        while slow is not fast:
            slow, fast = slow.next, fast.next
        return slow



# 146LRUCache.py

from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = OrderedDict()  # Feature: OrderedDict to maintain insertion order
        self.slots = capacity
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.cache:
            return -1
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.cache:
            self.cache.pop(key)
            self.slots += 1
        if self.slots == 0:
            self.cache.popitem(last=False)
        else:
            self.slots -=1
        self.cache[key] = value

        


class Node:  # Note: can be replaced by a list, which is simpler and faster
    def __init__(self, key, val, prev=None, next=None):
        self.key = key
        self.val = val
        self.prev, self.next = prev, next

    def pop(self):
        self.next.prev = self.prev
        self.prev.next = self.next

    def insert(self, node):
        node.prev.next = self
        self.prev = node.prev
        self.next = node
        node.prev = self


class LRUCache:  # Algorithm(data structure): doubly linked list
                 # (which is exactly the way OrderedDict was implemented)

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.dic = {}
        self.empty = capacity
        self.head = Node(0, 0)
        self.tail = Node(0, 0, self.head, None)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.dic:
            return -1
        self.dic[key].pop()
        self.dic[key].insert(self.tail)
        return self.dic[key].val

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.dic:
            self.dic[key].val = value
            self.dic[key].pop()
            self.dic[key].insert(self.tail)
            return
        self.dic[key] = Node(key, value)
        if self.empty > 0:
            self.empty -= 1
        else:
            self.dic.pop(self.head.next.key)
            self.head.next.pop()
        self.dic[key].insert(self.tail)


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)




# 148SortList.py

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def sortList(self, head):  # Algorithm: Merge Sort
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        prev = slow
        slow = slow.next
        prev.next = None
        h1, h2 = self.sortList(head), self.sortList(slow)
        head = ListNode(0)
        tail = head
        while h1 and h2:
            if h1.val < h2.val:
                tail.next = h1
                h1, tail = h1.next, tail.next
            else:
                tail.next = h2
                h2, tail = h2.next, tail.next
        if h1:
            while h1:
                tail.next = h1
                h1, tail = h1.next, tail.next
        elif h2:
            while h2:
                tail.next = h2
                h2, tail = h2.next, tail.next
        tail.next = None
        return head.next




# 152MaximumProductSubarray.py

class Solution:
    def maxProduct(self, nums):  # Algorithm: DP (O(n) Time, O(1) Space)
        """
        :type nums: List[int]
        :rtype: int
        """
        dp, res = (nums[0], nums[0]), nums[0]
        for n in nums[1:]:
            tmp = [dp[0] * n, dp[1] * n, n]
            dp = (max(tmp), min(tmp))
            res = max(res, dp[0])
        return res




# 153Sum.py

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




# 155MinStack.py

class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.array = []
        self.mins = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.array.append(x)
        if not self.mins or x <= self.mins[-1]:
            self.mins.append(x)

    def pop(self):
        """
        :rtype: void
        """
        x = self.array.pop()
        if x == self.mins[-1]:
            self.mins.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.array[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.mins[-1]


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()




# 160IntersectionOfTwoLinkedLists.py

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




# 169MajorityElement.py

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





# 17LetterCombinationsOfAPhoneNumber.py

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




# 198HouseRobber.py

class Solution:
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        norob, rob = 0, 0
        for n in nums:
            norob, rob = rob, max(norob + n, rob)
        return rob



# 19RemoveNthNodeFromEndOfList.py

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        slow, fast = head, head
        for i in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            slow, fast = slow.next, fast.next
        slow.next = slow.next.next
        return head




# 1TwoSum.py

class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dic = {}
        for i in range(len(nums)):
            if target - nums[i] in dic:
                return [dic[target - nums[i]], i]
            else:
                dic[nums[i]] = i



# 200NumberOfIslands.py

class Solution:
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        def dfs(row, col):
            if row < 0 or col < 0 or row >= len(grid) or col >= len(grid[0]):
                return
            if grid[row][col] == '1':
                grid[row][col] = '0'
                for i in range(4):
                    dfs(row + dx[i], col + dy[i])

        dx, dy = [-1, 1, 0, 0], [0, 0, 1, -1]
        res = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    dfs(i, j)
                    res += 1
        return res




# 206ReverseLinkedList.py

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head
        while curr:
            latt = curr.next
            curr.next = prev
            prev = curr
            curr = latt
        return prev




# 207CourseSchedule.py

class Solution:
    def canFinish(self, numCourses, prerequisites):  # Algorithm: BFS TopSort
                                                     # Complexity: O(|V| + |E|)
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses
        for t, s in prerequisites:
            graph[s].append(t)
            indegree[t] += 1
        queue = [t for t in range(numCourses) if indegree[t] == 0]
        visited = 0
        while queue:
            visited += 1
            for t in graph[queue[0]]:
                indegree[t] -= 1
                if indegree[t] == 0:
                    queue.append(t)
            queue.pop(0)
        return visited == numCourses





# 208.ImplementTrie.py

class TrieNode:

    def __init__(self):
        self.next = {}


class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        self.root.next['$'] = TrieNode()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        ptr = self.root
        for c in word + '$':
            if c not in ptr.next:
                ptr.next[c] = TrieNode()
            ptr = ptr.next[c]

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        ptr = self.root
        for c in word + '$':
            if c not in ptr.next:
                return False
            ptr = ptr.next[c]
        return True

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        ptr = self.root
        for c in prefix:
            if c not in ptr.next:
                return False
            ptr = ptr.next[c]
        return True


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)




# 20ValidParentheses.py

class Solution:
    def isValid(self, s):  # Algorithm(data structure): stack
                           # Note: expr is valid <=> valid stack operations
        """
        :type s: str
        :rtype: bool
        """
        dic = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for c in s:
            if c in dic:
                stack.append(c)
            else:
                if not stack or dic[stack.pop()] != c:
                    return False
        return not stack




# 215KthLargestElementInAnArray.py

class Solution:
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return sorted(nums, reverse=True)[k - 1]




# 218TheSkylineProblem.py

class Solution:
    def getSkyline(self, buildings):  # Algorithm(data structure): PriorityQueue
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        import heapq

        points = set([b[0] for b in buildings] + [b[1] for b in buildings])
        p, active, res = 0, [], []

        for x in sorted((points)):
            while p < len(buildings) and buildings[p][0] <= x:
                heapq.heappush(active, (-buildings[p][2], buildings[p][1]))
                p += 1

            while active and active[0][1] <= x:
                heapq.heappop(active)

            curr_height = -active[0][0] if active else 0
            if not res or curr_height != res[-1][1]:
                res.append([x, curr_height])

        return res


class SolutionB:
    def getSkyline(self, buildings):
        """
        :type buildings: List[List[int]]
        :rtype: List[List[int]]
        """
        import heapq

        if not buildings:
            return []

        heap = [(0, -float('inf'))]
        rightmost = max(list(zip(*buildings))[1])
        curr_height, res = 0, []

        buildings.append([rightmost, rightmost, 0])

        for li, ri, hi in buildings:
            while -heap[0][1] <= li:
                h, r = heapq.heappop(heap)
                h, r = -h, -r
                while -heap[0][1] <= r:
                    heapq.heappop(heap)
                if r == li and curr_height == hi:
                    continue
                curr_height = -heap[0][0] if r < li else hi
                res.append([r, curr_height])

            if hi > curr_height:
                if res and li == res[-1][0]:
                    res[-1][1] = hi
                else:
                    res.append([li, hi])
                curr_height = hi
            heapq.heappush(heap, (-hi, -ri))

        return res




# 21MergeTwoSortedLists.py

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




# 221MaximalSquare.py

class Solution:

    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix) == 0:
            return 0
        ans = 0
        matrix = [[int(x) for x in row] for row in matrix]
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if row > 0 and col > 0 and matrix[row][col] == 1:
                    matrix[row][col] = 1 + min(matrix[row - 1][col - 1],
                                               matrix[row][col - 1], matrix[row - 1][col])
                ans = max(ans, matrix[row][col])
        return ans * ans




# 226InvertBinaryTree.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is not None:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root



# 22GenerateParentheses.py

from collections import deque


class Solution:
    def generateParenthesis(self, n):
        # Note: S is valid when and only when '('.count >= ')'.count
        #       for any prefix of S
        """
        :type n: int
        :rtype: List[str]
        """
        queue = deque([' ' * (2 * n)])
        while queue[0].count('(') < n:
            curr = queue.popleft()
            lp = curr.find(' ')
            curr = curr[:lp] + '(' + curr[lp + 1:]
            rp = lp
            while rp < 2 * n - 1 and curr[rp + 1] == ' ':
                rp += 1
            for p in range(rp, lp, -2):
                queue.append(curr[:p] + ')' + curr[p + 1:])
        return list(queue)




# 234PalindromeLinkedList.py

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        return vals == vals[::-1]
        



# 236LowestCommonAncestorOfABinaryTree.py

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if not root or root is p or root is q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right


class SolutionB(object):
    ans = []
    paths = []
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def dfs(root):
            if not root:
                return
            self.ans.append(root)
            if root is p or root is q:
                self.paths.append(self.ans[:])
            dfs(root.left)
            dfs(root.right)
            self.ans.pop()

        if p is q:
            return p
        self.paths = []
        dfs(root)
        minlen = min(len(self.paths[0]), len(self.paths[1]))
        for i in range(minlen - 1, -1, -1):
            if self.paths[0][i] is self.paths[1][i]:
                return self.paths[0][i]




# 238ProductOfArrayExceptSelf.py

class Solution:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        ans1, ans2, size = 1, 1, len(nums)
        res = [1] * size
        for i in range(size):
            res[i] *= ans1
            res[size - i - 1] *= ans2
            ans1 *= nums[i]
            ans2 *= nums[size - i - 1]
        return res


class SolutionB:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        prod, zero = 1, 0
        for n in nums:
            if n != 0:
                prod *= n
            else:
                zero += 1
        if zero == 0:
            return [prod // n for n in nums]
        elif zero == 1:
            return [prod if n == 0 else 0 for n in nums]
        else:
            return [0] * len(nums)




# 239SlidingWindowMaximum.py

class Solution:
    def maxSlidingWindow(self, nums, k):  # Complexity: O(n)
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        if not nums:  # Algorithm: maintaining a ordered deque
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




# 23MergeKSortedLists.py

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





# 240SearchA2DMatrixII.py

class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix:
            return False
        h, w = len(matrix), len(matrix[0])
        row, col = h - 1, 0
        while row >= 0 and col < w:
            if matrix[row][col] == target:  # Algorithm
                return True
            elif matrix[row][col] < target:
                col += 1
            else:
                row -= 1
        return False




# 279PerfectSquares.py

from math import sqrt

class Solution:
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        while n % 4 == 0:
            n = int(n / 4)
        if n % 8 == 7: # Math: Legendre's three-square theorem
            return 4
        elif int(sqrt(n))**2 == n:
            return 1
        for i in range(1, int(sqrt(n) + 1)):
            if int(sqrt(n - i**2))**2 == n - i**2:
                return 2
        return 3 # Math: Lagrange's four-square theorem+



# 283MoveZeroes.py

class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        size = len(nums)
        for i in range(nums.count(0)):
            nums.remove(0)
        nums += [0] * (size - len(nums))



# 287FindTheDuplicateNumber.py

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



# 297SerializeAndDeserializeBinaryTree.py

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Codec:
    def serialize(self, root):
        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        if not root:
            return '^$'
        else:
            return '^' + str(root.val) + self.serialize(root.left) + \
                self.serialize(root.right) + '$'

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        if data == '^$':
            return None
        else:
            p1 = data[1:].index('^') + 1
            cnt, p2 = 1, p1 + 1
            while cnt > 0:
                if data[p2] == '^':
                    cnt += 1
                elif data[p2] == '$':
                    cnt -= 1
                p2 += 1
            root = TreeNode(int(data[1:p1]))
            root.left = self.deserialize(data[p1:p2])
            root.right = self.deserialize(data[p2:-1])
            return root

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.deserialize(codec.serialize(root))




# 2AddTwoNumbers.py

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head, zero = ListNode(0), ListNode(0)
        ptr, carryout = head, 0
        while l1 or l2:
            l1 = zero if not l1 else l1
            l2 = zero if not l2 else l2
            carryout, num = divmod(l1.val + l2.val + carryout, 10)
            ptr.next = ListNode(num)
            ptr, l1, l2 = ptr.next, l1.next, l2.next
        if carryout:
            ptr.next = ListNode(carryout)
            ptr = ptr.next
        ptr.next = None
        return head.next




# 300LongestIncreasingSubsequence.py

class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        size, dp = 0, [0] * len(nums)
        for n in nums:
            left, right = 0, size
            while left < right:  # Algorithm: Binary Search
                mid = (left + right) // 2
                if dp[mid] >= n:
                    right = mid
                else:
                    left = mid + 1
            dp[left] = n
            size = max(left + 1, size)
        return size


class SolutionB:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0] * len(nums)
        res = 0
        for i in range(len(nums)):
            dp[i] = 1 + max([0] + [dp[j] for j in range(i) if nums[j] < nums[i]])
            res = max(res, dp[i])
        return res




# 301RemoveInvalidParentheses.py

class Solution:
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def revise(k, expr, pars):  # Note: expr is valid 
                                    # <=> (lcount == rcount) and (lcount >= rcount during forward pass)
                                    # <=> (lcount >= rcount during forward pass) and (rcount >= lcount during backward pass)
            cnt = 0
            while k < len(expr):
                cnt += int(expr[k] == pars[0])
                cnt -= int(expr[k] == pars[1])
                if cnt < 0:
                    break
                k += 1
            else:
                if pars[0] == '(':
                    revise(0, expr[::-1], (')', '('))
                else:
                    res.append(expr[::-1])
                return

            for i in range(0, k + 1):
                if expr[i] == pars[1] and (i == 0 or expr[i - 1] != pars[1]):
                    revise(k, expr[:i] + expr[i + 1:], pars)

        res = []
        revise(0, s, ('(', ')'))
        return list(set(res))




# 309BestTimeToBuyAndSellStockWithCooldown.py

class Solution:
    def maxProfit(self, prices):  # DP, Complexity: O(n)
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n < 2:
            return 0
        profit, profit_with_stock = [0] * (n + 2), [0] * (n + 2)
        profit_with_stock[0] = -prices[0]
        for i in range(1, n):
            profit[i] = max(profit[i - 1], profit_with_stock[i - 1] + prices[i])
            profit_with_stock[i] = max(profit_with_stock[i - 1], profit[i - 2] - prices[i])
        return profit[n - 1]


class SolutionB:
    def maxProfit(self, prices):  # DP, Complexity: O(n ^ 2)
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        prices.insert(0, 0)
        dp = [0] * (n + 2)
        for i in range(1, n + 1):
            dp[i] = dp[i - 1]
            for k in range(1, i):
                ans = dp[k - 2] + prices[i] - prices[k]
                if ans > dp[i]:
                    dp[i] = ans
        return dp[n]




# 312BurstBalloons.py

class Solution:
    ans = 0
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        nums = [1] + nums + [1]
        dp = [[0] * (n + 2) for _ in range(n + 2)]  # Algorithm: DP (key: treat k as the last balloon burst)
        for l in range(1, n + 1):
            for i in range(1, n + 2 - l):
                j = i + l
                for k in range(i, j):
                    ans = dp[i][k] + dp[k + 1][j] + nums[k] * nums[i - 1] * nums[j]
                    if ans > dp[i][j]:
                        dp[i][j] = ans
        return dp[1][n + 1]




# 315.CountOfSmallerNumbersAfterSelf.py

class BinaryIndexedTree:
    @staticmethod
    def lowbit(x):
        return x & (-x)

    def __init__(self, n):
        self._size = n
        self._array = [0] * (n + 1)

    def __str__(self):
        return str(self._array[1:])

    def __repr__(self):
        return str(self._array[1:])

    def __len__(self):
        return self._size

    def update(self, index, delta):
        while index <= self._size:
            self._array[index] += delta
            index += BinaryIndexedTree.lowbit(index)

    def getsum(self, index):  # index begins from 1
        res = 0
        while index >= 1:
            res += self._array[index]
            index -= BinaryIndexedTree.lowbit(index)
        return res


class Solution:
    def countSmaller(self, nums):  # Algorithm(data structure): Binary Indexed Tree
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        def update(index):
            while index <= size:
                BITree[index] += 1
                index += index & -index

        def getsum(index):
            res = 0
            while index >= 1:
                res += BITree[index]
                index -= index & -index
            return res

        size = len(nums)
        numpos = sorted((zip(nums, range(size))))
        BITree = [0] * (size + 1)
        res = [0] * size
        for n, pos in numpos:
            res[pos] = getsum(size - pos)
            update(size - pos)
        return res




# 31NextPermutation.py

class Solution:
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        def reverse(left, right):
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left, right = left + 1, right - 1

        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i + 1]:
                for j in range(len(nums) - 1, i, -1):
                    if nums[j] > nums[i]:
                        nums[i], nums[j] = nums[j], nums[i]
                        reverse(i + 1, len(nums) - 1)
                        break
                break
        else:
            reverse(0, len(nums) - 1)




# 322CoinChange.py

class Solution:
    ans = float('inf')
    def coinChange(self, coins, amount):  # Algorithm: DFS with pruning
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        def dfs(index, target, used):
            if index == len(coins) or used + target // coins[index] >= self.ans:
                if target == 0 and used < self.ans:
                    self.ans = used
                return
            for k in range(target // coins[index], -1, -1):
                dfs(index + 1, target - coins[index] * k, used + k)

        coins.sort(reverse=True)
        dfs(0, amount, 0)
        return self.ans if self.ans != float('inf') else -1




# 32LongestValidParentheses.py

class Solution:
    def longestValidParentheses(self, s):  # Algorithm: DP, O(n) time
                                           # * there's another solution using stack
        """
        :type s: str
        :rtype: int
        """
        dp, res = [0] * len(s), 0
        for i, char in enumerate(s):
            p = i - dp[i - 1] - 1
            if char == ')' and p >= 0 and s[p] == '(':
                dp[i] = dp[i - 1] + 2 + (dp[p - 1] if p > 0 else 0)
            else:
                dp[i] = 0
            res = max(res, dp[i])
        return res




# 337HouseRobberIII.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def solve(node):
            if node:
                lnmax, lmax = solve(node.left)
                rnmax, rmax = solve(node.right)
                return lmax + rmax, max(lmax + rmax, node.val + lnmax + rnmax)
            else:
                return 0, 0

        return solve(root)[1]




# 338CountingBits.py

class Solution:
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        dp = [0]
        for i in range(1, num + 1):
            dp.append(1 + dp[i & (i - 1)])
        return dp




# 33SearchInRotatedSortedArray.py

class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        if not nums:
            return -1
        left, right = 0, len(nums)
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] > nums[left] and \
                    nums[left] <= target < nums[mid] or \
                    nums[mid] < nums[left] and \
                    (target >= nums[left] or target < nums[mid]):
                    # Feature: a < b < c
                right = mid
            else:
                left = mid
        return left if nums[left] == target else -1


class SolutionB:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def myBinarySearch(i, j):
            if i + 1 == j:
                return i if nums[i] == target else -1
            mid = (i + j) // 2
            if nums[mid] > nums[i] and \
                    nums[i] <= target < nums[mid] or \
                    nums[mid] < nums[i] and \
                    (target >= nums[i] or target < nums[mid]):
                return myBinarySearch(i, mid)
            else:
                return myBinarySearch(mid, j)
        return myBinarySearch(0, len(nums)) if nums else -1




# 347TopKFrequentElements.py

from collections import Counter

class Solution:
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        counter = Counter(nums)
        res = sorted(counter.keys(), key=lambda x: -counter[x])[0: k]
                # Equivilant to: res = [x[0] for x in counter.most_common(k)]
                # Feature: counter.most_common(n) counter[x]
        return res





# 34SearchForARange.py

class Solution:
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        if not nums:
            return [-1, -1]
        left, right = 0, len(nums)
        while left + 1 < right:
            mid = (left + right - 1) // 2
            if nums[mid] >= target:
                right = mid + 1
            else:
                left = mid + 1
        start = left if nums[left] == target else -1
        left, right = start, len(nums)
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] > target:
                right = mid
            else:
                left = mid
        end = left if nums[left] == target else -1
        return [start, end]



# 394DecodeString.py

class Solution:
    def decodeString(self, s):  # Algorithm: using stack
        """
        :type s: str
        :rtype: str
        """
        stack, curr_num, curr_str = [], 0, ''
        for c in s:
            if c == '[':
                stack.append(curr_str)
                stack.append(curr_num)
                curr_num, curr_str = 0, ''
            elif c == ']':
                num = stack.pop()
                curr_str = stack.pop() + curr_str * num
            elif c.isdigit():
                curr_num = curr_num * 10 + int(c)
            else:
                curr_str += c
        return curr_str


class SolutionB:
    def decodeString(self, s):  # Recursive version
        """
        :type s: str
        :rtype: str
        """
        l = s.find('[')
        if l == -1:
            return s
        p = l - 1
        while p > 0 and s[p - 1].isdigit():
            p -= 1
        r = l
        lcnt, rcnt = 1, 0
        while lcnt > rcnt:
            r += 1
            if s[r] == '[':
                lcnt += 1
            elif s[r] == ']':
                rcnt += 1
        return s[0: p] + self.decodeString(s[l + 1: r]) * int(s[p: l]) + self.decodeString(s[r + 1:])




# 39CombinationSum.py

class Solution:
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        def dfs(k, s, curr):
            if k == len(candidates):
                if s == target:
                    res.append(curr.copy())
                return
            tmp = curr.copy()
            for i in range((target - s) // candidates[k] + 1):
                dfs(k + 1, s, tmp)
                s += candidates[k]
                tmp.append(candidates[k])

        res = []
        dfs(0, 0, [])
        return res




# 3LongestSubstringWithoutRepeatingCharacters.py

class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, res, last = 0, 0, {}  # Algorithm: optimized sliding window
        for right, c in enumerate(s):  # Feature: enumerate, cleaner and
                                       #          faster than frequent s[i]
            if c in last and left <= last[c]:
                left = last[c] + 1
            last[c] = right
            res = max(res, right - left + 1)
        return res


class SolutionB:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        from collections import defaultdict
        counts = defaultdict(int)
        left, res = 0, 0
        for right in range(len(s)):  # Algorithm: Sliding Window
            counts[s[right]] += 1
            while counts[s[right]] > 1:
                counts[s[left]] -= 1
                left += 1
            res = max(res, right - left + 1)
        return res




# 406QueueReconstructionByHeight.py

class Solution:
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda x: (-x[0], x[1]))
        queue = []
        for person in people:
            queue.insert(person[1], person)
        return queue
        
class SolutionB:
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        n = len(people)
        people.sort(key=lambda x: (x[0], -x[1]))
        queue = [None] * n
        for person in people:
            pos = [i for i in range(n) if queue[i] is None][person[1]]
            queue[pos] = person
        return queue
        
        # Complexity: O(n^2)



# 416PartitionEqualSubsetSum.py

class Solution:
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        target, mod = divmod(sum(nums), 2)
        if mod:
            return False
        dp = [False] * (target + 1)
        dp[0] = True
        for n in nums:
            for i in range(target, n - 1, -1):
                dp[i] = dp[i] or dp[i - n]
        return dp[target]


class Solution:
    def canPartition(self, nums):  # Algorithm: DFS with pruning
        """
        :type nums: List[int]
        :rtype: bool
        """
        nums.sort(reverse=True)  # Sorting before DFS, important!
        target, mod = divmod(sum(nums), 2)  # Feature: divmod
        if mod:
            return False
        ans = 0
        rsum = [0] * len(nums)
        for i in range(len(nums) - 1, -1, -1):
            ans += nums[i]
            rsum[i] = ans

        def dfs(k, sofar):
            if k == len(nums) or sofar >= target:  # Pruning 1
                return sofar == target
            elif sofar + rsum[k] <= target:  # Prunning 2
                return sofar + rsum[k] == target
            return dfs(k + 1, sofar) or dfs(k + 1, sofar + nums[k])
        return dfs(0, 0)




# 42TrappingRainWater.py

class Solution:
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0

        n = len(height)
        outline, outline2 = [height[0]] * n, [height[n - 1]] * n
        for i in range(1, n):
            outline[i] = max(height[i], outline[i - 1])
        for i in range(n - 2, -1, -1):
            outline2[i] = max(height[i], outline2[i + 1])
        outline = [min(outline[i], outline2[i]) for i in range(n)]

        res = 0
        for i in range(n):
            res += outline[i] - height[i]
        return res




# 437PathSumIII.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    ans = 0
    cache = {0: 1}

    def pathSum(self, root, sum): # Algorithm: PATH(a, b) = PATH(root, b) - PATH(root, a) -> DFS
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """

        def dfs(t, sofar):  # Algorithm: DFS
            if not t:
                return
            sofar += t.val
            self.ans += self.cache.get(sofar - sum, 0)
            self.cache[sofar] = self.cache.get(sofar, 0) + 1
            dfs(t.left, sofar)
            dfs(t.right, sofar)

            self.cache[sofar] -= 1






        dfs(root, 0)
        return self.ans




# 438FindAllAnagramsInAString.py

class Solution:
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        if len(s) < len(p):
            return []
        window, pcount, res = [0] * 26, [0] * 26, []
        for c in p:
            pcount[ord(c) - 97] += 1
        for c in s[:len(p)]:
            window[ord(c) - 97] += 1
        for i in range(len(p), len(s)):
            if window == pcount:
                res.append(i - len(p))
            window[ord(s[i]) - 97] += 1
            window[ord(s[i - len(p)]) - 97] -= 1
        if window == pcount:
            res.append(len(s) - len(p))
        return res





# 442FindAllDuplicatesInAnArray.py

from collections import Counter

class Solution:
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        counter = Counter(num for num in nums)
        return [x for x in counter if counter[x] == 2]



# 448FindAllNumbersDisappearedInAnArray.py

class Solution:
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        return list(set(range(1, len(nums) + 1)).difference(nums))




# 461HammingDistance.py

class Solution:
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        ans = x ^ y
        res = 0
        while ans > 0:
            ans -= ans & (-ans)  # Equivilant to: ans &= ans - 1
            res += 1
        return res
        
        # Another solution: return bin(x ^ y).count('1')



# 46Permutations.py

class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        from itertools import permutations
        return list(permutations(nums))  # Feature: permutation(iterable, r)




# 48RotateImage.py

class Solution:
    def rotate(self, matrix):  # Space Complexity: O(1)
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        h = n // 2
        w = n - h
        for i in range(h):
            for j in range(w):
                tmp = matrix[i][j]
                matrix[i][j] = matrix[n - 1 - j][i]
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j]
                matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i]
                matrix[j][n - 1 - i] = tmp


class SolutionB:
    def rotate(self, matrix):  # Space Complexity: O(n^2)
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        rotated = list(zip(*matrix))
        for i in range(len(rotated)):
            matrix[i] = rotated[i][::-1]




# 494TargetSum.py

class Solution:
    def findTargetSumWays(self, nums, S):   # Algorithm: DP
                                            # Time: O(n*l), Space: O(l)
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        maxsum = sum(nums)
        S += maxsum
        if S < 0 or S > 2 * maxsum:
            return 0
        dp = [0] * (2 * maxsum + 1)
        dp[nums[0] + maxsum] += 1
        dp[-nums[0] + maxsum] += 1
        for i in range(1, len(nums)):
            next = [0] * (2 * maxsum + 1)
            for j in range(2 * maxsum + 1):
                if j - nums[i] >= 0:
                    next[j] += dp[j - nums[i]]
                if j + nums[i] <= 2 * maxsum:
                    next[j] += dp[j + nums[i]]
            dp = next
        return dp[S]

class SolutionB:
    def findTargetSumWays(self, nums, S):  # Time Complexity: O(2^n) (Bad!)
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        from collections import deque
        
        nums.sort(reverse=True)
        maxsum = [0] * (len(nums) + 1)
        sum = 0
        for i in range(len(nums) - 1, -1, -1):
            sum += nums[i]
            maxsum[i] = sum
        
        queue = deque([(0, 0)])
        res = 0
        while queue:
            n, k = queue.popleft()
            if k == len(nums):
                if n == S:
                    res += 1
            else:
                if n - maxsum[k] <= S and n + maxsum[k] >= S:
                    queue.append((n + nums[k], k + 1))
                    queue.append((n - nums[k], k + 1))
        return res




# 49GroupAnagrams.py

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




# 4MedianOfTwoSortedArrays.py

class Solution:
    def findMedianSortedArrays(self, nums1, nums2):  # *there's an O(log(min(m, n))) solution
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        nums = sorted(nums1 + nums2)
        size = len(nums)
        if size % 2 == 1:
            return float(nums[size // 2])
        else:
            return (nums[size // 2 - 1] + nums[size // 2]) / 2



# 538ConvertBSTToGreaterTree.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    ans = 0
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def traverse(node):  # In-order traversal
            if node:
                traverse(node.right)
                node.val += self.ans
                self.ans = node.val
                traverse(node.left)
        traverse(root)
        return root

class SolutionB:
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        def convertNode(node, sum):  # Divide-and-Conquer
            if node:
                ans = sum + convertNode(node.right, sum) + node.val
                node.val = ans
                return ans + convertNode(node.left, ans) - sum
            else:
                return 0
        convertNode(root, 0)
        return root



# 53MaximumSubarray.py

class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ans, res = nums[0], nums[0]
        for n in nums[1:]:
            ans = ans + n if ans > 0 else n
            res = max(ans, res)
        return res



# 543DiameterOfBinaryTree.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def solve(node):
            if node:
                ldepth = solve(node.left)
                rdepth = solve(node.right)
                self.ans = max(self.ans, ldepth + rdepth)
                return max(ldepth, rdepth) + 1
            else:
                return 0

        self.ans = 0
        solve(root)
        return self.ans




# 55JumpGame.py

class Solution:
    def canJump(self, nums):  # Algorithm: Greedy / O(n) DP
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        smallest = n - 1
        for pos in range(n - 2, -1, -1):
            if pos + nums[pos] >= smallest:
                smallest = pos
        return nums[0] >= smallest


class SolutionB:
    def canJump(self, nums):  # Algorithm: Greedy
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) == 1:
            return True
        pos = 0
        while nums[pos] > 0:
            if pos + nums[pos] >= len(nums) - 1:
                return True
            ans = 1
            for step in range(1, nums[pos] + 1):
                if nums[pos + step] + step > nums[pos + ans] + ans:
                    ans = step
            pos = pos + ans
        return False




# 560SubarraySumEqualsK.py

class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        from collections import defaultdict
        cache = defaultdict(int)  # Feature: defaultdict(<callable object>)
        cache[0] = 1
        ans, res = 0, 0
        for n in nums:
            ans += n
            res += cache[ans - k]
            cache[ans] += 1
        return res




# 56MergeIntervals.py

# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e


class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        res = []
        for interval in sorted(intervals, key=lambda x: x.start):
            if res and interval.start <= res[-1].end:
                res[-1].end = max(res[-1].end, interval.end)
            else:
                res.append(interval)
        return res


class SolutionB:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return []

        from collections import Counter
        start_counts = Counter([x.start for x in intervals])
        end_counts = Counter([x.end for x in intervals])
        timeline = sorted({x.start for x in intervals} | {x.end for x in intervals})
        res = []
        curr_start = timeline[0]
        curr_counts = 0
        for t in timeline:
            if curr_counts == 0:
                curr_start = t
            curr_counts = curr_counts + start_counts[t] - end_counts[t]
            if curr_counts == 0:
                res.append(Interval(curr_start, t))
        return res



# 572SubtreeOfAnotherTree.py

class Solution:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """

        # Algorithm: mapping binary trees to strings using pre-order traversal
        # -> mapping recursive structures to CFG strings for comparison
        def preorder(t):
            if not t:
                return '^$'
            else:
                return '^' + str(t.val) + preorder(t.left) + preorder(t.right) + '$'

        return preorder(t) in preorder(s)


class SolutionB:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """

        def isSame(s, t):
            if not s and not t:
                return True
            elif s and t:
                return s.val == t.val and isSame(s.left, t.left) and isSame(s.right, t.right)
            else:
                return False

        if not s and not t:
            return True
        elif s and t:
            return self.isSubtree(s.left, t) or self.isSubtree(s.right, t) or isSame(s, t)
        else:
            return False




# 581ShortestUnsortedContinuousSubarray.py

class Solution:
    def findUnsortedSubarray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        low = min([i for i in range(1, n) if nums[i - 1] > nums[i]] + [n])
        if low == n:
            return 0
        up = max([i for i in range(1, n) if nums[i - 1] > nums[i]] + [0])
        nmin, nmax = min(nums[low - 1:up + 1]), max(nums[low - 1:up + 1])
        start = max([i for i in range(low) if nums[i] <= nmin] + [-1])
        end = min([i for i in range(up, n) if nums[i] >= nmax] + [n])
        return end - start - 1




# 5LongestPalindromicSubstring.py

class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        left, right = 0, 0
        res = ''
        while left < len(s):  # Algorithm: traversing all palindromic substrings
            while right < len(s) and s[right] == s[left]:
                right += 1
            lp, rp = left - 1, right
            while lp >= 0 and rp < len(s) and s[lp] == s[rp]:
                lp -= 1
                rp += 1
            if rp - lp - 1 > len(res):
                res = s[lp + 1:rp]
            left = right
        return res




# 617MergeTwoBinaryTrees.py

class Solution:
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if t1 is not None and t2 is not None:
            t1.val += t2.val
            t1.left = self.mergeTrees(t1.left, t2.left)
            t1.right = self.mergeTrees(t1.right, t2.right)
            return t1
        else:
            return t1 if t1 is not None else t2




# 621TaskScheduler.py

class Solution:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import Counter  # Algorithm: Math, O(1) Time (Without counting the time of reading inputs)
        counts = list(Counter(tasks).values())
        nmax = max(counts)
        ans = counts.count(nmax)
        return max((nmax - 1) * (n + 1) + ans, len(tasks))


class SolutionB:
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        from collections import Counter  # Algorithm: Greedy
        counts = Counter(tasks)
        ntask, res = len(counts), 0
        while ntask:
            if n + 1 >= ntask:
                iter = counts.most_common()[-1][1]
                for task in counts:
                    counts[task] -= iter
            else:
                elems = counts.most_common(n + 2)
                iter = elems[n][1] - elems[n + 1][1] + 1
                for task, _ in elems[0: -1]:
                    counts[task] -= iter
            for x in list(counts):
                if counts[x] == 0:
                    counts.pop(x)
            res += (n + 1) * iter if len(counts) > 0 else (n + 1) * (iter - 1) + ntask
            ntask = len(counts)
        return res




# 62UniquePaths.py

class Solution:
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        from scipy.special import comb
        return int(comb(m + n - 2, n - 1))  # Feature: comb, perm




# 647PalindromicSubstrings.py

class Solution:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        left, right = 0, 0
        res = 0
        while left < len(s):
            while right < len(s) and s[right] == s[left]:
                right += 1
            res += int((right - left) * (right - left + 1) / 2)
            lp, rp = left - 1, right
            while lp >= 0 and rp < len(s) and s[lp] == s[rp]:
                res += 1
                lp -= 1
                rp += 1
            left = rp
        return res


# Complexity: O(n^2)
class SolutionB:
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        dp = [[False] * (len(s) + 1) for i in range(len(s))]
        res = len(s)
        for i in range(len(s)):
            dp[i][i] = True
            dp[i][i + 1] = True
        for length in range(2, len(s) + 1):
            for offset in range(0, len(s) - length + 1):
                dp[offset][offset + length] = s[offset] == s[offset + length - 1] \
                                              and dp[offset + 1][offset + length - 1]
                res += int(dp[offset][offset + length])
        return res




# 64MinimumPathSum.py

class Solution:
    def minPathSum(self, grid):  # Space Complexity: O(n)
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        h, w = len(grid), len(grid[0])
        dp = [float('inf')] * (w + 1)  # Feature: float('inf')
        dp[w - 1] = 0
        for i in range(h - 1, -1, -1):
            for j in range(w - 1, -1, -1):
                dp[j] = min(dp[j], dp[j + 1]) + grid[i][j]
        return dp[0]




# 70ClimbingStairs.py

class Solution:
    def climbStairs(self, n):  # Complexity: O(n)
        """
        :type n: int
        :rtype: int
        """
        a, b = 1, 2
        for i in range(0, n - 2):
            a, b = b, a + b
        return b if n > 1 else 1


class SolutionB:
    def climbStairs(self, n):  # Complexity: O(n^2)
        """
        :type n: int
        :rtype: int
        """
        from math import factorial
        def comb(N, k):
            return factorial(N) // factorial(k) // factorial(N - k)

        res = 0
        for x in range(n // 2 + 1):
            res += int(comb(n - x, x))
        return res




# 72EditDistance.py

class Solution:
    def minDistance(self, word1, word2):  # Algorithm: DP
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        dp = [[i] * (len(word2) + 1) for i in range(len(word1) + 1)]
        for j in range(len(word2) + 1):
            dp[0][j] = j
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j - 1],
                                       dp[i - 1][j], dp[i][j - 1])
        return dp[len(word1)][len(word2)]




# 732MyCalendarIII.py

from collections import Counter

class MyCalendarThree:

    def __init__(self):
        self.counter = Counter()

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: int
        """
        self.counter[start] += 1 # Python Feature: use of Counter
        self.counter[end] -= 1
        active, ans = 0, 0
        # Algorithm: counting overlapping segments
        for i in sorted(self.counter):
            active += self.counter[i] 
            ans = max(ans, active)
        return ans


# Your MyCalendarThree object will be instantiated and called as such:
# obj = MyCalendarThree()
# param_1 = obj.book(start,end)



# 75SortColors.py

class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        p0, p1, p2 = 0, 0, 0  # Algorithm: in-place partition using ptrs
        while p2 < len(nums):
            print(p0, p1, p2, nums)
            if nums[p2] == 0:
                tmp, nums[p2] = nums[p2], nums[p1]
                nums[p1] = nums[p0]
                nums[p0] = tmp
                p0, p1, p2 = p0 + 1, p1 + 1, p2 + 1
            elif nums[p2] == 1:
                nums[p1], nums[p2] = nums[p2], nums[p1]
                p1, p2 = p1 + 1, p2 + 1
            else:
                p2 += 1




# 765CouplesHoldingHands.py

class Solution:
    def minSwapsCouples(self, row):
        """
        :type row: List[int]
        :rtype: int
        """
        pos = {}
        for i in range(len(row)):
            pos[row[i]] = i
        swap_count = 0
        # Algorighm: greedy
        for i in range(0, len(row), 2):
            if row[i + 1] == (row[i] - 1 if row[i] % 2 else row[i] + 1):
                continue
            swap_count += 1
            p = row[i + 1]
            q = row[i] - 1 if row[i] % 2 else row[i] + 1
            row[i + 1], row[pos[q]] = q, p
            pos[p], pos[q] = pos[q], pos[p]
        return swap_count
        



# 76MinimumWindowSubstring.py

class Solution:
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        from collections import defaultdict
        left = 0
        counts = defaultdict(int)
        total_match = 0
        for c in t:
            counts[c] += 1  # Algorithm: using dict and total_match to track 
                            #            the letters matched
                            #            (all counts <= zero means already matched)
        res = ''
        for right in range(len(s)):  # Algorithm: sliding window using two pointers
            counts[s[right]] -= 1
            if counts[s[right]] >= 0:
                total_match += 1

            if total_match == len(t):
                while counts[s[left]] < 0:
                    counts[s[left]] += 1
                    left += 1
                if res == '' or right - left + 1 < len(res):
                    res = s[left:right + 1]

                counts[s[left]] += 1
                total_match -= 1
                left += 1
        return res




# 78Subsets.py

class Solution:  # BFS version
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        from collections import deque
        queue = deque([([], 0)])
        while queue[0][1] < len(nums):
            ans, n = queue.popleft()
            queue.append((ans, n + 1))
            queue.append((ans + [nums[n]], n + 1))
        return [x[0] for x in queue]


class SolutionB:  # DFS version
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def gen(ans, n):
            if n == len(nums):
                res.append(ans)
            else:
                gen(ans.copy(), n + 1)
                gen(ans + [nums[n]], n + 1)

        res = []
        gen([], 0)
        return list(res)




# 79WordSearch.py

class Solution:
    ans = ''
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        def dfs(row, col):
            self.ans += board[row][col]
            if self.ans == word:
                return True
            elif not word.startswith(self.ans):
                self.ans = self.ans[:-1]
                return False

            tmp = board[row][col]
            board[row][col] = '*'
            for i in range(4):
                x, y = row + dx[i], col + dy[i]
                if x >= 0 and y >= 0 and x < len(board) and y < len(board[0])\
                        and board[x][y] != '*' and dfs(x, y):
                    return True
            board[row][col] = tmp
            self.ans = self.ans[:-1]
            return False

        dx = [1, -1, 0, 0]
        dy = [0, 0, 1, -1]
        for row in range(len(board)):
            for col in range(len(board[0])):
                if dfs(row, col):
                    return True
        return False




# 807MaxIncreaseToKeepCitySkyline.py

class Solution:
    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        left_skyline = list(map(max, grid))
        up_skyline = list(map(max, *grid)) # Python Feature: map(max, *grid)
                                           # equivilant to: up_skyline = [max(col) for col in zip(*grid)]
                                           # can be used to transpose a matrix
        result = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                result += min(up_skyline[j], left_skyline[i]) - grid[i][j]
        return result

if __name__ == '__main__':
    grid = [[3,0,8,4],[2,4,5,7],[9,2,6,3],[0,3,1,0]]
    print(Solution().maxIncreaseKeepingSkyline(grid))



# 84LargestRectangleInHistogram.py

class Solution:
    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        stack, res = [(0, -1, 0)], 0  # Algorithm: Ordered Stack(Deque)
        heights.append(0)
        for i in range(len(heights)):
            while stack and stack[-1][0] > heights[i]:
                h, pos, l = stack.pop()
                res = max(res, h * (i - pos + l))
            stack.append((heights[i], i, i - stack[-1][1] - 1))
        return res




# 85MaximalRectangle.py

class Solution:
    def maximalRectangle(self, matrix):  # Complexity: O(n*m)
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        def largestRectangleArea(heights):
            stack, res = [(0, -1, 0)], 0
            heights.append(0)
            for i in range(len(heights)):
                while stack and stack[-1][0] > heights[i]:
                    h, pos, l = stack.pop()
                    res = max(res, h * (i - pos + l))
                stack.append((heights[i], i, i - stack[-1][1] - 1))
            return res

        if len(matrix) == 0:
            return 0
        h, w, res = len(matrix), len(matrix[0]), 0
        heights = [0] * w
        for i in range(h):
            heights = [heights[j] + 1 if matrix[i][j] == '1' else 0 for j in range(w)]
            res = max(res, largestRectangleArea(heights))
        return res


class SolutionB:
    def maximalRectangle(self, matrix):  # Complexity: O(n*m*m)
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix) == 0:
            return 0
        h, w, res = len(matrix), len(matrix[0]), 0
        dp = [[0] * w for _ in range(w)]
        for l in range(h):
            for i in range(w):
                dp[i][i] = dp[i][i] + 1 if matrix[l][i] == '1' else 0
                res = max(res, dp[i][i])
                for j in range(i + 1, w):
                    dp[i][j] = dp[i][j] + \
                        1 if (matrix[l][j] == '1' and dp[i][j - 1]) else 0
                    res = max(res, dp[i][j] * (j - i + 1))
        return res




# 94BinaryTreeInorderTraversal.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stk = []
        ptr = root
        while stk or ptr:
            if ptr:
                ptr = stk.pop()
                res.append(ptr.val)
                ptr = ptr.right
            else:
                stk.append(ptr)
                ptr = ptr.left
        return res

class SolutionB:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        def visit(node):
            if node:
                visit(node.left)
                res.append(node.val)
                visit(node.right)

        res = []
        visit(root)
        return res





# 96UniqueBinarySearchTrees.py

class Solution:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        from math import factorial
        return factorial(2 * n) // (factorial(n)) ** 2 // (n + 1)  # Three times faster than comb(N, k)


class SolutionB:
    def numTrees(self, n):
        """
        :type n: int
        :rtype: int
        """
        from scipy.special import comb
        eps = 0.01
        return int(comb(2 * n, n) + eps) // (n + 1)  # Note: comb(N, k) returns float64




# 98ValidateBinarySearchTree.py

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def valid(root, lower_bound, upper_bound):
            if not root:
                return True
            if root.val <= lower_bound or root.val >= upper_bound:
                return False
            return valid(root.left, lower_bound, root.val) and\
                valid(root.right, root.val, upper_bound)
        return valid(root, -float('inf'), float('inf'))


class SolutionB:
    ans = True
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def dfs(root):
            if not root:
                return (-float('inf'), float('inf'))
            left = dfs(root.left)
            right = dfs(root.right)
            if root.val <= left[0] or root.val >= right[1]:
                self.ans = False
            return (max(root.val, right[0]), min(root.val, left[1]))

        dfs(root)
        return self.ans




