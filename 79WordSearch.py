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
