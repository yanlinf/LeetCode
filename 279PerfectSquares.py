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