# The problem is: A string s is good if it is possible to
# rearrange the characters of s such that the new string
# formed contains "prep" as **substring**. So the string
# "proper" is good but "poor" is not. Given an integer n,
# find the number of only lowercase English characters. As
# this number might be too large, return it modulo (10^9 +7)


def func(n: int):
    # N = TOTAL_NUM = 26 ** n
    # n1 = NUM[0A]
    # n2 = NUM[1A]

    # n3 = NUM[0B]
    # n4 = NUM[0A && 0B]
    # n5 = NUM[1A && 0B]
    # n6 = NUM[>=2A && 0B] = n3 - n4 - n5

    # n7 = NUM[>=2A && 0C] = n6
    # n8 = NUM[0B && 0C]
    # n9 = NUM[0A && 0B && 0C]
    # n10 = NUM[1A && 0B && 0C]
    # n11 = NUM[>=2A && 0B && 0C] = n8 - n9 - n10
    # n12 = NUM[>=2A && >=1B && 0C] = n7 - n11

    # answer = N - n1 - n2 - n6 - n12

    N = 26 ** n
    n1 = 25 ** n
    n2 = n * 25 ** (n - 1)
    n3 = n1
    n4 = 24 ** n
    n5 = n * 24 ** (n - 1)
    n6 = n3 - n4 - n5
    n7 = n6
    n8 = n4
    n9 = 23 ** n
    n10 = n * 23 ** (n - 1)
    n11 = n8 - n9 - n10
    n12 = n7 - n11
    ans = N - n1 - n2 - n6 - n12
    return ans % (10 ** 9 + 7)


if __name__ == '__main__':
    test_cases = [
        (4, 12),
        (5, 1460),
        (10, 83943898),
    ]
    for n, fn in test_cases:
        pred = func(n)
        print(f'n: {n}   pred: {pred}   gold: {fn}')
        assert pred == fn
    print('All test passed!')
