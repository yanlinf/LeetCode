class Solution {
public:
	bool **dp;
	bool isMatch(string s, string p) {
		dp = new bool *[s.size() + 1];
		for (int i = 0; i <= s.size(); i++)
			dp[i] = new bool[p.size() + 1];

		dp[s.size()][p.size()] = true;
		for (int i = 0; i < s.size(); i++)
			dp[i][p.size()] = false;

		for (int j = p.size() - 1; j >= 0; j--)
			for (int i = s.size(); i >= 0; i--){
				if (p.size() - j >= 2 && p[j + 1] == '*') {
					dp[i][j] = 0;
					for (int k = i; k <= s.size() && (k == i || s[k - 1] == p[j] || p[j] == '.'); k++)
						if (dp[k][j + 2])
							dp[i][j] = true;
				}
				else if (i == s.size())
					dp[i][j] = false;
				else if (p[j] == '.')
					dp[i][j] = dp[i + 1][j + 1];
				else if (p[j] == '*')
					dp[i][j] = false;
				else
					dp[i][j] = (s[i] == p[j]) && dp[i + 1][j + 1];
			}
		
		return dp[0][0];
	}
};