class Solution {
public:
	bool isMatch(string s, string p) {
		if (s.size() == 0 && p.size() == 0)
			return true;
		else if (p.size() == 0)
			return false;

		if (p.size() >= 2 && p[1] == '*') {
			for (int i = 0; i <= s.size() && (i == 0 || s[i - 1] == p[0] || p[0] == '.'); i++)
				if (isMatch(s.substr(i), p.substr(2)))
					return true;
			return false;
		}
		else if (s.size() == 0)
			return false;
		else if (p[0] == '.')
			return isMatch(s.substr(1), p.substr(1));
		else
			return s[0] == p[0] && isMatch(s.substr(1), p.substr(1));
	}
};