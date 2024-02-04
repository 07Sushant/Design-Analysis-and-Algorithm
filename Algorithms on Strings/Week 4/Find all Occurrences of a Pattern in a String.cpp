#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

static vector<int> compute_prefix_fun(const string &p)
{
	int n = (int)p.size();
	vector<int> pref(n);
	int border = 0;

	pref[0] = 0;

	for (int i = 1; i < n; i++) {
		while (border > 0 && p[i] != p[border]) {
			border = pref[border - 1];
		}

		if (p[i] == p[border]) {
			border++;
		} else {
			border = 0;
		}
		pref[i] = border;
	}

	return pref;
}

static vector<int> find_pattern(const string &pattern, const string &text)
{
	vector<int> result;
	string s = pattern + "$" + text;
	vector<int> pref = compute_prefix_fun(s);

	for (size_t i = pattern.size() + 1; i < s.size(); i++) {
		if (pref[i] == pattern.size()) {
			result.push_back(i - 2*pattern.size());
		}
	}
	cout << endl;

	return result;
}

int main()
{
	string pattern, text;
	cin >> pattern;
	cin >> text;

	vector<int> result = find_pattern(pattern, text);

	for (int i = 0; i < result.size(); ++i) {
		cout << result[i] << " ";
	}
	cout << endl;

	return 0;
}