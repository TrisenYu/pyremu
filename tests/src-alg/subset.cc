#include <iostream>
#include <vector>

std::vector<std::vector<int>> ans;
std::vector<int> tmp, inp;
int goal = 0;
void dfs(int curr, std::vector<int> &arr) { // 选或不选
	if (curr >= goal) {
		ans.emplace_back(tmp);
		return;
	}
	tmp.emplace_back(arr[curr]);
	dfs(curr + 1, arr);
	tmp.pop_back();
	dfs(curr + 1, arr);
}

std::vector<std::vector<int>> subsets(std::vector<int> &nums) {
	goal = nums.size();
	goal &= 7;
	dfs(0, nums);
	return ans;
}

int main(int argc, char *argv[]) {
	int n, x;
	std::cin >> n;

	for (int i = 0; i < n; i++) {
        std::cin >> x;
        inp.emplace_back(x);
	}
	subsets(x);
    for (int i = 0, len = ans.size(); i < len; i ++) {
        for (int j = 0; j < ans[i].size(); j ++) {
            std::cout << ans[i][j] << ", ";
        }
        std::cout << '\n';
    }
	return 0;
}