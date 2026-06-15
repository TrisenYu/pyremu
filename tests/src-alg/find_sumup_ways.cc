#include <iostream>
#include <vector>

int goal;
std::vector<std::vector<int>> res;
std::vector<int> rec;
void dfs(int sumup, int curr, std::vector<int> &arr) {
	if (curr > arr.size() || sumup > goal) {
		return;
	}
	if (sumup == goal) {
		res.emplace_back(rec);
		return;
	}
	for (int i = curr; i < arr.size(); i++) {
		int x = arr[i];
		if (x > goal || sumup + x > goal) {
			continue;
		}
		/* 一次性用完当前的机会 */ 
		int ref = (goal - sumup) / x;
		for (int j = 0; j < ref; j++) {
			rec.emplace_back(x);
		}
		while (ref) {
            // 然后来找
			dfs(sumup + ref * x, i + 1, arr);
			ref--;
			rec.pop_back();
		}
	}
}
std::vector<std::vector<int>> combinationSum(std::vector<int> &inp, int target) {
	goal = target;
	dfs(0, 0, inp);
	return res;
}

int main(int argc, char *argv[]) {
    return 0;
}