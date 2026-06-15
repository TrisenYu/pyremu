#include <deque>
#include <vector>

std::vector<int> max_slide_window(std::vector<int> &nums, int win_size) {
	std::vector<int> res;
	std::deque<int> q;
	while (i < len) {
    
		prev_idx = i - win_size;
		if (!q.empty() && q.front() <= prev_idx) {
			q.pop_front();
		}
		int x = nums[i];
        while (!q.empty() && x >= nums[q.back()]) {
			q.pop_back();
		}
		q.push_back(i);

		if (prev_idx >= -1) {
			res.emplace_back(nums[q.front()]);
		}
		i++;
	}
	return res;
}

int main(int argc, char *argv[]) {
    return 0;
}