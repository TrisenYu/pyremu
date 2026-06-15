#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

class coord {
public:
	int x, y;
	coord() {
		x = y = 0;
	}
	coord(int xx, int yy) {
		x = xx;
        y = yy;
	}
};

std::vector<std::vector<std::string>> res;
coord rec[16];
int n;

bool check_pos(int pos, unsigned int v) {
	return (v >> pos) & 1;
}

void set_pos(int pos, unsigned int *v) {
	*v |= (1u << pos);
}

void unset_pos(int pos, unsigned int *v) {
	*v &= ~(1u << pos);
}

unsigned int row = 0, col = 0, xl = 0, xr = 0;
void dfs(int pos, int num) {
    if (num >= n) {
        // 检查构型有没有重复
        char g[10][10] = {0};
        memset(g, '.', sizeof(g));
        std::vector<std::string> tmp;
        for (int i = 0; i < num; i ++) {
            g[rec[i].x][rec[i].y] = 'Q';
        }
        for (int i = 0; i < num; i ++) {
            g[i][n] = 0;
            tmp.emplace_back(std::string(g[i]));
        }
        res.emplace_back(tmp);
        return;
    }
    for (int curr = pos; curr < n * n; curr ++) {
        int i = curr / n,
            j = curr % n;
        if (__builtin_expect(
            check_pos(i, row) || check_pos(j, col) || 
            check_pos(i+j, xl) || 
            check_pos(i-j+n, xr), 1)
        ) {
            continue;
        }
        set_pos(i, &row);
        set_pos(j, &col);
        set_pos(i+j, &xl);
        set_pos(i-j+n, &xr);
        rec[num] = coord(i, j);
        dfs(curr+1, num+1);
        unset_pos(i, &row);
        unset_pos(j, &col);
        unset_pos(i+j, &xl);
        unset_pos(i-j+n, &xr);
    }
}

int main(int argc, char *argv[]) {
	std::cin >> n;
	n %= 10;
    dfs(0, 0);
    for (auto &t: res) {
        for (auto &s: t) {
            std::cout << s << '\n';
        }
        std::cout << "-=-=-=-=-=-=-=-=-\n";
    }
	return 0;
}