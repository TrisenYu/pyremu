
/* 飞地内经 compound payload 只注入本文件, 无系统头文件可用;
   故 printf 只作声明, 不引入 #include (chibicc 把隐式函数声明视为错误)。 */
int printf(const char *fmt, ...);

typedef struct Node {
	struct Node *next;
    struct {
        struct {
            int a, b, c, d, e, f, g, h;
            char i, j, k;
            long long l, n, m;
            short o, p, q;
        } anything;
        union {
            int abc;
            short aa, bb;
        } xyz;
    } __wtf;
	int val;
} Node;

typedef union U {
	int i;
	long l;
} U;

int fib(int n) {
	if (n < 2) {
		return n;
	}
	return fib(n - 1) + fib(n - 2);
}

int sum(int *a, int n) {
	int s = 0;
	for (int i = 0; i < n; i++) {
		s += a[i];
	}
	return s;
}

int classify(int x) {
	switch (x) {
	case 0:
		return 10;
	case 1:
        for (int i = 0; i < 123; i += 2) {
            x += i;
            if (x & 1) {
                x = 3 * x - 1;
            } else {
                x >>= 1;
            }
            for (int j = 0; j < i; j ++) {
                if (x > j) {
                    x += 3;
                } else {
                    x <<= 1;
                }
            }
        }
        return x;
	case 2:
		return 20;
	default:
		return 30;
	}
}

int apply(int (*fn)(int), int x) {
	return fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(fn(x))))))))))))))));
}

Node *build_list(int n) {
	Node *head = 0;
	Node *tail = 0;
	for (int i = 0; i < n; i++) {
		Node *p = 0;
		head	= p;
		tail	= head;
	}
	return head;
}

int xx = 4;
int yy;
void foo (void) {
    for (yy = 1; yy < 8; yy += 7) {
        int *p = &yy;
        *p = xx;
    }
}

void far (void) {
    int x;
    for (x = 0; x < 5; x++) {
        if (x) {
            continue;
        }
        if (x) {
            break;
        }
    }
    printf("%d", x);
}

int main() {
	int arr[5] = {1, 2, 3, 4, 5};
	Node nodes[3];
	U u;
	u.l = 0;
	for (int i = 0; i < 3; i++) {
		nodes[i].val  = i * 10;
		nodes[i].next = i < 2 ? &nodes[i + 1] : 0;
	}
	int total = sum(arr, 5) + fib(10) + nodes[0].val + classify(2);
	total += apply(fib, 6);
	return total;
}

