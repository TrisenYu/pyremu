/**
------------------------------------------------------------------------------
  HOMFLY.C
  Main functions for the library
------------------------------------------------------------------------------
*/

#include <stdlib.h>

#include "control.h"
#include "homfly.h"
#include "knot.h"
#include "order.h"
#include "poly.h"
#include "standard.h"

char *homfly_str(char *argv) {
	Poly *answer; /* HOMFLY polynomial for the original link */
	char *out;

	answer = homfly(argv);
	out	   = p_show(answer); /* display the answer */
	if (answer) {
		p_kill(answer);		  /* 释放结果项数组 */
		free((char *)answer); /* 释放结果结构 */
	}

	return out;
}

Poly *homfly(char *argv) {
	Link *link;
	Poly *answer;

	if (!k_read(&link, argv)) { /* read link; 非法 Gauss code */
		return (Poly *)0;
	}
	answer = c_homfly(link);
	k_free(link); /* 释放 Link 及其 dllink 节点 */
	return answer;
}

/**
 * Compute the homfly polynomial and return the result as the polynomial answer.
 */
Poly *c_homfly(Link *link) {
	Instruct *plan; /* list of instructions */
	Poly *answer;

	c_init(); /* initialize variables */

	o_make(link, &plan);						  /* make plan for attacking the link */
	answer = c_follow(plan, link->num_crossings); /* follow the plan */
	free((char *)plan);							  /* 释放指令数组 */

	/* c_init 每次调用都会为这 5 个全局多项式重新分配 term 数组, 用完即释放避免累积泄漏。
   * p_add/p_mult 在下一次调用时都会用新 malloc 覆盖 outp->term, 不会读取旧指针, 故此处释放安全。 */
	free(llplus.term);
	free(lplusm.term);
	free(lminusm.term);
	free(llminus.term);
	free(mll.term);

	return answer;
}
