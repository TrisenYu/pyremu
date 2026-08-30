/*
------------------------------------------------------------------------------
  By Bob Jenkins, 1989, in relation to my Master's Thesis.  Public Domain.
  This module handles entire knots.
------------------------------------------------------------------------------
*/
#include <stdlib.h>
#include <string.h>

#include "dllink.h"
#include "knot.h"
#include "standard.h"

/*
------------------------------------------------------------------------------
  Assumes the dllinks all form cycles
  This displays each string once by listing its crossings in order.
  After that, it displays the HANDedness of each crossing.
------------------------------------------------------------------------------
*/
void k_show(Link *link) {
	word i, j;
	word tab[MAXCROSS];
	dllink *count, *start;

	for (i = 0; i < link->num_crossings; ++i) {
		tab[i] = 0;
	}
	for (j = 0; j < 2; ++j) {
		for (i = 0; i < link->num_crossings; ++i) {
			if (!link->data[i].hand
				|| !((tab[i] == 0) || (tab[i] == 1) || (tab[i] == 10))) {
				continue;
			}
			if (tab[i] != 10) {
				count = link->data[i].o;
			} else {
				count = link->data[i].u;
			}
			start = count;

			if (link->data[count->c].o == count) {
				tab[count->c] += 10;
			} else {
				tab[count->c] += 1;
			}
			if (count != 0) {
				count = count->z;
			}
			while (count != start) {
				if (link->data[count->c].o == count) {
					tab[count->c] += 10;
				} else {
					tab[count->c] += 1;
				}

				count = count->z;
			}
		}
	}
}

/*
------------------------------------------------------------------------------
  Assumes the file given by the user exists and contains a legal knot.
------------------------------------------------------------------------------
*/
boolean k_read(Link **link, char *f) {
	// char name[20];
	word links, startwhere, where, startover, over, i, j;
	int num_crossings, pos;
	crossing *kk = (crossing *)0;
	crossing k[MAXCROSS];

	for (i = 0; i < MAXCROSS; i++) {
		k[i].hand = 0;
		k[i].o	  = (dllink *)0; /* 零初始化, 避免对未出现在穿越串中的交叉做未定义读 */
		k[i].u	  = (dllink *)0;
	}

	if (f == 0) {
		goto fail;
	}
	sscanf(f, "%d%n", &links, &pos);
	f += pos;
	for (i = 0; i < links; ++i) /* how many pieces of string */
	{
		sscanf(f, "%d%n", &num_crossings, &pos);
		f += pos;
		sscanf(f, "%d %d%n", &startwhere, &startover, &pos);
		f += pos;
		if (startover == 1) {
			l_add((dllink *)0, startwhere, &k[startwhere].o);
		} else {
			l_add((dllink *)0, startwhere, &k[startwhere].u);
		}
		for (j = 1; j < num_crossings; ++j) {
			sscanf(f, "%d %d%n", &where, &over, &pos);
			f += pos;

			/* check that OVER is legal */
			if ((over != 1) && (over != -1)) {
				goto fail;
			}
			dllink **goal = &k[where].u;
			if (over == 1) {
				goal = &k[where].o;
			}
			if (startover == 1) {
				l_add(k[startwhere].o, where, goal);
			} else {
				l_add(k[startwhere].u, where, goal);
			}
		}
	}
	num_crossings = 0;
	while (sscanf(f, "%d %d%n", &where, &over, &pos) == 2) {
		f += pos;
		k[where].hand = over;
		if (where > num_crossings) {
			num_crossings = where;
		}
	}
	++num_crossings;

	kk = (crossing *)malloc(sizeof(crossing) * num_crossings);
	for (i = 0; i < num_crossings; ++i) {
		kk[i] = k[i];
	}

	/* check that every crossing has an overpass and underpass */
	for (i = 0; i < num_crossings; ++i) {
		if (!k[i].o || !k[i].u || !(k[i].hand == 1 || k[i].hand == -1)) {
			goto fail;
		}
	}

	*link				   = (Link *)malloc(sizeof(Link));
	(*link)->num_crossings = num_crossings;
	(*link)->data		   = kk;
	return TRUE;

fail:
	/* 非法输入: 释放 k_read 已分配但尚未移交的部分内存 (dllink 节点 + crossing 数组) */
	for (i = 0; i < MAXCROSS; i++) {
		free((char *)k[i].o);
		free((char *)k[i].u);
	}
	free((char *)kk);
	return FALSE;
}

/* 释放 k_read 构造的 Link: 每个交叉的 over/under dllink 节点 + crossing 数组 + Link 本身。
 * 每个 dllink 节点在 k_read 中恰好存入一个 crossing 的 o 或 u 字段, 故逐个 free 不会重复。 */
void k_free(Link *link) {
	int i;

	if (!link) {
		return;
	}
	for (i = 0; i < link->num_crossings; ++i) {
		free((char *)link->data[i].o);
		free((char *)link->data[i].u);
	}
	free((char *)link->data);
	free((char *)link);
}
