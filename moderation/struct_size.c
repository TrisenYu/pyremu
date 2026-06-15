#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef enum { Ident1, Ident2, Ident3, Ident4, Ident5 } Enumeration;

typedef int OneToThirty;
typedef int OneToFifty;
typedef char CapitalLetter;
typedef char String30[31];
typedef int Array1Dim[51];
typedef int Array2Dim[51][51];
typedef int boolean;

struct Record {
	struct Record *PtrComp;
	Enumeration Discr;
	Enumeration EnumComp;
	OneToFifty IntComp;
	String30 StringComp;
};

typedef struct Record RecordType;
typedef RecordType *RecordPtr;

int main(int argc, char *argv[]) {
	printf("%zu\n", sizeof(RecordPtr) * 2);
	return 0;
}