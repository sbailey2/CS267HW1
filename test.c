#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "dgemm-blocked.c"
#include "dgemm-packed.c"

int main(int argc, char *argv[])
{
    const int lda = 31;
    double A[lda*lda];
    double B[lda*lda];
    srand(time(0));
    for (int i = 0; i < lda * lda; ++i) {
	A[i] = 2 * drand48() - 1;
	B[i] = 2 * drand48() - 1;
    }

    /*printf("A:\n");
    for (int i = 0; i < lda; ++i) {
	for (int j = 0; j < lda; ++j) {
	    printf("%f ", A[j + i * lda]);
	}
	printf("\n");
    }
    printf("B:\n");
    for (int i = 0; i < lda; ++i) {
	for (int j = 0; j < lda; ++j) {
	    printf("%f ", B[j + i * lda]);
	}
	printf("\n");
	}*/

    double c[lda*lda];
    double z[lda*lda];
    for (int i = 0; i < lda*lda; ++i) {
	c[i] = 0;
        z[i] = 0;
    }

    square_dgemm(lda, A, B, c);
    square_packed_dgemm(lda, A, B, z);

    /*printf("C:\n");
    for (int i = 0; i < lda; ++i) {
	for (int j = 0; j < lda; ++j) {
	    printf("%f ", c[j + i * lda]);
	}
	printf("\n");
    }
    printf("Z:\n");
    for (int i = 0; i < lda; ++i) {
	for (int j = 0; j < lda; ++j) {
	    printf("%f ", z[j + i * lda]);
	}
	printf("\n");
	}*/

    for (int i = 0; i < lda; ++i) {
	for (int j = 0; j < lda; ++j) {
	    if (z[j + i * lda] != c[j + i * lda]) {
		printf("Error: mismatch at (%d,%d)\n", i, j);
		printf("Z=%f,C=%f\n", z[j + i * lda], c[j + i * lda]);
		return 0;
	    }
	}
    }

    printf("Matrices match\n");

    return 0;
}
