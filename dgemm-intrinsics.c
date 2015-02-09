#include <stdlib.h>
#include <string.h>
#include <emmintrin.h>
#include <pmmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm, with SSE intrinsics.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  __m128d cij, bcol, arow, c1, arow2, c2; 
  /* For each row i of A */
 for (int j = 0; j < N; ++j)
  {
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=2) 
    {
      /* Compute C(i,j) */
      cij = _mm_load_pd(C+i+j*lda);
      for (int k = 0; k < K; k+=2)
      {
	bcol = _mm_load_pd(B+k+j*lda);
	arow = _mm_loadl_pd(arow,A+i+k*lda);
	arow = _mm_loadh_pd(arow,A+i+(k+1)*lda);
	c1 = _mm_mul_pd(arow,bcol);

	arow2 = _mm_loadl_pd(arow2,A+i+1+k*lda);
	arow2 = _mm_loadh_pd(arow2,A+i+1+(k+1)*lda);
	c2 = _mm_mul_pd(arow2,bcol);

        cij = _mm_add_pd(_mm_hadd_pd(c1,c2),cij);
     }
     _mm_store_pd(C+i+j*lda,cij);
   }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  // pad matrices if dimensions are odd, to avoid alignment issues later
  double *newA, *newB, *newC;
  int md;
  newA = (double*)calloc((lda+1)*(lda+1),sizeof(double));
  newB = (double*)calloc((lda+1)*(lda+1),sizeof(double));
  newC = (double*)calloc((lda+1)*(lda+1),sizeof(double));

  if (lda % 2 != 0)
  {
    for (int r = 0; r < lda; ++r)
    {
      memcpy(newA+r*(lda+1),A+r*lda,lda*sizeof(double));
      memcpy(newB+r*(lda+1),B+r*lda,lda*sizeof(double));
      memcpy(newC+r*(lda+1),C+r*lda,lda*sizeof(double));
    }
    md = lda + 1;
  } else {
    memcpy(newA,A,lda*lda*sizeof(double));
    memcpy(newB,B,lda*lda*sizeof(double));
    memcpy(newC,C,lda*lda*sizeof(double));
    md = lda;
  }
    

  /* For each block-row of A */ 
  for (int i = 0; i < md; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < md; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < md; k += BLOCK_SIZE)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, md-i);
	int N = min (BLOCK_SIZE, md-j);
	int K = min (BLOCK_SIZE, md-k);

	/* Perform individual block dgemm */
	do_block(md, M, N, K, newA + i + k*md, newB + k + j*md, newC + i + j*md);
      }

  // remove padding
  if (lda % 2 != 0)
  {
    for (int r = 0; r < lda; ++r)
    {
      memcpy(A+r*lda,newA+r*(lda+1),lda*sizeof(double));
      memcpy(B+r*lda,newB+r*(lda+1),lda*sizeof(double));
      memcpy(C+r*lda,newC+r*(lda+1),lda*sizeof(double));
    }
  } else {
    memcpy(A,newA,lda*lda*sizeof(double));
    memcpy(B,newB,lda*lda*sizeof(double));
    memcpy(C,newC,lda*lda*sizeof(double));
  }

  free(newA);
  free(newB);
  free(newC);
}
