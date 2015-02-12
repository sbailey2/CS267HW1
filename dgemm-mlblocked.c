#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <string.h>

const char* dgemm_desc = "Blocked dgemm with intrinsics, row order packing for A, col order packing for B, C, and two levels of blocking";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 40
#endif

// BLOCK_SIZE2 must be a multiple of BLOCK_SIZE`
#if !defined(BLOCK_SIZE2)
#define BLOCK_SIZE2 200
#endif

#if !defined(min)
#define min(a,b) (((a)<(b))?(a):(b))
#endif

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int M, int N, int K, double* A, double* B, double* C)
{
  __m128d cij, bcol, arow, c1, arow2, c2; 
  /* For each row i of A */
  for (int j = 0; j < N; ++j)
  {
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=2) 
    {
      /* Compute C(i,j) */
      cij = _mm_load_pd(C+i+j*M);
      for (int k = 0; k < K; k+=2)
      {
	bcol = _mm_load_pd(B+k+j*K);
        arow = _mm_load_pd(A+k+i*K);
	c1 = _mm_mul_pd(arow,bcol);

        arow2 = _mm_load_pd(A+k+(i+1)*K);
	c2 = _mm_mul_pd(arow2,bcol);

        cij = _mm_add_pd(_mm_hadd_pd(c1,c2),cij);
      }
    _mm_store_pd(C+i+j*M,cij);
    }
  }
}

/* If lda is not even, pad matrix with extra row and extra column. Then
   copies matrix old into matrix new where each block of BLOCK_SIZE occupies
   continugous memory, in column order */
void pad_and_pack_col(int lda, double *new, double *old)
{
  int blockrowind, blockcolind, rowind, colind, cwidth, rwidth;
  int newlda = lda;

  if (lda % 2 != 0)
    newlda = lda + 1; // if odd dimensioned matrix, pad

  for (int j = 0; j < lda; ++j)
    for (int i = 0; i < lda; ++i)
    {
      blockrowind = i/BLOCK_SIZE;
      blockcolind = j/BLOCK_SIZE;
      rowind = i-blockrowind*BLOCK_SIZE;
      colind = j-blockcolind*BLOCK_SIZE;
      cwidth = min (BLOCK_SIZE, newlda-blockcolind*BLOCK_SIZE);
      rwidth = min (BLOCK_SIZE, newlda-blockrowind*BLOCK_SIZE);
      new[blockcolind*BLOCK_SIZE*newlda+blockrowind*BLOCK_SIZE*cwidth+colind*rwidth+rowind] = old[i+j*lda];
    } 
}

/* If lda is not even, then remove the extra row and column padding. Then
   copy matrix packed back into unpacked in column major order. */
void unpad_and_unpack_col(int lda, double *packed, double *unpacked)
{
  int blockrowind, blockcolind, rowind, colind, cwidth, rwidth;
  int newlda = lda;

  if (lda % 2 != 0)
    newlda = lda + 1;

  for (int j = 0; j < lda; ++j)
    for (int i = 0; i < lda; ++i)
    {
      blockrowind = i/BLOCK_SIZE;
      blockcolind = j/BLOCK_SIZE;
      rowind = i-blockrowind*BLOCK_SIZE;
      colind = j-blockcolind*BLOCK_SIZE;
      cwidth = min (BLOCK_SIZE, newlda-blockcolind*BLOCK_SIZE);
      rwidth = min (BLOCK_SIZE, newlda-blockrowind*BLOCK_SIZE);
      unpacked[i+j*lda] = packed[blockcolind*BLOCK_SIZE*newlda+blockrowind*BLOCK_SIZE*cwidth+colind*rwidth+rowind];
    }
} 

/* Same as pad_and_pack_col, except each block is packed in row order */
void pad_and_pack_row(int lda, double *new, double *old)
{
  int blockrowind, blockcolind, rowind, colind, cwidth, rwidth;
  int newlda = lda;

  if (lda % 2 != 0)
    newlda = lda + 1; // if odd dimensioned matrix, pad

  for (int j = 0; j < lda; ++j)
    for (int i = 0; i < lda; ++i)
    {
      blockrowind = i/BLOCK_SIZE;
      blockcolind = j/BLOCK_SIZE;
      rowind = i-blockrowind*BLOCK_SIZE;
      colind = j-blockcolind*BLOCK_SIZE;
      cwidth = min (BLOCK_SIZE, newlda-blockcolind*BLOCK_SIZE);
      rwidth = min (BLOCK_SIZE, newlda-blockrowind*BLOCK_SIZE);
      new[blockrowind*BLOCK_SIZE*newlda+blockcolind*BLOCK_SIZE*rwidth+rowind*cwidth+colind] = old[i+j*lda];
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  double *newA, *newB, *newC;
  int newlda = lda;
  if (lda % 2 != 0)
    newlda = lda + 1;
  newA = (double*)_mm_malloc(sizeof(double)*newlda*newlda,128);
  newB = (double*)_mm_malloc(sizeof(double)*newlda*newlda,128);
  newC = (double*)_mm_malloc(sizeof(double)*newlda*newlda,128);
  memset(newA,0,sizeof(double)*newlda*newlda);
  memset(newB,0,sizeof(double)*newlda*newlda);
  memset(newC,0,sizeof(double)*newlda*newlda);
  pad_and_pack_row(lda,newA,A);
  pad_and_pack_col(lda,newB,B);
  pad_and_pack_col(lda,newC,C);

  for (int j = 0; j < newlda; j += BLOCK_SIZE2)
    for (int i = 0; i < newlda; i += BLOCK_SIZE2)
      for (int k = 0; k < newlda; k += BLOCK_SIZE2)
      {
	int M = min (BLOCK_SIZE2, newlda-i);
        int N = min (BLOCK_SIZE2, newlda-j);
	int K = min (BLOCK_SIZE2, newlda-k);

	for (int s = j; s < j + N; s += BLOCK_SIZE)
	  for (int r = i; r < i + M; r += BLOCK_SIZE)
	    for (int t = k; t < k + K; t += BLOCK_SIZE)
	    {
	      int R = min (BLOCK_SIZE, newlda-r);
	      int S = min (BLOCK_SIZE, newlda-s);
	      int T = min (BLOCK_SIZE, newlda-t);
	      do_block(R, S, T, newA + r*newlda + t*R,
		newB + s*newlda + t*S, newC + s*newlda + r*S);
	    }
      }

  unpad_and_unpack_col(lda,newC,C);

  _mm_free(newA);
  _mm_free(newB);
  _mm_free(newC); 
}
