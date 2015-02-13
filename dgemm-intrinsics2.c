#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <string.h>

const char* dgemm_desc = "Blocked dgemm with intrinsics, register blocking, inner product version";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 60
#endif

#if !defined(min)
#define min(a,b) (((a)<(b))?(a):(b))
#endif

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. 
 * Assume that block dimensions are divisible by 4 */
static void do_block (int M, int N, int K, double* A, double* B, double* C)
{
  __m128d c1, c2, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, temp1, temp2, temp3; 
  /* For each row i of A */
  for (int j = 0; j < N; ++j)
  {
    /* For each column j of B */ 
    for (int i = 0; i < M; i+=4) 
    {
      /* Compute C(i,j) */
      c1 = _mm_load_pd(C+i+j*M);
      c2 = _mm_load_pd(C+i+2+j*M);
      for (int k = 0; k < K; k+=4)
      {
	b1 = _mm_load_pd(B+k+j*K);
	b2 = _mm_load_pd(B+k+2+j*K);
	
	a1 = _mm_load_pd(A+k+i*K);
	a3 = _mm_load_pd(A+k+(i+1)*K);
	temp1 = _mm_mul_pd(a1,b1);
	temp2 = _mm_mul_pd(a3,b1);
	temp3 = _mm_hadd_pd(temp1,temp2);
	c1 = _mm_add_pd(temp3,c1);

	a2 = _mm_load_pd(A+k+2+i*K);
	a4 = _mm_load_pd(A+k+2+(i+1)*K);
	temp1 = _mm_mul_pd(a2,b2);
	temp2 = _mm_mul_pd(a4,b2);
	temp3 = _mm_hadd_pd(temp1,temp2);
	c1 = _mm_add_pd(temp3,c1);

	a5 = _mm_load_pd(A+k+(i+2)*K);
	a7 = _mm_load_pd(A+k+(i+3)*K);
	temp1 = _mm_mul_pd(a5,b1);
	temp2 = _mm_mul_pd(a7,b1);
	temp3 = _mm_hadd_pd(temp1,temp2);
	c2 = _mm_add_pd(temp3,c2);

	a6 = _mm_load_pd(A+k+2+(i+2)*K);
	a8 = _mm_load_pd(A+k+2+(i+3)*K);
	temp1 = _mm_mul_pd(a6,b2);
	temp2 = _mm_mul_pd(a8,b2);
	temp3 = _mm_hadd_pd(temp1,temp2);
	c2 = _mm_add_pd(temp3,c2);
     }
    _mm_store_pd(C+i+j*M,c1);
    _mm_store_pd(C+i+2+j*M,c2);
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

  if (lda % 4 != 0)
    newlda = (lda/4)*4 + 4; // if odd dimensioned matrix, pad

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

  if (lda % 4 != 0)
    newlda = (lda/4)*4 + 4;

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

  if (lda % 4 != 0)
    newlda = (lda/4)*4 + 4; // if odd dimensioned matrix, pad

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
  if (lda % 4 != 0)
    newlda = (lda/4)*4 + 4;
  newA = (double*)_mm_malloc(sizeof(double)*newlda*newlda,128);
  newB = (double*)_mm_malloc(sizeof(double)*newlda*newlda,128);
  newC = (double*)_mm_malloc(sizeof(double)*newlda*newlda,128);
  memset(newA,0,sizeof(double)*newlda*newlda);
  memset(newB,0,sizeof(double)*newlda*newlda);
  memset(newC,0,sizeof(double)*newlda*newlda);
  pad_and_pack_row(lda,newA,A);
  pad_and_pack_col(lda,newB,B);
  pad_and_pack_col(lda,newC,C);

  for (int j = 0; j < newlda; j += BLOCK_SIZE)
    for (int i = 0; i < newlda; i += BLOCK_SIZE)
      for (int k = 0; k < newlda; k += BLOCK_SIZE)
      {
	int M = min (BLOCK_SIZE, newlda-i);
        int N = min (BLOCK_SIZE, newlda-j);
	int K = min (BLOCK_SIZE, newlda-k);

	/* Perform individual block dgemm */
	do_block(M, N, K, newA + i*newlda + k*M, 
          newB + j*newlda + k*N, newC + j*newlda + i*N);
      }

  unpad_and_unpack_col(lda,newC,C);

  _mm_free(newA);
  _mm_free(newB);
  _mm_free(newC); 
}
