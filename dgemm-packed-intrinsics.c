#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <pmmintrin.h>

const char* dgemm_desc = "Blocked dgemm with matrix packing and intrinsics.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 44
#endif

#if !defined(min)
#define min(a,b) (((a)<(b))?(a):(b))
#endif

/* This auxiliary subroutine performs a smaller dgemm operation on a packed matrix
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_packed_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i+j*lda];
      for (int k = 0; k < K; ++k)
	cij += A[i+k*BLOCK_SIZE] * B[k+j*BLOCK_SIZE];
      C[i+j*lda] = cij;
    }
}

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
	bcol = _mm_load_pd(B+k+j*BLOCK_SIZE);
	arow = _mm_loadl_pd(arow,A+i+k*BLOCK_SIZE);
	arow = _mm_loadh_pd(arow,A+i+(k+1)*BLOCK_SIZE);
	c1 = _mm_mul_pd(arow,bcol);

	arow2 = _mm_loadl_pd(arow2,A+i+1+k*BLOCK_SIZE);
	arow2 = _mm_loadh_pd(arow2,A+i+1+(k+1)*BLOCK_SIZE);
	c2 = _mm_mul_pd(arow2,bcol);

        cij = _mm_add_pd(_mm_hadd_pd(c1,c2),cij);
     }
     _mm_store_pd(C+i+j*lda,cij);
   }
  }
 }


/* This auxiliary routine copies the block i,j from matrix X and stores
   it in a continuous piece of memory in Y */
void pack_block(int lda, int i, int j, double *X, double *Y)
{
    int M = min (BLOCK_SIZE, lda-i*BLOCK_SIZE);
    int N = min (BLOCK_SIZE, lda-j*BLOCK_SIZE);
    int blocks = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_square = BLOCK_SIZE * BLOCK_SIZE;
    for (int y = 0; y < M; ++y) {
	for (int x = 0; x < N; ++x) {
	    int yi = (j + i * blocks) * block_square + x + y * BLOCK_SIZE;
	    int xi = (j + i * lda) * BLOCK_SIZE + x + y * lda;
	    Y[yi] = X[xi];
	}
    }
}

/* This auxiliary routine copies the block i,j from the reorganized matrix Y
   and stores it in a row-major format in X */
void unpack_block(int lda, int i, int j, double *X, double *Y)
{
    int M = min (BLOCK_SIZE, lda-i*BLOCK_SIZE);
    int N = min (BLOCK_SIZE, lda-j*BLOCK_SIZE);
    int blocks = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_square = BLOCK_SIZE * BLOCK_SIZE;
    for (int y = 0; y < M; ++y) {
	for (int x = 0; x < N; ++x) {
	    int yi = (j + i * blocks) * block_square + x + y * BLOCK_SIZE;
	    int xi = (j + i * lda) * BLOCK_SIZE + x + y * lda;
	    X[xi] = Y[yi];
	}
    }
}

/* This routine reorganizes the data so that each block is in
 * one contiguous piece of memory */
void pack(int lda, double *X, double *Y)
{
    int blocks = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < blocks; ++i) {
	for (int j = 0; j < blocks; ++j) {
	    pack_block(lda, i, j, X, Y);
	}
    }
}

/* This routine unpackes the data from
 * one contiguous piece of memory in Y */
void unpack(int lda, double *X, double *Y)
{
    int blocks = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < blocks; ++i) {
	for (int j = 0; j < blocks; ++j) {
	    unpack_block(lda, i, j, X, Y);
	}
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
    double *Y, *X, *oldC;
    int oldlda = lda;
    int blocks = (lda + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int block_square = BLOCK_SIZE * BLOCK_SIZE;
    Y = (double*)_mm_malloc(sizeof(double) * blocks * blocks * BLOCK_SIZE * BLOCK_SIZE,128);
    X = (double*)_mm_malloc(sizeof(double) * blocks * blocks * BLOCK_SIZE * BLOCK_SIZE,128);
    if (lda % 2 != 0)
    {
      oldC = C;
      C = (double*) _mm_malloc(sizeof(double)*(lda+1)*(lda+1),128);
      lda = lda + 1;
      for (int i = 0; i < oldlda; ++i)
	for (int j = 0; j < oldlda; ++j)
	  C[j+i*lda] = oldC[j+i*oldlda];
    }
    pack(oldlda, A, X);
    pack(oldlda, B, Y);

  /* For each block-row of A */ 
  for (int i = 0; i < blocks; i += 1)
    /* For each block-column of B */
    for (int j = 0; j < blocks; j += 1)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < blocks; k += 1)
      {
	/* Correct block dimensions if block "goes off edge of" the matrix */
	int M = min (BLOCK_SIZE, oldlda-i*BLOCK_SIZE);
	int N = min (BLOCK_SIZE, oldlda-j*BLOCK_SIZE);
	int K = min (BLOCK_SIZE, oldlda-k*BLOCK_SIZE);

	/* Perform individual block dgemm */
	do_block(lda, M, N, K, X + (i + k * blocks) * block_square,
	  Y + (k + j * blocks) * block_square, C + (i + j*lda) * BLOCK_SIZE);
      }

  if (lda != oldlda)
  {
    for (int i = 0; i < oldlda; ++i)
      for (int j = 0; j < oldlda; ++j)
        oldC[j+i*oldlda] = C[j+i*lda];
    oldC = C;
    _mm_free(oldC);
  }
  _mm_free(X);
  _mm_free(Y);
  
}
