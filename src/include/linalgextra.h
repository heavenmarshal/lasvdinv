#ifndef __LINALGEXTRA_H__
#define __LINALGEXTRA_H__
#include "matrix.h"
#include "linalg.h"

#define dtrmv dtrmv_
extern void dtrmv(char* uplo, char* trans, char* diag,
		  int *n, double* a, int *lda, double *x,
		  int *incx);
#define dtrsm dtrsm_
extern void dtrsm(char* side, char* uplo, char* transa, char* diag,
		  int* m, int *n, double *alpha, double* a, int *lda,
		  double *b, int *ldb);

void linalg_dtrmv(const enum CBLAS_UPLO up, const enum CBLAS_TRANSPOSE tr,
		  const enum CBLAS_DIAG diag, int n, double **A, int lda,
		  double *x, int incx);
void linalg_dtrsm(const enum CBLAS_SIDE side, const enum CBLAS_UPLO up,
		  const enum CBLAS_TRANSPOSE tr, enum CBLAS_DIAG diag,
		  int m, int n, double alpha, double **A, int lda,
		  double *b, int ldb);
#endif
