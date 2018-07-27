#include"linalgextra.h"

void linalg_dtrmv(const enum CBLAS_UPLO up, const enum CBLAS_TRANSPOSE tr,
		  const enum CBLAS_DIAG diag, int n, double **A, int lda,
		  double *x, int incx)
{
  char uplo, trans, isdiag;
  uplo = (up==CblasUpper)? 'U':'L';
  trans = (tr==CblasTrans)? 'T':'N';
  isdiag = (diag==CblasUnit)? 'U':'N';
  dtrmv(&uplo, &trans, &isdiag, &n, *A, &lda, x, &incx);
}
void linalg_dtrsm(const enum CBLAS_SIDE side, const enum CBLAS_UPLO up,
		  const enum CBLAS_TRANSPOSE tr, enum CBLAS_DIAG diag,
		  int m, int n, double alpha, double **A, int lda,
		  double *b, int ldb)
{
  char isleft, uplo, trans, isdiag;
  isleft = (side==CblasLeft)? 'L': 'R';
  uplo = (up==CblasUpper)? 'U':'L';
  trans = (tr==CblasTrans)? 'T':'N';
  isdiag = (diag==CblasUnit)? 'U':'N';
  dtrsm(&isleft, &uplo, &trans, &isdiag, &m, &n, &alpha, *A, &lda, b, &ldb);
}
