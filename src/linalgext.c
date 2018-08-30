#include <stdlib.h>
#include "linalgext.h"

const char jobz = 'S';

void linalg_dgesdd(double **X, int nrow, int ncol,
		   double *s, double *u, double **vt)
{
  int info = 0, lwork = -1;
  int nsv = nrow<ncol? nrow : ncol;
  int *iwork = (int *) malloc(8*(size_t)(nsv)*sizeof(int));
  double tmp, *work;

  dgesdd(&jobz,&nrow,&ncol,*X,&nrow,s,u,&nrow,
	 *vt,&nsv,&tmp,&lwork,iwork, &info);
  /* if(info != 0) */
  /*   error("error code %d from Lapack routine '%s'", info, "dgesdd"); */
  lwork = (int) tmp;

  work = (double*) malloc(lwork * sizeof(double));

  dgesdd(&jobz,&nrow,&ncol,*X,&nrow,s,u,&nrow,
	 *vt,&nsv,work,&lwork,iwork,&info);
  free(work);
  free(iwork);
  /* if(info != 0) */
  /*   error("error code %d from Lapack routine '%s'", info, "dgesdd"); */
}

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
