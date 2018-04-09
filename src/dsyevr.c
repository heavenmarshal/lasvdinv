#include <stdlib.h>
#include "matrix.h"
#include "dsyevr.h"

char uplo = 'U';

int linalg_dsyevr(enum CBLAS_JOBZ jobz, enum CBLAS_RANGE range, int n, double **A,
		  int lda, double vl, double vu, int il, int iu, double abstol,
		  int *m, double *values, double **vectors, int ldvec)
{
  char job, rg;
  int lwork, liwork, info, itmp, *isuppz, *iwork;
  double tmp, *work, **Acopy;
  switch(jobz){
  case CblasValue: job = 'N'; break;
  case CblasBoth: job = 'V'; break;
  default: return INERR;
  }
  switch(range){
  case CblasAll: rg = 'A'; break;
  case CblasReal: rg = 'V'; break;
  case CblasInt: rg = 'I'; break;
  default: return INERR;
  }
  Acopy = new_dup_matrix(A, n, n); /* since dsyevr will destory input matrix */
  isuppz = new_ivector(2*n);
  lwork = -1; liwork = -1;
  dsyevr(&job, &rg, &uplo, &n, *Acopy, &n, &vl, &vu,
	 &il, &iu, &abstol, m, values, *vectors, &ldvec,
	 isuppz, &tmp, &lwork, &itmp, &liwork, &info);
  if(info != 0)
  {
    free(isuppz);
    delete_matrix(Acopy);
    return info;	/* error produced */
  }
  lwork = (int) tmp;
  liwork = itmp;
  work = new_vector(lwork);
  iwork = new_ivector(liwork);

  dsyevr(&job, &rg, &uplo, &n, *Acopy, &n, &vl, &vu,
	 &il, &iu, &abstol, m, values, *vectors, &ldvec,
	 isuppz, work, &lwork, iwork, &liwork, &info);
  free(isuppz);
  free(work);
  free(iwork);
  delete_matrix(Acopy);
  return info;
}

int linalg_dtrtrs(enum CBLAS_TRANSPOSE TA, enum CBLAS_DIAG DIAG, int n,
		    int nrhs, double **A, int lda, double **b, int ldb)
{
  char trans, diag;
  int info;
  switch(TA){
  case CblasNoTrans: trans = 'N'; break;
  case CblasTrans: trans = 'T'; break;
  case CblasConjTrans: trans = 'C'; break;
  default: return INERR;
  }
  switch(DIAG){
  case CblasNonUnit: diag = 'N'; break;
  case CblasUnit: diag = 'U'; break;
  default: return INERR;
  }
  dtrtrs(&uplo, &trans, &diag, &n, &nrhs, *A, &lda,
	 *b, &ldb, &info);
  return info;
}
