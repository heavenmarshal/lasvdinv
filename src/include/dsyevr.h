#ifndef __DSYEVR_H__
#define __DSYEVR_H__
#define INERR 9999
#include"linalg.h"

enum CBLAS_JOBZ {CblasValue=201, CblasBoth=202};
enum CBLAS_RANGE {CblasAll=211, CblasReal=212, CblasInt=213};

#define dsyevr dsyevr_
extern void dsyevr(const char *jobz, const char *range, const char *uplo,
		     const int *n, double *a, const int *lda,
		     const double *vl, const double *vu,
		     const int *il, const int *iu,
		     const double *abstol, int *m, double *w,
		     double *z, const int *ldz, int *isuppz,
		     double *work, const int *lwork,
		     int *iwork, const int *liwork,
		     int *info);
#define dtrtrs dtrtrs_
extern void dtrtrs(const char* uplo, const char* trans,
		     const char* diag, const int* n, const int *nrhs,
		     const double *a, const int *lda,
		     double *b, const int* ldb, int *info);

int linalg_dsyevr(enum CBLAS_JOBZ jobz, enum CBLAS_RANGE range, int n, double **A,
		    int lda, double vl, double vu, int il, int iu, double abstol,
		    int *m, double *values, double **vectors, int ldvec);

int linalg_dtrtrs(enum CBLAS_TRANSPOSE TA, enum CBLAS_DIAG DIAG, int n,
		    int nrhs, double **A, int lda, double **b, int ldb);
#endif
