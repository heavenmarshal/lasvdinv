#include <stdlib.h>
#include "linalg.h"
#include "matrix.h"
#include "matrixext.h"
#include "gp_sep.h"
#include "covar_sep.h"
#include "myjmlegpsep.h"
#include "lasvdgp.h"

static const double dab1 = 1.5;
static const double sqreps = 1.490116119384766E-8;
static double gab[2] = {0.0, 0.0};

static double * new_p_subvector(int *p, double *v, unsigned int len)
{
  int i;
  double *newv;
  newv = new_vector(len);
  for(i=0; i<len; ++i)
    newv[i] = v[p[i]];
  return newv;
}
static void flipvector(double *vec, unsigned int len)
{
  int i;
  for(i=0; i<len; ++i)
    vec[i] = -vec[i];
}
static void const_vector(double *vec, double scalar, unsigned int len)
{
  int i;
  for(i=0; i<len; ++i)
    vec[i] = scalar;
}

static void selectpoint(GPsep* gpsep, int nparam, int nfea, int *feaidx, double *param,
		 double **design, double *resp)
{
  int addidx, n0, cnfea, *feastart;
  double **xcand, *criter, **xadd, zadd;
  n0 = gpsep->n;
  cnfea = nfea - n0;
  feastart = feaidx + n0;
  xcand = new_p_submatrix_rows(feastart, design, cnfea, nparam, 0);
  criter = new_vector(cnfea);

  alcGPsep(gpsep, cnfea, xcand, 1, &param, 0, criter);
  flipvector(criter,cnfea);
  quick_select_index(criter,feastart, cnfea, 1);
  xadd = new_p_submatrix_rows(feastart, design, 1, nparam, 0);
  zadd = resp[feastart[0]];
  updateGPsep(gpsep, 1, xadd, &zadd, 0);
  delete_matrix(xcand);
  delete_matrix(xadd);
  free(criter);
}

GPsep* fitlagpsep(unsigned int nparam, unsigned int ndesign, unsigned int n0,
		  unsigned int nn, unsigned int nfea, unsigned int every,
		  double *param, double **design, double *resp, double gstart)
{
  GPsep *gpsep;
  int i, dits, gits, dconv, niter;
  int *feaidx, seqs[2] = {nfea, n0};
  double ds, dmin, dmax, dab2, dab[2], grange[2]={sqreps,gstart};
  double *dstart, *ddmin, *ddmax, **subdes, *subresp;
  feaidx = nearest_indices(nparam, 1, &param, ndesign, design,
			   seqs, 2);
  subdes = new_p_submatrix_rows(feaidx, design, n0, nparam, 0);
  subresp = new_p_subvector(feaidx, resp, n0);
  getDs(subdes, n0, nparam, &ds, &dmin, &dmax, &dab2);
  dstart = new_const_vector(ds, nparam);
  ddmin = new_const_vector(dmin, nparam);
  ddmax = new_const_vector(dmax, nparam);
  dab[0] = dab1;
  dab[1] = dab2;
  gpsep = newGPsep(nparam, n0, subdes, subresp, dstart, gstart, 1);
  jmleGPsep(gpsep, 100, ddmin, ddmax, grange, dab, gab, 0, &dits, &gits, &dconv, 1);
  niter = nn-n0;
  for(i=1; i<=niter; ++i)
  {
    selectpoint(gpsep, nparam, nfea, feaidx, param, design, resp);
    if(i % every == 0)
    {
      getDs(gpsep->X, gpsep->n, nparam, &ds, &dmin, &dmax, &dab2);
      dab[1] = dab2;
      const_vector(ddmin,dmin,nparam);
      const_vector(ddmax,dmax,nparam);
      jmleGPsep(gpsep, 100, ddmin, ddmax, grange, dab, gab, 0, &dits, &gits, &dconv, 1);
    }
  }
  delete_matrix(subdes);
  free(subresp);
  free(ddmin);
  free(ddmax);
  return gpsep;
}

void scalargpgradhess(GPsep* gpsep, int nparam, double *param,
		      double *grad, double **hess)
{
  int i, j, k, nn;
  double gradi, hessi, *corr, *KiZ, *d, **X;
  nn = gpsep->n;
  corr = new_vector(nn);
  X = gpsep->X;
  d = gpsep->d;
  covar_sep(nparam, &param, 1, X, nn, d, &corr);
  KiZ = gpsep->KiZ;
  for(i = 0; i < nparam; ++i)
  {
    gradi = 0.0;
    hessi = 0.0;
    for(k = 0; k < nn; ++k)
    {
      gradi += 2.0*(X[k][i] - param[i])/d[i]*corr[k]*KiZ[k];
      hessi += 2.0*corr[k]*(2.0*sq((X[k][i]-param[i])/d[i])-1.0/d[i])*KiZ[k];
    }
    grad[i] = gradi;
    hess[i][i] = hessi;
    for(j = 0; j< i; ++j)
    {
      hessi = 0.0;
      for(k = 0; k < nn; ++k)
	hessi += 4.0 *corr[k] *(X[k][i]-param[i])/d[i]*(X[k][j]-param[j])/d[j]*KiZ[k];
      hess[i][j] = hessi;
      hess[j][i] = hessi;
    }
  }
  free(corr);
}
