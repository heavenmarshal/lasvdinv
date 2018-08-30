#include "lasvdgp.h"
#include "matrix.h"
#include "linalg.h"
#include "matrixext.h"
#include "linalgext.h"
#include "quantile.hpp"
#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include "myjmlegpsep.h"
#include "rhelp.h"

static const double dab1 = 1.5;
static const double numdab2 = 3.907364;
static const double quanp = 0.1;
static const double sqreps = 1.490116119384766E-8;
static double gab[2] = {0.0, 0.0};

void getDs(double **X, unsigned int n, unsigned int m,
	   double *dstart, double *dmin, double *dmax, double *dab2)
{
  assert(X);
  assert(dstart);
  int distlen = n*(n-1)/2, poslen;
  double *distvec = new_vector(distlen);
  double ddmin, ddmax;
  distance_sym_vec(X,n,m,distvec);
  poslen = remove_nonpos(distvec,distlen);
  *dstart = quantile(distvec,quanp,poslen);
  if( dmin || dmax || dab2)
  {
    vector_minmax(distvec,poslen,&ddmin,&ddmax);
    ddmin*=0.5;
    ddmin = ddmin>sqreps? ddmin:sqreps;
    if(dmin) *dmin = ddmin;
    if(dmax) *dmax = ddmax;
    if(dab2) *dab2 = numdab2/ddmax;
  }
  free(distvec);
}

lasvdGP* newlasvdGP(double* xpred, double **design, double **resp,
		    unsigned int N, unsigned int m, unsigned int tlen,
		    unsigned int nn, unsigned int n0, unsigned int nfea,
		    unsigned int nsvd, unsigned int nadd, double frac,
		    double gstart)
{
  assert(design);
  assert(resp);
  assert(xpred);
  int lsvdi;
  int segs[3] = {nfea,nsvd,n0};
  lasvdGP* lasvdgp = (lasvdGP*) malloc(sizeof(lasvdGP));
  lasvdgp -> N = N;
  lasvdgp -> m = m;
  lasvdgp -> tlen = tlen;
  lasvdgp -> nn = nn;
  lasvdgp -> n0 = n0;
  lasvdgp -> nfea = nfea - n0;
/* n0 points include into the neighbor set in the initial step */
  lasvdgp -> nsvd = nsvd;
  lasvdgp -> nadd = nadd;
  lasvdgp -> frac = frac;
  lasvdgp -> gstart = gstart;
  lasvdgp -> design = design;
  lasvdgp -> resp = resp;
  lasvdgp -> basis = NULL;
  lasvdgp -> reds = NULL;
  lasvdgp -> coeff = NULL;
  /* allocate memory */

  lasvdgp -> xpred = new_dup_vector(xpred,m);
  lasvdgp -> feaidx = nearest_indices(m,1,&xpred,N,design,segs,3);
  lsvdi = nsvd + nn - n0;	/* largest possible  */
  lasvdgp -> svdidx = new_ivector(lsvdi);
  dupiv(lasvdgp -> svdidx, lasvdgp -> feaidx, nsvd);
  lasvdgp -> neigsvdidx = iseq(0,nn-1);
  buildBasis(lasvdgp);
  buildGPseps(lasvdgp);

  return lasvdgp;
}
void deletelasvdGP(lasvdGP* lasvdgp)
{
  assert(lasvdgp -> gpseps);
  int i, nbas;
  GPsep **gpseps = lasvdgp -> gpseps;
  nbas = lasvdgp -> nbas;
  for(i = 0; i < nbas; ++i)
    if(gpseps[i]) deleteGPsep(gpseps[i]);
  free(lasvdgp -> gpseps);

  assert(lasvdgp -> xpred);
  free(lasvdgp -> xpred);
  assert(lasvdgp -> feaidx);
  free(lasvdgp -> feaidx);
  assert(lasvdgp -> svdidx);
  free(lasvdgp -> svdidx);
  assert(lasvdgp -> neigsvdidx);
  free(lasvdgp -> neigsvdidx);

  assert(lasvdgp -> basis);
  free(lasvdgp -> basis);
  assert(lasvdgp -> reds);
  free(lasvdgp -> reds);
  assert(lasvdgp -> coeff);
  delete_matrix(lasvdgp -> coeff);

  free(lasvdgp);
}
void buildBasis(lasvdGP *lasvdgp)
{
  double **resp, **vt;
  double *u, *s;
  int nsvd, tlen, nv, nbas;
  nsvd = lasvdgp -> nsvd;
  tlen = lasvdgp -> tlen;
  nv = nsvd<tlen? nsvd : tlen;
  resp = new_p_submatrix_rows(lasvdgp->svdidx, lasvdgp->resp, lasvdgp->nsvd,
			      lasvdgp->tlen, 0);
  vt = new_matrix(nsvd,nv);
  u = new_vector(tlen * nv);
  s = new_vector(nv);
  linalg_dgesdd(resp,tlen,nsvd,s,u,vt);
  nbas = fracvlen(s,lasvdgp->frac,nv);
  if(lasvdgp->basis) free(lasvdgp->basis);
  lasvdgp->basis = new_vector(tlen * nbas);
  dupv(lasvdgp->basis,u,tlen * nbas);

  if(lasvdgp->reds) free(lasvdgp->reds);
  lasvdgp->reds = new_vector(nbas);
  dupv(lasvdgp->reds, s, nbas);

  if(lasvdgp->coeff) delete_matrix(lasvdgp->coeff);
  lasvdgp->coeff = new_dup_matrix(vt, nsvd, nbas);

  lasvdgp->nbas = nbas;
  lasvdgp->nappsvd = 0;
  delete_matrix(resp);
  delete_matrix(vt);
  free(u);
  free(s);
}
void buildGPseps(lasvdGP *lasvdgp)
{
  int i, nbas = lasvdgp -> nbas;
  double **subdes, *subv;
  double ds, *dstart;
  GPsep **gpseps;

  lasvdgp->gpseps = (GPsep **) malloc(nbas * sizeof(GPsep*));
  gpseps = lasvdgp -> gpseps;

  subdes = new_p_submatrix_rows(lasvdgp->feaidx,lasvdgp->design,lasvdgp->n0,
				lasvdgp->m, 0);
  subv = new_vector(lasvdgp->n0);
  getDs(subdes, lasvdgp->n0, lasvdgp->m, &ds, NULL, NULL, NULL);
  dstart = new_const_vector(ds,lasvdgp->m);
  for(i=0; i<nbas; ++i)
  {
    sub_p_matrix_rows_col(subv,lasvdgp->neigsvdidx,lasvdgp->coeff,i,lasvdgp->n0);
    gpseps[i] = newGPsep(lasvdgp->m, lasvdgp->n0, subdes,
			 subv, dstart, lasvdgp->gstart, 1);
  }
  lasvdgp -> hasfitted = 0;
  delete_matrix(subdes);
  free(subv);
  free(dstart);
}
void jmlelasvdGP(lasvdGP *lasvdgp, unsigned int maxit, unsigned int verb)
{
  double dab[2], grange[2]={sqreps,lasvdgp->gstart};
  double dstart, ddmin, ddmax, dab2;
  double *dmin, *dmax;
  int i, dits, gits, dconv;
  getDs(lasvdgp->gpseps[0]->X,lasvdgp->n0,lasvdgp->m, &dstart, &ddmin, &ddmax,
	&dab2);
  dab[0] = dab1;
  dab[1] = dab2;
  dmin = new_const_vector(ddmin,lasvdgp->m);
  dmax = new_const_vector(ddmax,lasvdgp->m);
  for(i=0; i<lasvdgp->nbas; ++i)
    myjmleGPsep(lasvdgp->gpseps[i], maxit, dmin, dmax,
		grange, dab, gab, verb, &dits,
		&gits, &dconv);
  lasvdgp->hasfitted = 1;
  free(dmin);
  free(dmax);
}
void selectNewPoints(lasvdGP *lasvdgp)
{
  int i, nbas, addidx, isvd, nadd, n0;
  int *feastart;
  double **xcand, *criter, *cordcriter, weight;
  double **xadd, **zadd, *zcord;
  GPsep *gpsep;
  n0 = lasvdgp->n0;
  nbas = lasvdgp->nbas;
  feastart = lasvdgp->feaidx + n0;
  xcand = new_p_submatrix_rows(feastart, lasvdgp->design,
			       lasvdgp->nfea, lasvdgp->m, 0);
  criter = new_zero_vector(lasvdgp->nfea);
  cordcriter = new_vector(lasvdgp->nfea);

  for(i = 0; i < nbas; ++i)
  {
    weight = -sq(lasvdgp->reds[i]);
    gpsep = lasvdgp -> gpseps[i];
    alcGPsep(gpsep,lasvdgp->nfea,xcand,1,&(lasvdgp->xpred),0,cordcriter);
    linalg_daxpy(lasvdgp->nfea,weight, cordcriter,1,criter,1);
  }
  nadd = lasvdgp -> nadd;
  quick_select_index(criter,feastart,lasvdgp->nfea,nadd);
  xadd = new_p_submatrix_rows(feastart, lasvdgp-> design,
			      nadd, lasvdgp->m, 0);
  zadd = new_matrix(nadd,nbas);

  for(i=0; i<nadd; ++i)
  {
    addidx = feastart[i];
    isvd = find_int(lasvdgp->svdidx,addidx,lasvdgp->nsvd);
    if(isvd != -1)
    {
      dupv(zadd[i], lasvdgp->coeff[isvd], nbas);
      lasvdgp -> neigsvdidx[n0] = isvd;
      n0 += 1;
      continue;
    }
    /* else */
    lasvdgp -> svdidx[lasvdgp->nsvd] = addidx;
    /* estimate the coefficient by least squares */
    linalg_dgemv(CblasTrans,lasvdgp->tlen,nbas,1.0,
		 &(lasvdgp->basis), lasvdgp->tlen, lasvdgp->resp[addidx],
		 1,0.0,zadd[i],1);
    divid_vector(zadd[i],lasvdgp->reds,nbas);
    lasvdgp -> neigsvdidx[n0] = lasvdgp->nsvd;
    n0 += 1;
    lasvdgp -> nsvd += 1;
    lasvdgp -> nappsvd += 1;
  }
  lasvdgp -> n0 = n0;
  lasvdgp -> nfea -= nadd;
  /* update the gp models */
  zcord = new_vector(nadd);
  for(i = 0; i < nbas; ++i)
  {
    get_col(zcord,zadd,i,nadd);
    updateGPsep(lasvdgp->gpseps[i],nadd,xadd,zcord,0);
  }
  lasvdgp->hasfitted = 0;
  delete_matrix(xcand);
  delete_matrix(xadd);
  delete_matrix(zadd);
  free(criter);
  free(cordcriter);
  free(zcord);
}
/* space for optimization since design set can be reused */
void renewlasvdGP(lasvdGP* lasvdgp)
{
  int i, nbas;
  assert(lasvdgp->gpseps);
  nbas = lasvdgp -> nbas;
  /* delete old gp models */
  for(i = 0; i < nbas; ++i)
    if(lasvdgp->gpseps[i]) deleteGPsep(lasvdgp->gpseps[i]);
  free(lasvdgp -> gpseps);

  buildBasis(lasvdgp);
  buildGPseps(lasvdgp);
}
void predlasvdGP(lasvdGP* lasvdgp, double* pmean, double* ps2)
{
  int i, n0, tlen, nbas, reslen;
  double **resid, **coeff;
  double *cmean, *cs2, *cdf, *bassq, ress2;
  GPsep **gpseps;

  assert(pmean);
  assert(ps2);
  gpseps = lasvdgp->gpseps;
  n0 = lasvdgp -> n0;
  tlen = lasvdgp -> tlen;
  nbas = lasvdgp -> nbas;
  coeff = new_zero_matrix(nbas,n0);
  for(i=0; i < nbas; ++i)
    linalg_daxpy(n0,lasvdgp->reds[i], gpseps[i]->Z,1,coeff[i],1);
  resid = new_p_submatrix_rows(lasvdgp->feaidx,lasvdgp->resp, n0, tlen, 0);
  linalg_dgemm(CblasNoTrans,CblasTrans,tlen,n0,nbas,-1.0,&(lasvdgp->basis),tlen,
	       coeff,n0,1.0,resid,tlen);
  /* Y-USV^T */
  /* ress2 = var_vector(*resid,(double)(n0*tlen+2), n0*tlen); */
  reslen = n0*tlen;
  ress2 = linalg_ddot(reslen,*resid,1,*resid,1);
  ress2 /= (reslen+2);
  cmean = new_vector(nbas);
  cs2 = new_vector(nbas);
  cdf = new_vector(nbas);
  for(i=0; i<nbas; ++i)
    predGPsep_lite(gpseps[i], 1, &(lasvdgp->xpred), cmean+i, cs2+i, cdf+i,NULL);
  prod_vector(cmean,lasvdgp->reds, nbas);
  prod_vector(cs2,lasvdgp->reds, nbas);
  prod_vector(cs2,lasvdgp->reds, nbas);
  linalg_dgemv(CblasNoTrans,tlen,nbas,1.0,&(lasvdgp->basis),tlen,cmean,1,0.0,pmean,1);
  bassq = new_sq_vector(lasvdgp->basis,tlen*nbas);
  linalg_dgemv(CblasNoTrans,tlen,nbas,1.0,&bassq,tlen,cs2,1,0.0,ps2,1);
  sum_vector_scalar(ps2,ress2,tlen);
  delete_matrix(coeff);
  delete_matrix(resid);
  free(cmean);
  free(cs2);
  free(cdf);
  free(bassq);
}
void iterlasvdGP(lasvdGP* lasvdgp, unsigned int resvdThres,
		 unsigned int every, unsigned int maxit, unsigned int verb)
{
  int i, n0, nn, niter, nadd, nrem;
  nn = lasvdgp -> nn;
  n0 = lasvdgp -> n0;
  nadd = lasvdgp -> nadd;
  niter = ceil_divide(nn-n0,nadd);
  for(i = 1; i <= niter; ++i)
  {
    n0 = lasvdgp->n0;
    nrem = nn - n0;
    nadd = lasvdgp-> nadd;
    nadd = nadd<nrem ? nadd : nrem;
    lasvdgp -> nadd = nadd;
    selectNewPoints(lasvdgp);
    if(lasvdgp -> nappsvd >= resvdThres)
    {
      renewlasvdGP(lasvdgp);
      jmlelasvdGP(lasvdgp,maxit,verb);
      continue;
    }
    if(i % every == 0)
      jmlelasvdGP(lasvdgp,maxit,verb);
  }
  /* finishing off */
  if(lasvdgp->nappsvd > 0)
    renewlasvdGP(lasvdgp);
  if(lasvdgp->hasfitted == 0)
    jmlelasvdGP(lasvdgp, maxit, verb);

}
