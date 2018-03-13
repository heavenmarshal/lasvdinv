#include<cmath>
#include<cstdlib>
#include<iostream>
#include"lagpLikelihood.hpp"

extern "C"{
  #include "gp_sep.h"
  #include "util.h"
  #include "lasvdgp.h"
  #include "matrix.h"
  #include "matrixext.h"
  #include "linalg.h"
}
#define SDEPS sqrt(DOUBLE_EPS)

static void sq_vector(double* vec, double* vold, unsigned int len)
{
  unsigned int i;
  for(i=0; i<len; ++i)
    vec[i] = vold[i]*vold[i];
}
static void diff_vector(double* vec, double* vdif, unsigned int len)
{
  unsigned int i;
  for(i=0; i<len; ++i)
    vec[i] -= vdif[i];
}

double lagpNaiveLikelihood::evalLogLikelihood(double* param)
{
  double dev, dtlen, loglik;
  double *xpred, *pmean, *ps2;
  dtlen = (double) tlen;
  xpred = new_dup_vector(param,nparam);
  pmean = new_vector(tlen);
  ps2 = new_vector(tlen);	//
  lasvdgp = newlasvdGP(xpred,design,resp,ndesign,nparam,tlen,nn,n0,
		       nfea,nn,1,frac,gstart);
  jmlelasvdGP(lasvdgp,100,0);
  iterlasvdGP(lasvdgp,resvdThres,every,100,0);
  predlasvdGP(lasvdgp,pmean,ps2);
  linalg_daxpy(tlen,-1.0,xi,1,pmean,1);
  dev = linalg_ddot(tlen,pmean,1,pmean,1);
  dev /= dtlen;
  loglik = -0.5 * LOG2PI - 0.5 * dtlen;
  loglik -= 0.5*dtlen*log(dev);
  deletelasvdGP(lasvdgp);
  free(pmean);
  free(ps2);
  free(xpred);
  return loglik;
}

double lagpProfileLikelihood::evalLogLikelihood(double* param)
{
  double dtlen, loglik, upb, sig2esp;
  double *xpred;
  dtlen = (double) tlen;
  xpred = new_dup_vector(param,nparam);
  lasvdgp = newlasvdGP(xpred,design,resp,ndesign,nparam,tlen,nn,n0,
		       nfea,nn,1,frac,gstart);
  jmlelasvdGP(lasvdgp,100,0);
  iterlasvdGP(lasvdgp,resvdThres,every,100,0);
  lagpLikInfo info(lasvdgp->nbas);
  predlasvdGPutil(lasvdgp, xpred, xi, &info);
  upb = info.sse/dtlen;
  sig2esp = Brent_fmin(0.0,upb, nloglikelihood, (void*) &info, SDEPS);
  loglik = nloglikelihood(sig2esp,(void*) &info)+0.5*LOG2PI;
  deletelasvdGP(lasvdgp);
  free(xpred);
  return -loglik;
}
void lagpProfileLikelihood::predlasvdGPutil(lasvdGP* lasvdgp, double *xpred, double* xi,
					    lagpLikInfo* info)
{
  int i, n0, nbas, tlen, reslen;
  double *cmean, *d2c, ress2;
  double **coeff, **resid;
  tlen = lasvdgp -> tlen;
  nbas = lasvdgp -> nbas;
  cmean = new_vector(nbas);
  d2c = new_vector(nbas);
  n0 = lasvdgp -> n0;
  coeff = new_zero_matrix(nbas,n0);
  for(i=0; i<nbas; ++i)
    linalg_daxpy(n0, lasvdgp->reds[i], lasvdgp->gpseps[i]->Z, 1, coeff[i], 1);
  resid = new_p_submatrix_rows(lasvdgp->feaidx,lasvdgp->resp, n0, tlen, 0);
  linalg_dgemm(CblasNoTrans,CblasTrans,tlen,n0,nbas,-1.0,&(lasvdgp->basis),tlen,
	       coeff,n0,1.0,resid,tlen);
  reslen = n0*tlen;
  ress2 = linalg_ddot(reslen,*resid,1,*resid,1);
  ress2 /= (reslen+2);
  for(i=0; i<nbas; ++i)
    predGPsep_lite(lasvdgp->gpseps[i], 1, &xpred, cmean+i, info->cs2+i, d2c+i, NULL);
  sq_vector(info->d2,lasvdgp->reds,nbas);
  dupv(d2c,info->d2,nbas);
  prod_vector(d2c,cmean,nbas);
  // evaluate info->dev
  linalg_dgemv(CblasTrans, tlen, nbas, 1.0, &(lasvdgp->basis), tlen,
	       xi, 1, 0.0, info->dev, 1);
  prod_vector(info->dev,lasvdgp->reds,nbas);
  info->tlen = tlen;
  info->sse = linalg_ddot(tlen,xi,1,xi,1);
  info->sse += linalg_ddot(nbas, d2c, 1, cmean, 1);
  info->sse -= 2.0 * linalg_ddot(nbas,info->dev,1,cmean,1);
  info->sig2hat = ress2;
  diff_vector(info->dev, d2c, nbas);
  delete_matrix(coeff);
  delete_matrix(resid);
  free(cmean);
  free(d2c);
}
double lagpProfileLikelihood::nloglikelihood(double sig2eps, void* info)
{
  int i, nbas;
  double sig2, tvar, loglik;
  lagpLikInfo* finfo = static_cast<lagpLikInfo*> (info);
  sig2 = sig2eps + finfo -> sig2hat;
  nbas = finfo -> nbas;
  loglik = (double)(finfo->tlen-nbas)*log(sig2);
  loglik += finfo->sse/sig2;
  for(i=0; i<nbas; ++i)
  {
    tvar = sig2+finfo->d2[i]*finfo->cs2[i];
    loglik += log(tvar) + finfo->cs2[i] * sq(finfo->dev[i])/tvar/sig2;
  }
  loglik *= 0.5;
  return loglik;
}
