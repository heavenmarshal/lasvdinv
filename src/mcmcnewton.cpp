#include <cstdlib>
#include "mcmcnewton.hpp"
#include "kernelNewton.hpp"
#include "likelihoodNewton.hpp"
#include <iostream>

extern "C"{
#include "linalg.h"
#include "matrix.h"
#include "covar_sep.h"
}
mcmcNewton::mcmcNewton(int nparam_, int nmc_, int nburn_, int nthin_,
		       unsigned int ndesign_, unsigned int tlen_, unsigned int n0_,
		       unsigned int nn_, unsigned int nfea_, unsigned int resvdThres_,
		       unsigned int every_, double frac_, double gstart_,
		       double *xi_, double **design_, double **resp_, double *x0, double post0,
		       priorBase *prior_, likelihoodNewton* likelihood_, kernelNewton *kernel_):
  mcmcBase(nparam_,nmc_,nburn_,nthin_, x0, post0, prior_), ndesign(ndesign_), tlen(tlen_), n0(n0_),
  nn(nn_), nfea(nfea_), resvdThres(resvdThres_), every(every_), frac(frac_), gstart(gstart_),
  xi(xi_), design(design_), resp(resp_), likelihoodn(likelihood_), kerneln(kernel_)
{
  grad = new_vector(nparam);
  pmean = new_vector(tlen);
  ps2 = new_vector(tlen);
  hess = new_matrix(nparam,nparam);
  cgrad = NULL;
  chess = NULL;
}
mcmcNewton::~mcmcNewton()
{
  free(grad);
  free(pmean);
  free(ps2);
  delete_matrix(hess);
  if(cgrad != NULL)
    free(cgrad);
  if(chess != NULL)
    delete_matrix(chess);
}
void mcmcNewton::fitlasvdgp(double *param)
{
  lasvdgp = newlasvdGP(param, design, resp, ndesign, nparam, tlen, nn, n0,
		       nfea, nn, 1, frac, gstart);
  jmlelasvdGP(lasvdgp,100,0);
  iterlasvdGP(lasvdgp,resvdThres,every,100,0);
  predlasvdGP(lasvdgp,pmean,ps2);
  evalgradhessian(param);
  deletelasvdGP(lasvdgp);
  lasvdgp = NULL;
}
void mcmcNewton::evalgradhessian(double *param)
{
  if(lasvdgp == NULL) return;
  int nbas, i, j, k, p, n;
  double  dc2dx2, *btdev, *kiy, *d;
  double **corr, **X, **dcdx;

  n = lasvdgp->n0;
  nbas = lasvdgp->nbas;
  btdev = new_vector(nbas);
  corr = new_matrix(nbas,n);
  dcdx = new_matrix(nparam,nbas);
  linalg_dgemv(CblasTrans, tlen, nbas, -1.0, &(lasvdgp->basis), tlen,
	       xi, 1, 0.0, btdev, 1);
  for(i = 0; i < nbas; ++i)
  {
    X = lasvdgp->gpseps[i]->X;
    d = lasvdgp->gpseps[i]->d;
    kiy = lasvdgp->gpseps[i]->KiZ;
    covar_sep(nparam, &param, 1, X, n, d,&corr[i]);
    btdev[i] += sq(lasvdgp->reds[i])*linalg_ddot(n,corr[i],1,kiy,1);
  }
  for(i=0; i<nparam; ++i)
  {
    grad[i] = 0.0;
    hess[i][i] = 0.0;
    for(j = 0; j < nbas; ++j)
    {
      dcdx[i][j] = 0.0;
      dc2dx2 = 0.0;
      X = lasvdgp->gpseps[j]->X;
      d = lasvdgp->gpseps[j]->d;
      kiy = lasvdgp->gpseps[j]->KiZ;
      for(k = 0; k < n; ++k)
      {
	dcdx[i][j] += 2.0 * corr[j][k] * (X[k][i]-param[i])/d[i]*kiy[k];
	dc2dx2 += 2.0 * corr[j][k] * (2.0*sq((X[k][i]-param[i])/d[i]) - 1.0/d[i]) *kiy[k];
      }
      grad[i] += btdev[j] * dcdx[i][j];
      hess[i][i] += btdev[j] * dc2dx2;
    }
    for(j = 0; j<i; ++j)
    {
      hess[i][j] = 0.0;
      for(k = 0; k < nbas; ++k)
      {
	dc2dx2 = 0.0;
	X = lasvdgp->gpseps[k]->X;
	d = lasvdgp->gpseps[k]->d;
	kiy = lasvdgp->gpseps[k]->KiZ;
	for(p = 0; p <n; ++p)
	  dc2dx2 += 4.0*corr[k][p] * (X[p][i]-param[i])/d[i]*(X[p][j]-param[j])/d[j] *kiy[p];
	hess[i][j] += btdev[k]*dc2dx2;
      }
    }
  }
  for(i=0; i<nparam; ++i)
  {
      for(k=0; k<nbas; ++k)
	  hess[i][i] += sq(lasvdgp->reds[k])*sq(dcdx[i][k]);
      for(j=0; j<i; ++j)
      {
	  for(k = 0; k < nbas; ++k)
	      hess[i][j] += sq(lasvdgp->reds[k])* dcdx[i][k] * dcdx[j][k];
	  hess[j][i] = hess[i][j];
      }
  }
  free(btdev);
  delete_matrix(corr);
  delete_matrix(dcdx);
}
void mcmcNewton::accept()
{
  if(cgrad != NULL)
    free(cgrad);
  if(chess != NULL)
    delete_matrix(chess);
  cgrad = grad;
  chess = hess;
  grad = new_vector(nparam);
  hess = new_matrix(nparam,nparam);
}
void mcmcNewton::run()
{
    int i, j, k;
  double logpost, logaccprob, logru;
  double *proposal;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  proposal = new_vector(nparam);
  fitlasvdgp(current);
  accept();
  for(i=0, j=-nburn, k=0; i<nmc; ++i, ++j)
  {
    kerneln -> procGradHess(cgrad,chess);
    kerneln -> propose(current,proposal);
    std::cout<<"proposal= [";
    for(int p = 0; p <nparam-1; ++p) std::cout<<proposal[p]<<',';
    std::cout<<proposal[nparam-1]<<']'<<std::endl;
    logaccprob = -clogpost - kerneln->logDensity(current, proposal);
    std::cout<<"clogpost= " << clogpost<<std::endl;
    std::cout<<"forward= "<< kerneln->logDensity(current, proposal)<<std::endl;
    fitlasvdgp(proposal);
    logpost = evalLogPosterior(proposal);
    kerneln -> procGradHess(grad,hess);
    logaccprob += logpost + kerneln ->logDensity(proposal,current);
    std::cout<<"logpost= "<<logpost<<std::endl;
    std::cout<<"backward= "<<kerneln ->logDensity(proposal,current)<<std::endl;
    
    logru = distribution(generator);
    logru = log(logru);
    if(logru < logaccprob)
    {
      dupv(current,proposal, nparam);
      clogpost = logpost;
      accept();
      naccept++;
    }
    if(j >= 0 && j % nthin ==0)
    {
      dupv(sample[k],current,nparam);
      k ++;
    }
  }
  free(proposal);
}
double mcmcNewton::evalLogPosterior(double *param)
{
  double post;
  post = likelihoodn -> evalLogLikelihood(param, pmean, ps2);
  post += prior->evalLogPrior(param);
  return post;
}
