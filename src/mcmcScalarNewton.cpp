#include "mcmcScalarNewton.hpp"
#include <cstdlib>
extern "C"{
  #include "linalg.h"
  #include "matrix.h"
  #include "fitlagpsep.h"
}

mcmcScalarNewton::mcmcScalarNewton(int nparam_, int nmc_, int nburn_, int nthin_, unsigned int ndesign_,
				   unsigned int n0_, unsigned int nn_, unsigned int nfea_,
				   unsigned int every_, unsigned int tlen_, unsigned int islog_, double gstart_,
				   double **design_, double *resp_, double *x0, double post0, priorBase *prior_,
				   likelihoodNewton *likelihood_, kernelNewton *kernel_):
  mcmcBase(nparam_, nmc_, nburn_, nthin_, x0, post0, prior_, NULL, NULL), ndesign(ndesign_), n0(n0_),
  nn(nn_), nfea(nfea_), every(every_), tlen(tlen_), islog(islog_), gstart(gstart_), design(design_), resp(resp_),
  likelihoodn(likelihood_), kerneln(kernel_)
{
  grad = new_vector(nparam);
  hess = new_matrix(nparam,nparam);
  cgrad = NULL;
  chess = NULL;
}

mcmcScalarNewton::~mcmcScalarNewton()
{
  free(grad);
  delete_matrix(hess);
  if(cgrad != NULL)
    free(cgrad);
  if(chess != NULL)
    delete_matrix(chess);
}

void mcmcScalarNewton::fitlagp(double *param)
{
  double df;
  GPsep *gpsep;
  gpsep = fitlagpsep(nparam, ndesign, n0, nn, nfea,
		     every, param, design, resp, gstart);
  predGPsep_lite(gpsep, 1, &param, &pmean, &ps2, &df, NULL);
  scalargpgradhess(gpsep, nparam, param, grad, hess);
  if(!islog)
    transloggradhess(nparam, tlen, pmean, grad, hess);
  deleteGPsep(gpsep);
  gpsep=NULL;
}
void mcmcScalarNewton::accept()
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
void mcmcScalarNewton::run()
{
  int i, j, k;
  double logpost, logaccprob, logru;
  double *proposal;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  proposal = new_vector(nparam);
  fitlagp(current);
  accept();
  for(i=0, j=-nburn, k=0; i<nmc; ++i, ++j)
  {
    kerneln -> procGradHess(cgrad,chess);
    kerneln -> propose(current,proposal);
    logaccprob = -clogpost - kerneln->logDensity(current, proposal);
    fitlagp(proposal);
    logpost = evalLogPosterior(proposal);
    kerneln -> procGradHess(grad,hess);
    logaccprob += logpost + kerneln ->logDensity(proposal,current);

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
double mcmcScalarNewton::evalLogPosterior(double *param)
{
  double post;
  post = likelihoodn -> evalLogLikelihood(param, &pmean, &ps2);
  post += prior->evalLogPrior(param);
  return post;
}

