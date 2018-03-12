#include<cassert>
#include<cstdlib>
#include<cmath>
#include<random>		// c++11
#include<algorithm>
#include "mcmcutil.hpp"

extern "C"{
  #include "matrix.h"
}

mcmcBase::mcmcBase(int nparam_,int nmc_, int nburn_, int nthin_,
		   double* x0, double* post0,
		   priorBase& prior_, likelihoodBase& likelihood_,
		   kernelBase& kernel_):
  nparam(nparam_), nmc(nmc_), nburn(nburn_), nthin(nthin_),naccept(0),
  prior(prior_), likelihood(likelihood_), kernel(kernel_)
{
  nsample = (nmc - nburn)/nthin;
  assert(x0 != NULL);
  current = new_dup_vector(x0, nparam);
  if(post0 == NULL)
    clogpost = evalLogPosterior(current);
  else
    clogpost = *post0;
  sample = new_matrix(nsample,nparam);
}

mcmcBase::~mcmcBase()
{
  free(current);
  delete_matrix(sample);
}

double mcmcBase::evalLogPosterior(double* param)
{
  double post;
  post = likelihood.evalLogLikelihood(param);
  post += prior.evalLogPrior(param);
  return post;
}
void mcmcBase::run()
{
  int i, j, k;
  double logpost, logaccprob, logru;
  double *proposal;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  proposal = new_vector(nparam);
  for(i=0, j=-nburn, k=0; i<nmc; ++i, ++j)
  {
    kernel.propose(current,proposal); // sampling a proposal parameter
    logpost = evalLogPosterior(proposal);
    logaccprob = logpost + kernel.logDensity(proposal,current);
    logaccprob -= clogpost + kernel.logDensity(current,proposal);
    logru = distribution(generator);
    logru = log(logru);
    if(logru < logaccprob)	// accept
    {
      dupv(current,proposal,nparam);
      naccept++;
    }
    if(j >= 0 && j % nthin == 0)
    {
      dupv(sample[k],current,nparam);
      k++;
    }
  }
  free(proposal);
}
void mcmcBase::getSample(double *output)
{
  int slen = nparam*nsample;
  dupv(output,sample[0],slen);
}

void normalKernel::propose(const double* from, double* to)
{
  int i;
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,sigma);
  for(i=0; i<nparam; i++)
    to[i] = from[i] + distribution(generator);
}
