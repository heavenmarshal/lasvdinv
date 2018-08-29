#include<cassert>
#include<cstdlib>
#include<cmath>
#include<ctime>
#include<random>		// c++11
#include<algorithm>
#include<iostream>
#include "mcmcutil.hpp"

extern "C"{
  #include "matrix.h"
}

mcmcBase::mcmcBase(int nparam_,int nmc_, double* x0, double post0,
		   priorBase* prior_, likelihoodBase* likelihood_,
		   kernelBase* kernel_):
  nparam(nparam_), nmc(nmc_), naccept(0),clogpost(post0),
  generator(time(NULL)), prior(prior_), likelihood(likelihood_),
  kernel(kernel_)
{
  assert(x0 != NULL);
  current = new_dup_vector(x0, nparam);
  sample = new_matrix(nmc,nparam);
}
mcmcBase::~mcmcBase()
{
  free(current);
  delete_matrix(sample);
}

double mcmcBase::evalLogPosterior(double* param)
{
  double post;
  post = likelihood->evalLogLikelihood(param);
  post += prior->evalLogPrior(param);
  return post;
}
void mcmcBase::run()
{
  int i;
  double logpost, logaccprob, logru;
  double *proposal;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  proposal = new_vector(nparam);
  for(i=0; i<nmc; ++i)
  {
    kernel->propose(current,proposal); // sampling a proposal parameter
    logpost = evalLogPosterior(proposal);
    logaccprob = logpost + kernel->logDensity(proposal,current);
    logaccprob -= clogpost + kernel->logDensity(current,proposal);
    logru = distribution(generator);
    logru = log(logru);
    if(logru < logaccprob)	// accept
    {
      dupv(current,proposal,nparam);
      clogpost = logpost;
      naccept++;
    }
    dupv(sample[i],current,nparam);
    kernel->update(i,this);
  }
  free(proposal);
}
void mcmcBase::getSample(double *output)
{
  int slen = nparam*nmc;
  dupv(output,sample[0],slen);
}

void normalKernel::propose(double* from, double* to)
{
  int i;
  std::normal_distribution<double> distribution(0.0,sigma);
  for(i=0; i<nparam; i++)
    to[i] = from[i] + distribution(generator);
}
