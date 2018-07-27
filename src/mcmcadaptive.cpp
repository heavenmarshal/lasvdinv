#include "mcmcadaptive.hpp"

void mcmcAdaptive::run()
{
  int i, j, k;
  double logpost, logaccprob, logru;
  double *proposal;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  proposal = new_vector(nparam);
  for(i=0, j=-nburn, k=0; i<nmc; ++i, ++j)
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
    if(j >= 0 && j % nthin == 0)
    {
      dupv(sample[k],current,nparam);
      kernel -> updatecov(k, sample);
      k++;
    }

  }
  free(proposal);
}
