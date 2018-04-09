#include<cmath>
#include<cstdlib>
#include "likelihoodNewton.hpp"
extern "C"{
  #include "linalg.h"
  #include "matrix.h"
}
double naivelikelihoodNewton::evalLogLikelihood(double *param, double* pmean, double *ps2)
{
  double dev, dtlen, loglik;
  double *dmcp;
  dtlen = (double) tlen;
  dmcp = new_dup_vector(pmean,tlen);
  linalg_daxpy(tlen,-1.0,xi,1,dmcp,1);
  dev = linalg_ddot(tlen,dmcp, 1, dmcp, 1);
  dev /= dtlen;
  loglik = -0.5 * LOG2PI - 0.5 *dtlen;
  loglik -= 0.5*dtlen*log(dev);
  free(dmcp);
  return loglik;
}
