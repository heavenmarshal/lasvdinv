#include "kerneladaptive.hpp"
extern "C"{
  #include "matrix.h"
  #include "linalg.h"
  #include "linalgextra.h"
}

void kernelAdaptive::updatecov(int niter, double **sample)
{
  citer = niter;
  if(niter < 1) return;
  int i, j;
  double diter, *xbarp, *xbar, *cx;
  diter = (double)niter;
  xbarp = new_zero_vector(nparam);
  xbar = new_vector(nparam);
  cx = sample[niter];
  for(i=0; i < niter; ++i)
    for(j=0; j < nparam; ++j)
      xbarp[j] += sample[i][j];
  dupv(xbar,xbarp,nparam);
  for(j=0; j < nparam; ++j)
  {
    xbar[j] += cx[j];
    xbar[j] /= (diter+1.0);
    xbarp[j] /= diter;
  }
  for(i=0; i<nparam; ++i)
  {
    covmat[i][i] *= (diter-1.0)/diter;
    covmat[i][i] += sval*xbarp[i]*xbarp[i];
    covmat[i][i] -= sval*(diter+1.0)/diter*xbar[i]*xbar[i];
    covmat[i][i] += sval*cx[i]*cx[i]/diter+sval*eps/diter;
    for(j=0; j<i; ++j)
    {
      covmat[i][j] *= (diter-1.0)/diter;
      covmat[i][j] += sval*xbarp[i]*xbarp[j];
      covmat[i][j] -= sval*(diter+1.0)/diter*xbar[i]*xbar[j];
      covmat[i][j] += sval*cx[i]*cx[i]/diter;
      covmat[j][i] = covmat[i][j];
    }
  }
  if(niter >= n0)
  {
    dup_matrix(umat, covmat, nparam, nparam);
    linalg_dpotrf(nparam, umat);
  }
  free(xbarp);
  free(xbar);
}

void kernelAdaptive::propose(double *from, double *to)
{
  if(citer<n0)
  {
    regularpropose(from,to);
    return;
  }
  adaptivepropose(from,to);
}
double kernelAdaptive::logDensity(double *from, double *to)
{
  if(citer < n0)
    return regularlogDensity(from, to);
  return adaptivelogDensity(from, to);
}

void kernelAdaptive::regularpropose(double *from, double *to)
{
  int i;
  std::normal_distribution<double> distribution(0.0,sigma0);
  for(i=0; i<nparam; i++)
    to[i] = from[i] + distribution(generator);

}
double kernelAdaptive::regularlogDensity(double *from, double *to)
{
  int i;
  double tmp, logden;
  tmp = 0.0;
  for(i=0; i<nparam; ++i)
    tmp -= SQ(from[i]-to[i]);
  logden = -0.5*LOG2PI-nparam*log(sigma);
  logden += 0.5*tmp/sigma/sigma;
  return logden;
}
void kernelAdaptive::adaptivepropose(double *from, double *to)
{
  int i;
  double tvec;
  tvec = new_vector(nparam);
  dupv(to,from,nparam);
  std::normal_distribution<double> distribution(0.0,1.0);
  for(i=0; i<nparam; ++i)
    tvec[i] = distribution(generator);
  linalg_dtrmv(CblasUpper, CblasTrans, CblasNoUnit, nparam,
	       umat, nparam, tvec, 1);
  linalg_daxpy(nparam, 1.0, tvec, 1, to, 1);
  free(tvec);
}
void kernelAdaptive::adaptivelogDensity(double *from, double *to)
{
  double logden, *tvec;
  tvec = new_vector(nparam);
  dupv(tvec,to,nparam);
  linalg_daxpy(nparam,-1.0,from, 1, tvec,1);
  linalg_dtrsm(CblasLeft, CblasUpper, CblasTrans, CblasNoUnit,
	       nparam, 1, 1.0, umat, nparam, tvec, nparam);
  logden = linalg_ddot(nparam, tvec, 1, tvec, 1);
  logden += LOG2PI;
  logden *= 0.5;
  for(i=0; i<nparam; ++i)
    logden += log(umat[i][i]);
  free(tvec);
  return -logden;
}
