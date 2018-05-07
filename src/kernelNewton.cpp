#include <cmath>
#include <cstdlib>
#include <random>
#include "kernelNewton.hpp"
#include <iostream>
extern "C"{
  #include "matrix.h"
  #include "linalg.h"
  #include "dsyevr.h"
}
void eigenkernelNewton::procGradHess(double *grad, double **hess)
{
  int i, info, m;
  double val, *worker;
  worker = new_vector(nparam);
  info = linalg_dsyevr(CblasBoth, CblasAll, nparam, hess, nparam,
		       0.0, 0.0, 0, 0, 0.0, &m, values, vectors, nparam);
  for(i=0; i<nparam; ++i)
  {
    val = values[i];
    val = val<0.0 ? -val: val;
    val = (val<thres)? thres: val;
    values[i] = val;
  }
  linalg_dgemv(CblasTrans, nparam, nparam, 1.0, vectors, nparam,
	       grad, 1, 0.0, worker, 1);
  for(i=0; i<nparam; ++i)
    worker[i] /= values[i];
  linalg_dgemv(CblasNoTrans, nparam, nparam, 1.0, vectors, nparam,
	       worker, 1, 0.0, moffset, 1);
  free(worker);
}
void eigenkernelNewton::propose(double* from, double* to)
{
  int i;
  double *nrand, *worker;
  std::normal_distribution<double> distribution(0.0,1.0);
  dupv(to,from,nparam);
  linalg_daxpy(nparam, -1.0, moffset, 1, to, 1);
  nrand = new_vector(nparam);
  worker = new_vector(nparam);
  for(i = 0; i < nparam; ++i)
    nrand[i] = distribution(generator);
  linalg_dgemv(CblasTrans, nparam, nparam, 1.0, vectors, nparam,
	       nrand, 1, 0.0, worker, 1);
  for(i=0; i<nparam; ++i)
    worker[i] /= sqrt(values[i]);
  linalg_dgemv(CblasNoTrans, nparam, nparam, 1.0, vectors, nparam,
	       worker, 1, 0.0, nrand, 1);
  linalg_daxpy(nparam, sdfrac, nrand, 1, to, 1);
  free(worker);
  free(nrand);
}
double eigenkernelNewton::logDensity(double *from, double *to)
{
  int i;
  double logden, *dev, *worker;
  worker = new_dup_vector(to, nparam);
  dev = new_vector(nparam);
  linalg_daxpy(nparam, -1.0, from, 1, worker, 1);
  linalg_daxpy(nparam, 1.0, moffset, 1, worker, 1);
  linalg_dgemv(CblasTrans, nparam, nparam, 1.0, vectors, nparam,
	       worker, 1, 0.0, dev, 1);
  for(i=0; i<nparam; ++i)
    dev[i] *= sqrt(values[i]);
  logden = linalg_ddot(nparam, dev, 1, dev, 1);
  logden *= -0.5/sq(sdfrac);
  for(i=0; i<nparam; ++i)
    logden += 0.5*log(values[i]);
  logden -= nparam*log(sdfrac);
  free(dev);
  return logden;    
}
