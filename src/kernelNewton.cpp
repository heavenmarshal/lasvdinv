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
static void adddiag(double **mat, int nrow, int ncol, double add)
{
  int i, nadd = nrow>ncol? ncol:nrow;
  for(i=0; i<nadd; ++i)
    mat[i][i] +=  add;
}

void normalkernelNewton::procGradHess(double* grad, double **hess)
{
  int info, m;
  double mineigen, offset;
  double *values, *vectors;
  std::cout<<"gradient"<<std::endl;
  printVector(grad, nparam, stdout, HUMAN);
  std::cout<<"hessian"<<std::endl;
  printMatrix(hess, nparam, nparam, stdout);
  values = new_vector(nparam);
  vectors=NULL;
  info=linalg_dsyevr(CblasValue,CblasInt, nparam, hess, nparam, 0.0, 0.0, 1,1,
		     0.0,&m,values,&vectors,nparam);
  //to do: error handling
  mineigen = values[0];
  dup_matrix(precision, hess, nparam, nparam);
  if(mineigen<thres){
    offset = thres-mineigen;
    adddiag(precision, nparam, nparam, offset);
  }
  std::cout<<"precision"<<std::endl;
  printMatrix(precision,nparam,nparam,stdout);
  dup_matrix(cholu, precision, nparam, nparam);
  info = linalg_dpotrf(nparam, cholu);
  dupv(moffset,grad,nparam);
  info = linalg_dtrtrs(CblasTrans, CblasNonUnit, nparam, 1, cholu,
		       nparam, &moffset, nparam);
  info = linalg_dtrtrs(CblasNoTrans, CblasNonUnit, nparam, 1, cholu,
		       nparam, &moffset, nparam);
  std::cout<<"moffset"<<std::endl;
  printVector(moffset,nparam,stdout,HUMAN);
  free(values);
}
void normalkernelNewton::propose(double *from, double* to)
{
  int i;
  double *nrand;
  std::normal_distribution<double> distribution(0.0,1.0);
  dupv(to,from,nparam);
  linalg_daxpy(nparam, -1.0, moffset, 1, to, 1);
  nrand = new_vector(nparam);
  for(i = 0; i < nparam; ++i)
    nrand[i] = distribution(generator);
  linalg_dtrtrs(CblasNoTrans, CblasNonUnit, nparam, 1, cholu,
		nparam, &nrand, nparam);
  linalg_daxpy(nparam, 1.0, nrand, 1, to, 1);
  free(nrand);
}
double normalkernelNewton::logDensity(double *from, double *to)
{
  int i, info;
  double logden, *dev;
  dev = new_dup_vector(to, nparam);
  linalg_daxpy(nparam, -1.0, from, 1, dev, 1);
  linalg_daxpy(nparam, 1.0, moffset, 1, dev, 1);
  info = linalg_dtrtrs(CblasTrans, CblasNonUnit, nparam, 1, cholu,
		       nparam, &dev, nparam);
  logden = linalg_ddot(nparam, dev, 1, dev, 1);
  logden *= -0.5;
  for(i = 0; i < nparam; ++i)
    logden += log(cholu[i][i]);
  free(dev);
  return logden;
}
