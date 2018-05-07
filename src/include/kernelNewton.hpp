#ifndef __KERNELNEWTON_HPP__
#define __KERNELNEWTON_HPP__
#include<cstdlib>
#include"mcmcutil.hpp"
extern "C"{
  #include "matrix.h"
}
class kernelNewton: public kernelBase{
public:
  kernelNewton(int nparam_): kernelBase(nparam_){}
  virtual ~kernelNewton(){};
  virtual void propose(double* from, double* to){}
  virtual double logDensity(double* from, double *to){return 0.0;}
  virtual void procGradHess(double *grad, double **hess){}
};


class eigenkernelNewton: public kernelNewton{
public:
  eigenkernelNewton(int nparam_, double thres_, double sdfrac_):
    kernelNewton(nparam_), thres(thres_), sdfrac(sdfrac_)
    {
      moffset = new_vector(nparam);
      values = new_vector(nparam);
      vectors = new_matrix(nparam, nparam);
    }
  ~eigenkernelNewton(){
    free(moffset);
    free(values);
    delete_matrix(vectors);
  }
  void propose(double* from, double* to);
  double logDensity(double* from, double *to);
  void procGradHess(double* grad, double **hess);
private:
  double thres;
  double sdfrac;
  double *moffset;
  double *values;		// truncated eigenvalues of the hessian matrix
  double **vectors; 		// eigenvectors of the hessian matrix
};

#endif
