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

class normalkernelNewton: public kernelNewton{
public:
  normalkernelNewton(int nparam_, double thres_): kernelNewton(nparam_), thres(thres_){
    moffset = new_vector(nparam);
    precision = new_matrix(nparam,nparam);
    cholu = new_matrix(nparam, nparam);
  }
  ~normalkernelNewton(){
    free(moffset);
    delete_matrix(precision);
    delete_matrix(cholu);
  }
  void propose(double* from, double* to);
  double logDensity(double* from, double *to);
  void procGradHess(double* grad, double **hess);
private:
  double thres;
  double *moffset;
  double **precision;		// precision matrix
  double **cholu; 		// upper triangular matrix of cholesky of precision
};
#endif
