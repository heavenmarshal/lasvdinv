#ifndef __KERNELADAPTIVE_HPP__
#define __KERNELADAPTIVE_HPP__
#include "mcmcutil.hpp"
extern "C"{
  #include "matrix.h"
}
class kernelAdaptive: public kernelBase{
public:
  kernelAdaptive(int nparam_, int n0_, double eps_,
		 double sval_, double sigma0_):
    kernelBase(nparam_), n0(n0_), citer(0),
    eps(eps_), sval(sval_), sigma0(sigma0_){
    covmat = new_zero_matrix(nparam,nparam);
    umat = new_matrix(nparam,nparam);
  };
  ~kernelAdaptive(){
    delete_matrix(covmat);
    delete_matrix(umat);
  };
  void init(){citer=0;}
  void propose(double *from, double *to);
  double logDensity(double *from, double *to);
  void update(int niter, mcmcBase *mcobj);
private:
  int n0;			// the threshold number of iterations for adaption
  int citer;			// record current number of iterations
  double eps, sval, sigma0;
  double **covmat, **umat;
  void updatecov(int niter, double **sample);
  void regularpropose(double *from, double *to);
  double regularlogDensity(double *from, double *to);
  void adaptivepropose(double *from, double *to);
  double adaptivelogDensity(double *from, double *to);
};
#endif
