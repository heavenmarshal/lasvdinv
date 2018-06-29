#ifndef __LIKELIHOODNEWTON_HPP__
#define __LIKELIHOODNEWTON_HPP__
#include <cmath>
#include "mcmcutil.hpp"

class likelihoodNewton: public likelihoodBase{
public:
  likelihoodNewton(int nparam_): likelihoodBase(nparam_){}
  virtual ~likelihoodNewton(){};
  virtual double evalLogLikelihood(double* param, double* pmean, double *ps2){
    return 0.0;}
};
class naivelikelihoodNewton: public likelihoodNewton{
public:
  naivelikelihoodNewton(int nparam_,int tlen_, double *xi_):
    likelihoodNewton(nparam_), tlen(tlen_), xi(xi_){}
  virtual ~naivelikelihoodNewton(){}
  double evalLogLikelihood(double* param, double* pmean, double *ps2);
private:
  int tlen;
  double *xi;
};
class scalarlikelihoodNewton: public likelihoodNewton{
public:
  scalarlikelihoodNewton(int nparam_, unsigned int tlen_, unsigned int islog_):
    likelihoodNewton(nparam_),tlen(tlen_), islog(islog_){};
  virtual ~scalarlikelihoodNewton(){}
  double evalLogLikelihood(double* param, double* pmean, double *ps2){
    double loglik;
    loglik = -(*pmean);
    if(!islog)
      loglik = -0.5* (double)tlen * log(-loglik);
    return loglik;
  }
private:
  unsigned int tlen, islog;
};
#endif
