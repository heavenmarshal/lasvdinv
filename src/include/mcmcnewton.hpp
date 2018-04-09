#ifndef __MCMCNEWTON_HPP__
#define __MCMCNEWTON_HPP__
#include "mcmcutil.hpp"
#include "likelihoodNewton.hpp"
#include "kernelNewton.hpp"
extern "C"{
#include"lasvdgp.h"
}

class mcmcNewton: public mcmcBase{
public:
  mcmcNewton(int nparam_, int nmc_, int nburn_, int nthin_,
	     unsigned int ndesign_, unsigned int tlen_, unsigned int n0_,
	     unsigned int nn_, unsigned int nfea_, unsigned int resvdThres_,
	     unsigned int every_, double frac_, double gstart_,
	     double *xi_, double **design_, double **resp_,
	     double* x0, double post0, priorBase *prior_,
	     likelihoodNewton* likelihood_, kernelNewton *kernel_);
  ~mcmcNewton();
  void run();
private:
  unsigned int ndesign, tlen, n0, nn, nfea;
  unsigned int resvdThres, every;
  double frac,  gstart;
  double *xi, **design, **resp;
  double *pmean, *ps2;
  double *grad, *cgrad;
  double **hess, **chess;
  lasvdGP *lasvdgp;
  likelihoodNewton *likelihoodn;
  kernelNewton *kerneln;
  void fitlasvdgp(double *newx);
  void evalgradhessian(double *param);
  double evalLogPosterior(double *param);
  void accept();
};

#endif
