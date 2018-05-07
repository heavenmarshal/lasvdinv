#ifndef __MCMCSCALARNEWTON_HPP__
#define __MCMCSCALARNEWTON_HPP__
#include "likelihoodNewton.hpp"
#include "kernelNewton.hpp"
#include "mcmcutil.hpp"

extern "C"{
  #include "gp_sep.h"
}
class mcmcScalarNewton: public mcmcBase{
public:
  mcmcScalarNewton(int nparam_, int nmc_, int nburn_, int nthin_, unsigned int ndesign_,
		   unsigned int n0_, unsigned int nn_, unsigned int nfea_,
		   unsigned int every_, double gstart_, double **design_,
		   double *resp_, double *x0, double post0, priorBase *prior_,
		   likelihoodNewton *likelihood_, kernelNewton *kernel_);
  ~mcmcScalarNewton();
  void run();
private:
  unsigned int ndesign, n0, nn, nfea, every;
  double gstart, pmean, ps2;
  double **design, *resp;
  double *grad, *cgrad;
  double **hess, **chess;
  GPsep *gpsep;
  likelihoodNewton *likelihoodn;
  kernelNewton *kerneln;
  static void selectpoint(GPsep* gpsep, int nparam, int nfea, int *feaidx,
			  double *param, double **design, double *resp);
  void fitlagp(double *param);
  void evalgradhessian(double *param);
  double evalLogPosterior(double *param);
  void accept();
};
#endif
