#ifndef __MCMCUTIL_HPP__
#define __MCMCUTIL_HPP__
#include<cmath>
#include<cstdlib>
extern "C"{
  #include "matrix.h"
}
#define LOG2PI 0.79817986835
#define SQ(x) ((x)*(x))

class priorBase{
public:
  priorBase(int nparam_):nparam(nparam_){};
  virtual ~priorBase(){};
  virtual double evalLogPrior(const double* param){return 0.0;};
protected:
  int nparam;
};
class kernelBase{
public:
  kernelBase(int nparam_): nparam(nparam_){};
  virtual ~kernelBase(){};
  virtual void propose(const double* from, double* to){};
  virtual double logDensity(const double* from, const double* to){return 0.0;};
protected:
  int nparam;
};

class likelihoodBase{
public:
  likelihoodBase(int nparam_): nparam(nparam_){};
  virtual ~likelihoodBase(){};
  virtual double evalLogLikelihood(const double* param){return 0.0;};
protected:
  int nparam;
};
class uniformPrior: public priorBase{
public:
  uniformPrior(int nparam_, double *lb, double *ub): priorBase(nparam_)
    {
      lowb = new_dup_vector(lb,nparam_);
      upb = new_dup_vector(ub,nparam_);
    }
  ~uniformPrior()
    {
      free(lowb);
      free(upb);
    }
  double evalLogPrior(const double* param)
    {
      bool isin = true;
      int i;
      double logprior;
      for(i=0; i<nparam; ++i)
	if(param[i] > upb[i] || param[i] < lowb[i])
	{
	  isin = false;
	  break;
	}
      logprior = isin? 0.0: -INFINITY;
      return logprior;
    }
private:
  double *lowb;
  double *upb;
};
class mcmcBase{
public:
  mcmcBase(int nparam_,int nmc_, int nburn_, int nthin_,
	   double* x0, double* post0,
	   priorBase& prior_, likelihoodBase& likelihood_,
	   kernelBase& kernel_);
  virtual ~mcmcBase();
  virtual void run();
  void getSample(double* output);
protected:
  int nparam;
  int nmc;
  int nburn;
  int nthin;
  int nsample;
  int naccept;
  double clogpost;
  double* current;
  double** sample;
  priorBase& prior;
  likelihoodBase& likelihood;
  kernelBase& kernel;
  double evalLogPosterior(double *param);
};
class normalKernel: public kernelBase{
public:
  normalKernel(int nparam_, double sigma_): kernelBase(nparam_), sigma(sigma_){};
  double logDensity(const double* from, const double *to)
  {
    int i;
    double tmp, logden;
    tmp = 0.0;
    for(i=0; i<nparam; ++i)
      tmp -= SQ(from[i]-to[i]);
    logden = -0.5*LOG2PI-nparam*log(sigma);
    logden -= 0.5*tmp/sigma/sigma;
    return logden;
  }
  void propose(const double* from, double* to);
private:
  double sigma;
};
#endif
