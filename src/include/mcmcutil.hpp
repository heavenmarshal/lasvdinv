#ifndef __MCMCUTIL_HPP__
#define __MCMCUTIL_HPP__
#include<cmath>
#include<cstdlib>
#include<ctime>
#include<random>
extern "C"{
  #include "matrix.h"
}
#define LOG2PI 1.83787706641
#define SQ(x) ((x)*(x))
class mcmcBase;

class priorBase{
public:
  priorBase(int nparam_):nparam(nparam_){};
  virtual ~priorBase(){};
  virtual double evalLogPrior(const double* param){
      return 0.0;};
protected:
  int nparam;
};
class kernelBase{
public:
  kernelBase(int nparam_): nparam(nparam_), generator(time(NULL)){};
  virtual ~kernelBase(){};
  virtual void init(){};
  virtual void propose(double* from, double* to){};
  virtual double logDensity(double* from, double* to){return 0.0;};
  virtual void update(int iter, mcmcBase *mcobj){};
protected:
    int nparam;
    std::default_random_engine generator;
};

class likelihoodBase{
public:
  likelihoodBase(int nparam_): nparam(nparam_){};
  virtual ~likelihoodBase(){};
  virtual double evalLogLikelihood(double* param){
    return 0.0;};
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
  mcmcBase(int nparam_,int nmc_, double* x0, double post0,
	   priorBase* prior_, likelihoodBase* likelihood_,
	   kernelBase* kernel_);
  virtual ~mcmcBase();
  virtual void run();
  double** getSample(){return sample;}
  void getSample(double* output);
protected:
  int nparam;
  int nmc;
  int naccept;
  double clogpost;
  double* current;
  double** sample;
  std::default_random_engine generator;
  priorBase* prior;
  likelihoodBase* likelihood;
  kernelBase* kernel;
  double evalLogPosterior(double *param);
};
class normalKernel: public kernelBase{
public:
  normalKernel(int nparam_, double sigma_): kernelBase(nparam_), sigma(sigma_){};
  double logDensity(double* from, double *to)
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
  void propose(double* from, double* to);
private:
  double sigma;
};
#endif
