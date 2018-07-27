#ifndef __MCMCADAPTIVE_HPP__
#define __MCMCADAPTIVE_HPP__
#icnlude "mcmcutil.hpp"

class mcmcAdaptive: public mcmcBase{
public:
  mcmcAdaptive(int nparam_, int nmc_, int nburn_, int nthin_,
	       double *x0, double post0, priorBase* prior_,
	       likelihoodBase* likelihood_, kernelAdaptive *kernel_):
    mcmcBase(nparam_, nmc_, nburn_, nthin_, x0, post0, prior_,
	     likelihood_, NULL), kernel(kernel_) {};
  void run();
private:
  kernelAdaptive *kernel;
};
#endif
