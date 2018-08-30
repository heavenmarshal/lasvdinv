#include "mcmcutil.hpp"
#include "lagpLikelihood.hpp"
#include "kerneladaptive.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
extern "C"{
  #include "matrix.h"
  #include "rhelp.h"
}
enum LIKTYPE {Profile=101, Fixvar=102};
enum KERTYPE {Normal=201, Adaptive=202};
extern "C"{
  void lasvdinv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *liktype_, unsigned int *kertype_,
		unsigned int *nstarts_, unsigned int* nmc_, unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
		unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_, unsigned int *nthread_,
		unsigned int *adpthres_, double *noiseVar_, double *frac_, double* gstart_, double* kersd, double *xi_,
		double *design_, double *resp_, double *xstarts_, double* poststarts_, double* lb_,
		double *ub_, double *eps_, double *sval_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    slen = (*nmc_) * nparam;
#ifdef _OPENMP
    mxth = omp_get_max_threads();
#else
    mxth = 1;
#endif
    if(*nthread_>mxth)
    {
      MYprintf(MYstdout, "NOTE: omp.threads(%d) > max(%d), using %d\n",
	       *nthread_, mxth, mxth);
      *nthread_ = mxth;
    }
#ifdef _OPENMP
#pragma omp parallel num_threads(*nthread_)
    {
      unsigned int i, start, step;
      start = omp_get_thread_num();
      step = *nthread_;
#else
      unsigned int i, start, step;
      start = 0; step = 1;
#endif
      likelihoodBase *likelihood;
      kernelBase* kernel;
      if(*liktype_ == Profile)
	likelihood = new lagpProfileLikelihood(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
					       *resvdThres_, *every_, *frac_, *gstart_,
					       xi_, design, resp);
      else
	likelihood = new lagpFixvarLikelihood(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
					      *resvdThres_, *every_, *noiseVar_, *frac_, *gstart_,
					      xi_, design, resp);
      if(*kertype_ == Normal)
	kernel = new normalKernel(nparam, *kersd);
      else
	kernel = new kernelAdaptive(nparam, *adpthres_, *eps_, *sval_, *kersd);

      uniformPrior prior(nparam, lb_, ub_);
      for(i = start; i < *nstarts_; i+=step)
      {

	mcmcBase mcmc(nparam, *nmc_, xstarts[i], poststarts_[i],
		      &prior, likelihood, kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
      delete likelihood;
      delete kernel;
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }
}
