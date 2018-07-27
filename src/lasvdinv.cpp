#include "mcmcutil.hpp"
#include "lagpLikelihood.hpp"
#include "mcmcnewton.hpp"
#include "kernelNewton.hpp"
#include "likelihoodNewton.hpp"
#include "mcmcScalarNewton.hpp"
#include "lagpScalarLikelihood.hpp"
#include "mcmcadaptive.hpp"
#include "kerneladaptive.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
extern "C"{
  #include "matrix.h"
  #include "rhelp.h"
}
extern "C"{
  void lagpNaiveInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
		    unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
		    unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
		    unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_,
		    unsigned int *nthread_, double *frac_, double* gstart_, double* kersd, double *xi_,
		    double *design_, double *resp_, double *xstarts_, double* poststarts_, double* lb_,
		    double *ub_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, nsample, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {

	lagpNaiveLikelihood naiveLik(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
				     *resvdThres_, *every_, *frac_, *gstart_,
				     xi_, design, resp);
	uniformPrior prior(nparam, lb_, ub_);
	normalKernel kernel(nparam, *kersd);
	mcmcBase mcmc(nparam, *nmc_, *nburn_, *nthin_, xstarts[i], poststarts_[i],
		      &prior, &naiveLik, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }
  void lagpProfileInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
		      unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
		      unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
		      unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_,
		      unsigned int *nthread_, double *frac_, double* gstart_, double* kersd, double *xi_,
		      double *design_, double *resp_, double *xstarts_, double* poststarts_, double* lb_,
		      double *ub_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, nsample, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	lagpProfileLikelihood profileLik(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
					 *resvdThres_, *every_, *frac_, *gstart_,
					 xi_, design, resp);
	uniformPrior prior(nparam, lb_, ub_);
	normalKernel kernel(nparam, *kersd);
	mcmcBase mcmc(nparam, *nmc_, *nburn_, *nthin_, xstarts[i], poststarts_[i],
		      &prior, &profileLik, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }
    void lagpFixvarInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
		       unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
		       unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
		       unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_,
		       unsigned int *nthread_, double *noiseVar_, double *frac_, double* gstart_,
		       double* kersd, double *xi_, double *design_, double *resp_, double *xstarts_,
		       double* poststarts_, double* lb_, double *ub_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, nsample, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	lagpFixvarLikelihood fixvarLik(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
				       *resvdThres_, *every_, *noiseVar_, *frac_, *gstart_,
				       xi_, design, resp);
	uniformPrior prior(nparam, lb_, ub_);
	normalKernel kernel(nparam, *kersd);
	mcmcBase mcmc(nparam, *nmc_, *nburn_, *nthin_, xstarts[i], poststarts_[i],
		      &prior, &fixvarLik, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }
  void lagpEigenNewtonInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
			  unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
			  unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
			  unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_,
			  unsigned int *nthread_, double *frac_, double* gstart_, double* kerthres_,
			  double* kersdfrac_, double *xi_,
			  double *design_, double *resp_, double *xstarts_, double* poststarts_, double* lb_,
			  double *ub_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, nsample, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	naivelikelihoodNewton likelihood(nparam, tlen, xi_);
	uniformPrior prior(nparam, lb_, ub_);
	eigenkernelNewton kernel(nparam, *kerthres_, *kersdfrac_);
	mcmcNewton mcmc(nparam, *nmc_, *nburn_, *nthin_,
			ndesign, tlen, *n0_, *nn_, *nfea_,
			*resvdThres_, *every_, *frac_, *gstart_,
			xi_, design, resp, xstarts[i], poststarts_[i],
			&prior, &likelihood, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }

  void lagpScalarNewtonInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
			   unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
			   unsigned int* n0_, unsigned int *nn_,
			   unsigned int *nfea_, unsigned int *every_, unsigned int *tlen_, unsigned int *islog_,
			   unsigned int *nthread_, double* gstart_, double* kerthres_,
			   double* kersdfrac_,
			   double *design_, double *resp_, double *xstarts_, double* poststarts_, double* lb_,
			   double *ub_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, nsample, slen;
    double **design, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    design = new_matrix_bones(design_,ndesign,nparam);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	scalarlikelihoodNewton likelihood(nparam,*tlen_, *islog_);
	uniformPrior prior(nparam, lb_, ub_);
	eigenkernelNewton kernel(nparam, *kerthres_, *kersdfrac_);
	mcmcScalarNewton mcmc(nparam, *nmc_, *nburn_, *nthin_, ndesign, *n0_, *nn_,
			      *nfea_, *every_, *tlen_, *islog_, *gstart_, design,
			      resp_, xstarts[i], poststarts_[i], &prior, &likelihood, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
  }


  void lagpScalarInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
		     unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
		     unsigned int* n0_, unsigned int *nn_, unsigned int *nfea_,
		     unsigned int *every_, unsigned int *nthread_, double* gstart_,
		     double* kersd_, double *design_, double *resp_, double *xstarts_,
		     double* poststarts_, double* lb_, double *ub_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, nsample, slen;
    double **design, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    design = new_matrix_bones(design_,ndesign,nparam);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	lagpScalarLikelihood likelihood(ndesign, nparam, *n0_, *nn_,
					*nfea_, *every_, *gstart_, design, resp_);
	uniformPrior prior(nparam, lb_, ub_);
	normalKernel kernel(nparam, *kersd_);
	mcmcBase mcmc(nparam, *nmc_, *nburn_, *nthin_, xstarts[i], poststarts_[i],
		      &prior, &likelihood, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
  }
    void lagpProfileAdaptiveInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
				unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
				unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
				unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_,
				unsigned int *nthread_, unsigned int *adpthres_, double *frac_, double* gstart_,
				double* kersd, double *xi_, double *design_, double *resp_, double *xstarts_,
				double* poststarts_, double* lb_, double *ub_, double *eps_, double* sval_,
				double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, nsample, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	lagpProfileLikelihood profileLik(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
					 *resvdThres_, *every_, *frac_, *gstart_,
					 xi_, design, resp);
	uniformPrior prior(nparam, lb_, ub_);
	kernelAdaptive kernel(nparam, *adpthres_, *eps_, *sval_, *kersd);
	mcmcAdaptive mcmc(nparam, *nmc_, *nburn_, *nthin_, xstarts[i], poststarts_[i],
			  &prior, &profileLik, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }
  void lagpFixvarAdaptiveInv(unsigned int* ndesign_, unsigned int *nparam_, unsigned int *nstarts_,
			     unsigned int* nmc_, unsigned int *nburn_, unsigned int *nthin_,
			     unsigned int *tlen_, unsigned int* n0_, unsigned int *nn_,
			     unsigned int *nfea_, unsigned int* resvdThres_, unsigned int *every_,
			     unsigned int *nthread_, unsigned int *adpthres_, double *noiseVar_, double *frac_, double* gstart_,
			     double* kersd, double *xi_, double *design_, double *resp_, double *xstarts_,
			     double* poststarts_, double* lb_, double *ub_, double *eps_, double *sval_, double *samples)
  {
    unsigned int mxth, ndesign, nparam, tlen, nsample, slen;
    double **design, **resp, **xstarts;
    ndesign = *ndesign_;
    nparam = *nparam_;
    tlen = *tlen_;
    design = new_matrix_bones(design_,ndesign,nparam);
    resp = new_matrix_bones(resp_, ndesign, tlen);
    xstarts = new_matrix_bones(xstarts_, *nstarts_, nparam);
    nsample = (*nmc_-*nburn_)/(*nthin_);
    slen = nsample * nparam;
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
      for(i = start; i < *nstarts_; i+=step)
      {
	lagpFixvarLikelihood fixvarLik(ndesign, nparam, tlen, *n0_, *nn_, *nfea_,
				       *resvdThres_, *every_, *noiseVar_, *frac_, *gstart_,
				       xi_, design, resp);
	uniformPrior prior(nparam, lb_, ub_);
	kernelAdaptive kernel(nparam, *adpthres_, *eps_, *sval_, *kersd);
	mcmcAdaptive mcmc(nparam, *nmc_, *nburn_, *nthin_, xstarts[i], poststarts_[i],
			  &prior, &fixvarLik, &kernel);
	mcmc.run();
	mcmc.getSample(samples+i*slen);
      }
#ifdef _OPENMP
    }
#endif
    free(xstarts);
    free(design);
    free(resp);
  }
}
