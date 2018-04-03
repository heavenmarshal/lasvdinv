#include <Rinternals.h>
#include "mcmcutil.hpp"
#include "oracleLikelihood.cpp"
extern "C"{
  #include "matrix.h"
}
extern "C"{
  SEXP oracleInv(SEXP simulator, SEXP nparam_, SEXP nmc_, SEXP nburn_, SEXP nthin_,
		 SEXP kersd_, SEXP xi_, SEXP timepoints_, SEXP xstarts_,
		 SEXP poststarts_, SEXP lb_, SEXP ub_, SEXP rho_)
  {
    int i, nparam, tlen, nstarts, nmc, nburn;
    int nthin, nsample, slen;
    double **xstarts;
    SEXP samples;
    nparam = *INTEGER(nparam_);
    tlen = length(xi_);
    nstarts = nrows(xstarts_);
    nmc = *INTEGER(nmc_);
    nburn = *INTEGER(nburn_);
    nthin = *INTEGER(nthin_);
    nsample = (nmc-nburn)/nthin;
    slen = nsample * nparam;
    PROTECT(samples = allocVector(REALSXP, nstarts*slen));
    xstarts = new_matrix_bones(REAL(xstarts_),nstarts,nparam);
    for(i = 0; i < nstarts; ++i)
    {
      oracleLikelihood oracleLik(nparam, tlen, simulator, timepoints_, xi_, rho_);
      uniformPrior prior(nparam, REAL(lb_), REAL(ub_));
      normalKernel kernel(nparam, *REAL(kersd_));
      mcmcBase mcmc(nparam, nmc, nburn, nthin, xstarts[i],
		    REAL(poststarts_)+i, prior, oracleLik, kernel);
      mcmc.run();
      mcmc.getSample(REAL(samples)+i*slen);
    }
    free(xstarts);
    UNPROTECT(1);

    return samples;
  }
}
