#ifndef __ORACLELIKELIHOOD_HPP__
#define __ORACLELIKELIHOOD_HPP__
#include <Rinternals.h>
#include "mcmcutil.hpp"
class oracleLikelihood: public likelihoodBase{
public:
  oracleLikelihood(int nparam, int tlen_, SEXP func_, SEXP timepoints_,
		   SEXP xi_,SEXP rho_):
    likelihoodBase(nparam), tlen(tlen_), func(func_), timepoints(timepoints_),
    xi(xi_), rho(rho_)
  {};
  double evalLogLikelihood(double *param);
private:
  int tlen;
  SEXP func;
  SEXP timepoints;
  SEXP xi;
  SEXP rho;
};

#endif
