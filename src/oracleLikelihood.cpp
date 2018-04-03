#include <cmath>
#include <Rinternals.h>
#include "oracleLikelihood.hpp"
extern "C"{
  #include "linalg.h"
  #include "matrix.h"
}

double oracleLikelihood::evalLogLikelihood(double *param)
{
  double dev, loglik, dtlen;
  double *pres, *pxi;
  SEXP R_fcall, res, x;
  PROTECT(x = allocVector(REALSXP,nparam));
  dupv(REAL(x), param, nparam);
  PROTECT(R_fcall = allocVector(LANGSXP,3));
  SETCAR(R_fcall,fun);
  SETCADR(R_fcall,x);
  SETCADDR(R_fcall,timepoints);
  PROTECT(res = eval(R_fcall,rho));
  pres = REAL(res);
  pxi = REAL(xi);
  dtlen = (double)tlen;
  linalg_daxpy(tlen, -1.0, pxi, 1, pres, 1);
  dev = linalg_ddot(tlen, pres, 1, pres, 1);
  dev /= dtlen;
  loglik = -0.5 * LOG2PI - 0.5 * dtlen;
  loglik -= 0.5*dtlen*log(dev);
  UNPROTECT(3);
  return loglik;
}
