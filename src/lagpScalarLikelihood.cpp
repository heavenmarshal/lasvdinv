#include "lagpScalarLikelihood.hpp"
extern "C"{
  #include "fitlagpsep.h"
  #include "gp_sep.h"
}
double lagpScalarLikelihood::evalLogLikelihood(double *param)
{
  double df, pmean;
  GPsep *gpsep;
  gpsep = fitlagpsep(nparam, ndesign, n0, nn, nfea,
		     every, param, design, resp, gstart);
  predGPsep_lite(gpsep, 1, &param, &pmean, NULL, &df, NULL);
  deleteGPsep(gpsep);
  return pmean;
}
