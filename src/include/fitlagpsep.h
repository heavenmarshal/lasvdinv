#ifndef __FITLAGPSEP_H__
#define __FITLAGPSEP_H__
#include "gp_sep.h"

GPsep* fitlagpsep(unsigned int nparam, unsigned int ndesign, unsigned int n0,
		  unsigned int nn, unsigned int nfea, unsigned int every,
		  double *param, double **design, double *resp, double gstart);

void scalargpgradhess(GPsep* gpsep, int nparam, double *param,
		      double *grad, double **hess);
#endif
