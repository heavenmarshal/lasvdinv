#ifndef __LAGPSCALARLIKELIHOOD_HPP__
#define __LAGPSCALARLIKELIHOOD_HPP__
#include "mcmcutil.hpp"

class lagpScalarLikelihood: public likelihoodBase{
public:
  lagpScalarLikelihood(unsigned int ndesign_, unsigned int nparam_, unsigned int n0_,
		       unsigned int nn_, unsigned int nfea_, unsigned int every_,
		       double gstart_, double **design_, double *resp_):
    likelihoodBase(nparam_), ndesign(ndesign_), n0(n0_), nn(nn_), nfea(nfea_),
    every(every_), gstart(gstart_), design(design_), resp(resp_) {};
  double evalLogLikelihood(double *param);
private:
  unsigned int ndesign, nparam, n0, nn, nfea, every;
  double gstart, **design, *resp;
};
#endif
