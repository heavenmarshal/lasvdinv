#ifndef __LAGPLIKELIHOOD_HPP__
#define __LAGPLIKELIHOOD_HPP__
#include "mcmcutil.hpp"
extern "C"{
#include"matrix.h"
#include"linalg.h"
#include"lasvdgp.h"
}
struct lagpLikInfo
{
  int nbas;
  int tlen;
  double sse;
  double sig2hat;
  double* dev;
  double* d2;
  double* cs2;
  lagpLikInfo(int nbas_):nbas(nbas_){
    dev = new_vector(nbas_);
    d2 = new_vector(nbas_);
    cs2 = new_vector(nbas_);
  }
  ~lagpLikInfo(){
    free(dev);
    free(d2);
    free(cs2);
  }
};

class lagpLikelihood: public likelihoodBase{
public:
  lagpLikelihood(unsigned int ndesign_, unsigned int nparam_, unsigned int tlen_,
		 unsigned int n0_, unsigned int nn_, unsigned int nfea_,
		 unsigned int resvdThres_, unsigned int every_, double frac_,
		 double gstart_, double *xi_, double **design_, double **resp_):
    likelihoodBase(nparam_), ndesign(ndesign_), tlen(tlen_), n0(n0_), nn(nn_),
    nfea(nfea_), resvdThres(resvdThres_), every(every_), frac(frac_), gstart(gstart_),
    xi(xi_), design(design_), resp(resp_) {};
  virtual double evalLogLikelihood(const double* param){return 0.0;};

protected:
  unsigned int ndesign;
  unsigned int tlen;
  unsigned int n0;
  unsigned int nn;
  unsigned int nfea;
  unsigned int resvdThres;
  unsigned int every;
  double frac;
  double gstart;
  double *xi;			// field observation
  double **design;
  double **resp;
  lasvdGP *lasvdgp;
};

class lagpNaiveLikelihood: public lagpLikelihood{
public:
  lagpNaiveLikelihood(unsigned int ndesign_, unsigned int nparam_, unsigned int tlen_,
		      unsigned int n0_, unsigned int nn_, unsigned int nfea_,
		      unsigned int resvdThres_, unsigned int every_, double frac_,
		      double gstart_, double *xi_, double **design_, double **resp_):
    lagpLikelihood(ndesign_, nparam_, tlen_, n0_, nn_, nfea_, resvdThres_, every_, frac_,
		   gstart_, xi_, design_, resp_){};
  double evalLogLikelihood(double* param);
};

class lagpProfileLikelihood: public lagpLikelihood{
public:
  lagpProfileLikelihood(unsigned int ndesign_, unsigned int nparam_, unsigned int tlen_,
			unsigned int n0_, unsigned int nn_, unsigned int nfea_,
			unsigned int resvdThres_, unsigned int every_, double frac_,
			double gstart_, double *xi_, double **design_, double **resp_):
    lagpLikelihood(ndesign_, nparam_, tlen_, n0_, nn_, nfea_, resvdThres_, every_, frac_,
		   gstart_, xi_, design_, resp_){};
  double evalLogLikelihood(double* param);
private:
  static double nloglikelihood(double sig2eps, void* info);
  static void predlasvdGPutil(lasvdGP* lasvdgp, double* xpred, double* xi,
			      lagpLikInfo* info);

};
#endif
