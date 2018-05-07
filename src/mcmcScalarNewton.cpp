#include "mcmcScalarNewton.hpp"
#include <cstdlib>
extern "C"{
  #include "linalg.h"
  #include "matrix.h"
  #include "covar_sep.h"
  #include "lasvdgp.h"
  #include "matrixext.h"
  #include "myjmlegpsep.h"
}
static const double dab1 = 1.5;
static const double sqreps = 1.490116119384766E-8;
static double gab[2] = {0.0, 0.0};

static double * new_p_subvector(int *p, double *v, unsigned int len)
{
  int i;
  double *newv;
  newv = new_vector(len);
  for(i=0; i<len; ++i)
    newv[i] = v[p[i]];
  return newv;
}
static void flipvector(double *vec, unsigned int len)
{
  int i;
  for(i=0; i<len; ++i)
    vec[i] = -vec[i];
}
static void const_vector(double *vec, double scalar, unsigned int len)
{
  int i;
  for(i=0; i<len; ++i)
    vec[i] = scalar;
}

mcmcScalarNewton::mcmcScalarNewton(int nparam_, int nmc_, int nburn_, int nthin_, unsigned int ndesign_,
				   unsigned int n0_, unsigned int nn_, unsigned int nfea_,
				   unsigned int every_, double gstart_, double **design_,
				   double *resp_, double *x0, double post0, priorBase *prior_,
				   likelihoodNewton *likelihood_, kernelNewton *kernel_):
  mcmcBase(nparam_, nmc_, nburn_, nthin_, x0, post0, prior_, NULL, NULL), ndesign(ndesign_), n0(n0_),
  nn(nn_), nfea(nfea_), every(every_), gstart(gstart_), design(design_), resp(resp_),
  likelihoodn(likelihood_), kerneln(kernel_)
{
  grad = new_vector(nparam);
  hess = new_matrix(nparam,nparam);
  cgrad = NULL;
  chess = NULL;
}

mcmcScalarNewton::~mcmcScalarNewton()
{
  free(grad);
  delete_matrix(hess);
  if(cgrad != NULL)
    free(cgrad);
  if(chess != NULL)
    delete_matrix(chess);
}
void mcmcScalarNewton::selectpoint(GPsep* gpsep, int nparam, int nfea, int *feaidx,
				   double *param, double **design, double *resp)
{
  int addidx, ndes, cnfea, *feastart;
  double **xcand, *criter, **xadd, zadd;
  ndes = gpsep->n;
  cnfea = nfea - ndes;
  feastart = feaidx + ndes;
  xcand = new_p_submatrix_rows(feastart, design, cnfea, nparam, 0);
  criter = new_vector(cnfea);

  alcGPsep(gpsep, cnfea, xcand, 1, &param, 0, criter);
  flipvector(criter,cnfea);
  quick_select_index(criter,feastart, cnfea, 1);
  xadd = new_p_submatrix_rows(feastart, design, 1, nparam, 0);
  zadd = resp[feastart[0]];
  updateGPsep(gpsep, 1, xadd, &zadd, 0);
  delete_matrix(xcand);
  delete_matrix(xadd);
  free(criter);

}

void mcmcScalarNewton::fitlagp(double *param)
{
  int i, dits, gits, dconv, niter;
  int *feaidx, seqs[2] = {nfea, n0};
  double ds, dmin, dmax, dab2, df, dab[2], grange[2]={sqreps,gstart};
  double *dstart, *ddmin, *ddmax, **subdes, *subresp;
  feaidx = nearest_indices(nparam, 1, &param, ndesign, design,
			   seqs, 2);
  subdes = new_p_submatrix_rows(feaidx, design, n0, nparam, 0);
  subresp = new_p_subvector(feaidx, resp, n0);
  getDs(subdes, n0, nparam, &ds, &dmin, &dmax, &dab2);
  dstart = new_const_vector(ds, nparam);
  ddmin = new_const_vector(dmin, nparam);
  ddmax = new_const_vector(dmax, nparam);
  dab[0] = dab1;
  dab[1] = dab2;
  gpsep = newGPsep(nparam, n0, subdes, subresp, dstart, gstart, 1);
  myjmleGPsep(gpsep, 100, ddmin, ddmax, grange, dab, gab, 0, &dits, &gits, &dconv);
  niter = nn-n0;
  for(i=1; i<=niter; ++i)
  {
    selectpoint(gpsep, nparam, nfea, feaidx, param, design, resp);
    if(i % every == 0)
    {
      getDs(gpsep->X, gpsep->n, nparam, &ds, &dmin, &dmax, &dab2);
      dab[1] = dab2;
      const_vector(ddmin,dmin,nparam);
      const_vector(ddmax,dmax,nparam);
      myjmleGPsep(gpsep, 100, ddmin, ddmax, grange, dab, gab, 0, &dits, &gits, &dconv);
    }
  }
  predGPsep_lite(gpsep, 1, &param, &pmean, &ps2, &df, NULL);
  evalgradhessian(param);
  deleteGPsep(gpsep);
  gpsep=NULL;
  delete_matrix(subdes);
  free(feaidx);
  free(subresp);
  free(ddmin);
  free(ddmax);
}
void mcmcScalarNewton::evalgradhessian(double *param)
{
  int i, j, k, nn;
  double gradi, hessi, *corr, *KiZ, *d, **X;
  nn = gpsep->n;
  corr = new_vector(nn);
  X = gpsep->X;
  d = gpsep->d;
  covar_sep(nparam, &param, 1, X, nn, d, &corr);
  KiZ = gpsep->KiZ;
  for(i = 0; i < nparam; ++i)
  {
    gradi = 0.0;
    hessi = 0.0;
    for(k = 0; k < nn; ++k)
    {
      gradi += 2.0*(X[k][i] - param[i])/d[i]*corr[k]*KiZ[k];
      hessi += 2.0*corr[k]*(2.0*sq((X[k][i]-param[i])/d[i])-1.0/d[i])*KiZ[k];
    }
    grad[i] = gradi;
    hess[i][i] = hessi;  
    for(j = 0; j< i; ++j)
    {
      hessi = 0.0;
      for(k = 0; k < nn; ++k)
	hessi += 4.0 *corr[k] *(X[k][i]-param[i])/d[i]*(X[k][j]-param[j])/d[j]*KiZ[k];
      hess[i][j] = hessi;
      hess[j][i] = hessi;
    }
  }
  free(corr);
}
void mcmcScalarNewton::accept()
{
  if(cgrad != NULL)
    free(cgrad);
  if(chess != NULL)
    delete_matrix(chess);
  cgrad = grad;
  chess = hess;
  grad = new_vector(nparam);
  hess = new_matrix(nparam,nparam);
}
void mcmcScalarNewton::run()
{
  int i, j, k;
  double logpost, logaccprob, logru;
  double *proposal;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  proposal = new_vector(nparam);
  fitlagp(current);
  accept();
  for(i=0, j=-nburn, k=0; i<nmc; ++i, ++j)
  {
    kerneln -> procGradHess(cgrad,chess);
    kerneln -> propose(current,proposal);
    logaccprob = -clogpost - kerneln->logDensity(current, proposal);
    fitlagp(proposal);
    logpost = evalLogPosterior(proposal);
    kerneln -> procGradHess(grad,hess);
    logaccprob += logpost + kerneln ->logDensity(proposal,current);
    
    logru = distribution(generator);
    logru = log(logru);
    if(logru < logaccprob)
    {
      dupv(current,proposal, nparam);
      clogpost = logpost;
      accept();
      naccept++;
    }
    if(j >= 0 && j % nthin ==0)
    {
      dupv(sample[k],current,nparam);
      k ++;
    }
  }
  free(proposal);
}

double mcmcScalarNewton::evalLogPosterior(double *param)
{
  double post;
  post = likelihoodn -> evalLogLikelihood(param, &pmean, &ps2);
  post += prior->evalLogPrior(param);
  return post;
}

