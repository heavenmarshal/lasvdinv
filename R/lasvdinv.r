evallogpost <- function(y,xi,tlen)
{
    dev <- drop(crossprod(y-xi))/tlen
    loglik <- -0.5*log(2*pi)-0.5*tlen
    loglik <- loglik - 0.5 * tlen* log(dev)
}
lasvdinv <- function(design, resp, xi, nstarts, nmc, n0, nn, noiseVar=0,
                     liktype=c("profile","fixvar"), kertype=c("normal","adaptive"),
                     nfea = min(1000,nrow(design)), nsvd=nn,resvdThres = min(5, nn-n0),
                     every = min(5,nn-n0), frac = .95, gstart = 0.0001,
                     nthread = 4, adpthres=100*ncol(design), eps=1e-5, sval=5.76/ncol(design)^2,
                     lb = rep(0,ncol(design)), ub = rep(1,ncol(design)), kersigfrac=.05)
{
    liktype <- match.arg(liktype)
    kertype <- match.arg(kertype)
    likcode <- if(liktype=="profile") 101 else 102
    kercode <- if(kertype=="normal") 201 else 202
    ndesign <-  nrow(design)
    nparam <-  ncol(design)
    tlen <- nrow(resp)
    if(ncol(resp) != ndesign) stop("size response matrix does not aggree with design matrix!")
    diam <-  min(ub-lb)
    if(diam<0) stop("misspecified the upper or lower bound of the design domain")
    kersd <- diam * kersigfrac
    logpost <- apply(resp,2,evallogpost,xi,tlen)
    post <- exp(logpost)
    idx <- sample(1:ndesign,nstarts,replace=TRUE,prob=post)
    xstarts <- design[idx,,drop=FALSE]
    poststarts <- logpost[idx]
    out <- .C("lasvdinv",as.integer(ndesign), as.integer(nparam),
              as.integer(likcode), as.integer(kercode), as.integer(nstarts),
              as.integer(nmc), as.integer(tlen), as.integer(n0),
              as.integer(nn), as.integer(nfea), as.integer(resvdThres),
              as.integer(every), as.integer(nthread), as.integer(adpthres),as.double(noiseVar),
              as.double(frac), as.double(gstart), as.double(kersd), as.double(xi),
              as.double(t(design)), as.double(resp), as.double(t(xstarts)),
              as.double(poststarts), as.double(lb), as.double(ub), as.double(eps),
              as.double(sval),samples=double(nstarts*nparam*nmc))
    samples <- matrix(out$samples,nrow=nstarts*nmc,byrow=TRUE)
    return(samples)
}
