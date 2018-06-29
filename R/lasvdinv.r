evallogpost <- function(y,xi,tlen)
{
    dev <- drop(crossprod(y-xi))/tlen
    loglik <- -0.5*log(2*pi)-0.5*tlen
    loglik <- loglik - 0.5 * tlen* log(dev)
}
scalarlogpost <- function(y,xi,tlen)
{
    dev <- drop(crossprod(y-xi))
    loglik <- -0.5*tlen*log(dev)
}
scalardev <- function(y,xi,tlen)
{
    dev <- drop(crossprod(y-xi))
}
lasvdinv <- function(design, resp, xi, nstarts, nmc, n0, nn,
                     liktype=c("naive","profile"), nfea = min(1000,nrow(design)),
                     nsvd=nn, nburn=0, nthin=1,resvdThres = min(5, nn-n0),
                     every = min(5,nn-n0), frac = .95, gstart = 0.0001,
                     nthread = 4, lb = rep(0,ncol(design)), ub = rep(1,ncol(design)),
                     kersigfrac=.05)
{
    liktype <- match.arg(liktype)
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
    nsample <- floor((nmc-nburn)/nthin)
    funname <- if(liktype=="naive") "lagpNaiveInv" else "lagpProfileInv"
    out <- .C(funname,as.integer(ndesign), as.integer(nparam),
              as.integer(nstarts), as.integer(nmc), as.integer(nburn),
              as.integer(nthin), as.integer(tlen), as.integer(n0),
              as.integer(nn), as.integer(nfea), as.integer(resvdThres),
              as.integer(every), as.integer(nthread), as.double(frac),
              as.double(gstart), as.double(kersd), as.double(xi),
              as.double(t(design)), as.double(resp), as.double(t(xstarts)),
              as.double(poststarts), as.double(lb), as.double(ub),
              samples=double(nstarts*nparam*nsample))
    samples <- matrix(out$samples,nrow=nstarts*nsample,byrow=TRUE)
    return(samples)
}

lasvdnewtoninv <- function(design, resp, xi, nstarts, nmc, n0, nn,
                           nfea = min(1000,nrow(design)), nsvd=nn, nburn=0,
                           nthin=1,resvdThres = min(5, nn-n0),
                           every = min(5,nn-n0), frac = .95, gstart = 0.0001,
                           nthread = 4, lb = rep(0,ncol(design)), ub = rep(1,ncol(design)),
                           kerthres=1e-5,kersdfrac=1.0)
{
    ndesign <-  nrow(design)
    nparam <-  ncol(design)
    tlen <- nrow(resp)
    if(ncol(resp) != ndesign) stop("size response matrix does not aggree with design matrix!")
    diam <-  min(ub-lb)
    if(diam<0) stop("misspecified the upper or lower bound of the design domain")
    logpost <- apply(resp,2,evallogpost,xi,tlen)
    post <- exp(logpost)
    idx <- sample(1:ndesign,nstarts,replace=TRUE,prob=post)
    xstarts <- design[idx,,drop=FALSE]
    poststarts <- logpost[idx]
    nsample <- floor((nmc-nburn)/nthin)
    out <- .C("lagpEigenNewtonInv",as.integer(ndesign), as.integer(nparam),
              as.integer(nstarts), as.integer(nmc), as.integer(nburn),
              as.integer(nthin), as.integer(tlen), as.integer(n0),
              as.integer(nn), as.integer(nfea), as.integer(resvdThres),
              as.integer(every), as.integer(nthread), as.double(frac),
              as.double(gstart), as.double(kerthres), as.double(kersdfrac),as.double(xi),
              as.double(t(design)), as.double(resp), as.double(t(xstarts)),
              as.double(poststarts), as.double(lb), as.double(ub),
              samples=double(nstarts*nparam*nsample))
    samples <- matrix(out$samples,nrow=nstarts*nsample,byrow=TRUE)
    return(samples)
}

lagpscalarnewtoninv <- function(design, resp, xi, nstarts, nmc, n0, nn,
                                nfea = min(1000,nrow(design)), nburn=0,
                                nthin=1, every = min(5,nn-n0), islog=TRUE,
                                gstart = 0.0001, nthread = 4, lb = rep(0,ncol(design)),
                                ub = rep(1,ncol(design)),
                                kerthres=1e-5,kersdfrac=1.0)
{
    ndesign <-  nrow(design)
    nparam <-  ncol(design)
    tlen <- nrow(resp)
    if(ncol(resp) != ndesign) stop("size response matrix does not aggree with design matrix!")
    diam <-  min(ub-lb)
    if(diam<0) stop("misspecified the upper or lower bound of the design domain")
    logpost <- apply(resp,2,scalarlogpost,xi,tlen)
    resp <- if(islog) -logpost else apply(resp,2,scalardev,xi,tlen)
    post <- exp(logpost)
    idx <- sample(1:ndesign,nstarts,replace=TRUE,prob=post)
    xstarts <- design[idx,,drop=FALSE]
    poststarts <- logpost[idx]
    nsample <- floor((nmc-nburn)/nthin)
    out <- .C("lagpScalarNewtonInv",as.integer(ndesign), as.integer(nparam),
              as.integer(nstarts), as.integer(nmc), as.integer(nburn),
              as.integer(nthin),  as.integer(n0),
              as.integer(nn), as.integer(nfea),
              as.integer(every), as.integer(tlen), as.integer(islog), as.integer(nthread),
              as.double(gstart), as.double(kerthres), as.double(kersdfrac),
              as.double(t(design)), as.double(resp), as.double(t(xstarts)),
              as.double(poststarts), as.double(lb), as.double(ub),
              samples=double(nstarts*nparam*nsample))
    samples <- matrix(out$samples,nrow=nstarts*nsample,byrow=TRUE)
    return(samples)
}

lagpscalarinv <- function(design, resp, xi, nstarts, nmc, n0, nn,
                          nfea = min(1000,nrow(design)), nburn=0,
                          nthin=1, every = min(5,nn-n0), gstart = 0.0001,
                          nthread = 4, lb = rep(0,ncol(design)),
                          ub = rep(1,ncol(design)),
                          kersigfrac=0.05)
{
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
    nsample <- floor((nmc-nburn)/nthin)
    out <- .C("lagpScalarInv",as.integer(ndesign), as.integer(nparam),
              as.integer(nstarts), as.integer(nmc), as.integer(nburn),
              as.integer(nthin),  as.integer(n0),
              as.integer(nn), as.integer(nfea),
              as.integer(every), as.integer(nthread),
              as.double(gstart), as.double(kersd),
              as.double(t(design)), as.double(logpost), as.double(t(xstarts)),
              as.double(poststarts), as.double(lb), as.double(ub),
              samples=double(nstarts*nparam*nsample))
    samples <- matrix(out$samples,nrow=nstarts*nsample,byrow=TRUE)
    return(samples)
}
