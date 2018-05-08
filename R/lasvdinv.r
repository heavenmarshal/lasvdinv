## assume the input domain is [0,1]^m
## lasvdinv <- function(design, resp, yobs, nn, n0=ceiling(0.5*nn),
##                      npurs = ceiling(.05*nrow(design)), nclose = 1,
##                      nneig=4*ncol(design),
##                      nfea = min(1000,nrow(design)), nsvd=nn,
##                      nadd=1, frac=.95, gstart=0.001,
##                      resvdThres=min(5, nn-n0), every=min(5, nn-n0),
##                      maxit=100, verb=0, nthread=4)
## {
##     N <- nrow(design)
##     m <- ncol(design)
##     if(ncol(resp) != N) stop("number of design points and responses must be consistent")
##     tlen <- nrow(resp)
##     if(length(yobs) != tlen) stop("inconsistent number of timepoints in ybos and resp")

##     eucdist <- sqrt(apply((resp-yobs)^2,2,sum))
##     centidx <- order(eucdist)[1:npurs]
##     centers <- design[centidx,,drop=FALSE]
##     cl <- makeCluster(nthread)
##     ret <- tryCatch(parApply(cl,centers,1, evalesl2, design, resp, yobs, N,
##                              m, tlen, nn, n0, nclose, nneig, nfea, nsvd, nadd,
##                              frac, gstart, resvdThres, every, maxit, verb),
##                     finally=stopCluster(cl))
##     xopt <- matrix(unlist(sapply(ret,`[`,"xopt")),ncol=m,byrow=TRUE)
##     esl2opt <- unlist(sapply(ret,`[`,"esl2opt"))
##     optidx <- which.min(esl2opt)
##     bestx <- xopt[optidx,]
##     ret <- list(xopt=xopt,esl2opt=esl2opt,bestx=bestx)
##     return(ret)
## }

## lasvdinvms <- function(design, resp, yobs, nn, n0=ceil(0.5*nn),
##                        npurs = ceil(.05*nrow(design)), nclose = 1,
##                        nneig=4*ncol(design),
##                        nfea = min(1000,nrow(design)), nsvd=nn,
##                        nadd=1, frac=.95, gstart=0.001,
##                        resvdThres=min(5, nn-n0), every=min(5, nn-n0),
##                        numstarts=5, maxit=100, verb=0, nthread=4)
## {
##     N <- nrow(design)
##     m <- ncol(design)
##     if(ncol(resp) != N) stop("number of design points and responses must be consistent")
##     tlen <- nrow(resp)
##     if(length(yobs) != tlen) stop("inconsistent number of timepoints in ybos and resp")

##     eucdist <- sqrt(apply((resp-yobs)^2,2,sum))
##     centidx <- order(eucdist)[1:npurs]
##     centers <- design[centidx,,drop=FALSE]
##     cl <- makeCluster(nthread)
##     ret <- tryCatch(parApply(cl,centers,1, evalesl2ms, design, resp, yobs, N,
##                              m, tlen, nn, n0, nclose, nneig, nfea, nsvd, nadd,
##                              frac, gstart, resvdThres, every, numstarts, maxit, verb),
##                     finally=stopCluster(cl))
##     xopt <- matrix(unlist(sapply(ret,`[`,"xopt")),ncol=m,byrow=TRUE)
##     esl2opt <- unlist(sapply(ret,`[`,"esl2opt"))
##     optidx <- which.min(esl2opt)
##     bestx <- xopt[optidx,]
##     ret <- list(xopt=xopt,esl2opt=esl2opt,bestx=bestx)
##     return(ret)
## }
evallogpost <- function(y,xi,tlen)
{
    dev <- drop(crossprod(y-xi))/tlen
    loglik <- -0.5*log(2*pi)-0.5*tlen
    loglik <- loglik - 0.5 * tlen* log(dev)
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
                                nthin=1, every = min(5,nn-n0), gstart = 0.0001,
                                nthread = 4, lb = rep(0,ncol(design)),
                                ub = rep(1,ncol(design)),
                                kerthres=1e-5,kersdfrac=1.0)
{
    ndesign <-  nrow(design)
    nparam <-  ncol(design)
    tlen <- nrow(resp)
    if(ncol(resp) != ndesign) stop("size response matrix does not aggree with design matrix!")
    diam <-  min(ub-lb)
    if(diam<0) stop("misspecified the upper or lower bound of the design domain")
    logpost <- apply(resp,2,evallogpost,xi,tlen)
    nlogpost <- -logpost
    post <- exp(logpost)
    idx <- sample(1:ndesign,nstarts,replace=TRUE,prob=post)
    xstarts <- design[idx,,drop=FALSE]
    poststarts <- logpost[idx]
    nsample <- floor((nmc-nburn)/nthin)
    out <- .C("lagpScalarNewtonInv",as.integer(ndesign), as.integer(nparam),
              as.integer(nstarts), as.integer(nmc), as.integer(nburn),
              as.integer(nthin),  as.integer(n0),
              as.integer(nn), as.integer(nfea),
              as.integer(every), as.integer(nthread),
              as.double(gstart), as.double(kerthres), as.double(kersdfrac),
              as.double(t(design)), as.double(nlogpost), as.double(t(xstarts)),
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
