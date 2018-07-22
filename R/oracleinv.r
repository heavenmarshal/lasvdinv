evalloglik <- function(x,xi,timepoints,simulator)
{
    tlen <- length(timepoints)
    y <- simulator(x,timepoints)
    dev <- drop(crossprod(y-xi))/tlen
    loglik <- -0.5*log(2*pi)-0.5*tlen
    loglik <- loglik - 0.5 * tlen * log(dev)
}
evallogprior <- function(x, lb, ub)
{
    logprior <- if(all(x<=ub & x>=lb)) 0.0 else -Inf
}

nkpropose <- function(from, sigma, nparam)
{
    to <- from+rnorm(nparam,0,sigma)
}
nklogdensity <- function(from, to, sigma, nparam)
{
    logden <- sum(dnorm(to,from,sigma,TRUE))
}
oraclemcmc <- function(xstart,poststart,xi,simulator,timepoints,nmc,nburn,nthin,
                       lb,ub,sigma)
{
    nparam <- length(xstart)
    nsample <- floor((nmc-nburn)/nthin)
    samples <- matrix(nrow=nsample,ncol=nparam)
    clogpost <- poststart
    cx <- xstart
    j <- -nburn
    k <- 1
    for(i in 1:nmc)
    {
        xnew <- nkpropose(cx,sigma,nparam)
        logpost <- evalloglik(xnew,xi,timepoints,simulator)
        logpost <- logpost+evallogprior(xnew,lb,ub)
        logaccprob <- logpost+nklogdensity(xnew,cx,sigma,nparam)
        logaccprob <- logaccprob-clogpost-nklogdensity(cx,xnew,sigma,nparam)
        if(is.na(logaccprob)) logaccprob <- -Inf
        logru <- log(runif(1))
        if(logru<logaccprob)
        {
            cx <- xnew
            clogpost <- logpost
        }
        if(j>=0 && j %% nthin==0)
        {
            samples[k,]=cx
            k <- k+1
        }
        j <- j+1
    }
    return(samples)
}
oracleinv <- function(design,resp,xi,simulator,timepoints,nstarts,nmc,nburn=0,nthin=1,
                      lb=rep(0,ncol(design)), ub=rep(1,ncol(design)),
                      kersigfrac=0.05)
{
    ndesign <-  nrow(design)
    nparam <-  ncol(design)
    tlen <- nrow(resp)
    if(ncol(resp) != ndesign) stop("size response matrix does not aggree with design matrix!")
    diam <-  min(ub-lb)
    if(diam<0) stop("misspecified the upper or lower bound of the design domain")
    sigma <- diam * kersigfrac
    logpost <- apply(resp,2,evallogpost,xi,tlen)
    post <- exp(logpost)
    idx <- sample(1:ndesign,nstarts,replace=TRUE,prob=post)
    xstarts <- design[idx,,drop=FALSE]
    poststart <- logpost[idx]
    xstarts <- as.list(as.data.frame(t(xstarts)))
    poststart <- as.list(poststart)
    splist <- mapply(oraclemcmc,xstarts,poststart,
                     MoreArgs=list(xi=xi,simulator=simulator,
                     timepoints=timepoints,nmc=nmc,nburn=nburn,
                     nthin=nthin,lb=lb,ub=ub,sigma=sigma),
                     SIMPLIFY=FALSE)
    samples <- do.call(rbind,splist)
    return(samples)
}
