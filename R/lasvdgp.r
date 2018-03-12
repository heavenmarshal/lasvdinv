lasvdgp <- function(center, xpred, design, resp,
                    n0, nn, nfea = min(1000, nrow(design)),
                    nsvd = nn, nadd = 1,
                    frac = .95, gstart = .001,
                    resvdThres = min(5, nn-n0),
                    every = min(5, nn-n0),
                    maxit = 100, verb = 0)
{
    if(!is.matrix(design)) stop("design must be a matrix")
    if(!is.matrix(resp)) stop("resp must be a matrix")
    N <- nrow(design)
    m <- ncol(design)
    if(length(center) != m) stop("illegal center point!")
    if(ncol(resp) != N)
        stop("number of design points and responses are not consistent")
    tlen <- nrow(resp)
    if(!is.matrix(xpred) && length(xpred) != m)
        stop("illegal form of prediction set")
    if(!is.matrix(xpred)) xpred <- matrix(xpred,ncol=m)
    if(ncol(xpred) != m) stop("dimensions of design and prediction set are not consistent")
    npred = nrow(xpred)
    if(nfea > N || nfea <0) stop("illegal nfea")
    if(n0>nfea || n0 <0) stop("illegal n0")
    if(nn<n0 || nn>nfea) stop("illegal nn")
    if(nsvd<n0 || nsvd > nfea) stop("illegal nsvd")
    if(nadd < 0) stop("illegal nadd")
    if(resvdThres < 0) stop("illegal resvdThres")
    if(every < 0) stop("illegal every")
    out <- .C("lasvdGP_R",as.double(center), as.double(t(xpred)), as.double(t(design)),
              as.double(resp), as.integer(N), as.integer(m), as.integer(npred),
              as.integer(tlen), as.integer(nn), as.integer(n0), as.integer(nfea),
              as.integer(nsvd), as.integer(nadd), as.double(frac), as.double(gstart),
              as.integer(resvdThres), as.integer(every), as.integer(maxit),
              as.integer(verb), pmean = double(tlen*npred), ps2 = double(tlen*npred),
              PACKAGE="lasvdinv")
    ret <- list(pmean = matrix(out$pmean,nrow=tlen), ps2 = matrix(out$ps2,nrow=tlen))
    return(ret)
}

lasvdgpms <- function(center, xpred, design, resp,
                      n0, nn, nfea = min(1000, nrow(design)),
                      nsvd = nn, nadd = 1,
                      frac = .95, gstart = .001,
                      resvdThres = min(5, nn-n0),
                      every = min(5, nn-n0),
                      numstarts = 5, maxit = 100, verb = 0)
{
    if(!is.matrix(design)) stop("design must be a matrix")
    if(!is.matrix(resp)) stop("resp must be a matrix")
    N <- nrow(design)
    m <- ncol(design)
    if(length(center) != m) stop("illegal center point!")
    if(ncol(resp) != N)
        stop("number of design points and responses are not consistent")
    tlen <- nrow(resp)
    if(!is.matrix(xpred) && length(xpred) != m)
        stop("illegal form of prediction set")
    if(!is.matrix(xpred)) xpred <- matrix(xpred,ncol=m)
    if(ncol(xpred) != m) stop("dimensions of design and prediction set are not consistent")
    npred = nrow(xpred)
    if(nfea > N || nfea <0) stop("illegal nfea")
    if(n0>nfea || n0 <0) stop("illegal n0")
    if(nn<n0 || nn>nfea) stop("illegal nn")
    if(nsvd<n0 || nsvd > nfea) stop("illegal nsvd")
    if(nadd < 0) stop("illegal nadd")
    if(resvdThres < 0) stop("illegal resvdThres")
    if(every < 0) stop("illegal every")
    if(numstarts < 0) stop("illegal numstarts")

    out <- .C("lasvdGPms_R", as.double(center), as.double(t(xpred)),
              as.double(t(design)), as.double(resp), as.integer(N),
              as.integer(m), as.integer(npred), as.integer(tlen),
              as.integer(nn), as.integer(n0), as.integer(nfea),
              as.integer(nsvd), as.integer(nadd), as.double(frac),
              as.double(gstart), as.integer(resvdThres), as.integer(every),
              as.integer(numstarts), as.integer(maxit), as.integer(verb),
              pmean = double(tlen*npred), ps2 = double(tlen*npred),
              PACKAGE="lasvdinv")
    ret <- list(pmean = matrix(out$pmean,nrow=tlen), ps2 = matrix(out$ps2,nrow=tlen))
    return(ret)
}
evalesl2 <- function(center, design, resp, yobs, N, m, tlen, nn, n0,
                     nclose, nneig, nfea, nsvd, nadd, frac, gstart,
                     resvdThres, every, maxit, verb)
{
    cidx <- nearestNN(center, design, nclose)
    radi <-  sqrt(apply((t(design[cidx,])-center)^2,2,sum))
    radi <- max(radi)
    ## generate feasible set
    xx <- lhs::maximinLHS(nneig, m+1)
    rr <- xx[,m+1]
    xx <- qnorm(xx[,1:m])
    xx <- xx/sqrt(apply(xx^2,1,sum))
    xx <- xx*rr^(1/m)*radi
    xpred <- t(xx)+center
    valid <- xpred >=0 & xpred <= 1
    valid <- apply(valid,2,all)
    xpred <- cbind(center, xpred[,valid])
    npred <- ncol(xpred)
    out <- .C("evalesl2_R", as.double(center), as.double(xpred), as.double(t(design)),
              as.double(resp), as.double(yobs), as.integer(N), as.integer(m),
              as.integer(npred), as.integer(tlen), as.integer(nn), as.integer(n0),
              as.integer(nfea), as.integer(nsvd), as.integer(nadd), as.double(frac),
              as.double(gstart), as.integer(resvdThres), as.integer(every),
              as.integer(maxit), as.integer(verb), esl2=double(npred))
    esl2 <- out$esl2
    optidx <- which.min(esl2)
    xopt <- xpred[,optidx]
    esl2opt <- min(esl2)
    ret <- list(xopt=xopt,esl2opt=esl2opt)
}

evalesl2ms <- function(center, design, resp, yobs, N, m, tlen, nn, n0,
                       nclose, nneig, nfea, nsvd, nadd, frac, gstart,
                       resvdThres, every, numstarts, maxit, verb)
{
    cidx <- nearestNN(center, design, nclose)
    radi <-  sqrt(apply((t(design[cidx,])-center)^2,2,sum))
    radi <- max(radi)
    ## generate feasible set
    xx <- lhs::maximinLHS(nneig, m+1)
    rr <- xx[,m+1]
    xx <- qnorm(xx[,1:m])
    xx <- xx/sqrt(apply(xx^2,1,sum))
    xx <- xx*rr^(1/m)*radi
    xpred <- t(xx)+center
    valid <- xpred >=0 & xpred <= 1
    valid <- apply(valid,2,all)
    xpred <- cbind(center, xpred[,valid])
    npred <- ncol(xpred)
    out <- .C("evalesl2ms_R", as.double(center), as.double(xpred), as.double(t(design)),
              as.double(resp), as.double(yobs), as.integer(N), as.integer(m),
              as.integer(npred), as.integer(tlen), as.integer(nn), as.integer(n0),
              as.integer(nfea), as.integer(nsvd), as.integer(nadd), as.double(frac),
              as.double(gstart), as.integer(resvdThres), as.integer(every),
              as.integer(numstarts), as.integer(maxit), as.integer(verb),
              esl2=double(npred))
    esl2 <- out$esl2
    optidx <- which.min(esl2)
    xopt <- xpred[,optidx]
    esl2opt <- min(esl2)
    ret <- list(xopt=xopt,esl2opt=esl2opt)
}
