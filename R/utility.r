nearestNN <- function(xref, X, nclose)
{
    if(!is.matrix(X)) stop("X must be a matrix")
    nn <- nrow(X)
    m <- ncol(X)
    if(!is.matrix(xref) && length(xref) != m)
        stop("illegal specification of xref")
    if(!is.matrix(xref)) xref <- matrix(xref,ncol=m)
    if(ncol(xref) != m) stop("the dimensions of xref and X do not match")
    if(nclose > nn) stop("nclose must be no larger than n")
    nref <- nrow(xref)
    out <- .C("nearestNN_R", as.double(t(xref)), as.double(t(X)),
              as.integer(nref), as.integer(nn), as.integer(m),
              as.integer(nclose), ans = integer(nn),
              PACKAGE="lasvdinv")
    ret <- out$ans[1:nclose]+1
    return(ret)
}
