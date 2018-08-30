/********************************************************************************
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2005, University of California
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
 *
 ********************************************************************************/


#include "rhelp.h"

#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>


#define DEBUG


/*
 * replace matrix with zeros
 */

void zero(double **M, unsigned int n1, unsigned int n2)
{
  unsigned int i, j;
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) M[i][j] = 0;
}

/*
 * replace square matrix with identity
 */

void id(double **M, unsigned int n)
{
  unsigned int i;
  zero(M, n, n);
  for(i=0; i<n; i++) M[i][i] = 1.0;
}


/*
 * same as new_matrix below, but for creating
 * n x n identity matrices
 */

double ** new_id_matrix(unsigned int n)
{
  unsigned int i;
  double** m = new_zero_matrix(n, n);
  for(i=0; i<n; i++) m[i][i] = 1.0;
  return m;
}


/*
 * same as new_matrix below, but zeros out the matrix
 */

double ** new_zero_matrix(unsigned int n1, unsigned int n2)
{
  unsigned int i, j;
  double **m = new_matrix(n1, n2);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) m[i][j] = 0.0;
  return m;
}

/*
 * create a new n1 x n2 matrix which is allocated like
 * an n1*n2 array, but can be referenced as a 2-d array
 */

double ** new_matrix(unsigned int n1, unsigned int n2)
{
  int i;
  double **m;

  if(n1 == 0 || n2 == 0) return NULL;

  m = (double**) malloc(sizeof(double*) * n1);
  assert(m);
  m[0] = (double*) malloc(sizeof(double) * (n1*n2));
  assert(m[0]);

  for(i=1; i<n1; i++) m[i] = m[i-1] + n2;

  return m;
}


/*
 * create a double ** Matrix from a double * vector
 * should be freed with the free command, rather than
 * delete_matrix
 */

double ** new_matrix_bones(double *v, unsigned int n1, unsigned int n2)
{
  double **M;
  int i;
  M = (double **) malloc(sizeof(double*) * n1);
  M[0] = v;
  for(i=1; i<n1; i++) M[i] = M[i-1] + n2;
  return(M);
}

/*
 * create a new n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 */

double ** new_dup_matrix(double** M, unsigned int n1, unsigned int n2)
{
  double **m;

  if(n1 <= 0 || n2 <= 0) {
    /* assert(M == NULL); */
    return NULL;
  }

  m = new_matrix(n1, n2);
  dup_matrix(m, M, n1, n2);
  return m;
}


/*
 * copy M2 to M1
 */

void dup_matrix(double** M1, double **M2, unsigned int n1, unsigned int n2)
{
  unsigned int i;
  if(n1 == 0 || n2 == 0) return;
  assert(M1 && M2);
  for(i=0; i<n1; i++) dupv(M1[i], M2[i], n2);
}

/*
 * create a bigger n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 * deletes the old matrix
 */

double ** new_bigger_matrix(double** M, unsigned int n1, unsigned int n2,
		unsigned int n1_new, unsigned int n2_new)
{
  int i;
  double **m;

  assert(n1_new >= n1);
  assert(n2_new >= n2);

  if(n1_new <= 0 || n2_new <= 0) {
    assert(M == NULL);
    return NULL;
  }

  if(M == NULL) {
    assert(n1 == 0 || n2 == 0);
    return new_zero_matrix(n1_new, n2_new);
  }

  if(n2 == n2_new) {
    m = (double**) malloc(sizeof(double*) * n1_new);
    assert(m);
    m[0] = realloc(M[0], sizeof(double) * n1_new * n2_new);
    free(M);
    assert(m[0]);
    for(i=1; i<n1_new; i++) m[i] = m[i-1] + n2_new;
    zerov(m[n1], (n1_new-n1)*n2_new);
  } else {
    m = new_zero_matrix(n1_new, n2_new);
    dup_matrix(m, M, n1, n2);
    delete_matrix(M);
  }
  return m;
}

/*
 * delete a matrix allocated as above
 */

void delete_matrix(double** m)
{
  if(m == NULL) return;
  assert(*m);
  free(*m);
  assert(m);
  free(m);
}


/*
 * print an n x col matrix allocated as above out an opened outfile.
 * actually, this routine can print any double**
 */

void printMatrix(double **M, unsigned int n, unsigned int col, FILE *outfile)
{
  int i,j;
  assert(outfile);
  if(n > 0 && col > 0) assert(M);
  for(i=0; i<n; i++) {
    for(j=0; j<col; j++) {
#ifdef DEBUG
      if(j==col-1) MYprintf(outfile, "%.20f\n", M[i][j]);
      else MYprintf(outfile, "%.20f ", M[i][j]);
#else
      if(j==col-1) MYprintf(outfile, "%g\n", M[i][j]);
      else MYprintf(outfile, "%g ", M[i][j]);
#endif
    }
  }
}

/*
 * wmean_of_columns:
 *
 * fill mean[n2] with the weighted mean of the columns of M (n1 x n2);
 * weight vector should have length n1;
 */

void wmean_of_columns(double *mean, double **M, unsigned int n1, unsigned int n2,
		      double *weight)
{
  unsigned int i,j;
  double sw;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(mean && M);

  /* find normailzation constant */
  if(weight) sw = sumv(weight, n1);
  else sw = (double) n1;

  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    mean[i] = 0;
    if(weight) for(j=0; j<n1; j++) mean[i] += weight[j] * M[j][i];
    else for(j=0; j<n1; j++) mean[i] += M[j][i];
    mean[i] = mean[i] / sw;
  }
}
/*
 * min_of_columns:
 *
 * fill s[n1] with the min of the columns of M (n1 x n2);
 */

void min_of_columns(double *s, double **M, unsigned int n1, unsigned int n2)
{
  unsigned int i,j;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(s && M);

  /* calculate sum of columns */
  for(i=0; i<n2; i++) {
    s[i] = M[0][i];
    for(j=1; j<n1; j++) if(M[j][i] < s[i]) s[i] = M[j][i];
  }
}

/*
 * data structure for sorting weighted samples
 * to estimate quantiles
 */
/*
 * allocate and return an array of length n with scale*1 at
 * each entry
 */

double* ones(unsigned int n, double scale)
{
  double *o;
  unsigned int i;
  /* o = (double*) malloc(sizeof(double) * n); */
  o = new_vector(n);
  /* assert(o); */
  for(i=0; i<n; i++) o[i] = scale;
  return o;
}

/*
 * allocate and return an array containing
 * the integer seqence [from...to]
 */

int* iseq(double from, double to)
{
  unsigned int n,i;
  int by;
  int *s = NULL;

  if(from <= to) {
    n = (unsigned int) (to - from) + 1;
    by = 1;
  } else {
    assert(from > to);
    n = (unsigned int) (from - to) + 1;
    by = -1;
  }

  if(n == 0) return NULL;

  s = new_ivector(n);
  s[0] = from;
  for(i=1; i<n; i++) {
    s[i] = s[i-1] + by;
  }
  return s;
}

/*
 * Returns the kth smallest value in the array arr[1..n].  The input
 * array will be rearranged to have this value in location arr[k] ,
 * with all smaller elements moved to arr[1..k-1] (in arbitrary order)
 * and all larger elements in arr[k+1..n] (also in arbitrary order).
 * (from Numerical Recipies in C)
 *
 * This Quickselect routine is based on the algorithm described in
 * "Numerical recipes in C", Second Edition, Cambridge University
 * Press, 1992, Section 8.5, ISBN 0-521-43108-5 This code by Nicolas
 * Devillard - 1998. Public domain.
 */

#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }
#define IELEM_SWAP(a,b) { register int t=(a);(a)=(b);(b)=t; }

double quick_select_index(double arr[], int iarr[], int n, int k)
{
  int low, high ;
  int middle, ll, hh;

  low = 0 ; high = n-1 ;
  assert(k >= low && k <= high);
  for (;;) {
    if (high <= low) /* One element only */
      return arr[k] ;

    if (high == low + 1) {  /* Two elements only */
      if (arr[low] > arr[high]) {
        ELEM_SWAP(arr[low], arr[high]) ;
        if(iarr) IELEM_SWAP(iarr[low], iarr[high]) ;
      }
      return arr[k] ;
    }

    /* Find kth of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high]) {
      ELEM_SWAP(arr[middle], arr[high]) ;
      if(iarr) IELEM_SWAP(iarr[middle], iarr[high]) ;
    }
    if (arr[low] > arr[high]) {
      ELEM_SWAP(arr[low], arr[high]) ;
      if(iarr) IELEM_SWAP(iarr[low],iarr[high]) ;
    }
    if (arr[middle] > arr[low])  {
      ELEM_SWAP(arr[middle], arr[low]) ;
      if(iarr) IELEM_SWAP(iarr[middle],iarr[low]) ;
    }

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;
    if(iarr) IELEM_SWAP(iarr[middle], iarr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
        if(iarr) IELEM_SWAP(iarr[ll], iarr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;
    if(iarr) IELEM_SWAP(iarr[low], iarr[hh]) ;

    /* Re-set active partition */
    if (hh <= k)
        low = ll;
        if (hh >= k)
        high = hh - 1;
    }
}

#undef ELEM_SWAP

/*
 * open file with the given name
 * and print the passed matrix to it
 */

void matrix_to_file(const char* file_str, double** matrix, unsigned int n1, unsigned int n2)
{
  FILE* MOUT;

  MOUT = fopen(file_str, "w");
  assert(MOUT);
  printMatrix(matrix, n1, n2, MOUT);
  fclose(MOUT);
}

/*
 * sub_p_matrix:
 *
 * copy the cols v[1:n1][p[n2]] to V.
 * must have nrow(v) == nrow(V) and ncol(V) >= lenp
 * and ncol(v) >= max(p)
 */

void sub_p_matrix(double **V, int *p, double **v,
		  unsigned int nrows, unsigned int lenp,
		  unsigned int col_offset)
{
  int i,j;
  assert(V); assert(p); assert(v); assert(nrows > 0 && lenp > 0);
  for(i=0; i<nrows; i++) for(j=0; j<lenp; j++)
    V[i][j+col_offset] = v[i][p[j]];
}


/*
 * sub_p_matrix_rows:
 *
 * copy the rows v[1:n1][p[n2]] to V.
 * must have ncol(v) == ncol(V) and nrow(V) >= lenp
 * and nrow(v) >= max(p)
 */

void sub_p_matrix_rows(double **V, int *p, double **v,
		       unsigned int ncols, unsigned int lenp,
		       unsigned int row_offset)
{
  int i;
  assert(V); assert(p); assert(v); assert(ncols > 0 && lenp > 0);
  for(i=0; i<lenp; i++)
    dupv(V[i+row_offset], v[p[i]], ncols);
}


/*
 * new_p_submatrix_rows:
 *
 * create a new matrix from the rows of v, specified
 * by p.  Must have have ncol(v) == ncol(V) and nrow(V) >= nrows
 * and nrow(v) >= max(p)
 */

double **new_p_submatrix_rows(int *p, double **v, unsigned int nrows,
			      unsigned int ncols, unsigned int row_offset)
{
  double **V;
  if(nrows+row_offset == 0 || ncols == 0) return NULL;
  V = new_matrix(nrows + row_offset, ncols);
  if(nrows > 0) sub_p_matrix_rows(V, p, v, ncols, nrows, row_offset);
  return(V);
}


/*
 * allocates a new double array of size n1
 */

double* new_vector(unsigned int n)
{
  double *v;
  if(n == 0) return NULL;
  v = (double*) malloc(sizeof(double) * n);
  return v;
}


/*
 * allocates a new double array of size n1
 * and fills it with zeros
 */

double* new_zero_vector(unsigned int n)
{
  double *v;
  v = new_vector(n);
  zerov(v, n);
  return v;
}


/*
 * allocates a new double array of size n1
 * and fills it with the contents of vold
 */

double* new_dup_vector(double* vold, unsigned int n)
{
  double *v;
  v = new_vector(n);
  dupv(v, vold, n);
  return v;
}


/*
 * copies vold to v
 * (assumes v has already been allcocated)
 */

void dupv(double *v, double* vold, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) v[i] = vold[i];
}


/*
 * sumv:
 *
 * return the sum of the contents of the vector
 */

double sumv(double *v, unsigned int n)
{
  unsigned int i;
  double s;
  if(n==0) return 0;
  assert(v);
  s = 0;
  for(i=0; i<n; i++) s += v[i];
  return(s);
}


/*
 * sumiv:
 *
 * return the sum of the contents of the integer vector
 */

int sumiv(int *iv, unsigned int n)
{
  unsigned int i;
  int s;
  if(n==0) return 0;
  assert(iv);
  s = 0;
  for(i=0; i<n; i++) s += iv[i];
  return(s);
}


/*
 * meaniv:
 *
 * return the mean of the contents of the integer vector
 */

int meaniv(int *iv, unsigned int n)
{
  return((int) (sumiv(iv, n)/n));
}

/*
 * zeros out v
 * (assumes that it has already been allocated)
 */

void zerov(double*v, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) v[i] = 0;
}

/*
 * printing a vector out to outfile
 */

void printVector(double *v, unsigned int n, FILE *outfile, PRINT_PREC type)
{
  unsigned int i;
  if(type==HUMAN) for(i=0; i<n; i++) MYprintf(outfile, "%g ", v[i]);
  else if(type==MACHINE) for(i=0; i<n; i++) MYprintf(outfile, "%.20f ", v[i]);
  else error("bad PRINT_PREC type");
  MYprintf(outfile, "\n");
}


/*
 * new vector of integers of length n
 */

int *new_ivector(unsigned int n)
{
  int *iv;
  if(n == 0) return NULL;
  iv = (int*)  malloc(sizeof(int) * n);
  assert(iv);
  return iv;
}


/*
 * duplicate the integer contents of iv of length n into the already
 * allocated vector iv_new, also of length n
 */

void dupiv(int *iv_new, int *iv, unsigned int n)
{
  unsigned int i;
  if(n > 0) assert(iv && iv_new);
  for(i=0; i<n; i++) iv_new[i] = iv[i];
}


/*
 * zeros out v
 * (assumes that it has already been allocated)
 */

void zeroiv(int*v, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) v[i] = 0;
}

/*
 * sq:
 *
 * calculate the square of x
 */

double sq(double x)
{
  return x*x;
}
