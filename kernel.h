/*
  SUPPORTING FUNCTIONALITY TO EVALUATE KERNELS APPEARING IN PDF FITS
  ---BOTH FOR ITD->PDF & pITD->PDF
*/

#ifndef _kernel_h_
#define _kernel_h_

namespace PITD
{
  /*
    A structure to hold results for evaluation of a generalized hypergeometric function via Arb
  */
  struct pfq_t
  {
    double real, imag;
  };

  
  /*
    Evaluate the generalized hypergeometric function pFq
  */
  pfq_t GenHypGeomEval(double val);

  /*
    Evaluate the tildeB Kernel appearing in pITD->PDF fits
    Catching the pdfType ( 0 - qval, 1 - qplus ) as a second argument
  */
  double tildeBKernel(double u, int type);

  /*
    Evaluate the tildeD Kernel appearing in pITD->PDF fits
    Catching the pdfType ( 0 - qval, 1 - qplus ) as a second argument
  */
  double tildeDKernel(double u, int type);

  /*
    A Beta function call
  */
  double betaFn(double v, double w);

  /* /\* */
  /*   Jacobi coefficients */
  /*   w_{n,j}^{(a,b)} = (-1)^j/n! * (n_C_j) * { Gamma(a+n+1)*Gamma(a+b+n+j+1) }/{ Gamma(a+j+1)*Gamma(a+b+n+1) } */
  /* *\/ */
  /* double coeffJacobi(int n, int j, double a, double b); */

  /* /\* */
  /*   Jacobi Polynomial:  \omega_n^{(a,b)}(x) = \sum_{j=0}^n coeffJacobi(n,j,a,b)*x^j */
  /* *\/ */
  /* double jacobi(int n, double a, double b, double x); */

  /* /\* */
  /*   Support for Taylor expansion of NLO pITD->PDF Matching Kernel */
  /* *\/ */
  /* double texp_cn(int n, int z); */

  /* /\* */
  /*   Support for Taylor expansion of DGLAP Kernel */
  /* *\/ */
  /* double texp_gn(int n); */

  /* /\* */
  /*   Support for Taylor expansion of Lattice-MSbar matching kernel */
  /* *\/ */
  /* double texp_dn(int n); */

  /* /\* */
  /*   Support for Taylor expansion of (pITD->PDF Kernel)*(Jacobi PDF Parametrization) */
  /* *\/ */
  /* double pitd_texp_sigma_n(int n, int trunc, double a, double b, double nu); */

}
#endif
