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

  /*
    Jacobi Polynomial
  */
  double jacobi(int n, double a, double b, double x);


}
#endif
