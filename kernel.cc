/*
  SUPPORTING FUNCTIONALITY TO EVALUATE KERNELS APPEARING IN PDF FITS
  ---BOTH FOR ITD->PDF & pITD->PDF
*/
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h> // Evaluaton of Gamma/Beta functions
#include <gsl/gsl_sf_hyperg.h> // hypergeometric fun
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_psi.h> // polygamma evalutions

/*
  Grab arb headers for generalized hypergeometric function evaluation
*/
#include "arb.h"
#include "arb_hypgeom.h"
#include "acb.h"
#include "acb_hypgeom.h"

#include "pitd_util.h"
#include "kernel.h"

namespace PITD
{

#ifdef PSEUDOCONVOL
  double pseudoConvol2F3IntegralRep(double x, void * p)
  {
    pseudoConvolParams_t * locPtr = (pseudoConvolParams_t *)p;

    double res2F3(0.0);
    if ( locPtr->reality )
      res2F3 = cos(x*locPtr->nu)*
        (gsl_sf_gamma(2+locPtr->a+locPtr->b)/(gsl_sf_gamma(1+locPtr->a)*gsl_sf_gamma(1+locPtr->b)))*
        pow(x,locPtr->a)*pow(1-x,locPtr->b);
    if ( !locPtr->reality )
      res2F3 = sin(x*locPtr->nu)*locPtr->norm*pow(x,locPtr->a)*pow(1-x,locPtr->b);

    return res2F3;
  }
#endif

  /*
    Evaluate a 2F3 generalized hypergeometric function to convolve of (rough) cosine-transform
    of pseudo-PDF fits w/ DGLAP/matching kernels
  */
  pfq_t pseudoPDFCosineTransform(double a, double b, double val)
  {
    // Will compute a 2F3 order generalized hypergeometric function
    slong p = 2; slong q = 3;
    // Precision of result
    slong prec = 64; // double precision

    // Convert the val reference into a acb_t struct
    arb_t zRe; arb_init(zRe); arb_set_d(zRe,val);

    // Construct vectors for the numerator/denominators of pFq
    arb_t aRe;
    arb_init(aRe);
    arb_set_d(aRe,0.5);
    arb_t bRe;
    arb_init(bRe);
    arb_set_d(bRe,0.5);

    arb_struct * aZ = _arb_vec_init(p);
    arb_struct * bZ = _arb_vec_init(q);
  
    // Now set each component of numerator/denominator vectors
    for ( slong i = 0; i < p; i++ )
      {
	arb_t rescaleRe; arb_init(rescaleRe);
	arb_set_d(rescaleRe,0.5*(a+i+1));
	arb_set(aZ+i,rescaleRe);
	arb_clear(rescaleRe);

	flint_cleanup();
      }
    
    slong j = 0;

    arb_t bz;  arb_init(bz); arb_set_d(bz,0.5);
    arb_set(bZ+j, bz); j++;
    arb_set_d(bz,1.0+(a+b)/2);
    arb_set(bZ+j, bz); j++;
    arb_set_d(bz,1.5+(a+b)/2);
    arb_set(bZ+j, bz);

    // Free some memory
    arb_clear(aRe);
    arb_clear(bRe);

    arb_t res; arb_init(res);

    // THE CALL
    arb_hypgeom_pfq(res, aZ, p, bZ, q, zRe, 0, prec); // 0 INDICATES NON-REGULARIZED PFQ

    char * hypRealChar;
    hypRealChar = arb_get_str(res,prec,ARB_STR_NO_RADIUS);

    // Free more memory
    arb_clear(zRe);
    arb_clear(bz);
    arb_clear(res);
    _arb_vec_clear(aZ,p);
    _arb_vec_clear(bZ,q);
    
    pfq_t resHypGeom;
    resHypGeom.real = atof(hypRealChar);

    // Must free these or memory leaks!
    free(hypRealChar);
    flint_cleanup(); // free associated flint memory
    
    return resHypGeom;
  }


  /*
    Evaluate a 2F3 generalized hypergeometric function to convolve of (rough) sine-transform
    of pseudo-PDF fits w/ DGLAP/matching kernels
  */
  pfq_t pseudoPDFSineTransform(double a, double b, double val)
  {
    // Will compute a 2F3 order generalized hypergeometric function
    slong p = 2; slong q = 3;
    // Precision of result
    slong prec = 64; // double precision

    // Convert the val reference into a acb_t struct
    arb_t zRe; arb_init(zRe); arb_set_d(zRe,val);

    // Construct vectors for the numerator/denominators of pFq
    arb_t aRe;
    arb_init(aRe);
    arb_set_d(aRe,0.5);
    arb_t bRe;
    arb_init(bRe);
    arb_set_d(bRe,1.5);

    arb_struct * aZ = _arb_vec_init(p);
    arb_struct * bZ = _arb_vec_init(q);
  
    // Now set each component of numerator/denominator vectors
    for ( slong i = 0; i < p; i++ )
      {
        arb_t rescaleRe; arb_init(rescaleRe);
        arb_set_d(rescaleRe,0.5*(a+i+2));
        arb_set(aZ+i,rescaleRe);
	arb_clear(rescaleRe);
	
	flint_cleanup();
      }
    
    slong j = 0;

    arb_t bz;  arb_init(bz); arb_set_d(bz,1.5);
    arb_set(bZ+j, bz); j++;
    arb_set_d(bz,(3+a+b)/2);
    arb_set(bZ+j, bz); j++;
    arb_set_d(bz,(4+a+b)/2);
    arb_set(bZ+j, bz);

    // Free some memory
    arb_clear(aRe);
    arb_clear(bRe);

    arb_t res; arb_init(res);
    // THE CALL
    arb_hypgeom_pfq(res, aZ, p, bZ, q, zRe, 0, prec); // 1 INDICATES REGULARIZED PFQ

    char * hypRealChar;
    hypRealChar = arb_get_str(res,prec,ARB_STR_NO_RADIUS);

    // Free more memory
    arb_clear(zRe);
    arb_clear(bz);
    arb_clear(res);
    _arb_vec_clear(aZ,p);
    _arb_vec_clear(bZ,q);

    pfq_t resHypGeom;
    resHypGeom.real = atof(hypRealChar);

    // Must free these or memory leaks!
    free(hypRealChar);  
    flint_cleanup(); // free associated flint memory
    
    return resHypGeom;
  }

#ifdef PSEUDOCONVOL
  /*
    Effectively evaluate the 2F3 generalized hypergeometric functions by leaving them as explicit
    cosine-/sine-integral transforms of pseudo-PDF
  */
  double pseudoPDFCosineTransform_IntegralRep(double a, double b, double ioffe)
  {
    pseudoConvolParams_t pseudoConvolParams(ioffe, a, b, 0.0, true);

    gsl_function F;
    F.function = &pseudoConvol2F3IntegralRep;
    F.params = &pseudoConvolParams;

    size_t n = 500;
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(n);
    
    double dum, dumerr;
    size_t nevals;
    double epsabs=0.0000001;
    double epsrel = 0.0;

    int success = gsl_integration_cquad(&F,0.0,1.0,epsabs,epsrel,w,&dum,&dumerr,&nevals);
    // Free associated memory
    gsl_integration_cquad_workspace_free(w);

    return dum;
  }

  double pseudoPDFSineTransform_IntegralRep(double a, double b, double ioffe, double c)
  {
    pseudoConvolParams_t pseudoConvolParams(ioffe, a, b, c, false);

    gsl_function F;
    F.function = &pseudoConvol2F3IntegralRep;
    F.params = &pseudoConvolParams;

    size_t n = 500;
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(n);
    
    double dum, dumerr;
    size_t nevals;
    double epsabs=0.0000001;
    double epsrel = 0.0;

    int success = gsl_integration_cquad(&F,0.0,1.0,epsabs,epsrel,w,&dum,&dumerr,&nevals);
    // Free associated memory
    gsl_integration_cquad_workspace_free(w);

    return dum;
  }
#endif


  /*
    Evaluate the generalized hypergeometric function pFq
  */
  pfq_t GenHypGeomEval(double val)
  {
    // Convert the val reference into a acb_t struct
    arb_t zRe; arb_init(zRe); arb_set_d(zRe,0.0);
    arb_t zIm; arb_init(zIm); arb_set_d(zIm,-1.0*val);
    // Pack real/imag components into a complex struct
    acb_t z; acb_init(z);
    acb_set_arb_arb(z,zRe,zIm);
    // std::cout << "  Complex argument passed to be passed to hypergeometric function   =   ";
    // acb_print(z); std::cout << "\n";

    // Free memory for the components
    arb_clear(zRe); arb_clear(zIm);
   

    // Will compute a 3F3 order generalized hypergeometric function
    slong p = 3; slong q = 3;
    // Precision of result
    slong prec = 64; // double precision


    /*
    Start constructing vectors for the numerator/denominators of pFq
    */
    arb_t aRe, aIm;
    arb_init(aRe); arb_init(aIm);
    arb_set_d(aRe,1); arb_set_d(aIm,0);
    arb_t bRe, bIm;
    arb_init(bRe); arb_init(bIm);
    arb_set_d(bRe,2); arb_set_d(bIm,0);
  
    acb_t az;  acb_init(az); acb_set_arb_arb(az,aRe,aIm);
    acb_t bz;  acb_init(bz); acb_set_arb_arb(bz,bRe,bIm);

    acb_struct * aZ = _acb_vec_init(p);
    acb_struct * bZ = _acb_vec_init(q);

    for ( slong i = 0; i < p; i++ )
      {
	acb_one(aZ+i);
	acb_set(bZ+i,bz);
      }
    /*
    Vectors for numerator/denominator now set
    */
    
    // std::cout << "Printing contents of aZ array ... " << std::endl;
    // acb_print(&aZ[0]); std::cout << "\n";
    // acb_print(&aZ[1]); std::cout << "\n";
    // acb_print(&aZ[2]); std::cout << "\n";
    // std::cout << "Printing contents of bZ array ... " << std::endl;
    // acb_print(&bZ[0]); std::cout << "\n";
    // acb_print(&bZ[1]); std::cout << "\n";
    // acb_print(&bZ[2]); std::cout << "\n";

    // Free some memory
    arb_clear(aRe); arb_clear(aIm);
    arb_clear(bRe); arb_clear(bIm);


    // std::cout << "Right before the hypergeometric call " << std::endl;

    /*
    res contains:
         ---> a pointer to an array of coefficients (coeffs)
         ---> the used length (length)
         ---> the allocated size of the array (alloc)

    An acb_poly_t is defined as an array of length one of type acb_poly_struct
    */
    acb_t res;
    acb_init(res);

    // arb_poly_t res;
    // arb_poly_init(res);
  
    // THE CALL
    acb_hypgeom_pfq(res, aZ, p, bZ, q, z, 0, prec);
  
    // std::cout << " Printing contents for hypgeom res   =    " << std::endl;
    // acb_print(res);

    arb_t hypImag; arb_init(hypImag);
    acb_get_imag(hypImag,res);
    arb_t hypReal; arb_init(hypReal);
    acb_get_real(hypReal,res);
    char * hypRealChar; char * hypImagChar;
    hypRealChar = arb_get_str(hypReal,prec,ARB_STR_NO_RADIUS);
    hypImagChar = arb_get_str(hypImag,prec,ARB_STR_NO_RADIUS);
    // std::cout << "\n\nHypergeometric real = " << hypRealChar << std::endl;
    // std::cout << "\n\nHypergeometric imag = " << hypImagChar << std::endl;

    
    /*
    Doing the complex multiplication w/ Arb functions seems to lead to
    an incorrect result, so let's do it by hand outside of this function...
    */
    // arb_t phaseRe; arb_init(phaseRe); arb_set_d(phaseRe,0);
    // arb_t phaseIm; arb_init(phaseIm); arb_set_d(phaseIm,val);
    // std::cout << "    The argument of applied phase factor = " << val << std::endl;


    // Free more memory
    acb_clear(z);
    acb_clear(az);
    acb_clear(bz);
    acb_clear(res);
    arb_clear(hypImag);
    arb_clear(hypReal);
    _acb_vec_clear(aZ,p);
    _acb_vec_clear(bZ,q);
  

    pfq_t resHypGeom;
    resHypGeom.real = atof(hypRealChar);
    resHypGeom.imag = atof(hypImagChar);

    // Must free these or memory leaks!
    free(hypRealChar);
    free(hypImagChar);


    flint_cleanup(); // free associated flint memory

    return resHypGeom;
  }

  /*
    Evaluate the tildeB Kernel appearing in pITD->PDF fits
    Catching the pdfType ( 0 - qval, 1 - qplus ) as a second argument
  */
  double tildeBKernel(double u, int pdfType)
  {
    double si = gsl_sf_Si(u); // Grab the sine-integral
    double ci = gsl_sf_Ci(u); // Grab the cosine-integral
    if ( pdfType == 0 )
      return (1-cos(u))/pow(u,2)+2*sin(u)*((u*si-1)/u)+((3-4*M_EULER)/2)*cos(u)+2*cos(u)*(ci-log(u));
    if ( pdfType == 1 )
      return -1.0*((sin(u)+u)/pow(u,2))+((3-4*M_EULER)/2)*sin(u)+2*cos(u)*((1-u*si)/u)+2*sin(u)*(ci-log(u));
  }

  /*
    Evaluate the tildeD Kernel appearing in pITD->PDF fits
    Catching the pdfType ( 0 - qval, 1 - qplus ) as a second argument
  */
  double tildeDKernel(double u, int pdfType)
  {
    pfq_t HypGeom = GenHypGeomEval(u);
    double reArg = cos(u)*HypGeom.real-sin(u)*HypGeom.imag; // hand determine real(phase*hypgeom)
    double imArg = cos(u)*HypGeom.imag+sin(u)*HypGeom.real; // hand determine imag(phase*hypgeom)
    if ( pdfType == 0 )
      return -4*u*(imArg)-((2-(2+pow(u,2))*cos(u))/pow(u,2));
    if ( pdfType == 1 )
      return 4*u*(reArg)+sin(u)*(1+2/pow(u,2))-2/u;
  }

  /*
    A Beta function call
  */
  double betaFn(double v, double w)
  {
    return gsl_sf_beta(v,w);
  }

  // /*
  //   Jacobi coefficients
  //   w_{n,j}^{(a,b)} = (-1)^j/n! * (n_C_j) * { Gamma(a+n+1)*Gamma(a+b+n+j+1) }/{ Gamma(a+j+1)*Gamma(a+b+n+1) }
  // */
  // double coeffJacobi(int n, int j, double a, double b)
  // {
  //   return (pow(-1,j)/gsl_sf_fact(n))*gsl_sf_choose(n,j)
  //     *( (gsl_sf_gamma(a+n+1)*gsl_sf_gamma(a+b+n+j+1)) / (gsl_sf_gamma(a+j+1)*gsl_sf_gamma(a+b+n+1)) );
  // }

  // /*
  //   Jacobi Polynomial:  \omega_n^{(a,b)}(x) = \sum_{j=0}^n coeffJacobi(n,j,a,b)*x^j
  // */
  // double jacobi(int n, double a, double b, double x)
  // {
  //   double sum(0.0);
  //   for ( int j = 0; j <= n; j++ )
  //     sum += coeffJacobi(n,j,a,b)*pow(x,j);
  //   return sum;
  // }


  // /*
  //   Support for Taylor expansion of DGLAP Kernel
  // */
  // double texp_gn(int n)
  // {
  //   double sum(0.0);
  //   // Sum part of what would be a divergent p-series
  //   for ( int k = 1; k <= n; k++ )
  //     sum += 1.0/k;
  //   sum*=2;

  //   return 3/2 - 1/(1+n) - 1/(2+n) - sum;
  // }

  // /*
  //   Support for Taylor expansion of Lattice-MSbar matching kernel
  // */
  // double texp_dn(int n)
  // {
  //   double sum(0.0);
  //   // Sum part of what would be a divergent p-series
  //   for ( int k = 1; k <= n; k++ )
  //     sum += 1.0/k;
  //   sum = pow(sum, 2);

  //   return 2*( sum + (2*pow(M_PI,2)+n*(n+3)*(3+pow(M_PI,2)))/(6*(n+1)*(n+2)) - gsl_sf_psi_1_int(n+1) );
  // }

  // /*
  //   Support for Taylor expansion of NLO pITD->PDF Matching Kernel
  // */
  // double texp_cn(int n, int z)
  // {
  //   return 1-((alphaS*Cf)/(2*M_PI))*( texp_gn(n)*log( (exp(2*M_EULER+1)/4)*pow(MU*z,2) ) + texp_dn(n) );
  // }

  // /*
  //   Support for Taylor expansion of (pITD->PDF Kernel)*(Jacobi PDF Parametrization)
  // */
  // double pitd_texp_sigma_n(int n, int trunc, double a, double b, double nu)
  // {
  //   double sum(0.0);
  //   for ( int j = 0; j <= n; j++ )
  //     {
  // 	for ( int k = 0; k <= trunc; k++ )
  // 	  {
  // 	    sum += (pow(-1,k)/gsl_sf_fact(2*k))*texp_cn(2*k,z)*
  // 	      coeffJacobi(n,j,a,b)*betaFn(a+2*k+j+1,b+1)*pow(nu,2*k);
  // 	  }
  //     }
  //   return sum;
  // }



}
