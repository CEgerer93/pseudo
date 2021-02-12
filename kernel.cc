/*
  SUPPORTING FUNCTIONALITY TO EVALUATE KERNELS APPEARING IN PDF FITS
  ---BOTH FOR ITD->PDF & pITD->PDF
*/
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_gamma.h> // Evaluaton of Gamma/Beta functions
#include <gsl/gsl_sf_hyperg.h> // hypergeometric fun
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_sf_psi.h> // digamma evalutions

/*
  Grab arb headers for generalized hypergeometric function evaluation
*/
#include "arb.h"
#include "arb_hypgeom.h"
#include "acb.h"
#include "acb_hypgeom.h"


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

    flint_cleanup(); // free associated flint memory

    return resHypGeom;
  }

  /*
    Evaluate the tildeB Kernel appearing in pITD->PDF fits
  */
  double tildeBKernel(double u)
  {
    double si = gsl_sf_Si(u); // Grab the sine-integral
    double ci = gsl_sf_Ci(u); // Grab the cosine-integral
    return (1-cos(u))/pow(u,2)+2*sin(u)*((u*si-1)/u)+((3-4*M_EULER)/2)*cos(u)+2*cos(u)*(ci-log(u));
  }

  /*
    Evaluate the tildeD Kernel appearing in pITD->PDF fits
  */
  double tildeDKernel(double u)
  {
    pfq_t HypGeom = GenHypGeomEval(u);
    double imArg = cos(u)*HypGeom.imag+sin(u)*HypGeom.real; // hand determine imag(phase*hypgeom)
    return -4*u*(imArg)-((2-(2+pow(u,2))*cos(u))/pow(u,2));
  }

  /*
    A Beta function call
  */
  double betaFn(double v, double w)
  {
    return gsl_sf_beta(v,w);
  }


}
