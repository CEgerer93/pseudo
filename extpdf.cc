/*
  --NUMERICALLY INTEGRATE CONVOLUTION OF PERTURBATIVE KERNELS AND PHENO PDFS TO GENERATE
  A IOFFE-TIME PSEUDO-STRUCTURE FUNCTION
  --MINIMIZE DIFFERENCE BETWEEN FIT CURVE AND BOOTSTRAP(OR JACKKNIFE) SAMPLES
 */
#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<sstream>
#include<vector>
#include<iomanip>
#include<cmath> // math.h
#include<map> // use maps to grab kenels for current pairs
#include<unordered_map>
#include<chrono>
/*
  Grab headers for gsl function calls
*/
#include <gsl/gsl_sf_gamma.h> // Evaluaton of Gamma/Beta functions
#include <gsl/gsl_rng.h> // Random numbers
#include <gsl/gsl_integration.h> // Numerical integration
#include <gsl/gsl_multimin.h> // multidimensional minimization
#include <gsl/gsl_vector.h> // allocating/accessing gsl_vectors for passing params to minimizer
#include <gsl/gsl_matrix.h> // matrix routines for inversion of data covariance
#include <gsl/gsl_permutation.h> // permutation header for matrix inversions
#include <gsl/gsl_linalg.h> // linear algebra
#include <gsl/gsl_sf_hyperg.h> // hypergeometric fun
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_sf_psi.h> // digamma evalutions

/*
  Grab arb headers for generalized hypergeometric function evaluation
*/
#include "arb.h"
#include "arb_hypgeom.h"
#include "acb.h"
#include "acb_hypgeom.h"

/*
 Headers for threading
*/
#include<omp.h>

/*
  HDF5 Header
*/
#include "hdf5.h"

#include "pitd_util.h"


// Default values of p & z to cut on
int zmin,zmax,pmin,pmax,Nmom,Nz;

#define ReIm 2
// If we are building st. alpha_s is not a fitted parameter, then set a value here
#ifndef FITALPHAS
#define alphaS 0.303
#endif


using namespace PITD;


// Parameters describing pheno PDF
struct pdfParams
{
#ifndef QPLUS
  double alpha, beta;
  // Conditionally allo qplus distribution to be simultanouesly fit
#elif defined QPLUS
  double normP,alphaP,betaP;
#endif
  // Conditionally allow alphaS to be a fitted parameter at compile time
#ifdef FITALPHAS
  double alphaS;
#endif

  // Set barriers
  std::pair<int,int> alphaRestrict = std::make_pair(-1,1);
  std::pair<int,int> betaRestrict = std::make_pair(0.5,5); // 4
};

// Structure to hold all needed parameters to perform convolution of a trial pdf
struct convolParams
{
  pdfParams p;
  double nu;
  int z;
};

// A structure to hold results for evaluation of a generalized hypergeometric function via Arb
struct pfq
{
  double real, imag;
};


pfq GenHypGeomEval(double val)
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
  

  pfq resHypGeom;
  resHypGeom.real = atof(hypRealChar);
  resHypGeom.imag = atof(hypImagChar);

  flint_cleanup(); // free associated flint memory

  return resHypGeom;
}



// Evaluate the B kernel
double tildeBKernel(double u)
{
  double si = gsl_sf_Si(u); // Grab the sine-integral
  double ci = gsl_sf_Ci(u); // Grab the cosine-integral

  return (1-cos(u))/pow(u,2)+2*sin(u)*((u*si-1)/u)+((3-4*M_EULER)/2)*cos(u)+2*cos(u)*(ci-log(u));
}

// Evaluate the D kernel
double tildeDKernel(double u)
{
  // std::cout << "Within the tildeDKernel function " << std::endl;

  pfq HypGeom = GenHypGeomEval(u);
  double imArg = cos(u)*HypGeom.imag+sin(u)*HypGeom.real; // hand determine imag(phase*hypgeom)
  // std::cout << " Hand mult = " << imArg << std::endl;
  
  return -4*u*(imArg)-((2-(2+pow(u,2))*cos(u))/pow(u,2));
}


// Hold the entire NLO kernel
#ifdef FITALPHAS
double NLOKernel(double x, double ioffeTime, int z, double alphaS)
#else
double NLOKernel(double x, double ioffeTime, int z)
#endif
{
  double xnu = x * ioffeTime;
  return cos(xnu)-(alphaS/(2*M_PIl))*Cf*(log( (exp(2*M_EULER+1)/4)*pow(MU*z,2) )
					 *tildeBKernel(xnu)+tildeDKernel(xnu)  );
}



// Return the beta function evaluated at x & y
double betaFn(double v, double w)
{
  return gsl_sf_beta(v,w);
}

// Convolution
double NLOKernelPhenoPDFConvolution(double x, void * p)
{
  convolParams * cp = (convolParams *)p;
  // Friendly local copies
#ifndef QPLUS
  double alpha = (cp->p.alpha);
  double beta = (cp->p.beta);
  double normalization = betaFn(alpha+1,beta+1);
#elif defined QPLUS
  double normP = (cp->p.normP);
  double alphaP = (cp->p.alphaP);
  double betaP = (cp->p.betaP);
#endif

  // Allow for conditional fitting of alphaS at compile time
#ifdef FITALPHAS
  double alphaS = (cp->p.alphaS);
#endif
  double ioffeTime = (cp->nu);
  int z = (cp->z);


  /*
    Return normalized PDF N^-1 * x^alpha * (1-x)^beta
    convolved with perturbative NLO kernel or Cosine/Sine
  */
#ifdef CONVOLK
#warning "Kernel will be NLO matching kernel  --  assuming pITD data will be fit"
#warning "!!!Imaginary pITD --> PDF kernel not ready!!!"
#ifdef FITALPHAS
  return NLOKernel(x,ioffeTime,z,alphaS)*(1/normalization)*pow(x,alpha)*pow(1-x,beta);
#else
  return NLOKernel(x,ioffeTime,z)*(1/normalization)*pow(x,alpha)*pow(1-x,beta);
#endif
#endif

#ifdef CONVOLC
#ifndef QPLUS
#warning "Kernel for qv will be cos(x\nu) --  assuming ITD data will be fit!"
  return cos(x*ioffeTime)*(1/normalization)*pow(x,alpha)*pow(1-x,beta);
#elif defined QPLUS
#warning "Kernel for q+ will be sin(x\nu)  --  assuming ITD data will be fit!"
  return sin(x*ioffeTime)*normP*pow(x,alphaP)*pow(1-x,betaP);
#endif
#endif
}

// Perform numerical convolution of NLO matching kernel & pheno PDF
// double convolution(pdfParams& params)
double convolution(pdfParams& params, double nu, int z)
{
  /*
    Compute approximation to definite integral of form

          I = \int_a^b f(x)w(x)dx

    Where w(x) is a weight function ( = 1 for general integrands)
    Absolute/relative error bounds provided by user
  */

  // Collect and package the parameters to do the convolution
  convolParams cParams; cParams.p = params;
  cParams.nu = nu; cParams.z = z;
  

  // Define the function to integrate - selecting kernel based on passed comp
  gsl_function F;
  F.function = &NLOKernelPhenoPDFConvolution;
  F.params = &cParams;
  
  // hold n double precision intervals of integrand, their results and error estimates
  size_t n = 500; // 1000; // 250

  /*
    Allocate workspace sufficient to hold data for size_t n intervals
  */
  gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(n);

  // Set some parameters for the integration
  double result, abserr;
  size_t nevals; // evaluations to solve
  double epsabs=0.0000001; // Set absolute error of integration
  double epsrel = 0.0; // To compute a specified absolute error, setting epsrel to zero
  /*
    Perform the numerical convolution here
    Pass success boolean as an int
  */ 
#ifdef CONVOLK // need to exclude 0 for kernel convolution to even proceed
  int success = gsl_integration_cquad(&F,0.00000000001,1.0,epsabs,epsrel,w,&result,&abserr,&nevals);
#elif CONVOLC
  int success = gsl_integration_cquad(&F,0.0,1.0,epsabs,epsrel,w,&result,&abserr,&nevals);
#endif

  // Free associated memory
  gsl_integration_cquad_workspace_free(w);

  return result;  
}

/*
  PERFORM INVERSIONS OF PASSED MATRICES - return # of SVs removed
*/
int matrixInv(gsl_matrix* cov, gsl_matrix* invcov, size_t dataDim)
{
  // int status; // status of operations
  // // Instantiate a null gsl_permutation pointer
  // gsl_permutation * p = gsl_permutation_calloc(dataDim);
  // int signum; // sign of permutation (-1)^n
  // /*
  //   Determine the LU decomposition of provided matrix
  //   --> diagonal & upper triangular components of input matrix are now U (upper)
  //   --> lower triangular components of input matrix are now L (lower)
  // */
  // std::cout << "Got to the decomposition" << std::endl;
  // status = gsl_linalg_LU_decomp(cov,p,&signum);
  // std::cout << "Got past the decomposition" << std::endl;
  
  
  // // Compute inverse of matrix from its LU decomposition
  // status = gsl_linalg_LU_invert(cov,p,invcov);
  // // std::FILE* dumOutput = std::fopen("dummy.dat","w");
  // // status = gsl_matrix_fprintf(dumOutput,invcov,"%f");

  // // std::cout << gsl_matrix_get(invcov,0,0) << std::endl;
  // // std::cout << gsl_matrix_get(invcov,1,1) << std::endl;


  gsl_matrix * V = gsl_matrix_alloc(dataDim,dataDim);
  gsl_vector *S = gsl_vector_alloc(dataDim);
  gsl_vector *work = gsl_vector_alloc(dataDim);


  /*
    PERFORM THE SINGULAR VALUE DECOMPOSITION OF DATA COVARIANCE MATRIX (A)
    
    A = USV^T
        ---> A is an MxN matrix
	---> S is the singular value matrix (diagonal w/ singular values along diag - descending)
	---> V is returned in an untransposed form
  */
  gsl_linalg_SV_decomp(cov,V,S,work);
  // On output cov is replaced w/ U


  // Define an svd cut
  double svdCut = 1e-11;
  // Initialize the inverse of the S diag
  gsl_vector *pseudoSInv = gsl_vector_alloc(dataDim);
  gsl_vector_set_all(pseudoSInv,0.0);


  // Vector of singular values that are larger than specified cut
  std::vector<double> aboveCutVals;

  std::cout << "The singular values above SVD Cut = " << svdCut << " are..." << std::endl;
  for ( int s = 0; s < dataDim; s++ )
    {
      double dum = gsl_vector_get(S,s);
      if ( dum >= svdCut )
	{
	  aboveCutVals.push_back(dum);
	  // std::cout << dum << std::endl;
	}
    }

  
  // Assign the inverse of aboveCutVals to the pseudoSInv vector
  for ( std::vector<double>::iterator it = aboveCutVals.begin(); it != aboveCutVals.end(); ++it )
    {
      gsl_vector_set(pseudoSInv,it-aboveCutVals.begin(),1.0/(*it));
    }

  // Promote this pseudoSInv vector to a matrix, where entries are placed along diagonal
  gsl_matrix * pseudoSInvMat = gsl_matrix_alloc(dataDim,dataDim);
  gsl_matrix_set_zero(pseudoSInvMat);
  for ( int m = 0; m < dataDim; m++ )
    {
      gsl_matrix_set(pseudoSInvMat,m,m,gsl_vector_get(pseudoSInv,m));
      std::cout << gsl_vector_get(pseudoSInv,m) << std::endl;
    }
  
  
  /*
    With singular values that are zero, best we can do is construct a pseudo-inverse
  */
  // In place construct the transpose of U (cov was modified in place to U above)_
  gsl_matrix_transpose(cov);


  gsl_matrix * SinvUT = gsl_matrix_alloc(dataDim,dataDim); gsl_matrix_set_zero(SinvUT);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,pseudoSInvMat,cov,0.0,SinvUT);


  // Now make the covariance inverse
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,V,SinvUT,0.0,invcov);

  // // Free memory associated with gsl_matrix
  // gsl_permutation_free(p);


  // Return the number of singular values removed
  return pseudoSInv->size - aboveCutVals.size();
}



/*
  MULTIDIMENSIONAL MINIMIZATION - CHI2 
  MINIMIZE LEAST-SQUARES BETWEEN DATA AND NUMERICALLY INTEGRATED MATCHING KERNEL AND pITD
*/
double chi2Func(const gsl_vector * x, void *data)
{

  // Get a pointer to the void reducedPITD class
  reducedPITD * cpyPITD = (reducedPITD *)data;
#ifndef QPLUS
  gsl_matrix * invCov = (cpyPITD->invCovR);
#elif defined QPLUS
  gsl_matrix * invCov = (cpyPITD->invCovI);
#endif

  // Get a pointer to the map
#ifndef QPLUS
  std::map<int,rPITD> * cpyMap = &(cpyPITD->real.disps);
#elif defined QPLUS
  std::map<int,rPITD> * cpyMap = &(cpyPITD->imag.disps);
#endif

  // // Locally determine Nmom
  // int Nmom = invCov->size1/cpyPITD->real.disps.size();
  // std::cout << " NMOM = " << Nmom << std::endl;


  pdfParams pdfp;
#ifndef QPLUS
  pdfp.alpha = gsl_vector_get(x,0);
  pdfp.beta = gsl_vector_get(x,1);
  // Conditionally allow alphaS to be fit at compile time
#ifdef FITALPHAS
  pdfp.alphaS = gsl_vector_get(x,2);
#endif
#elif defined QPLUS
  pdfp.normP = gsl_vector_get(x,0);
  pdfp.alphaP = gsl_vector_get(x,1);
  pdfp.betaP = gsl_vector_get(x,2);
#ifdef FITALPHAS
  pdfp.alphaS = gsl_vector_get(x,3);
#endif
#endif


  /*
    Evaluate the kernel for all points and store
  */
  // std::cout << "ZMIN chi2 = " << zmin << std::endl;
  // std::vector<double> convols(Nmom*cpyMap->size());
  std::vector<double> convols(Nmom*Nz);
  for ( auto cm = cpyMap->begin(); cm != cpyMap->end(); ++cm )
    {
#pragma omp parallel for num_threads(Nmom)
      for ( int m = 0; m < Nmom; m++ )
	{
	  double nu_tmp = cm->second.ensem.IT[m];
	  double z_tmp = cm->first;

	  double convolTmp;
	  if ( nu_tmp == 0 ) { convolTmp = 1; }
	  else { convolTmp = convolution(pdfp,nu_tmp,z_tmp); }

	  convols[m+(cm->first-zmin)*Nmom] = convolTmp;

	}
    }

//   // Conditionally construct the q+ convols
// #ifdef QPLUS
//   std::vector<double> convolsI(Nmom*cpyMapI->size());
//   for ( auto cm = cpyMapI->begin(); cm != cpyMapI->end(); ++cm )
//   {
// #pragma omp parallel for num_threads(6)
//     for ( int m = 0; m < Nmom; m++ )
//       {
// 	double nu_tmp = cm->second.ensem.IT[m];
// 	double z_tmp = cm->first;

// 	double convolTmp;
// 	if ( nu_tmp == 0 ) { convolTmp = 0; }
// 	else { convolTmp = convolution(pdfp,nu_tmp,z_tmp); }

//       }
//   }
// #endif
  


  /*
    FOR A SLOW MAN'S EVALUATION OF THE CONVOLUTION
  */
  // for ( int a = 0; a < Nz; a++ )
  //   {
  //     for ( int m = 0; m < Nmom; m++ )
  // 	{
  // 	  double nu_tmp = (*cpyMapR)[a].ensem.IT[m];
  // 	  double z_tmp = a;

  // 	  double convolTmp;
  // 	  if ( nu_tmp == 0 ) { convolTmp = 1; }
  // 	  else { convolTmp = convolution(pdfp,nu_tmp,z_tmp); }

  // 	  convols[m+a*Nmom]=convolTmp;
  // 	}
  //   }
  // // END OF THE SLOW MAN'S EVALUATION
  
  double chi2(0.0);
  gsl_vector *iDiffVec = gsl_vector_alloc(Nmom*Nz);//cpyMap->size());
  gsl_vector *jDiffVec = gsl_vector_alloc(Nmom*Nz);//cpyMap->size());

  for ( auto d1 = cpyMap->begin(); d1 != cpyMap->end(); ++d1 )
    {
#pragma omp parallel for num_threads(Nmom)
      for ( int m1 = 0; m1 < Nmom; m1++ )
	{
	  int I = m1 + (d1->first - zmin)*Nmom;

	  gsl_vector_set(iDiffVec,I,convols[I] - d1->second.ensem.avgM[m1]);
	}
    }

  /*
    FOR A SLOW MAN'S SETTING OF DIFFERENCE VECTOR BETWEEN CONVOLUTION AND DATA
  */
  // for ( int d = 0; d < Nz; d++ )
  //   {
  //     for ( int m1 = 0; m1 < Nmom; m1++ )
  // 	{
  // 	  int I = m1+d*Nmom;
  // 	  gsl_vector_set(iDiffVec,I,convols[I]-(*cpyMapR)[d].ensem.avgM[m1]);
  // 	}
  //   }
  // // END OF THE SLOW MAN'S DIFFERENCE VECTOR SETTING


  // The difference vector need only be computed once, so make a second copy to form correlated chi2
  gsl_vector_memcpy(jDiffVec,iDiffVec);

  // Initialize cov^-1 right multiplying jDiffVec
  gsl_vector *invCovRightMult = gsl_vector_alloc(Nmom*Nz);//cpyMap->size());
  gsl_blas_dgemv(CblasNoTrans,1.0,invCov,jDiffVec,0.0,invCovRightMult);

  // Form the scalar dot product of iDiffVec & result of invCov x jDiffVec
  gsl_blas_ddot(iDiffVec,invCovRightMult,&chi2);


//   // Conditionally repeat chi2 determination for imaginary data
//   // For now reusing gsl_vector/matrix pointers, by assuming real/imag use same number of zseps
// #ifdef QPLUS
//   for ( auto d1 = cpyMapI->begin(); d1 != cpyMapI->end(); ++d1 )
//     {
// #pragma omp parallel for num_threads(Nmom)
//       for ( int m1 = 0; m1 < Nmom; m1++ )
// 	{
// 	  gsl_vector_set(iDiffVec,I,convolsI[I] - d1->second.ensem.avgM[m1]);
// 	}
//     }
  
//   gsl_vector_memcpy(jDiffVec,iDiffVec);
//   gsl_blas_dgemv(CblasNoTrans,1.0,invCovI,jDiffVec,0.0,invCovRightMult);
//   gsl_blas_ddot(iDiffVec,invCovRightMult,&chi2I);
// #endif



  /*
    BRUTE FORCE TALLEY THE CHI2
  */
  // for ( int i = 0; i < Nmom*Nz; i++ )
  //   {
  //     for ( int j = 0; j < Nmom*Nz; j++ )
  // 	{

  // 	  chi2+=gsl_vector_get(iDiffVec,i)*gsl_matrix_get(invCov,i,j)*gsl_vector_get(jDiffVec,j);
  // 	}
  //   }



  // Free some memory
  gsl_vector_free(iDiffVec);
  gsl_vector_free(jDiffVec);
  gsl_vector_free(invCovRightMult);


  // // Sum the real/imag chi2 always, where chi2 imag may be zero if QPLUS is not defined
  // chi2=chi2R+chi2I;

#ifndef QPLUS
#ifdef CONSTRAINED
  /*
    CHECK FOR VALUES OF {ALPHA,BETA} OUTSIDE ACCEPTABLE RANGE AND INFLATE CHI2
  */
  if ( pdfp.alpha < pdfp.alphaRestrict.first || pdfp.alpha > pdfp.alphaRestrict.second )
    {
      chi2+=1000000;
    }
  if ( pdfp.beta < pdfp.betaRestrict.first || pdfp.beta > pdfp.betaRestrict.second )
    {
      chi2+=1000000;
    }
#endif
#endif

  return chi2;
}


// Method to set the data covariance for Real/Imag/or both pITDs
void setCovariance(gsl_matrix *c,compPITD &e, std::vector<reducedPITD> &j, int comp)//, int Nmom, int zmin)
{
  for ( auto d1 = e.disps.begin(); d1 != e.disps.end(); ++d1 )
    {
      for ( int m1 = 0; m1 < Nmom; m1++ )
	{
	  int I = m1 + (d1->first-zmin)*Nmom; // get the index by moding out by zmin

	  for ( auto d2 = e.disps.begin(); d2 != e.disps.end(); ++d2 )
	    {
	      for ( int m2 = 0; m2 < Nmom; m2++ )
		{
		  int J = m2 + (d2->first-zmin)*Nmom; // get the index by moding out by zmin


		  double val = 0.0;
		  for ( int g = 0; g < j.size(); g++ )
		    {
		      // Determine this entry of the covariance matrix
		      if ( comp == 0 )
			{
			  val += ( j[g].real.disps[d1->first].ensem.avgM[m1] - d1->second.ensem.avgM[m1] )*
			    ( j[g].real.disps[d2->first].ensem.avgM[m2] - d2->second.ensem.avgM[m2] );
			}
		      if ( comp == 1 )
			{
			  val += ( j[g].imag.disps[d1->first].ensem.avgM[m1] - d1->second.ensem.avgM[m1] )*
			    ( j[g].imag.disps[d2->first].ensem.avgM[m2] - d2->second.ensem.avgM[m2] );
			}
		    }

		  // Set the entry and proceed
#ifdef UNCORRELATED
		  if ( I == J ) // modify diagonal elements from zero
		    {
		      gsl_matrix_set(c,I,J,val*((j.size()-1)/(1.0*j.size())));
		    }
#else
		  gsl_matrix_set(c,I,J,val*((j.size()-1)/(1.0*j.size())));
#endif

		} // end m2
	    } // end auto d2
	} // end m1
    } // end auto d1 
  
} // end setCovariance


int main( int argc, char *argv[] )
{

  if ( argc != 12 )
    {
      std::cout << "Usage: $0 <alpha_i> <beta_i> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <pmin> <pmax> <zmin> <zmax>" << std::endl;
      exit(1);
    }

  // Enable nested parallelism
  omp_set_nested(true);

  std::stringstream ss;

  // Allow for conditional fitting of alpha_s at compile time
#ifdef FITALPHAS
  double alpha_s_i = 0.3;
#endif
#ifndef QPLUS
  double alpha_i, beta_i;
  ss << argv[1]; ss >> alpha_i; ss.clear(); ss.str(std::string());
  ss << argv[2]; ss >> beta_i;  ss.clear(); ss.str(std::string());
#elif defined QPLUS
  double normP_i = 1.0;
  double alphaP_i;
  double betaP_i;
  ss << argv[1]; ss >> alphaP_i; ss.clear(); ss.str(std::string());
  ss << argv[2]; ss >> betaP_i;  ss.clear(); ss.str(std::string());
#endif
  int gauge_configs, jkStart, jkEnd;
  
  std::string matelemType;

  ss << argv[4];  ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[5];  ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[6];  ss >> jkStart;       ss.clear(); ss.str(std::string());
  ss << argv[7];  ss >> jkEnd;         ss.clear(); ss.str(std::string());
  ss << argv[8];  ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[9];  ss >> pmax;          ss.clear(); ss.str(std::string());
  ss << argv[10]; ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[11]; ss >> zmax;          ss.clear(); ss.str(std::string());

  // Potentially modify the num elements for initializations
  Nmom = (pmax-pmin)+1;
  Nz   = (zmax-zmin)+1;
  
  // Append Qval or Qplus to matelemType
#ifdef QPLUS
  matelemType="q+_"+matelemType;
#else
  matelemType="qv_"+matelemType;
#endif


  // Set an output file for jackknife fit results
#ifdef UNCORRELATED
#warning "   Performing an uncorrelated fit"
#ifdef CONVOLC
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+".2-parameter.convolC.uncorrelated";
#else
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+".2-parameter.convolK.uncorrelated";
#endif
#else
#warning "   Performing a correlated fit"
#ifdef CONVOLC
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+".2-parameter.convolC.correlated";
#else
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+".2-parameter.convolK.correlated";
#endif
#endif
  output += ".pmin"+std::to_string(pmin)+"_pmax"+std::to_string(pmax)+
    "_zmin"+std::to_string(zmin)+"_zmax"+std::to_string(zmax)+".txt";
  std::ofstream OUT(output.c_str(), std::ofstream::in | std::ofstream::app );
  

  /*
    INITIALIZE STRUCTURE FOR DISTRIBUTION TO FIT (WHERE DISTRIBUTION IS EITHER ITD OR pITD)
  */
  reducedPITD distribution = reducedPITD(gauge_configs);

  // Read from H5 file
  H5Read(argv[3],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"itd"); // pitd

  /*
    Determine full data covariance
  */
  distribution.calcCov();    std::cout << "Computed the full data covariance" << std::endl;
  distribution.calcInvCov(); std::cout << "Computed the inverse of full data covariance" << std::endl;




  /*
    CONSTRUCT THE DATA COVARIANCE OF FITTED REDUCED PSEUDO IOFFE-TIME DISTRIBUTION
    FROM PURELY THE ENSEM INSTANCE OF reducedPITD
  */
  // Initialize gsl_matrices for covariance and inverse and set all entries to zero
#ifndef QPLUS
  gsl_matrix * Cov = gsl_matrix_calloc(ensem->real.disps.size()*Nmom,ensem->real.disps.size()*Nmom);
  gsl_matrix * invCov = gsl_matrix_calloc(ensem->real.disps.size()*Nmom,ensem->real.disps.size()*Nmom);
  setCovariance(Cov,ensem->real,jack,0);//,Nmom,zmin);
#elif defined QPLUS
  gsl_matrix * CovP = gsl_matrix_calloc(ensem->imag.disps.size()*Nmom,ensem->imag.disps.size()*Nmom);
  gsl_matrix * invCovP = gsl_matrix_calloc(ensem->imag.disps.size()*Nmom,ensem->imag.disps.size()*Nmom);
  setCovariance(CovP,ensem->imag,jack,1);//,Nmom,zmin);
#endif
  
  std::cout << "Inverting the data covariance matrix" << std::endl;

  // Do the matrix inversion via SVD, catching the number of singular values
  // removed below SV cut upon return
#ifndef QPLUS
  int svsBelowCut = matrixInv(Cov,invCov,Nmom*ensem->real.disps.size());
  std::cout << "Number of singular values removed from qv covariance = " << svsBelowCut << std::endl;
#elif defined QPLUS
  int svsBelowCutP = matrixInv(CovP,invCovP,Nmom*ensem->imag.disps.size());
  std::cout << "Number of singular values removed from q+ covariance = " << svsBelowCutP << std::endl;
#endif

  // // Covariance matrix is no longer needed
  // gsl_matrix_free(Cov);


  // std::cout << "Checking suitable inverse was found" << std::endl;
  // gsl_matrix * id = gsl_matrix_alloc(Nmom*Nz,Nmom*Nz); gsl_matrix_set_zero(id);
  // gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,Cov,invCov,0.0,id);
	      


  // Store the computed inverse of data covariance matrix
#ifndef QPLUS
  ensem->invCovR = invCov;
#elif defined QPLUS
  ensem->invCovI = invCovP;
#endif




//   /*
//     SCAN OVER ALPHA AND BETA AND EVALUATE CHI2
//     HOPEFULLY USE THESE DETERMINATIONS TO SET INITIAL FIT PARAMS
//   */
//   {
//     std::string scanName = "b_b0xDA__J0_A1pP."+matelemType+"_scan.2-parameter.txt";
//     std::ofstream scanOUT(scanName.c_str(), std::ofstream::in | std::ofstream::app );
//     for ( int ascan = 1; ascan <= 500; ascan++ )
//       {
// // #pragma omp parallel for num_threads(16)
// 	for ( int bscan = 1; bscan <= 500; bscan++ )
// 	  {
// 	    double alphaScan = -0.8 + ascan*(2.0/500);
// 	    double betaScan  = 0.5 + bscan*(3.5/500);

// 	    gsl_vector *scan = gsl_vector_alloc(2);
// 	    gsl_vector_set(scan,0,alphaScan);
// 	    gsl_vector_set(scan,1,betaScan);

// 	    double redChi2 = chi2Func(scan,&ensemReal)/(Nmom*ensemReal.disps.size() - scan->size - svsBelowCut);
// 	    scanOUT << "SCAN:: " << redChi2 << " " << alphaScan << " " << betaScan << "\n";
	    
// 	    gsl_vector_free(scan);
// 	  }
//       }
//     scanOUT.close();
//   }
//   exit(30);
	    








  /*
    SET THE STARTING PARAMETER VALUES AND INITIAL STEP SIZES ONCE
  */
  // double pdfp_ini[3] = { alpha_i, beta_i, alpha_s_i};
  /*
    QPLUS START
  */
#ifdef QPLUS
  // Set initial values in parameter space
#ifdef FITALPHAS
  gsl_vector *pdfp_ini = gsl_vector_alloc(4);
  gsl_vector *pdfpSteps = gsl_vector_alloc(4);

  gsl_vector_set(pdfp_ini,3,alpha_s_i);  // ALPHA_S
  gsl_vector_set(pdfpSteps,3,0.1);
#else
  gsl_vector *pdfp_ini = gsl_vector_alloc(3);
  gsl_vector *pdfpSteps = gsl_vector_alloc(3);
#endif
  gsl_vector_set(pdfp_ini,0,normP_i);    // q+ NORMALIZATION
  gsl_vector_set(pdfp_ini,1,alphaP_i);   // q+ ALPHA
  gsl_vector_set(pdfp_ini,2,betaP_i);    // q+ BETA
  // Set initial step sizes in parameter space
  gsl_vector_set(pdfpSteps,0,0.5);       // q+ NORMALIZATION step
  gsl_vector_set(pdfpSteps,1,0.1);       // q+ ALPHA step
  gsl_vector_set(pdfpSteps,2,0.2);       // q+ BETA step
#endif
  // END QPLUS

  /*
    QVALENCE START
  */
#ifndef QPLUS
#ifdef FITALPHAS
  gsl_vector *pdfp_ini = gsl_vector_alloc(3);
  gsl_vector *pdfpSteps = gsl_vector_alloc(3);

  gsl_vector_set(pdfp_ini,2,alpha_s_i);  // ALPHA_S
  gsl_vector_set(pdfpSteps,2,0.1);
#else
  gsl_vector *pdfp_ini = gsl_vector_alloc(2);
  gsl_vector *pdfpSteps = gsl_vector_alloc(2);
#endif
  gsl_vector_set(pdfp_ini,0,alpha_i);    // q valence ALPHA
  gsl_vector_set(pdfp_ini,1,beta_i);     // q valence BETA
  // Set initial step sizes in parameter space
  gsl_vector_set(pdfpSteps,0,0.05);      // q valence ALPHA step
  gsl_vector_set(pdfpSteps,1,0.3);       // q valence BETA step
#endif


  /*
    INITIALIZE THE SOLVER HERE, SO REPEATED CALLS TO SET, RETURN A NEW NMRAND2 SOLVER
  */
  /*
    Initialize a multidimensional minimzer without relying on derivatives of function
  */
  // Select minimization type
  const gsl_multimin_fminimizer_type *minimizer = gsl_multimin_fminimizer_nmsimplex2rand;
  // const gsl_multimin_fminimizer_type *minimizer = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer * fmin = gsl_multimin_fminimizer_alloc(minimizer,pdfp_ini->size);




  /*
    LOOP OVER ALL JACKKNIFE SAMPLES AND DO THE FIT - ITERATING UNTIL COMPLETION
  */
  for ( auto itJ = jack.begin() + jkStart; itJ <= jack.begin() + jkEnd; ++itJ )
    {
      // Time the duration of this fit
      auto jackTimeStart = std::chrono::steady_clock::now();

      // Hold a counter of which jackknife sample
      int JackNum = itJ - jack.begin();

      // Assign the full data covariance
#ifndef QPLUS
      itJ->invCovR = invCov;
#elif defined QPLUS
      itJ->invCovI = invCovP;
#endif

      // Define the gsl_multimin_function
      gsl_multimin_function Chi2;
      // Dimension of the system
      Chi2.n = pdfp_ini->size;
      // Function to minimize
      Chi2.f = &chi2Func;
      Chi2.params = &(*itJ);
      
      
      std::cout << "Establishing initial state for minimizer..." << std::endl;
      // Set the state for the minimizer
      // Repeated call the set function to ensure nelder-mead random minimizer
      // starts w/ a different random simplex for each jackknife sample
      int status = gsl_multimin_fminimizer_set(fmin,&Chi2,pdfp_ini,pdfpSteps);

      std::cout << "Minimizer established..." << std::endl;
      

      // Iteration count
      int k = 1;
      double tolerance = 0.0000001; // 0.0001
      int maxIters = 10000;      // 1000
      
      
      while ( gsl_multimin_test_size( gsl_multimin_fminimizer_size(fmin), tolerance) == GSL_CONTINUE )
	{
	  // End after maxIters
	  if ( k > maxIters ) { break; }
	  
	  // Iterate
	  gsl_multimin_fminimizer_iterate(fmin);
	  
	  std::cout << "Current params  (" << JackNum << "," << k << ") ::" << std::setprecision(14)
#ifndef QPLUS
		    << "   alpha (qv) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),0)
		    << "   beta (qv) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),1)
#ifdef FITALPHAS
		    << "   alpha_s (qv) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),2)
#endif
#endif
#ifdef QPLUS
		    << "   N (q+) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),0)
		    << "   alpha (q+) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),1)
		    << "   beta (q+) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),2)
#ifdef FITALPHAS
		    << "   alpha_s (q+) = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),3)
#endif
#endif
		    << std::endl;
	  
	  k++;
	}
      
      
      
      // Return the best fit parameters
      gsl_vector *bestFitParams = gsl_vector_alloc(pdfp_ini->size);
      bestFitParams = gsl_multimin_fminimizer_x(fmin);

      // Return the best correlated Chi2
      double chiSq = gsl_multimin_fminimizer_minimum(fmin);
      // Determine the reduced chi2
#ifndef QPLUS
      double reducedChiSq = chiSq / (Nmom*ensem->real.disps.size() - pdfp_ini->size - svsBelowCut);
#else
      double reducedChiSq = chiSq / (Nmom*ensem->imag.disps.size() - pdfp_ini->size - svsBelowCutP );
#endif

      
      
      std::cout << " For jackknife sample J = " << JackNum << ", Converged after " << k
		<<" iterations, Optimal Chi2/dof = " << reducedChiSq << std::endl;
#ifndef QPLUS
      std::cout << std::setprecision(6) << "  alpha (qv) =  " << gsl_vector_get(bestFitParams,0) << std::endl;
      std::cout << std::setprecision(6) << "  beta (qv) =  " << gsl_vector_get(bestFitParams,1) << std::endl;
#ifdef FITALPHAS
      std::cout << std::setprecision(6) << "  alpha_s (qv) =  " << gsl_vector_get(bestFitParams,2) << std::endl;
#endif
#endif
#ifdef QPLUS
      std::cout << std::setprecision(6) << "  N (q+) =  " << gsl_vector_get(bestFitParams,0) << std::endl;
      std::cout << std::setprecision(6) << "  alpha (q+) =  " << gsl_vector_get(bestFitParams,1) << std::endl;
      std::cout << std::setprecision(6) << "  beta (q+) =  " << gsl_vector_get(bestFitParams,2) << std::endl;
#ifdef FITALPHAS
      std::cout << std::setprecision(6) << "  alpha_s (q+) =  " << gsl_vector_get(bestFitParams,3) << std::endl;
#endif
#endif
      
      // Write the fit results to a file
      OUT << std::setprecision(10) << reducedChiSq << " "
	  << gsl_vector_get(bestFitParams,0) << " " << gsl_vector_get(bestFitParams,1) << " "
#ifndef QPLUS
#ifdef FITALPHAS
	  << gsl_vector_get(bestFitParams,2)
#endif
#endif
#ifdef QPLUS
	  << gsl_vector_get(bestFitParams,2) << " "
#ifdef FITALPHAS
	  << gsl_vector_get(bestFitParmas,3)
#endif
#endif
	  << "\n";

      OUT.flush();

      // // Free memory
      // gsl_vector_free(bestFitParams);

      
      // Determine/print the total time for this fit
      auto jackTimeEnd = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = jackTimeEnd-jackTimeStart;
      std::cout << "           --- Time to complete jackknife fit " << JackNum << "  =  " << elapsed_seconds.count() << "s\n";
      
    } // End loop over jackknife samples
  

  // Close the output file containing jack fit results
  OUT.close();

  return 0;
}
