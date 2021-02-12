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
#include <gsl/gsl_rng.h> // Random numbers
#include <gsl/gsl_integration.h> // Numerical integration
#include <gsl/gsl_multimin.h> // multidimensional minimization
#include <gsl/gsl_vector.h> // allocating/accessing gsl_vectors for passing params to minimizer
#include <gsl/gsl_matrix.h> // matrix routines for inversion of data covariance
#include <gsl/gsl_permutation.h> // permutation header for matrix inversions
#include <gsl/gsl_linalg.h> // linear algebra
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

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


/*
  Parameters describing pheno PDF fit
*/
struct pdfFitParams_t
{
  double norm, alpha, beta, gamma, delta;

  // Set barriers
  std::pair<int,int> alphaRestrict = std::make_pair(-1,1);
  std::pair<int,int> betaRestrict = std::make_pair(0.5,5);

  // Default/Parametrized constructor w/ initializer lists
  // pdfFitParams_t() : norm(1.0), alpha(-0.3), beta(2.5), gamma(0.0), delta(0.0) {}
  pdfFitParams_t(double _n = 1.0, double _a = -0.3, double _b = 2.5, double _g = 0.0,
		 double _d = 0.0) : norm(_n), alpha(_a), beta(_b), gamma(_g), delta(_d) {}
};

// Structure to hold all needed parameters to perform convolution of a trial pdf
struct convolParams
{
  pdfFitParams_t p;
  double nu;
  int z;
};


#if 0
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
// double convolution(pdfFitParams_t& params)
double convolution(pdfFitParams_t& params, double nu, int z)
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


  pdfFitParams_t pdfp;
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
#endif


int main( int argc, char *argv[] )
{
  
  if ( argc != 12 )
    {
      std::cout << "Usage: $0 <PDF (QVAL -or- QPLUS)> <nparam fit> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <zmin> <zmax> <pmin> <pmax>" << std::endl;
      // std::cout << "Usage: $0 <alpha_i> <beta_i> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <zmin> <zmax> <pmin> <pmax>" << std::endl;
      exit(1);
    }
  
  // Enable nested parallelism
  omp_set_nested(true);
  
  std::stringstream ss;
  
  //  // Allow for conditional fitting of alpha_s at compile time
  // #ifdef FITALPHAS
  //  double alpha_s_i = 0.3;
  // #endif
  // #ifndef QPLUS
  //  double alpha_i, beta_i;
  //  ss << argv[1]; ss >> alpha_i; ss.clear(); ss.str(std::string());
  //  ss << argv[2]; ss >> beta_i;  ss.clear(); ss.str(std::string());
  // #elif defined QPLUS
  //  double normP_i = 1.0;
  //  double alphaP_i;
  //  double betaP_i;
  //  ss << argv[1]; ss >> alphaP_i; ss.clear(); ss.str(std::string());
  //  ss << argv[2]; ss >> betaP_i;  ss.clear(); ss.str(std::string());
  // #endif
  int gauge_configs, jkStart, jkEnd, nParams, pdfType;
  
  std::string matelemType;
  enum PDFs { QVAL, QPLUS };
  
  ss << argv[1];  ss >> pdfType;       ss.clear(); ss.str(std::string());
  ss << argv[2];  ss >> nParams;        ss.clear(); ss.str(std::string());
  ss << argv[4];  ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[5];  ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[6];  ss >> jkStart;       ss.clear(); ss.str(std::string());
  ss << argv[7];  ss >> jkEnd;         ss.clear(); ss.str(std::string());
  ss << argv[8];  ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[9];  ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[10]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[11]; ss >> pmax;          ss.clear(); ss.str(std::string());

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


#if 0
  std::cout << "Checking suitable inverse was found" << std::endl;
  gsl_matrix * id = gsl_matrix_alloc(distribution.data.covR->size1,distribution.data.covR->size1);
  gsl_matrix_set_zero(id);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,distribution.data.covR,distribution.data.invCovR,0.0,id);
  printMat(id);
  exit(8);
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
  gsl_vector *pdfp_ini, *pdfpSteps;

  switch(pdfType)
    {
    case QVAL:
      pdfp_ini  = gsl_vector_alloc(nParams);
      pdfpSteps = gsl_vector_alloc(nParams);
      for ( int s = 0; s < nParams; s++ )
	{
	  gsl_vector_set(pdfp_ini, s, pdfFitParams_t().alpha);
	}

    case QPLUS:
      pdfp_ini  = gsl_vector_alloc(nParams+1);
      pdfpSteps = gsl_vector_alloc(nParams+1);
    }

  // std::cout << pdfp_ini->size() << std::endl;
  std::cout << gsl_vector_get(pdfp_ini,0) << std::endl;



#if 0
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
#endif
  return 0;
}
