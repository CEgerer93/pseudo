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
#include "kernel.h"


// If we are building st. alpha_s is not a fitted parameter, then set a value here
#ifndef FITALPHAS
#define alphaS 0.303
#endif

// Macros for maximum z and p in computed data
#define DATMAXZ 16
#define DATMAXP 6


using namespace PITD;


// Default values of p & z to cut on
int zmin,zmax,pmin,pmax;
int nParams, nParamsLT, nParamsAZ, pdfType; // Determine pdf to fit & # params

/*
  Parameters describing pheno PDF fit
*/
struct pdfFitParams_t
{
  double alpha, beta;           // the leading parameters i.e. x^alpha*(1-x)^beta

  gsl_vector *lt_sigmaN, *lt_fitParams;
  gsl_vector *az_sigmaN, *az_fitParams;

  std::map<int, std::string> pmap; // map to print fit parameter string and values during fit

  // Set barriers
  std::pair<int,int> alphaRestrict = std::make_pair(-1,1);
  std::pair<int,int> betaRestrict = std::make_pair(0.5,5);


  void setSigmaN(double nu, int z)
  {
    for ( size_t l = 0; l < lt_sigmaN->size; l++ )
      gsl_vector_set(lt_sigmaN, l, pitd_texp_sigma_n(l, 85, alpha, beta, nu, z) );
  }

  void setCorrectionSigmaN(double nu, int z)
  {
    for ( size_t l = 0; l < az_sigmaN->size; l++ )
      gsl_vector_set(az_sigmaN, l, pow((1.0/z),2)*pitd_texp_sigma_n_treelevel(l+1, 85, alpha, beta, nu) );
    // Correction starts for n=1 -> \infty
  }

  // Return the fit predicted pITD
  double pitdFit(double nu, int z)
  {
    double sumLT(0.0), sumAZ(0.0);
    setSigmaN(nu, z);
    setCorrectionSigmaN(nu, z);
    gsl_blas_ddot(lt_sigmaN, lt_fitParams, &sumLT); //  (leading-twist sigma_n)^T \cdot (leading-twist params)
    gsl_blas_ddot(az_sigmaN, az_fitParams, &sumAZ); //  (a^2/z^2 sigma_n)^T \cdot (a^2/z^2 params)
    return sumLT+sumAZ;
  }

  // Print the current fit values
  void printFit(gsl_vector *v)
  {
    for ( auto p = pmap.begin(); p != pmap.end(); ++p )
      std::cout << std::setprecision(10) << "  " << p->second << " =  " << gsl_vector_get(v,p->first);
    std::cout << "\n";
  }

  // Write best fit values to file
  void write(std::ofstream &os, double redChi2, gsl_vector *v)
  {
    os << std::setprecision(10) << redChi2 << " ";
    for ( auto p = pmap.begin(); p != pmap.end(); ++p )
      os << gsl_vector_get(v,p->first) << " ";
    os << "\n";
  }

  // Default/Parametrized constructor w/ initializer lists
  pdfFitParams_t() : alpha(0.0), beta(0.0) {}
  pdfFitParams_t(double _a, double _b, gsl_vector *lt, gsl_vector *az)
    : alpha(_a), beta(_b)// , lt_fitParams{lt}, az_fitParams{az}
  {
    // Set the param/jacobi poly vectors
    lt_fitParams = lt; az_fitParams = az;
    lt_sigmaN = gsl_vector_alloc(lt->size);
    az_sigmaN = gsl_vector_alloc(az->size);
    // Now set the parameter map for easy printing
    std::string qtype;
    if ( pdfType == 0 )
      qtype = "qv";
    if ( pdfType == 1 )
      qtype = "q+";

    pmap[0] = "alpha (" + qtype + ")"; pmap[1] = "beta (" + qtype + ")";
    int p;
    if ( pdfType == 0 )
      {
	for ( p = 2; p < 2+lt_sigmaN->size-1; p++ )
	  pmap[p] = "C[" + std::to_string(p-1) + "] (" + qtype +")";
	for ( p = 2+lt_sigmaN->size-1; p < nParams; p++ )
	  pmap[p] = "C_az[" + std::to_string(p-2-lt_sigmaN->size+1) + "] (" + qtype + ")";
      }
    if ( pdfType == 1 )
      {
	for ( p = 2; p < 2+lt_sigmaN->size; p++ )
	  pmap[p] = "C[" + std::to_string(p-2) + "] (" + qtype +")";
	for ( p = 2+lt_sigmaN->size; p < nParams; p++ )
	  pmap[p] = "C_az[" + std::to_string(p-2-lt_sigmaN->size) + "] (" + qtype + ")";
      }
  }
};

/*
  MULTIDIMENSIONAL MINIMIZATION - CHI2 
  MINIMIZE LEAST-SQUARES BETWEEN DATA AND NUMERICALLY INTEGRATED MATCHING KERNEL AND pITD
*/
double chi2Func(const gsl_vector * x, void *data)
{
  // Get a pointer to the (void) "thisJack" reducedPITD class instance
  reducedPITD * ptrJack = (reducedPITD *)data;
  gsl_matrix * invCov;
  if ( pdfType == 0 )
    invCov = (ptrJack->data.invCovR);
  if ( pdfType == 1 )
    invCov = (ptrJack->data.invCovI);

  // Get the current pdf params
  double dumA = gsl_vector_get(x,0); // alpha
  double dumB = gsl_vector_get(x,1); // beta

  gsl_vector *dumLTFit;
  gsl_vector *dumAZFit = gsl_vector_alloc(nParamsAZ);
  switch(pdfType)
    {
    case 0: // QVAL
      dumLTFit = gsl_vector_alloc(nParamsLT);
      gsl_vector_set(dumLTFit, 0, 1.0/betaFn(dumA+1,dumB+1) );
      for ( int d = 2; d < 2+nParamsLT-1; d++ )
	gsl_vector_set(dumLTFit, d-1, gsl_vector_get(x, d) );

      // Now grab the correction coefficients
      for ( int d = 2+dumLTFit->size-1; d < nParams; d++ )
	gsl_vector_set(dumAZFit, d-1-dumLTFit->size, gsl_vector_get(x, d));

      break;

    case 1: // QPLUS
      dumLTFit = gsl_vector_alloc(nParamsLT);
      
      for ( int d = 2; d < 2 + nParamsLT; d++ )
	gsl_vector_set(dumLTFit, d-2, gsl_vector_get(x, d));
      // Nor grab the correction coefficients
      for ( int d = 2 + nParamsLT; d < nParams; d++ )
	gsl_vector_set(dumAZFit, d-2-nParamsLT, gsl_vector_get(x, d));

      break;
    }
  // Now set the pdf params within a pdfFitParams_t struct instance
  pdfFitParams_t pdfp(dumA, dumB, dumLTFit, dumAZFit);
  

  /*
    Initialize chi2 and vectors to store differences between jack data and convolution
  */
  double chi2(0.0);
  gsl_vector *iDiffVec = gsl_vector_alloc(invCov->size1);
  gsl_vector *jDiffVec = gsl_vector_alloc(invCov->size1);

  /*
    Evaluate the kernel for all points and store
    ALSO
    Set the difference btwn convolution and jack data
  */
#pragma omp parallel num_threads(ptrJack->data.disps.size())
// #pragma omp parallel
// #pragma omp for
  for ( auto zz = ptrJack->data.disps.begin(); zz != ptrJack->data.disps.end(); ++zz )
    {
#pragma omp parallel num_threads(zz->second.moms.size())
// #pragma omp parallel
// #pragma omp for
      for ( auto mm = zz->second.moms.begin(); mm != zz->second.moms.end(); ++mm )
	{
	  // The index
	  int I = std::distance(zz->second.moms.begin(),mm) +
	    std::distance(ptrJack->data.disps.begin(),zz)*zz->second.moms.size();

	  double convolTmp;
	  if ( mm->second.IT == 0 )
	    {
	      if ( pdfType == 0 )
		convolTmp = 1;
	      if ( pdfType == 1 )
		convolTmp = 0;
	    }
	  else
	    {
	      convolTmp = pdfp.pitdFit(mm->second.IT,zz->first);
	    }

	  if ( pdfType == 0 )
	    gsl_vector_set(iDiffVec,I,convolTmp - mm->second.mat[0].real());
	  if ( pdfType == 1 )
	    gsl_vector_set(iDiffVec,I,convolTmp - mm->second.mat[0].imag());

	}
    }


  // The difference vector need only be computed once, so make a second copy to form correlated chi2
  gsl_vector_memcpy(jDiffVec,iDiffVec);

  // Initialize cov^-1 right multiplying jDiffVec
  gsl_vector *invCovRightMult = gsl_vector_alloc(invCov->size1);
  gsl_blas_dgemv(CblasNoTrans,1.0,invCov,jDiffVec,0.0,invCovRightMult);

  // Form the scalar dot product of iDiffVec & result of invCov x jDiffVec
  gsl_blas_ddot(iDiffVec,invCovRightMult,&chi2);


  // Free some memory
  gsl_vector_free(iDiffVec);
  gsl_vector_free(jDiffVec);
  gsl_vector_free(invCovRightMult);
  // gsl_matrix_free(invCov);


#ifdef CONSTRAINED
  /*
    CHECK FOR VALUES OF {ALPHA,BETA} OUTSIDE ACCEPTABLE RANGE AND INFLATE CHI2
  */
  if ( pdfp.alpha <= pdfp.alphaRestrict.first || pdfp.alpha >= pdfp.alphaRestrict.second )
    chi2+=1000000;
  if ( pdfp.beta <= pdfp.betaRestrict.first || pdfp.beta >= pdfp.betaRestrict.second )
    chi2+=1000000;
#endif

  return chi2;
}


int main( int argc, char *argv[] )
{
  
  if ( argc != 13 )
    {
      std::cout << "Usage: $0 <PDF (0 [QVAL] -or- 1 [QPLUS])> <lt n-jacobi> <az n-jacobi> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <zmin cut> <zmax cut> <pmin cut> <pmax cut>" << std::endl;
      // std::cout << "Usage: $0 <alpha_i> <beta_i> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <zmin> <zmax> <pmin> <pmax>" << std::endl;
      exit(1);
    }
  
  // Enable nested parallelism
  omp_set_nested(true);
  
  std::stringstream ss;
  
  int gauge_configs, jkStart, jkEnd;

  gsl_vector *dumLTIni, *dumAZIni;
  
  std::string matelemType;
  enum PDFs { QVAL, QPLUS };
  
  ss << argv[1];  ss >> pdfType;       ss.clear(); ss.str(std::string());
  ss << argv[2];  ss >> nParamsLT;     ss.clear(); ss.str(std::string());
  ss << argv[3];  ss >> nParamsAZ;     ss.clear(); ss.str(std::string());
  ss << argv[5];  ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[6];  ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[7];  ss >> jkStart;       ss.clear(); ss.str(std::string());
  ss << argv[8];  ss >> jkEnd;         ss.clear(); ss.str(std::string());
  ss << argv[9];  ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[10]; ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[11]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[12]; ss >> pmax;          ss.clear(); ss.str(std::string());

  /*
    Require minimally nParamsLT >= 1
  */
  if ( nParamsLT < 1 || nParamsAZ < 0 )
    {
      std::cout << "Cannot minimize with those parameters" << std::endl;
      exit(1);
    }
  
 
  // Append Qval or Qplus to matelemType
  if ( pdfType == 1 )
    {
      matelemType="q+_"+matelemType;
      nParams = 2 + nParamsLT + nParamsAZ; // C_0 coeff not fixed by PDF normalization
      dumLTIni = gsl_vector_alloc(nParamsLT);
    }
  if ( pdfType == 0 )
    {
      matelemType="qv_"+matelemType;
      nParams = 2 + (nParamsLT - 1) + nParamsAZ; // C_0 coeff fixed by PDF normalization
      dumLTIni = gsl_vector_alloc(nParamsLT); //  - 1);
    }
  dumAZIni = gsl_vector_alloc(nParamsAZ);


  // Set an output file for jackknife fit results
#ifdef UNCORRELATED
#warning "   Performing an uncorrelated fit"
#ifdef CONVOLC
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter.convolC.uncorrelated";
#else
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter.convolK.uncorrelated";
#endif
#else
#warning "   Performing a correlated fit"
#ifdef CONVOLC
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter.convolC.correlated";
#else
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter.convolK.correlated";
#endif
#endif
  output += ".pmin"+std::to_string(pmin)+"_pmax"+std::to_string(pmax)+
    "_zmin"+std::to_string(zmin)+"_zmax"+std::to_string(zmax)+".txt";
  std::ofstream OUT(output.c_str(), std::ofstream::in | std::ofstream::app );
  

  /*
    INITIALIZE STRUCTURE FOR DISTRIBUTION TO FIT (WHERE DISTRIBUTION IS EITHER ITD OR pITD)
  */
  reducedPITD distribution = reducedPITD(gauge_configs, zmin, zmax, pmin, pmax);

  // Read from H5 file (all z's & p's)
#ifdef CONVOLC
  H5Read(argv[4],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"itd"); // pitd
#endif
#ifdef CONVOLK
  H5Read(argv[4],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"pitd");
#endif
  

  /*
    Determine full data covariance
  */
  distribution.calcCov();    std::cout << "Computed the full data covariance" << std::endl;
  // distribution.cutOnPZ(zmin,zmax,pmin,pmax);
  distribution.calcInvCov(); std::cout << "Computed the inverse of full data covariance" << std::endl;
  std::cout << "Cut on {zmin, zmax} = { " << zmin << " , " << zmax << " }  &  {pmin, pmax} = { "
  	    << pmin << " , " << pmax << " }" << std::endl;


  int numCut = ( (DATMAXP - pmax) + pmin - 1 )*DATMAXZ + ( (DATMAXZ - zmax) + zmin )*DATMAXP;
  std::cout << "*** Removed " << numCut << " data points from fit" << std::endl;


#if 0
  std::cout << "Checking suitable inverse was found" << std::endl;
  gsl_matrix * id = gsl_matrix_alloc(distribution.data.covR->size1,distribution.data.covR->size1);
  gsl_matrix_set_zero(id);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,distribution.data.covR,distribution.data.invCovR,0.0,id);
  printMat(id);
  exit(8);
#endif



  /*
    SET THE STARTING PARAMETER VALUES AND INITIAL STEP SIZES ONCE
  */
  gsl_vector *pdfp_ini, *pdfpSteps;
  // pdfFitParams_t dumPfp(0.1,0,pdfp_ini,pdfpSteps); // collect the master starting variables
  pdfFitParams_t *dumPfp;

  pdfp_ini  = gsl_vector_alloc(nParams);
  pdfpSteps = gsl_vector_alloc(nParams);
  dumPfp = new pdfFitParams_t(0.1,2.0,dumLTIni,dumAZIni); // pass dummy LT & AZ vectors for
                                                          // pmap member initialization
  for ( int s = 0; s < 2; s++ ) { gsl_vector_set(pdfpSteps, s, 0.1); } // alpha, beta step sizes
  
  gsl_vector_set(pdfp_ini, 0, dumPfp->alpha); // alpha
  gsl_vector_set(pdfp_ini, 1, dumPfp->beta);  // beta
  
  // Set the remainder of fit params
  for ( int s = 2; s < pdfp_ini->size; s++ )
    {
      gsl_vector_set(pdfpSteps, s, 0.05);
      gsl_vector_set(pdfp_ini, s, 0.0);
    }


  /*
    INITIALIZE THE SOLVER HERE, SO REPEATED CALLS TO SET, RETURN A NEW NMRAND2 SOLVER
  */
  /*
    Initialize a multidimensional minimizer without relying on derivatives of function
  */
  // Select minimization type
  const gsl_multimin_fminimizer_type *minimizer = gsl_multimin_fminimizer_nmsimplex2rand;
  // const gsl_multimin_fminimizer_type *minimizer = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer * fmin = gsl_multimin_fminimizer_alloc(minimizer,pdfp_ini->size);

  if ( pdfType == 0 )
    std::cout << "Performing " << nParams << " parameter fit to QVAL" << std::endl;
  if ( pdfType == 1 )
    std::cout << "Performing " << nParams << " parameter fit to QPLUS" << std::endl;


  
  /*
    Collection for the fit results
  */
  std::vector<pdfFitParams_t> fitResults(gauge_configs);


  /*
    LOOP OVER ALL JACKKNIFE SAMPLES AND DO THE FIT - ITERATING UNTIL COMPLETION
  */
  for ( int itJ = jkStart; itJ < jkEnd; itJ++ )
    {
      // Time the duration of this fit
      auto jackTimeStart = std::chrono::steady_clock::now();


      /*
	BEGIN EXTRACTION OF INFO FOR THIS JACKKNIFE
      */
      // Instantiate a reducedPITD object for this jackknife
      reducedPITD thisJack(1, zmin, zmax, pmin, pmax);
      thisJack.data.invCovR = distribution.data.invCovR;
      thisJack.data.invCovI = distribution.data.invCovI;
      // Extract
      for ( auto z = distribution.data.disps.begin(); z != distribution.data.disps.end(); ++z )
	{
	  // Extract the jackknife value and associate with a momentum
	  zvals jackMoms;
	  for ( auto m = z->second.moms.begin(); m != z->second.moms.end(); ++m )
	    {
	      // A momVals for this jackknife
	      momVals jMomVals; jMomVals.mat.resize(1);
	      jMomVals.IT     = m->second.IT;
	      jMomVals.mat[0] = m->second.mat[itJ];

	      std::pair<std::string, momVals> amomJack (m->first, jMomVals);
	      jackMoms.moms.insert(amomJack);
	    } // end auto m

	  // Pack all of the jackMoms into this jackknife jDisps map
	  std::pair<int, zvals> azJack(z->first, jackMoms);
	  thisJack.data.disps.insert(azJack);
	} // end auto z
      /*
	DONE EXTRACTING INFO FOR THIS JACKKNIFE
      */
    

  
      // Define the gsl_multimin_function
      gsl_multimin_function Chi2;
      // Dimension of the system
      Chi2.n = pdfp_ini->size;
      // Function to minimize
      Chi2.f = &chi2Func;
      Chi2.params = &thisJack;
  
  
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
	  
	  std::cout << "Current params  (" << itJ << "," << k << ") ::";
	  dumPfp->printFit(gsl_multimin_fminimizer_x(fmin));
	  
	  k++;
	}
      
      
      
      // Return the best fit parameters
      gsl_vector *bestFitParams = gsl_vector_alloc(pdfp_ini->size);
      bestFitParams = gsl_multimin_fminimizer_x(fmin);

      // Return the best correlated Chi2
      double chiSq = gsl_multimin_fminimizer_minimum(fmin);
      // Determine the reduced chi2
      double reducedChiSq;
      // [02/16/2021] Replace substraction of singular values, with datapts cut from fit
      if ( pdfType == 0 )
	reducedChiSq = chiSq / (distribution.data.covR->size1 - pdfp_ini->size - distribution.data.svsFullR); 
      // distribution.data.svsFullR)  -OR- numCut ?!?!?!?!
      if ( pdfType == 1 )
	reducedChiSq = chiSq / (distribution.data.covI->size1 - pdfp_ini->size - distribution.data.svsFullI);
      // distribution.data.svsFullI)  -OR- numCut ?!?!?!?!

      
      
      std::cout << " For jackknife sample J = " << itJ << ", Converged after " << k
		<<" iterations, Optimal Chi2/dof = " << reducedChiSq << std::endl;
      dumPfp->printFit(bestFitParams);
      



      // // Pack the best fit values
      // // (n.b. this is so functionality of pdfFitParams_t object can be used)
      // pdfFitParams_t * best;
      // switch (pdfType)
      // 	{
      // 	case 0:
      // 	  switch (nParams)
      // 	    {
      // 	    case 2:
      // 	      best = new pdfFitParams_t(false, gsl_vector_get(bestFitParams, 0),
      // 					gsl_vector_get(bestFitParams, 1)); break;
      // 	    case 3:
      // 	      best = new pdfFitParams_t(false, gsl_vector_get(bestFitParams, 0),
      // 					gsl_vector_get(bestFitParams, 1), 0.0, gsl_vector_get(bestFitParams, 2));
      // 	      break;
      // 	    case 4:
      // 		best = new pdfFitParams_t(false, gsl_vector_get(bestFitParams, 0),
      // 					  gsl_vector_get(bestFitParams, 1), gsl_vector_get(bestFitParams, 2),
      // 					  gsl_vector_get(bestFitParams, 3)); break;
      // 	    }
      // 	  break; // case 0 pdfType
      // 	case 1:
      // 	  switch(nParams)
      // 	    {
      // 	    case 2:
      // 	      best = new pdfFitParams_t(true, gsl_vector_get(bestFitParams, 1),
      // 					gsl_vector_get(bestFitParams, 2)); break;
      // 	    case 3:
      // 	      best = new pdfFitParams_t(true, gsl_vector_get(bestFitParams, 1),
      // 					gsl_vector_get(bestFitParams, 2), 0.0, gsl_vector_get(bestFitParams, 3));
      // 	      break;
      // 	    case 4:
      // 	      best = new pdfFitParams_t(true, gsl_vector_get(bestFitParams, 1),
      // 					gsl_vector_get(bestFitParams, 2), gsl_vector_get(bestFitParams, 3),
      // 					gsl_vector_get(bestFitParams, 4)); break;
      // 	    }
      // 	  best->norm = gsl_vector_get(bestFitParams, 0);
      // 	  break; // case 1 pdfType
      // 	}

      // fitResults[itJ] = *best;

      
      // Write the fit results to a file
      // best->write(OUT, reducedChiSq);
      OUT.flush();

      // delete best;
      
      // Determine/print the total time for this fit
      auto jackTimeEnd = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = jackTimeEnd-jackTimeStart;
      std::cout << "           --- Time to complete jackknife fit "
		<< itJ << "  =  " << elapsed_seconds.count() << "s\n";
      
    } // End loop over jackknife samples


#if 0   //#ifdef CONVOLK
  // Write the pitd->PDF fit results for later plotting
  reducedPITD pitdPDFFit(gauge_configs);
  for ( int z = zmin; z <= zmax; z++ )
    {
      zvals dumZ;
      for ( int m = pmin; m <= pmax; m++ )
	{

	  // A dummy momVals struct to pack
	  momVals dumM(gauge_configs);
	  // Fetch the same Ioffe-time from distribution
	  dumM.IT = distribution.data.disps[z].moms["pz"+std::to_string(m)].IT;

#pragma omp parallel
#pragma omp for
	  for ( int j = 0; j < gauge_configs; j++ )
	    {
	      // Now evaluate the convolution for this jk's best fit params, for this {nu, z}
	      double bestMat = convolution( fitResults[j], dumM.IT, z );
	      if ( pdfType == 0 )
		dumM.mat[j].real( bestMat );
	      if ( pdfType == 1 )
		dumM.mat[j].imag( bestMat );
	    }

	  std::pair<std::string, momVals> amom ( "pz"+std::to_string(m), dumM );

	  dumZ.moms.insert(amom);

	} // end m

      // Now make/insert the dumZ instance
      std::pair<int, zvals> az ( z, dumZ );
      pitdPDFFit.data.disps.insert(az);

    } // end z


  /*
    Now that pitdPDFFit has been populated, used H5Write to write to h5 file
  */  
  std::string out_pitdPDFFit = "b_b0xDA__J0_A1pP." + matelemType + ".pITD-PDF-Fit.h5";
  char * out_pitdPDFFit_h5 = &out_pitdPDFFit[0];
  H5Write(out_pitdPDFFit_h5, &pitdPDFFit, gauge_configs, zmin, zmax, pmin, pmax, "pitd_PDF_Fit");
#endif
  

  // Close the output file containing jack fit results
  OUT.close();

  return 0;
}
