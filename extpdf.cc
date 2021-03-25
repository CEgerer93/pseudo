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
#include "varpro.h"
#include "pdf_fit_util.h"

// If we are building st. alpha_s is not a fitted parameter, then set a value here
#ifndef FITALPHAS
#define alphaS 0.303
#endif

// Macros for maximum z and p in computed data
#define DATMAXZ 16
#define DATMAXP 6


using namespace PITD;
using namespace VarPro;

// Default values of p & z to cut on
int zmin,zmax,pmin,pmax;
int nParams, nParamsLT, nParamsAZ, nParamsT4, nParamsT6, pdfType; // Determine pdf to fit & # params

// Define non-linear priors and prior widths
std::vector<double> nlPriors {0.0, 3.0}; // {alpha,beta}
std::vector<double> nlWidths {0.25, 0.5}; // 0.25, 0.25


// std::vector<double> nl_mu {0.0, 3.0};
// std::vector<double> nl_sig {0.25, 0.5};
// std::vector<double> nlPriors(2);
// std::vector<double> nlWidths(2);


// nlPriors[0] = log( pow( nl_mu[0], 2) / sqrt( pow(nl_mu[0], 2) + pow(nl_sig[0], 2) ) );
// nlPriors[1] = log( pow( nl_mu[1], 2) / sqrt( pow(nl_mu[1], 2) + pow(nl_sig[1], 2) ) );
// nlWidths[0] = log( 1 + ( pow(nl_sig[0], 2) / pow(nl_mu[0], 2) ) );
// nlWidths[1] = log( 1 + ( pow(nl_sig[1], 2) / pow(nl_mu[1], 2) ) );


// std::vector<double> nlPriors { log( pow( 0.1, 2) / sqrt( pow(0.1, 2) + pow(0.25, 2) ) ), log( pow( 3, 2) / sqrt( pow(3, 2) + pow(0.5, 2) ) )};
// std::vector<double> nlWidths { log( 1 + ( pow(0.25, 2) / pow(0.1, 2) ) ), log( 1 + ( pow(0.5, 2) / pow(3.0, 2) ))};




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

  gsl_vector *dumLTFit = gsl_vector_alloc(nParamsLT);
  gsl_vector *dumAZFit = gsl_vector_alloc(nParamsAZ);
  gsl_vector *dumT4Fit = gsl_vector_alloc(nParamsT4);
  gsl_vector *dumT6Fit = gsl_vector_alloc(nParamsT6);
#if 0
  switch(pdfType)
    {
    case 0: // QVAL
      dumLTFit = gsl_vector_alloc(nParamsLT);
      gsl_vector_set(dumLTFit, 0, 1.0/betaFn(dumA+1,dumB+1) );
      break;
    case 1: // QPLUS
      dumLTFit = gsl_vector_alloc(nParamsLT);
      break;
    }
#endif
  // Now set the pdf params within a pdfFitParams_t struct instance
  pdfFitParams_t pdfp(pdfType, dumA, dumB, dumLTFit, dumAZFit, dumT4Fit, dumT6Fit);
  

  /*
    Initialize chi2 and vectors to store differences between jack data and convolution
  */
  double chi2(0.0);
#ifdef VARPRO
  // collect pairs of nu and z for construction of basis functions
  std::vector<std::pair<int, double> > nuz(invCov->size1);
  gsl_vector *dataVec = gsl_vector_alloc(invCov->size1);
#else
  gsl_vector *iDiffVec = gsl_vector_alloc(invCov->size1);
  gsl_vector *jDiffVec = gsl_vector_alloc(invCov->size1);
#endif

  /*
    Evaluate the kernel for all points and store
    ALSO
    Set the difference btwn convolution and jack data
  */
#pragma omp parallel num_threads(ptrJack->data.disps.size())
  for ( auto zz = ptrJack->data.disps.begin(); zz != ptrJack->data.disps.end(); ++zz )
    {
#pragma omp parallel num_threads(zz->second.moms.size())
      for ( auto mm = zz->second.moms.begin(); mm != zz->second.moms.end(); ++mm )
	{
	  // The index
	  int I = std::distance(zz->second.moms.begin(),mm) +
	    std::distance(ptrJack->data.disps.begin(),zz)*zz->second.moms.size();

#ifndef VARPRO
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

#elif defined VARPRO
	  std::pair<int, double> nuzTmp = std::make_pair(zz->first, mm->second.IT);
	  nuz[I] = nuzTmp;
	  if ( pdfType == 0 )
	    gsl_vector_set(dataVec,I,mm->second.mat[0].real());
	  if ( pdfType == 1 )
	    gsl_vector_set(dataVec,I,mm->second.mat[0].imag());
#endif
	}
    }


#ifdef VARPRO
  varPro VP(nParamsLT, nParamsAZ, nParamsT4, nParamsT6, nuz.size(), pdfType);

  VP.makeBasis(dumA, dumB, nuz);
  VP.makeY(dataVec, invCov, pdfp);
  VP.makePhi(invCov, pdfp);
  VP.getInvPhi(); // compute inverse of Phi matrix

  // Collect result of data vectors sandwiched between inverse of data covariance
  double dataSum(0.0);
  // Collect result of varPro matrix/vector operations
  double varProSum(0.0);

  // Identity
  gsl_matrix *id = gsl_matrix_alloc(nParamsLT+nParamsAZ+nParamsT4+nParamsT6, nParamsLT+nParamsAZ+nParamsT4+nParamsT6);
  gsl_matrix_set_identity(id);
  gsl_matrix *inner = gsl_matrix_calloc(nParamsLT+nParamsAZ+nParamsT4+nParamsT6, nParamsLT+nParamsAZ+nParamsT4+nParamsT6);


  /*
    Determine VarPro solution for chi^2
  */
  gsl_matrix *product = gsl_matrix_alloc(VP.invPhi->size1,VP.invPhi->size2);
  gsl_matrix_memcpy(product, VP.invPhi);

  gsl_blas_dgemm(CblasTrans,CblasNoTrans,-1.0,VP.invPhi,id,2.0,product);
  gsl_vector *rightMult = gsl_vector_alloc(nParamsLT+nParamsAZ+nParamsT4+nParamsT6);
  gsl_blas_dgemv(CblasNoTrans,1.0,product,VP.Y,0.0,rightMult);
  gsl_blas_ddot(VP.Y,rightMult,&varProSum);
  varProSum *= -1;
  /*
    End of VarPro solution for chi^2
  */

  // Now compute (data)^T x Cov^-1 x (data)
  gsl_vector *dataRMult = gsl_vector_alloc(invCov->size1);
  gsl_blas_dgemv(CblasNoTrans,1.0,invCov,dataVec,0.0,dataRMult);
  gsl_blas_ddot(dataVec,dataRMult,&dataSum);


  // FINAL CHI2 FROM VARPRO
  chi2 = dataSum + varProSum;

#else
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
#endif



#ifdef CONSTRAINED
  // Log-normal on alpha, beta
  chi2 += (pow( (log(dumA + 1) - nlPriors[0]), 2))/pow(nlWidths[0],2)
    + (pow( (log(dumB + 0) - nlPriors[1]), 2))/pow(nlWidths[1],2);
  // A const piece remains from VarPro w/ priors
  for ( int c = 0; c < VP.numCorrections; c++ )
    {
      if ( c < VP.numLT )
	chi2 += pow(pdfp.prior[c],2)/pow(pdfp.width[c],2);
      if ( c >= VP.numLT && c < VP.numLT + VP.numAZ )
	chi2 += pow(pdfp.az_prior[c-VP.numLT],2)/pow(pdfp.az_width[c-VP.numLT],2);
      if ( c >= VP.numLT + VP.numAZ )
	chi2 += pow(pdfp.t4_prior[c-VP.numLT-VP.numAZ],2)/pow(pdfp.t4_width[c-VP.numLT-VP.numAZ],2);
    }
#endif
  return chi2;
}


int main( int argc, char *argv[] )
{
  
  if ( argc != 15 )
    {
      std::cout << "Usage: $0 <PDF (0 [QVAL] -or- 1 [QPLUS])> <lt n-jacobi> <az n-jacobi> <Twist-4 n-jacobi> <Twist-6 n-jacobi> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <zmin cut> <zmax cut> <pmin cut> <pmax cut>" << std::endl;
      exit(1);
    }
  
  // Enable nested parallelism
  omp_set_nested(true);
  
  std::stringstream ss;
  
  int gauge_configs, jkStart, jkEnd;

  gsl_vector *dumLTIni, *dumAZIni, *dumT4Ini, *dumT6Ini;
  
  std::string matelemType;
  enum PDFs { QVAL, QPLUS };
  
  ss << argv[1];  ss >> pdfType;       ss.clear(); ss.str(std::string());
  ss << argv[2];  ss >> nParamsLT;     ss.clear(); ss.str(std::string());
  ss << argv[3];  ss >> nParamsAZ;     ss.clear(); ss.str(std::string());
  ss << argv[4];  ss >> nParamsT4;     ss.clear(); ss.str(std::string());
  ss << argv[5];  ss >> nParamsT6;     ss.clear(); ss.str(std::string());
  ss << argv[7];  ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[8];  ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[9];  ss >> jkStart;       ss.clear(); ss.str(std::string());
  ss << argv[10];  ss >> jkEnd;         ss.clear(); ss.str(std::string());
  ss << argv[11]; ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[12]; ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[13]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[14]; ss >> pmax;          ss.clear(); ss.str(std::string());

  /*
    Require minimally nParamsLT >= 1
  */
  if ( nParamsLT < 1 || nParamsAZ < 0 || nParamsT4 < 0 || nParamsT6 < 0 )
    {
      std::cout << "Cannot minimize with those parameters" << std::endl;
      exit(1);
    }
  
 
  // Append Qval or Qplus to matelemType
  if ( pdfType == 1 )
    {
      matelemType="q+_"+matelemType;
      nParams = 2 + nParamsLT + nParamsAZ + nParamsT4 + nParamsT6; // C_0 coeff not fixed by PDF normalization
      dumLTIni = gsl_vector_alloc(nParamsLT);
    }
  if ( pdfType == 0 )
    {
      matelemType="qv_"+matelemType;
      // nParams = 2 + (nParamsLT - 1) + nParamsAZ + nParamsT4 + nParamsT6; // C_0 coeff fixed by PDF normalization
      nParams = 2 + nParamsLT + nParamsAZ + nParamsT4 + nParamsT6;
      dumLTIni = gsl_vector_alloc(nParamsLT); //  - 1);
    }
  dumAZIni = gsl_vector_alloc(nParamsAZ);
  dumT4Ini = gsl_vector_alloc(nParamsT4);
  dumT6Ini = gsl_vector_alloc(nParamsT6);


  // Set an output file for jackknife fit results
#ifdef UNCORRELATED
#warning "   Performing an uncorrelated fit"
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter_"+
    std::to_string(nParamsLT)+"lt_"+std::to_string(nParamsAZ)+"az_"+
    std::to_string(nParamsT4)+"t4_"+std::to_string(nParamsT6)+"t6_"+
    ".convolJ.uncorrelated";
#else
#warning "   Performing a correlated fit"
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter_"+
    std::to_string(nParamsLT)+"lt_"+std::to_string(nParamsAZ)+"az_"+
    std::to_string(nParamsT4)+"t4_"+std::to_string(nParamsT6)+"t6_"+
    ".convolJ.correlated";
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
  H5Read(argv[6],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"itd"); // pitd
#endif
#ifdef CONVOLK
  H5Read(argv[6],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"pitd");
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
  pdfFitParams_t *dumPfp;

  pdfp_ini  = gsl_vector_alloc(2);
  pdfpSteps = gsl_vector_alloc(2);
  for ( int s = 0; s < 2; s++ ) { gsl_vector_set(pdfpSteps, s, 0.15); } // alpha, beta step sizes


  /*
    INITIALIZE THE SOLVER HERE, SO REPEATED CALLS TO SET, RETURN A NEW NMRAND2 SOLVER

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

      /*
	Set up an instance of pdfFitParams_t struct for fit param printing
	pass dummy LT, AZ, T4 vectors for pmap member initialization
      */
      pdfFitParams_t *dumPfp = new pdfFitParams_t(pdfType, 0.05,2.0,dumLTIni,dumAZIni,dumT4Ini,dumT6Ini);
      gsl_vector_set(pdfp_ini, 0, dumPfp->alpha); // alpha
      gsl_vector_set(pdfp_ini, 1, dumPfp->beta);  // beta


      // Time the duration of this fit
      auto jackTimeStart = std::chrono::steady_clock::now();


      /*
	BEGIN EXTRACTION OF INFO FOR THIS JACKKNIFE
      */
      std::vector<std::pair<int, double> > nuzJK(distribution.data.invCovR->size1);
      gsl_vector *dataVecJK = gsl_vector_alloc(distribution.data.invCovR->size1);
      // Instantiate a reducedPITD object for this jackknife (only 1cfg)
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
	      int idx = std::distance(z->second.moms.begin(),m) +
		std::distance(distribution.data.disps.begin(),z)*z->second.moms.size();

	      // A momVals for this jackknife
	      momVals jMomVals; jMomVals.mat.resize(1);
	      jMomVals.IT     = m->second.IT;
	      jMomVals.mat[0] = m->second.mat[itJ];

	      std::pair<std::string, momVals> amomJack (m->first, jMomVals);
	      jackMoms.moms.insert(amomJack);


	      /*
		Don't forget to pack {nuzJK, dataVecJK} for final varPro solution
	      */
	      nuzJK[idx] = std::make_pair(z->first,jMomVals.IT);
	      if ( pdfType == 0 )
		gsl_vector_set(dataVecJK, idx, jMomVals.mat[0].real());
	      if ( pdfType == 1 )
		gsl_vector_set(dataVecJK, idx, jMomVals.mat[0].imag());
	      
	      
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
      Chi2.n = pdfp_ini->size;   // (dim = 2, as only alpha/beta are treated w/ nelder-mead)
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
      gsl_vector *fminBest = gsl_vector_alloc(pdfp_ini->size);
      fminBest = gsl_multimin_fminimizer_x(fmin);

      // Return the best correlated Chi2
      double chiSq = gsl_multimin_fminimizer_minimum(fmin);
      // Determine the reduced chi2
      double reducedChiSq;
      // [02/16/2021] Replace substraction of singular values, with datapts cut from fit
      if ( pdfType == 0 )
	reducedChiSq = chiSq / (distribution.data.covR->size1 - nParams - distribution.data.svsFullR); 
      if ( pdfType == 1 )
	reducedChiSq = chiSq / (distribution.data.covI->size1 - nParams - distribution.data.svsFullI);


      /*
	With optimal {alpha,beta}, make one last varPro instance and determine solution vector of constants
      */
      varPro solution(nParamsLT, nParamsAZ, nParamsT4, nParamsT6, nuzJK.size(), pdfType);
      solution.makeBasis(gsl_vector_get(fminBest,0),
			 gsl_vector_get(fminBest,1),nuzJK);
      if ( pdfType == 0 )
	{
	  solution.makeY(dataVecJK, thisJack.data.invCovR, *dumPfp);
	  solution.makePhi(thisJack.data.invCovR, *dumPfp);
	}
      if ( pdfType == 1 )
	{
	  solution.makeY(dataVecJK, thisJack.data.invCovI, *dumPfp);
	  solution.makePhi(thisJack.data.invCovI, *dumPfp);
	}
      solution.getInvPhi();
      gsl_blas_dgemv(CblasNoTrans,1.0,solution.invPhi,solution.Y,0.0,solution.soln);
      std::cout << "[BEST] CONST VEC = "; printVec(solution.soln);

      gsl_vector *bestFitParams = gsl_vector_alloc(nParams);
      gsl_vector_set(bestFitParams,0,gsl_vector_get(fminBest,0));
      gsl_vector_set(bestFitParams,1,gsl_vector_get(fminBest,1));
      for ( int b = 2; b < bestFitParams->size; b++ )
	gsl_vector_set(bestFitParams,b,gsl_vector_get(solution.soln,b-2));
      ///////////////////////////////////////////////////////////////////////////////////////////


      std::cout << " For jackknife sample J = " << itJ << ", Converged after " << k
		<<" iterations, Optimal Chi2/dof = " << reducedChiSq << std::endl;
      dumPfp->printBest(bestFitParams);
      



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
      dumPfp->write(OUT, reducedChiSq, bestFitParams);
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
