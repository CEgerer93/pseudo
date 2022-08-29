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

// // If we are building st. alpha_s is not a fitted parameter, then set a value here
// #ifndef FITALPHAS
// #define alphaS 0.303
// #endif

// Macros for maximum z and p in computed data
#warning "Setting DATMAXZ & DATMAXP from Makefile!***************"
#define DATMAXZ MAXZ
#define DATMAXP MAXP


using namespace PITD;
using namespace VarPro;

// Gamma matrix of insertion - in Chroma integer notation
int dirac;

// Default values of p & z to cut on
int zmin,zmax,pmin,pmax,dataDim;
int nParams, nParamsLT, nParamsAZ, nParamsT4, nParamsT6, nParamsT8, nParamsT10, pdfType; // Determine pdf to fit & # params

const double scale = 1.0; // rescale prior widths

// Define non-linear priors and prior widths
const int alpha0 = -1;
const int beta0  = 0; // AUG 3, 2022: Why the heck was this -1??
// std::vector<double> nlPriors {0.0, 3.0}; // {alpha,beta}
// std::vector<double> nlWidths {scale*0.25, scale*0.5}; // JK [0.4,1.0] // ME [0.25, 0.5]

// Priors of logarithm of alpha & beta --> these match JK's priors
const std::vector<int> logLowBound {alpha0, beta0};
const std::vector<double> logPriors {0.0, 3.0};
const std::vector<double> logWidths {scale*0.4,scale*1};

// These are the actual priors used in log-normal distributions
std::vector<double> nlPriors;
std::vector<double> nlWidths;

void setLogPriors()
{
  for ( int n = 0; n < logPriors.size(); ++n )
    {
      nlPriors.push_back( log( ( pow(logPriors[n]-logLowBound[n],2) )/sqrt(pow(logPriors[n]-logLowBound[n],2)+pow(logWidths[n],2)) ) );
      nlWidths.push_back( log( 1 + pow(logWidths[n],2)/pow(logPriors[n]-logLowBound[n],2) ) );
    }
}

/*
  MULTIDIMENSIONAL MINIMIZATION - CHI2 
  MINIMIZE LEAST-SQUARES BETWEEN DATA AND NUMERICALLY INTEGRATED MATCHING KERNEL AND pITD
*/
double costFunc(const gsl_vector * x, void *data)
{
  // Get a pointer to the (void) "thisJack" reducedPITD class instance
  reducedPITD * ptrJack = (reducedPITD *)data;
  gsl_matrix * invCov;
  if ( pdfType == 0 )
    invCov = (ptrJack->data.invCovR);
  if ( pdfType == 1 )
    invCov = (ptrJack->data.invCovI);

#if 0
#warning "Debug crap"
  for ( int zz = 1; zz < 9; ++zz )
    {
      std::cout << "For z = " << zz << "..." << std::endl;
      ptrJack->ensemPrintZ(zz,1);
    }
  exit(90);
#endif

  // Get the current pdf params
  double dumA = gsl_vector_get(x,0); // alpha
  double dumB = gsl_vector_get(x,1); // beta

  gsl_vector *dumLTFit = gsl_vector_alloc(nParamsLT);
  gsl_vector *dumAZFit = gsl_vector_alloc(nParamsAZ);
  gsl_vector *dumT4Fit = gsl_vector_alloc(nParamsT4);
  gsl_vector *dumT6Fit = gsl_vector_alloc(nParamsT6);
  gsl_vector *dumT8Fit = gsl_vector_alloc(nParamsT8);
  gsl_vector *dumT10Fit = gsl_vector_alloc(nParamsT10);
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
  pdfFitParams_t pdfp(pdfType, dumA, dumB, scale, dumLTFit, dumAZFit, dumT4Fit, dumT6Fit, dumT8Fit, dumT10Fit);
  

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
// #pragma omp parallel num_threads(ptrJack->data.disps.size())
  for ( auto zz = ptrJack->data.disps.begin(); zz != ptrJack->data.disps.end(); ++zz )
    {
// #pragma omp parallel num_threads(zz->second.moms.size())
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

  
#if 0
#warning "DEBUG"
  // std::cout << "DATA VEC (in costFunc) = " << &dataVec << std::endl;
  std::cout << "DATA VEC (in costFunc) = ";
  for ( size_t q = 0; q < dataVec->size; ++q )
    std::cout << gsl_vector_get(dataVec,q) << " ";
  exit(90);
#endif


#ifdef VARPRO
  varPro VP(nParamsLT, nParamsAZ, nParamsT4, nParamsT6, nParamsT8, nParamsT10, nuz.size(), pdfType);

  VP.makeBasis(dirac, dumA, dumB, nuz);
  VP.makeY(dataVec, invCov, pdfp);
  VP.makePhi(invCov, pdfp);
  VP.getInvPhi(); // compute inverse of Phi matrix

  // Collect result of data vectors sandwiched between inverse of data covariance
  double dataSum(0.0);
  // Collect result of varPro matrix/vector operations
  double varProSum(0.0);

  // Identity
  gsl_matrix *id = gsl_matrix_alloc(nParamsLT+nParamsAZ+nParamsT4+nParamsT6+nParamsT8+nParamsT10, nParamsLT+nParamsAZ+nParamsT4+nParamsT6+nParamsT8+nParamsT10);
  gsl_matrix_set_identity(id);
  gsl_matrix *inner = gsl_matrix_calloc(nParamsLT+nParamsAZ+nParamsT4+nParamsT6+nParamsT8+nParamsT10, nParamsLT+nParamsAZ+nParamsT4+nParamsT6+nParamsT8+nParamsT10);


  /*
    Determine VarPro solution for chi^2
  */
  gsl_matrix *product = gsl_matrix_alloc(VP.invPhi->size1,VP.invPhi->size2);
  gsl_matrix_memcpy(product, VP.invPhi);

  gsl_blas_dgemm(CblasTrans,CblasNoTrans,-1.0,VP.invPhi,id,2.0,product);
  gsl_vector *rightMult = gsl_vector_alloc(nParamsLT+nParamsAZ+nParamsT4+nParamsT6+nParamsT8+nParamsT10);
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
  // chi2 += (pow( (log(dumA - alpha0) - nlPriors[0]), 2))/pow(nlWidths[0],2) + (pow( (log(dumB - beta0) - nlPriors[1]), 2))/pow(nlWidths[1],2); # PREVIOUS COST
  chi2 += (pow( (log(dumA - alpha0) - nlPriors[0]), 2))/pow(nlWidths[0],2) + (pow( (log(dumB - beta0) - nlPriors[1]), 2))/pow(nlWidths[1],2)
    -2*( log(1.0/((dumA-alpha0)*nlWidths[0]*sqrt(2*M_PI))) + log(1.0/((dumB-beta0)*nlWidths[1]*sqrt(2*M_PI))) );
  // A const piece remains from VarPro w/ priors
  for ( int c = 0; c < VP.numCorrections; c++ )
    {
      if ( c < VP.numLT )
	chi2 += pow(pdfp.prior[c],2)/pow(pdfp.width[c],2);
      if ( c >= VP.numLT && c < VP.numLT + VP.numAZ )
	chi2 += pow(pdfp.az_prior[c-VP.numLT],2)/pow(pdfp.az_width[c-VP.numLT],2);
      if ( c >= VP.numLT + VP.numAZ && c < VP.numLT + VP.numAZ + VP.numT4 )
	chi2 += pow(pdfp.t4_prior[c-VP.numLT-VP.numAZ],2)/pow(pdfp.t4_width[c-VP.numLT-VP.numAZ],2);
      if ( c >= VP.numLT + VP.numAZ + VP.numT4 && c < VP.numLT + VP.numAZ + VP.numT4 + VP.numT6 )
	chi2 += pow(pdfp.t6_prior[c-VP.numLT-VP.numAZ-VP.numT4],2)/pow(pdfp.t6_width[c-VP.numLT-VP.numAZ-VP.numT4],2);
      if ( c>= VP.numLT + VP.numAZ + VP.numT4 + VP.numT6 && c < VP.numLT + VP.numAZ + VP.numT4 + VP.numT6 + VP.numT8 )
	chi2 += pow(pdfp.t8_prior[c-VP.numLT-VP.numAZ-VP.numT4-VP.numT6],2)/pow(pdfp.t8_width[c-VP.numLT-VP.numAZ-VP.numT4-VP.numT6],2);
      if ( c >= VP.numLT + VP.numAZ + VP.numT4 + VP.numT6 + VP.numT8 )
	chi2 += pow(pdfp.t10_prior[c-VP.numLT-VP.numAZ-VP.numT4-VP.numT6-VP.numT8],2)/pow(pdfp.t10_width[c-VP.numLT-VP.numAZ-VP.numT4-VP.numT6-VP.numT8],2);
    }
#endif

  gsl_vector_free(dumLTFit);
  gsl_vector_free(dumAZFit);
  gsl_vector_free(dumT4Fit);
  gsl_vector_free(dumT6Fit);
  gsl_vector_free(dumT8Fit);
  gsl_vector_free(dumT10Fit);
  gsl_vector_free(dataVec);
  gsl_matrix_free(id);
  gsl_matrix_free(inner);
  gsl_matrix_free(product);
  gsl_vector_free(rightMult);
  gsl_vector_free(dataRMult);


  return chi2;
}


/*
  CHI2FUNC above actually computes part of L^2 = -2Log( P[\theta|data,I] ) + C
  (i.e. negative logarithm of posterior probability distribution P[\theta|data,I] )
  needed to find most likely set of parameters \theta, given data and prior information 'I'

  --> where most likely set of parameters is found by maximizing posterior distribution,
      which is achieved numerically by minimizing L^2

  --> but L^2 determination that is CHI2FUNC, is missing normalizations of prior distributions

  --> SO, this function will determine correct L^2 and unconstrained \chi2
*/
std::vector<double> ell2Chi2(const double costFromFit, pdfFitParams_t * p, gsl_vector * bestParams,
			     double detCov, int dataDim)
{
  // vector to return true L^2 and unconstrained chi2
  std::vector<double> LC(2,costFromFit); // w/ LC[0] = L^2  &  LC[1] = chi2
  // Convenience
  double alpha = gsl_vector_get(bestParams,0);
  double beta  = gsl_vector_get(bestParams,1);
  int nLT = p->lt_fitParams->size;
  int nAZ = p->az_fitParams->size;
  int nT4 = p->t4_fitParams->size;
  int nT6 = p->t6_fitParams->size;
  int nT8 = p->t8_fitParams->size;
  int nT10 = p->t10_fitParams->size;

#ifdef DETCOV_IN_L2
  /*
    Account for dimension of dataset and determinant of data covariance --> irrelevant constant when comparing models on fixed data
    L^2 -= 2 log ( 1/ \sqrt[ (2\pi)^d * det[Cov] ] )
  */
  LC[0] -= 2*log(1.0/sqrt( pow(2*M_PI,dataDim)*detCov));
#endif

  // Next, remove prior infomation on \alpha, \beta from cost
  LC[1] -= ( pow( log(alpha-alpha0)-nlPriors[0], 2)/pow(nlWidths[0], 2) + pow( log(beta-beta0)-nlPriors[1], 2)/pow(nlWidths[1], 2) );
  LC[1] += 2*( log(1.0/((alpha-alpha0)*nlWidths[0]*sqrt(2*M_PI))) + log(1.0/((beta-beta0)*nlWidths[1]*sqrt(2*M_PI))) );

  // Arrive at final talley for L^2 and unconstrained chi2 by accounting for Gaussian priors of linear Jacobi coeffs.
  int i, offset;
  for ( i = 2; i < nLT + 2; i++ )
    {
      offset = 2;
      LC[1] -= pow( gsl_vector_get(bestParams,i) - p->prior[i-offset], 2)/pow( p->width[i-offset], 2); // correct cost
      LC[0] -= 2*log(1.0/(sqrt(2*M_PI)*p->width[i-offset]));                                           // compute L^2
    }
  for ( i = nLT + 2; i < nLT + nAZ + 2; i++ )
    {
      offset = nLT + 2;
      LC[1] -= pow( gsl_vector_get(bestParams,i) - p->az_prior[i-offset], 2)/pow( p->az_width[i-offset], 2); // correct cost
      LC[0] -= 2*log(1.0/(sqrt(2*M_PI)*p->az_width[i-offset]));                                             // compute L^2
    }
  for ( i = nLT + nAZ + 2; i < nLT + nAZ + nT4 + 2; i++ )
    {
      offset = nLT + nAZ + 2;
      LC[1] -= pow( gsl_vector_get(bestParams,i) - p->t4_prior[i-offset], 2)/pow( p->t4_width[i-offset], 2); // correct cost
      LC[0] -= 2*log(1.0/(sqrt(2*M_PI)*p->t4_width[i-offset]));                                              // compute L^2
    }
  for ( i = nLT + nAZ + nT4 + 2; i < nLT + nAZ + nT4 + nT6 + 2; i++ )
    {
      offset = nLT + nAZ + nT4 + 2;
      LC[1] -= pow( gsl_vector_get(bestParams,i) - p->t6_prior[i-offset], 2)/pow( p->t6_width[i-offset], 2); // correct cost
      LC[0] -= 2*log(1.0/(sqrt(2*M_PI)*p->t6_width[i-offset]));                                              // compute L^2
    }
  for ( i = nLT + nAZ + nT4 + nT6 + 2; i < nLT + nAZ + nT4 + nT6 + nT8 + 2; i++ )
    {
      offset = nLT + nAZ + nT4 + nT6 + 2;
      LC[1] -= pow( gsl_vector_get(bestParams,i) - p->t8_prior[i-offset], 2)/pow( p->t8_width[i-offset], 2); // correct cost
      LC[0] -= 2*log(1.0/(sqrt(2*M_PI)*p->t8_width[i-offset]));                                              // compute L^2
    }
  for ( i = nLT + nAZ + nT4 + nT6 + nT8 + 2; i < p->nParams + 2; i++ )
    {
      offset = nLT + nAZ + nT4 + nT6 + nT8 + 2;
      LC[1] -= pow( gsl_vector_get(bestParams,i) - p->t10_prior[i-offset], 2)/pow( p->t10_width[i-offset], 2); // correct cost
      LC[0] -= 2*log(1.0/(sqrt(2*M_PI)*p->t10_width[i-offset]));                                               // compute L^2
    }

  return LC;
}


int main( int argc, char *argv[] )
{
  if ( argc != 18 && argc != 19 )
    {
      std::cout << "Usage: $0 <PDF (0 [QVAL] -or- 1 [QPLUS])> <lt n-jacobi> <az n-jacobi> <Twist-4 n-jacobi> <Twist-6 n-jacobi> <Twist-8 n-jacobi> <Twist-10 n-jacobi> <h5 file> <matelem type - SR/Plat/L-summ> <gauge_configs> <jkStart> <jkEnd> <zmin cut> <zmax cut> <pmin cut> <pmax cut> <Dirac matrix of insertion - Chroma int notation> <h5 for systematic error>" << std::endl;
      exit(1);
    }
  
  // Enable nested parallelism
  omp_set_nested(true);
  
  std::stringstream ss;
  
  int gauge_configs, jkStart, jkEnd;

  gsl_vector *dumLTIni, *dumAZIni, *dumT4Ini, *dumT6Ini, *dumT8Ini, *dumT10Ini;
  
  std::string matelemType;
  enum PDFs { QVAL, QPLUS };
  
  ss << argv[1];  ss >> pdfType;       ss.clear(); ss.str(std::string());
  ss << argv[2];  ss >> nParamsLT;     ss.clear(); ss.str(std::string());
  ss << argv[3];  ss >> nParamsAZ;     ss.clear(); ss.str(std::string());
  ss << argv[4];  ss >> nParamsT4;     ss.clear(); ss.str(std::string());
  ss << argv[5];  ss >> nParamsT6;     ss.clear(); ss.str(std::string());
  ss << argv[6];  ss >> nParamsT8;     ss.clear(); ss.str(std::string());
  ss << argv[7];  ss >> nParamsT10;    ss.clear(); ss.str(std::string());
  ss << argv[9];  ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[10]; ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[11]; ss >> jkStart;       ss.clear(); ss.str(std::string());
  ss << argv[12]; ss >> jkEnd;         ss.clear(); ss.str(std::string());
  ss << argv[13]; ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[14]; ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[15]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[16]; ss >> pmax;          ss.clear(); ss.str(std::string());
  ss << argv[17]; ss >> dirac;         ss.clear(); ss.str(std::string());

  // set dataDim
  dataDim = (zmax-zmin+1)*(pmax-pmin+1);

  /*
    Big switches based on passed current
  */
  std::string redstarCurr;
  if ( dirac == 8 )  redstarCurr = "b_b0xDA__J0_A1pP";
  else if ( dirac == 11 ) redstarCurr = "a_a1xDA__J1_T1pM";
  else
    {
      std::cerr << "Insertion Gamma = " << dirac << " not supported";
      exit(4);
    }

  /*
    Require minimally nParamsLT >= 1
  */
  if ( nParamsLT < 1 || nParamsAZ < 0 || nParamsT4 < 0 || nParamsT6 < 0 || nParamsT8 < 0 || nParamsT10 < 0 )
    {
      std::cout << "Cannot minimize with those parameters" << std::endl;
      exit(1);
    }
  
 
  // Append Qval or Qplus to matelemType
  if ( pdfType == 1 )
    {
      matelemType="q+_"+matelemType;
      nParams = 2 + nParamsLT + nParamsAZ + nParamsT4 + nParamsT6 + nParamsT8 + nParamsT10; // C_0 coeff not fixed by PDF normalization
      dumLTIni = gsl_vector_alloc(nParamsLT);
    }
  if ( pdfType == 0 )
    {
      matelemType="qv_"+matelemType;
      // nParams = 2 + (nParamsLT - 1) + nParamsAZ + nParamsT4 + nParamsT6; // C_0 coeff fixed by PDF normalization
      nParams = 2 + nParamsLT + nParamsAZ + nParamsT4 + nParamsT6 + nParamsT8 + nParamsT10;
      dumLTIni = gsl_vector_alloc(nParamsLT); //  - 1);
    }
  dumAZIni = gsl_vector_alloc(nParamsAZ);
  dumT4Ini = gsl_vector_alloc(nParamsT4);
  dumT6Ini = gsl_vector_alloc(nParamsT6);
  dumT8Ini = gsl_vector_alloc(nParamsT8);
  dumT10Ini = gsl_vector_alloc(nParamsT10);


  // Set an output file for jackknife fit results
#ifdef UNCORRELATED
#warning "   Performing an uncorrelated fit"
  std::string output = redstarCurr+"."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter_"+
    std::to_string(nParamsLT)+"lt_"+std::to_string(nParamsAZ)+"az_"+
    std::to_string(nParamsT4)+"t4_"+std::to_string(nParamsT6)+"t6_"+
    std::to_string(nParamsT8)+"t8_"+std::to_string(nParamsT10)+"t10_"+
    ".convolJ.uncorrelated";
#else
#warning "   Performing a correlated fit"
  std::string output = redstarCurr+"."+matelemType+"_jack"+std::to_string(jkStart)+
    "-"+std::to_string(jkEnd)+"."+std::to_string(nParams)+"-parameter_"+
    std::to_string(nParamsLT)+"lt_"+std::to_string(nParamsAZ)+"az_"+
    std::to_string(nParamsT4)+"t4_"+std::to_string(nParamsT6)+"t6_"+
    std::to_string(nParamsT8)+"t8_"+std::to_string(nParamsT10)+"t10_"+
    ".convolJ.correlated";
#endif
  output += ".pmin"+std::to_string(pmin)+"_pmax"+std::to_string(pmax)+
    "_zmin"+std::to_string(zmin)+"_zmax"+std::to_string(zmax)+".alphas_"+std::to_string(alphaS)+".scale_"+std::to_string(scale)+"_priors"+".txt";
  std::ofstream OUT(output.c_str(), std::ofstream::in | std::ofstream::app );
  

  /*
    INITIALIZE STRUCTURE FOR DISTRIBUTION TO FIT (WHERE DISTRIBUTION IS EITHER ITD OR pITD)
  */
  reducedPITD distribution = reducedPITD(gauge_configs, zmin, zmax, pmin, pmax);
#ifdef MATELEMSYSTEMATIC
  reducedPITD distributionSysErr = reducedPITD(gauge_configs, zmin, zmax, pmin, pmax);
#endif

  // Read from H5 file (all z's & p's)
#ifdef CONVOLC
  H5Read(argv[8],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"itd",dirac); // pitd
#ifdef MATELEMSYSTEMATIC
  try {
    H5Read(argv[18],&distributionSysErr,gauge_configs,zmin,zmax,pmin,pmax,"itd",dirac); // pitd
  } catch (std::string &e) {
    std::cout << "H5 to estimate sys. error not set ... skipping..." << std::endl;
  }
#endif
#endif
#ifdef CONVOLK
  H5Read(argv[8],&distribution,gauge_configs,zmin,zmax,pmin,pmax,"pitd",dirac);
#ifdef MATELEMSYSTEMATIC
  try {
    H5Read(argv[18],&distributionSysErr,gauge_configs,zmin,zmax,pmin,pmax,"pitd",dirac);
  } catch (std::string &e) {
    std::cout << "H5 to estimate sys. error not set ... skipping..." << std::endl;
  }
#endif
#endif
  

  /*
    Determine full data covariance
  */
  distribution.calcCov();
  std::cout << "Computed the full data covariance" << std::endl;
#ifdef MATELEMSYSTEMATIC
  distribution.addSystematicCov(&distributionSysErr);
  std::cout << "Added sys error of matelem to diagonal of data covariance" << std::endl;
#endif

  // distribution.cutOnPZ(zmin,zmax,pmin,pmax);
  distribution.calcInvCov(); std::cout << "Computed the inverse of full data covariance" << std::endl;
  std::cout << "Cut on {zmin, zmax} = { " << zmin << " , " << zmax << " }  &  {pmin, pmax} = { "
  	    << pmin << " , " << pmax << " }" << std::endl;


  printMat(distribution.data.covR);

#if 0
  std::cout << "Checking suitable inverse was found" << std::endl;
  gsl_matrix * id = gsl_matrix_alloc(distribution.data.covR->size1,distribution.data.covR->size1);
  gsl_matrix_set_zero(id);
  gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,distribution.data.covR,distribution.data.invCovR,0.0,id);
  printMat(id);
  exit(8);
#endif

  /*
    INITIALIZE THE LOG NORMAL PRIORS
  */
  setLogPriors();




  /*
    OPTIONALLY SCAN THE COST FUNCTION IN {\alpha,\beta} SPACE
  */
#if ALPHABETASCAN
  std::string scanName = redstarCurr+"."+matelemType+
    "."+std::to_string(nParams)+"-parameter_"+
    std::to_string(nParamsLT)+"lt_"+std::to_string(nParamsAZ)+"az_"+
    std::to_string(nParamsT4)+"t4_"+std::to_string(nParamsT6)+"t6_"+
    std::to_string(nParamsT8)+"t8_"+std::to_string(nParamsT10)+"t10_"+
    ".convolJ.correlated"+
    ".pmin"+std::to_string(pmin)+"_pmax"+std::to_string(pmax)+
    "_zmin"+std::to_string(zmin)+"_zmax"+std::to_string(zmax)+".alphas_"+std::to_string(alphaS)+".scale_"+std::to_string(scale)+"_priors"+".SCAN.txt";
  std::ofstream scanOUT(scanName.c_str(), std::ofstream::in | std::ofstream::app );

  // for ( int ascan = 1; ascan <= 500; ascan++ )
  //   {
  //     for ( int bscan = 1; bscan <= 500; bscan++ )
  // 	{
  // 	  double alphaScan = -0.95 + ascan*(2.0/500);
  // 	  double betaScan  = 0.05 + bscan*(3.5/500);
  for ( int ascan = 1; ascan < 2; ascan++ )
    {
      for ( int bscan = 1; bscan < 2; bscan++ )
	{
	  double alphaScan = 0.0;
	  double betaScan = 3.0;
	  
	  gsl_vector * scan = gsl_vector_alloc(2);
	  gsl_vector_set(scan,0,alphaScan);
	  gsl_vector_set(scan,1,betaScan);


	  int dof;
	  if ( pdfType == 0 )
	    dof = distribution.data.covR->size1 - nParams - distribution.data.svsFullR;
	  if ( pdfType == 1 )
	    dof = distribution.data.covI->size1 - nParams - distribution.data.svsFullI;
	  
	  
	  double cost = costFunc(scan,&distribution);


	  pdfFitParams_t *scanPfp = new pdfFitParams_t(pdfType, alphaScan, betaScan,
						       scale, dumLTIni,dumAZIni,dumT4Ini,
						       dumT6Ini,dumT8Ini,dumT10Ini);

	  std::vector<std::pair<int, double> > nuzScan(distribution.data.invCovR->size1);
	  gsl_vector *scanDataVec = gsl_vector_alloc(distribution.data.invCovR->size1);
	  
	  for ( auto z = distribution.data.disps.begin(); z != distribution.data.disps.end(); ++z )
	    {
	      zvals scanMoms;
	      for ( auto m = z->second.moms.begin(); m != z->second.moms.end(); ++m )
		{
		  int idx = std::distance(z->second.moms.begin(),m) +
		    std::distance(distribution.data.disps.begin(),z)*z->second.moms.size();

		  momVals scanVals; scanVals.mat.resize(1);
		  scanVals.IT     = m->second.IT;
		  scanVals.mat[0] = m->second.matAvg;

		  nuzScan[idx] = std::make_pair(z->first,scanVals.IT);
		  if ( pdfType == 0 )
		    gsl_vector_set(scanDataVec, idx, scanVals.mat[0].real());
		  if ( pdfType == 1 )
		    gsl_vector_set(scanDataVec, idx, scanVals.mat[0].imag());
		} // auto m
	    } // auto z
	  

	  varPro scanSoln(nParamsLT, nParamsAZ, nParamsT4, nParamsT6, nParamsT8, nParamsT10,
			  nuzScan.size(), pdfType);
	  scanSoln.makeBasis(dirac, gsl_vector_get(scan,0),
			     gsl_vector_get(scan,1),nuzScan);
	  if ( pdfType == 0 )
	    {
	      scanSoln.makeY(scanDataVec, distribution.data.invCovR, *scanPfp);
	      scanSoln.makePhi(distribution.data.invCovR, *scanPfp);
	    }
	  if ( pdfType == 1 )
	    {
	      scanSoln.makeY(scanDataVec, distribution.data.invCovI, *scanPfp);
	      scanSoln.makePhi(distribution.data.invCovI, *scanPfp);
	    }
	  scanSoln.getInvPhi();
	  gsl_blas_dgemv(CblasNoTrans,1.0,scanSoln.invPhi,scanSoln.Y,0.0,scanSoln.soln);
	  
	  gsl_vector *scanFitParams = gsl_vector_alloc(nParams);
	  gsl_vector_set(scanFitParams,0,gsl_vector_get(scan,0));
	  gsl_vector_set(scanFitParams,1,gsl_vector_get(scan,1));
	  for ( int b = 2; b < scanFitParams->size; b++ )
	    gsl_vector_set(scanFitParams,b,gsl_vector_get(scanSoln.soln,b-2));




	  std::vector<double> scanL2Chi2 = ell2Chi2(cost,scanPfp,scanFitParams,1.0,dataDim);


	  scanOUT << "SCAN:: "; // << scanL2Chi2[0]/dof << " " << scanL2Chi2[1]/dof
		  // << " " << alphaScan << " " << betaScan << "\n";
	  scanPfp->printBest(scanFitParams);

	  scanPfp->write(scanOUT, scanL2Chi2[0], scanL2Chi2[1], scanL2Chi2[0]/dof,
			 scanL2Chi2[1]/dof, scanFitParams);
	  scanOUT.flush();
	  
	  gsl_vector_free(scan);
	}
    }
  scanOUT.close();

  if ( pdfType == 0 )
    {
      std::cout << "Printing real covariance" << std::endl;
      printMat(distribution.data.covR);
    }
  if ( pdfType == 1 )
    {
      std::cout << "Printing imag covariance" << std::endl;
      printMat(distribution.data.covI);
    }


  distribution.ensemPrintZ(1,0);
  distribution.ensemPrintZ(2,0);
  

  exit(30);

#endif





  /*
    SET THE STARTING PARAMETER VALUES AND INITIAL STEP SIZES ONCE
  */
  gsl_vector *pdfp_ini, *pdfpSteps;
  // pdfFitParams_t *dumPfp;

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


#if 0
#warning "Debug crap"
  for ( int zcheck = 0; zcheck < 9; ++zcheck )
    {
      std::cout << "For z = " << zcheck << " here is the ensemble average data with same z:" << std::endl;
      distribution.ensemPrintZ(zcheck,pdfType);
    }
  exit(90);
#endif
  
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
	pass dummy LT, AZ, T4, T6, T8, T10 vectors for pmap member initialization
      */
      pdfFitParams_t *dumPfp = new pdfFitParams_t(pdfType, -0.3, 0.2, scale, dumLTIni,dumAZIni,dumT4Ini,dumT6Ini,dumT8Ini,dumT10Ini); // 0.05 2.0
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
      gsl_matrix_memcpy(thisJack.data.invCovR,distribution.data.invCovR);
      gsl_matrix_memcpy(thisJack.data.invCovI,distribution.data.invCovI);
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
      gsl_multimin_function Cost;
      // Dimension of the system
      Cost.n = pdfp_ini->size;   // (dim = 2, as only alpha/beta are treated w/ nelder-mead)
      // Function to minimize
      Cost.f = &costFunc;
      Cost.params = &thisJack;
  
  
      std::cout << "Establishing initial state for minimizer..." << std::endl;
      // Set the state for the minimizer
      // Repeated call the set function to ensure nelder-mead random minimizer
      // starts w/ a different random simplex for each jackknife sample
      int status = gsl_multimin_fminimizer_set(fmin,&Cost,pdfp_ini,pdfpSteps);
  
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

      // Return the best correlated Cost
      double cost = gsl_multimin_fminimizer_minimum(fmin);
      std::cout << "---> THIS Cost = " << cost << std::endl;
      // Determine the reduced chi2
      double reducedChiSq, detCov;
      int dof;

      // [02/16/2021] Replace substraction of singular values, with datapts cut from fit
      // [06/23/2022] Compute determinant of data covariance
      if ( pdfType == 0 )
	{
	  dof = distribution.data.covR->size1 - nParams - distribution.data.svsFullR;
	  detCov = computeDet(distribution.data.covR);
	}
      if ( pdfType == 1 )
	{
	  dof = distribution.data.covI->size1 - nParams - distribution.data.svsFullI;
	  detCov = computeDet(distribution.data.covI);
	}
      // detCov = 1.0;
      std::cout << "DET(COV) = " << detCov << std::endl;

      // std::cout << " ---> SVS (R,I) = (" << distribution.data.svsFullR << ", " << distribution.data.svsFullI << ")" << std::endl;

      reducedChiSq = cost / dof; // This was formerlly how I computed a figure of merit!


      /*
	With optimal {alpha,beta}, make one last varPro instance and determine solution vector of constants
      */
      varPro solution(nParamsLT, nParamsAZ, nParamsT4, nParamsT6, nParamsT8, nParamsT10, nuzJK.size(), pdfType);
      solution.makeBasis(dirac, gsl_vector_get(fminBest,0),
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



      // Now use dumPfp to determine the true L^2 and unconstrained chi2
      std::vector<double> L2Chi2 = ell2Chi2(cost,dumPfp,bestFitParams,detCov,dataDim);

      std::cout << " For jackknife sample J = " << itJ << ", Converged after " << k
		<<" iterations, Optimal [L2, Chi2, L2/dof, Chi2/dof] = "
		<< L2Chi2[0] << " " << L2Chi2[1] << " " << L2Chi2[0]/dof << " " << L2Chi2[1]/dof
		<< " [formerly = " << reducedChiSq << " ] " << std::endl;
      dumPfp->printBest(bestFitParams);
      


      
      // Write the fit results to a file
      dumPfp->write(OUT, L2Chi2[0], L2Chi2[1], L2Chi2[0]/dof, L2Chi2[1]/dof, bestFitParams);
      OUT.flush();

      // delete best;
      
      // Determine/print the total time for this fit
      auto jackTimeEnd = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = jackTimeEnd-jackTimeStart;
      std::cout << "           --- Time to complete jackknife fit "
		<< itJ << "  =  " << elapsed_seconds.count() << "s\n";
      
    } // End loop over jackknife samples


  gsl_vector_free(dumLTIni);
  gsl_vector_free(dumAZIni);
  gsl_vector_free(dumT4Ini);
  gsl_vector_free(dumT6Ini);
  gsl_vector_free(dumT8Ini);
  gsl_vector_free(dumT10Ini);
  gsl_vector_free(pdfp_ini);
  gsl_vector_free(pdfpSteps);

  // Close the output file containing jack fit results
  OUT.close();

  return 0;
}
