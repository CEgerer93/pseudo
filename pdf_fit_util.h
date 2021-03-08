/*
  --NUMERICALLY INTEGRATE CONVOLUTION OF PERTURBATIVE KERNELS AND PHENO PDFS TO GENERATE
  A IOFFE-TIME PSEUDO-STRUCTURE FUNCTION
  --MINIMIZE DIFFERENCE BETWEEN FIT CURVE AND BOOTSTRAP(OR JACKKNIFE) SAMPLES
*/
#ifndef _pdf_fit_util_h_
#define _pdf_fit_util_h_

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

using namespace PITD;

/*
  Parameters describing pheno PDF fit
*/
struct pdfFitParams_t
{
  double alpha, beta;           // the leading parameters i.e. x^alpha*(1-x)^beta
  int pdfType;

  gsl_vector *lt_sigmaN, *lt_fitParams;
  gsl_vector *az_sigmaN, *az_fitParams;
  int nParams;


  // Let's try some Bayesian prior stuff
  std::vector<double> prior {0.0, 0.0, 0.0};
  std::vector<double> width {1.1, 0.5, 0.25};

  std::vector<double> az_prior {0.0, 0.0, 0.0};
  std::vector<double> az_width {0.25, 0.25, 0.25};
  

  // std::map<double, double> priors;
  // prior


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
pdfFitParams_t() : pdfType(-1), alpha(0.0), beta(0.0) {}
pdfFitParams_t(int _pt, double _a, double _b, gsl_vector *lt, gsl_vector *az)
: pdfType(_pt), alpha(_a), beta(_b)// , lt_fitParams{lt}, az_fitParams{az}
  {
    // Set the param/jacobi poly vectors
    lt_fitParams = lt; az_fitParams = az;
    lt_sigmaN = gsl_vector_alloc(lt->size);
    az_sigmaN = gsl_vector_alloc(az->size);

    nParams = lt->size + az->size;
    // Now set the parameter map for easy printing
    std::string qtype;
    if ( pdfType == 0 )
      qtype = "qv";
    if ( pdfType == 1 )
      qtype = "q+";

    pmap[0] = "alpha (" + qtype + ")"; pmap[1] = "beta (" + qtype + ")";
    /* int p; */
    /* if ( pdfType == 0 ) */
    /*   { */
    /* 	for ( p = 2; p < 2+lt->size-1; p++ ) */
    /* 	  pmap[p] = "C[" + std::to_string(p-1) + "] (" + qtype +")"; */
    /* 	for ( p = 2+lt_sigmaN->size-1; p < nParams; p++ ) */
    /* 	  pmap[p] = "C_az[" + std::to_string(p-2-lt_sigmaN->size+1) + "] (" + qtype + ")"; */
    /*   } */
    /* if ( pdfType == 1 ) */
    /*   { */
    /* 	for ( p = 2; p < 2+lt->size; p++ ) */
    /* 	  pmap[p] = "C[" + std::to_string(p-2) + "] (" + qtype +")"; */
    /* 	for ( p = 2+lt_sigmaN->size; p < nParams; p++ ) */
    /* 	  pmap[p] = "C_az[" + std::to_string(p-2-lt_sigmaN->size) + "] (" + qtype + ")"; */
    /*   } */
  }
};

#endif
