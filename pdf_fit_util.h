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
#include<vector>
#include<iomanip>
#include<map> // use maps to grab kenels for current pairs
#include<chrono>
/*
  Grab headers for gsl function calls
*/
#include <gsl/gsl_vector.h> // allocating/accessing gsl_vectors for passing params to minimizer
#include <gsl/gsl_matrix.h> // matrix routines for inversion of data covariance
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

  gsl_vector *lt_fitParams, *az_fitParams, *t4_fitParams, *t6_fitParams, *t8_fitParams, *t10_fitParams;
  int nParams;


  // Let's try some Bayesian prior stuff
  std::vector<double> prior {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> width {1.1, 0.75, 0.5, 0.25, 0.125, 0.1, 0.05, 0.025};

  std::vector<double> az_prior {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> az_width {0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.1, 0.1};
  std::vector<double> t4_prior {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> t4_width {0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.1, 0.1};
  std::vector<double> t6_prior {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> t6_width {0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.1, 0.1};
  std::vector<double> t8_prior {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> t8_width {0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.1, 0.1};
  std::vector<double> t10_prior {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> t10_width {0.25, 0.25, 0.25, 0.125, 0.125, 0.125, 0.1, 0.1};
  

  std::map<int, std::string> pmap; // map to print fit parameter string and values during fit

  // Print the current fit values
  void printFit(gsl_vector *v)
  {
    for ( auto p = pmap.begin(); p != pmap.end(); ++p )
      std::cout << std::setprecision(10) << "  " << p->second << " =  " << gsl_vector_get(v,p->first);
    std::cout << "\n";
  }

  // Print the best fit values
  void printBest(gsl_vector *v)
  {
    // Add best fit constants from VarPro before printing fit
    int b;
    for ( b = 2; b < lt_fitParams->size+2; b++ )
      pmap[b] = "C_lt" + std::to_string(b-2);
    for ( b = lt_fitParams->size + 2; b < lt_fitParams->size + az_fitParams->size + 2; b++ )
      pmap[b] = "C_az" + std::to_string(b-lt_fitParams->size-2);
    for ( b = lt_fitParams->size + az_fitParams->size + 2; b < lt_fitParams->size + az_fitParams->size + t4_fitParams->size + 2; b++ )
      pmap[b] = "C_t4" + std::to_string(b-lt_fitParams->size-az_fitParams->size-2);
    for ( b = lt_fitParams->size + az_fitParams->size + t4_fitParams->size + 2; b < lt_fitParams->size + az_fitParams->size + t4_fitParams->size + t6_fitParams->size + 2; b++ )
      pmap[b] = "C_t6" + std::to_string(b-lt_fitParams->size-az_fitParams->size-t4_fitParams->size-2);
    for ( b = lt_fitParams->size + az_fitParams->size + t4_fitParams->size + t6_fitParams->size + 2; b < lt_fitParams->size + az_fitParams->size + t4_fitParams->size + t6_fitParams->size + t8_fitParams->size + 2; b++ )
      pmap[b] = "C_t8" + std::to_string(b-lt_fitParams->size-az_fitParams->size-t4_fitParams->size-t6_fitParams->size-2);
    for ( b = lt_fitParams->size + az_fitParams->size + t4_fitParams->size + t6_fitParams->size + t8_fitParams->size + 2; b < 2 + nParams; b++ )
      pmap[b] = "C_t10" + std::to_string(b-lt_fitParams->size-az_fitParams->size-t4_fitParams->size-t6_fitParams->size-t8_fitParams->size-2);

    for ( auto p = pmap.begin(); p != pmap.end(); ++p )
      std::cout << std::setprecision(10) << "  " << p->second << " =  " << gsl_vector_get(v,p->first);
    std::cout << "\n";
  }

  // Write best fit values to file
  void write(std::ofstream &os, double L2, double Chi2, double L2dof, double Chi2dof, gsl_vector *v)
  {
    os << std::setprecision(10) << L2 << " " << Chi2 << " " << L2dof << " " << Chi2dof << " ";
    for ( auto p = pmap.begin(); p != pmap.end(); ++p )
      os << gsl_vector_get(v,p->first) << " ";
    os << "\n";
  }
  
  // Default/Parametrized constructor w/ initializer lists
pdfFitParams_t() : pdfType(-1), alpha(0.0), beta(0.0) {}
pdfFitParams_t(int _pt, double _a, double _b, gsl_vector *lt, gsl_vector *az, gsl_vector *t4, gsl_vector *t6, gsl_vector *t8, gsl_vector *t10)
: pdfType(_pt), alpha(_a), beta(_b)
  {
    // Set the param/jacobi poly vectors
    // passed lt, az, t4, t6, t8, t10 are allocated, but empty
    lt_fitParams = lt; az_fitParams = az; t4_fitParams = t4; t6_fitParams = t6; t8_fitParams = t8; t10_fitParams = t10;

    nParams = lt->size + az->size + t4->size + t6->size + t8->size + t10->size;
    // Now set the parameter map for easy printing
    std::string qtype;
    if ( pdfType == 0 )
      qtype = "qv";
    if ( pdfType == 1 )
      qtype = "q+";

    pmap[0] = "alpha (" + qtype + ")"; pmap[1] = "beta (" + qtype + ")";
  }
};

#endif
