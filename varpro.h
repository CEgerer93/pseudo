/*
  Define classes/structs/methods needed for variable projection
*/

#ifndef __varpro_h__
#define __varpro_h__

#include<vector>
#include<map>
#include<iostream>
#include<iomanip>
#include<complex>

#include<gsl/gsl_matrix.h>
#include<gsl/gsl_vector.h>

#include "hdf5.h"

#include "pdf_fit_util.h"

namespace VarPro
{
  /*
    Construct a model function from vector of coefficients and vector of basis functions

                     f(x_k)=\sum_j c_j \Phi_j(non-linear params, x_k)
  */
  class varPro
  {
  public:
    gsl_matrix * basis; // Non-linear Basis functions
    gsl_vector * Y;     // Y_i = \sum_k pITD(k)*Phi_i(non-linear params; nu, z) ; where k is a tuple of (nu,z)
    gsl_matrix * Phi;   // Outer product of basis functions;
    // Phi_{ij} = \sum_k Phi_i(non-linear params; nu, z)*Phi_j(non-linear params; nu, z) w/ k a tuple of (nu, z)
    gsl_matrix * invPhi;

    gsl_vector * soln; // The variable projection solution for linear constants

    // Ints to track numbers of diff types of non-linear functions
    int numLT, numAZ, numT4, numT6;
    int numCorrections;

    int pdfType;

    // Default
    varPro() {}
    // Parametrized
    varPro(int _numLT, int _numAZ, int _numT4, int _numT6, size_t numData, int _pdfType)
      {
	pdfType = _pdfType;
	numLT   = _numLT; numAZ = _numAZ; numT4 = _numT4; numT6 = _numT6;
	numCorrections = _numLT + _numAZ + _numT4 + _numT6;
	basis   = gsl_matrix_calloc(numCorrections,numData);
	Y       = gsl_vector_alloc(numCorrections);
	soln    = gsl_vector_alloc(numCorrections);
	Phi     = gsl_matrix_calloc(numCorrections,numCorrections);
	invPhi  = gsl_matrix_calloc(numCorrections,numCorrections);
      }
 
    // Destructor
    virtual ~varPro() {};

    // Populate the non-linear basis of functions
    void makeBasis(double a, double b, std::vector<std::pair<int, double> > &nuz);
    // Populate Y Solution vector
    void makeY(gsl_vector *data, gsl_matrix *invCov, pdfFitParams_t &fitParams);
    // Populate Phi matrix
    void makePhi(gsl_matrix *invCov, pdfFitParams_t &fitParams);
    // Get the inverse of Phi matrix
    void getInvPhi();
    
  };



}
#endif
