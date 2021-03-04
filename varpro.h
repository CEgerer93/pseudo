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

    // Ints to track numbers of diff types of non-linear functions
    int numLT, numAZ;
    int numFunc;

    // Default
    varPro() {}
    // Parametrized
    varPro(int _numLT, int _numAZ)
      {
	numLT(_numLT); numAZ(_numAZ);
	numFunc = _numLT + _numAZ;
	basis = gsl_vector_alloc(numFunc);
	Y     = gsl_vector_alloc(numFunc);
	Phi   = gsl_matrix_alloc(numFunc,numFunc);
      }
 
    // Destructor
    virtual ~varPro() {};

    // Populate the non-linear basis of functions
    void makeBasis(gsl_vector *d);


    void makeY(gsl_vector *data, gsl_matrix *invCov, double a, double b, std::vector<std::pair<int, double>> &nuz);
    void makePhi();
    
  private:
  };



}
#endif
