/*
  Define classes/structs/methods needed for variable projection
*/
#include "varpro.h"

namespace VarPro
{
  void varPro::makeBasis(double a, double b, std::vector<std::pair<int, double>> &nuz)
  {
    int l;
    for ( l = 0; l < numLT; l++ )
      {
	// Evaluate the l^th basis function at each {nu,z} combo
	for ( auto v = nuz.begin(); v != nuz.end(); ++v )
	  {
	    if ( l < numLT ) // Leading twist
	      gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),
			     pitd_texp_sigma_n(l, 85, a, b, v->second, v->first) );
	    if ( l >= numLT ) // (a/z)^2 corrections
	      gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),
			     pow((1.0/z),2)*pitd_texp_sigma_n_treelevel(l, 85, a, b, v->second) );
	  } // nuz
      } // l
  }

  void varPro::makeY(gsl_vector *data, gsl_matrix *invCov)
  {
    for ( int l = 0; l < numLT; l++ )
      {
	double sum1(0.0), sum2(0.0);

	gsl_vector *rMult = gsl_vector_alloc(invCov->size1);

	// Pull out l^th row of basis function matrix
	gsl_vector_view slice = gsl_matrix_row(basis, l);
	

	// Perform (covinv) x basis(l)
	gsl_blas_dgemv(CblasNoTrans,1.0,invCov,&slice.vector,0.0,rMult);
	// Perform (data)^T \cdot (rMult)
	gsl_blas_ddot(data,rMult,&sum1);


	// Perform (covinv) x (data)
	gsl_blas_dgemv(CblasNoTrans,1.0,invCov,data,0.0,rMult);
	// Perform (basis)^T \cdot (rMult)
	gsl_blas_ddot(&slice.vector,rMult,&sum2);

	// Now set the l^th entry of Y
	gsl_vector_set(Y, l, sum1+sum2);
      }
  }


  void makePhi(gsl_vector *basis, gsl_vector *invCov)
  {

  }

}
