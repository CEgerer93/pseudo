/*
  Define classes/structs/methods needed for variable projection
*/
#include "varpro.h"
#include "pitd_util.h"
#include "kernel.h"
#include <gsl/gsl_blas.h>

using namespace PITD;

namespace VarPro
{
  /*
    Populate the non-linear basis of functions
  */
  void varPro::makeBasis(double a, double b, std::vector<std::pair<int, double>> &nuz)
  {
    int l;
    for ( l = 0; l < numCorrections; l++ )
      {
	// Evaluate the l^th basis function at each {nu,z} combo
	for ( auto v = nuz.begin(); v != nuz.end(); ++v )
	  {
	    if ( l < numLT ) // Leading twist
	      gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),
			     pitd_texp_sigma_n(l, 85, a, b, v->second, v->first) );
	    if ( l >= numLT ) // (a/z)^2 corrections
	      gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),
			     pow((1.0/v->first),2)*pitd_texp_sigma_n_treelevel(l, 85, a, b, v->second) );
	  } // nuz
      } // l
  }

  /*
    Populate Y Solution vector
  */
  void varPro::makeY(gsl_vector *data, gsl_matrix *invCov, pdfFitParams_t &fitParams)
  {
    for ( int l = 0; l < numCorrections; l++ )
      {
	double sum(0.0), priorsSum(0.0);

	gsl_vector *rMult = gsl_vector_alloc(invCov->size1);

	// Pull out l^th row of basis function matrix
	gsl_vector_view slice = gsl_matrix_row(basis, l);


	// Perform (covinv) x basis(l)
	gsl_blas_dgemv(CblasNoTrans,1.0,invCov,&slice.vector,0.0,rMult);
	// Perform (data)^T \cdot (rMult)
	gsl_blas_ddot(data,rMult,&sum);


	// // Perform (covinv) x (data)
	// gsl_blas_dgemv(CblasNoTrans,1.0,invCov,data,0.0,rMult);
	// // Perform (basis)^T \cdot (rMult)
	// gsl_blas_ddot(&slice.vector,rMult,&sum2);


	// Collect the contributions from any priors
	if ( l < numLT )
	  {
	    priorsSum += fitParams.prior[l]/pow(fitParams.width[l],2);
	    // std::cout << fitParams.prior[l]/pow(fitParams.width[l],2) << std::endl;
	  }
	if ( l >= numLT )
	  {
	    // std::cout << fitParams.az_prior[l-numLT]/pow(fitParams.az_width[l-numLT],2) << std::endl;
	    priorsSum += fitParams.az_prior[l-numLT]/pow(fitParams.az_width[l-numLT],2);
	  }
	

	// Now set the l^th entry of Y
	gsl_vector_set(Y, l, sum+priorsSum);
      }
  }

  /*
    Populate Phi matrix
  */
  void varPro::makePhi(gsl_matrix *invCov, pdfFitParams_t &fitParams)
  {
    for ( int k = 0; k < numCorrections; k++ )
      {
	// Pull out k^th row of basis function matrix
	gsl_vector_view k_slice = gsl_matrix_row(basis, k);

	for ( int l = 0; l < numCorrections; l++ )
	  {
	    double sum(0.0);
	    // Pull out l^th row of basis function matrix
	    gsl_vector_view l_slice = gsl_matrix_row(basis, l);

	    gsl_vector * rMult = gsl_vector_alloc(invCov->size1);
	    // Perform (covinv) x basis(l)
	    gsl_blas_dgemv(CblasNoTrans,1.0,invCov,&l_slice.vector,0.0,rMult);
	    // Perform basis(k)^T \cdot (rMult)
	    gsl_blas_ddot(&k_slice.vector,rMult,&sum);


	    // Include contributions from priors!!!!!! - only appear on diagonal of Phi matrix
	    if ( k == l )
	      {
		if ( k < numLT )
		  sum += (1.0/pow(fitParams.width[k],2));
		if ( k >= numLT )
		  sum += (1.0/pow(fitParams.az_width[k-numLT],2));
	      }

	    // Insert this value into the Phi matrix
	    gsl_matrix_set(Phi, k, l, sum);

	  } // l
      } // k
  }

  /*
    Get the inverse of Phi matrix
  */
  void varPro::getInvPhi()
  {
    // The int and map because I don't want to rewrite the inversion
    int dumZi = -1;
    std::map<int, gsl_matrix *> dumMap;
    int catchSVs = matrixInv(Phi, dumMap, dumZi);
    // if ( catchSVs != 0 )
    //   std::cout << "---> In computing Phi^-1, removed " << catchSVs << " singular values" << std::endl;
    invPhi = dumMap.begin()->second;
  }

  // /*
  //   Return the solution of varpro
  // */
  // void varPro::getSoln()
  // {
  //   gsl_blas_dgemv(CblasNoTrans,1.0,VP.invPhi,VP.Y,0.0,soln);
  // }
}
