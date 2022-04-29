/*
  Define classes/structs/methods needed for variable projection
*/
#include "varpro.h"
#include "pitd_util.h"
#include "kernel.h"
#include <gsl/gsl_blas.h>

using namespace PITD;

const double LAMBDA = 0.286 * (aLat/hbarc); // GeV * ( a [fm] / hbarc [ GeV * fm ] )

namespace VarPro
{
  /*
    Populate the non-linear basis of functions
  */
  void varPro::makeBasis(int dirac, double a, double b, std::vector<std::pair<int, double>> &nuz)
  {
    int l;
    for ( l = 0; l < numCorrections; l++ )
      {
	// Evaluate the l^th basis function at each {nu,z} combo
	for ( auto v = nuz.begin(); v != nuz.end(); ++v )
	  {
	    /*
	      Leading Twist : \sum from l = 0 of pitd_texp_<sigma,eta>_n
	    */
	    if ( l < numLT )
	      {
		if ( pdfType == 0 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),
				 pitd_texp_sigma_n(l, 75, a, b, v->second, v->first, dirac) );
		if ( pdfType == 1 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),
				 pitd_texp_eta_n(l, 75, a, b, v->second, v->first, dirac) );
	      }
	    /*
	      (a/z)^2 corrections : \sum from l = 1 of pitd_texp_sigma_n_treelevel
	                            \sum from l = 0 of pitd_texp_eta_n_treelevel
	                            So l passed to pitd_texp_sigma_n_treelevel
				    needs to be shifted to l-(numLT-1)
	    */
	    else if ( l >= numLT && l < numLT + numAZ )
	      {
		// if ( pdfType == 0 )
		//   gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow((1.0/v->first),2)*
		// 		 pitd_texp_sigma_n_treelevel(l-(numLT-1), 75, a, b, v->second) );

		// Try a/mod(z) corrections in real component
		if ( pdfType == 0 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),1.0/abs(v->first)*
				 pitd_texp_sigma_n_treelevel(l-(numLT-1), 75, a, b, v->second) );
		if ( pdfType == 1 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow((1.0/v->first),1)*
				 pitd_texp_eta_n_treelevel(l-numLT, 75, a, b, v->second) );
	      }
	    /*
	      (z*Lambda_qcd)^2 corrections : \sum from l = 1 of pitd_texp_sigma_n_treelevel
	                                     \sum from l = 0 of pitd_texp_eta_n_treelevel
	                                     So l passed to pitd_texp_sigma_n_treelevel
					     needs to be shifted to l-(numLT+numAZ-1)
	    */
	    else if ( l >= numLT + numAZ && l < numLT + numAZ + numT4 )
	      {
		if ( pdfType == 0 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,2)*
				 pitd_texp_sigma_n_treelevel(l-(numLT+numAZ-1), 75, a, b, v->second) );
		if ( pdfType == 1 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,2)*
				 pitd_texp_eta_n_treelevel(l-(numLT+numAZ), 75, a, b, v->second) );
	      }
	    /*
	      (z*Lambda_qcd)^4 corrections : \sum from l = 1 of pitd_texp_sigma_n_treelevel
	                                     \sum from l = 0 of pitd_texp_eta_n_treelevel
					     So l passed pitd_texp_sigma_n_treelevel
					     needs to be shifted to l-(numLT+numAZ+numT4-1)
	    */
	    else if ( l >= numLT + numAZ + numT4 && l < numLT + numAZ + numT4 + numT6 )
	      {
		if ( pdfType == 0 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,4)*
				 pitd_texp_sigma_n_treelevel(l-(numLT+numAZ+numT4-1), 75, a, b, v->second) );
		if ( pdfType == 1 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,4)*
				 pitd_texp_eta_n_treelevel(l-(numLT+numAZ+numT4), 75, a, b, v->second) );
	      }
	    /*
	      (z*Lambda_qcd)^6 corrections : \sum from l = 1 of pitd_texp_sigma_n_treelevel
	                                     \sum from l = 0 of pitd_texp_eta_n_treelevel
					     So l passed pitd_texp_sigma_n_treelevel
					     needs to be shifted to l-(numLT+numAZ+numT4+numT6-1)
	    */
	    else if ( l >= numLT + numAZ + numT4 + numT6 && l < numLT + numAZ + numT4 + numT6 + numT8 )
	      {
		if ( pdfType == 0 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,6)*
				 pitd_texp_sigma_n_treelevel(l-(numLT+numAZ+numT4+numT6-1), 75, a, b, v->second) );
		if ( pdfType == 1 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,6)*
				 pitd_texp_eta_n_treelevel(l-(numLT+numAZ+numT4+numT6), 75, a, b, v->second) );
	      }
	    /*
	      (z*Lambda_qcd)^8 corrections : \sum from l = 1 of pitd_texp_sigma_n_treelevel
	                                     \sum from l = 0 of pitd_texp_eta_n_treelevel
					     So l passed pitd_texp_sigma_n_treelevel
					     needs to be shifted to l-(numLT+numAZ+numT4+numT6+numT8-1)
	    */
	    else if ( l >= numLT + numAZ + numT4 + numT6 + numT8 )
	      {
		if ( pdfType == 0 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,8)*
				 pitd_texp_sigma_n_treelevel(l-(numLT+numAZ+numT4+numT6+numT8-1), 75, a, b, v->second) );
		if ( pdfType == 1 )
		  gsl_matrix_set(basis, l, std::distance(nuz.begin(), v),pow(v->first*LAMBDA,8)*
				 pitd_texp_eta_n_treelevel(l-(numLT+numAZ+numT4+numT6+numT8), 75, a, b, v->second) );
	      }
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


	// Collect the contributions from any priors
	if ( l < numLT )
	  priorsSum += fitParams.prior[l]/pow(fitParams.width[l],2);
	else if ( l >= numLT && l < numLT + numAZ )
	  priorsSum += fitParams.az_prior[l-numLT]/pow(fitParams.az_width[l-numLT],2);
	else if ( l >= numLT + numAZ && l < numLT + numAZ + numT4 )
	  priorsSum += fitParams.t4_prior[l-numLT-numAZ]/pow(fitParams.t4_width[l-numLT-numAZ],2);
	else if ( l >= numLT + numAZ + numT4 && l < numLT + numAZ + numT4 + numT6 )
	  priorsSum += fitParams.t6_prior[l-numLT-numAZ-numT4]/pow(fitParams.t6_width[l-numLT-numAZ-numT4],2);
	else if ( l >= numLT + numAZ + numT4 + numT6 && l < numLT + numAZ + numT4 + numT6 + numT8 )
	  priorsSum += fitParams.t8_prior[l-numLT-numAZ-numT4-numT6]/pow(fitParams.t8_width[l-numLT-numAZ-numT4-numT6],2);
	else if ( l >= numLT + numAZ + numT4 + numT6 + numT8 )
	  priorsSum += fitParams.t10_prior[l-numLT-numAZ-numT4-numT6-numT8]/pow(fitParams.t10_width[l-numLT-numAZ-numT4-numT6-numT8],2);

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
		if ( k >= numLT && l < numLT + numAZ )
		  sum += (1.0/pow(fitParams.az_width[k-numLT],2));
		if ( k >= numLT + numAZ && l < numLT + numAZ + numT4 )
		  sum += (1.0/pow(fitParams.t4_width[k-numLT-numAZ],2));
		if ( k >= numLT + numAZ + numT4 && l < numLT + numAZ + numT4 + numT6 )
		  sum += (1.0/pow(fitParams.t6_width[k-numLT-numAZ-numT4],2));
		if ( k >= numLT + numAZ + numT4 + numT6 && l < numLT + numAZ + numT4 + numT6 + numT8 )
		  sum += (1.0/pow(fitParams.t8_width[k-numLT-numAZ-numT4-numT6],2));
		if ( k >= numLT + numAZ + numT4 + numT6 + numT8 )
		  sum += (1.0/pow(fitParams.t10_width[k-numLT-numAZ-numT4-numT6-numT8],2));
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
}
