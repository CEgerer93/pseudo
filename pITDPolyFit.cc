/*
  PERFORM CONSTANT Z^2 POLYNOMIAL FIT TO PSEUDO-ITD DATA FOR EACH JACKKNIFE SAMPLE
  FIT_real = 1 + \NU^2 + \NU^4 + \NU^6
 */
#include<iostream>
#include<fstream>
#include<string>
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
#include <gsl/gsl_integration.h> // Numerical integration
#include <gsl/gsl_vector.h> // allocating/accessing gsl_vectors for passing params to minimizer
#include <gsl/gsl_matrix.h> // matrix routines for inversion of data covariance
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h> // multidimensional minimization
#include <gsl/gsl_multifit_nlinear.h>
/*
 Headers for threading
*/
#include<omp.h>

#include "pitd_util.h"

using namespace PITD;


// A simple struct to hold parameters that are fit
struct fitParams
{
  double a, b, c;
  fitParams() : a(0.0), b(0.0), c(0.0) {} // constructor
  
  double func(bool reality, double ioffe) // return (Re/Im) polynomial evaluated at ioffe
  {
    if ( reality )
      return 1.0 + a*pow(ioffe, 2) + b*pow(ioffe, 4) + c*pow(ioffe,6);
    if ( !reality )
      return a*pow(ioffe, 1) + b*pow(ioffe, 3) + c*pow(ioffe, 5);
  }
};

struct fitStruc
{
  bool reality;
  std::vector<double> IT;
  std::vector<std::complex<double> > M;
  gsl_matrix * covinv;
  
  // Parametrized constructor w/ initializer list
  fitStruc(bool comp, gsl_matrix *cov_inv) : reality(comp), covinv(cov_inv) {}
};


// Compute the correlated (or uncorrelated) chi2 locally
double chi2Func(const gsl_vector * x, void *data)
{

  // Get a pointer to the void reducedPITD class
  fitStruc * jfitCpy = (fitStruc *)data;


  // The current fit parameters
  fitParams poly;
  poly.a = gsl_vector_get(x,0);
  poly.b = gsl_vector_get(x,1);
  poly.c = gsl_vector_get(x,2);

  // Evaluate the polynomial for these fit params at each Ioffe-time
  std::vector<double> polyRes;
  for ( auto nu = jfitCpy->IT.begin(); nu != jfitCpy->IT.end(); ++nu )
    {
      polyRes.push_back( poly.func(jfitCpy->reality, *nu) );
    }
  

  /*
    Begin chi2 computation
  */
  double chi2(0.0);
  gsl_vector *iDiffVec = gsl_vector_alloc(jfitCpy->IT.size());
  gsl_vector *jDiffVec = gsl_vector_alloc(jfitCpy->IT.size());

  for ( auto chiL = jfitCpy->M.begin(); chiL != jfitCpy->M.end(); ++chiL )
    {
      if ( jfitCpy->reality )
	gsl_vector_set(iDiffVec,std::distance(jfitCpy->M.begin(), chiL),
		       polyRes[std::distance(jfitCpy->M.begin(), chiL)] - chiL->real());
      if ( !jfitCpy->reality )
	gsl_vector_set(iDiffVec,std::distance(jfitCpy->M.begin(), chiL),
		       polyRes[std::distance(jfitCpy->M.begin(), chiL)] - chiL->imag());
    }
  // The difference vector need only be computed once, so make a second copy to form correlated chi2
  gsl_vector_memcpy(jDiffVec,iDiffVec);

  // Initialize cov^-1 right multiplying jDiffVec
  gsl_vector *invCovRightMult = gsl_vector_alloc(jfitCpy->IT.size());
  gsl_blas_dgemv(CblasNoTrans,1.0,jfitCpy->covinv,jDiffVec,0.0,invCovRightMult);

  // Form the scalar dot product of iDiffVec & result of invCov x jDiffVec
  gsl_blas_ddot(iDiffVec,invCovRightMult,&chi2);

  // Free some memory
  gsl_vector_free(iDiffVec);
  gsl_vector_free(jDiffVec);
  gsl_vector_free(invCovRightMult);


// #ifdef CONSTRAINED
//   /*
//     CHECK FOR VALUES OF {ALPHA,BETA} OUTSIDE ACCEPTABLE RANGE AND INFLATE CHI2
//   */
//   if ( pdfp.alpha < pdfp.alphaRestrict.first || pdfp.alpha > pdfp.alphaRestrict.second )
//     {
//       chi2+=1000000;
//     }
//   if ( pdfp.beta < pdfp.betaRestrict.first || pdfp.beta > pdfp.betaRestrict.second )
//     {
//       chi2+=1000000;
//     }
// #endif


  return chi2;
}




int main( int argc, char *argv[] )
{

  if ( argc != 8 )
    {
      std::cout << "Usage: $0 <h5 file> <configs> <matelemType> <zmin> <zmax> <pmin> <pmax>" << std::endl;
      exit(1);
    }

  int gauge_configs, zmin, zmax, pmin, pmax;
  std::string matelemType;

  std::stringstream ss;
  ss.clear(); ss.str(std::string());
  ss << argv[2]; ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[3]; ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[4]; ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[5]; ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[6]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[7]; ss >> pmax;          ss.clear(); ss.str(std::string());



  // reducedPITD *ensem = new reducedPITD(gauge_configs,0,8,6); // zmin = 0, zmax = 8, Nmom = 6
  // // The new call somehow packs 2049 = 349 x 6 entries into each IT, avgM, errM
  // // So passing std::vecs jackReal/Imag by reference to H5Read
  // std::vector<reducedPITD > jack(gauge_configs,reducedPITD(gauge_configs-1,9,6));//, new reducedPITD());


  reducedPITD rawPseudo(gauge_configs);


  H5Read(argv[1], &rawPseudo, gauge_configs, zmin, zmax, pmin, pmax);

  // for ( int J = 0 ; J < gauge_configs; J++ )
  //   {
  //     std::cout << rawPseudo.data.disps[1].moms["pz1"].mat[J] << std::endl;
  //   }


  /*
    Make data covariance within each zsep channel
  */
  rawPseudo.calcCovPerZ();    std::cout << "Got the covariances per z" << std::endl;
  // Inverses stored in place in covsR, covsI;
  rawPseudo.calcInvCovPerZ(); std::cout << "Got the inverses of covarianes per z" << std::endl;

  rawPseudo.viewZCovMat(1);
  rawPseudo.viewZCovInvMat(1);
  exit(9);


#if 0
  std::cout << "Checking suitable inverses were found" << std::endl;
  for ( auto ptr = rawPseudo.data.covsR.begin(); ptr != rawPseudo.data.covsR.end(); ++ptr )
    {
      gsl_matrix * id = gsl_matrix_alloc(ptr->second->size1,ptr->second->size1); gsl_matrix_set_zero(id);
      gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,ptr->second,rawPseudo.data.covsRInv[ptr->first],0.0,id);

      std::cout << "Compute ID" << std::endl;
      printMat(id);
      std::cout <<"$$$$$" << std::endl;
    }
#endif
  


  /*
    TO PERFORM EVOLUTION OF REDUCED PITD DATA FOR A CONSTANT Z^2,
    A 6TH DEGREE POLYNOMIAL IN NU IS FIT TO THE DATA, AND SUBSEQUENTLY
    CONVOLVED WITH THE DGLAP KERNEL

    THIS IS PERFORMED PER JACKKNIFE SAMPLE
  */



  // Fix some parameters constant throughout
  std::map<std::string, momVals>::const_iterator momsItr = rawPseudo.data.disps[1].moms.begin();
  size_t datumPerZ = momsItr->second.mat.size();
  size_t order = 3;



  // Open an output file to hold fit parameters for each z^2
  std::ofstream OUT;
  std::string output = "b_b0xDA__J0_A1pP."+matelemType+".POLYFIT.txt";
  OUT.open(output.c_str(), std::ofstream::trunc);







  /*
    SET THE INITIAL VALUES OF FITTED PARAMETERS + STEP SIZES
  */
  gsl_vector *_ini      = gsl_vector_alloc(order);
  gsl_vector *_iniSteps = gsl_vector_alloc(order);
  for ( size_t p = 0; p < order; p++ ) { gsl_vector_set(_ini,p,0.0); gsl_vector_set(_iniSteps,p,0.05); }


  /*
    INITIALIZE THE SOLVER HERE, SO REPEATED CALLS TO SET, RETURN A NEW NMRAND2 SOLVER
  */
  /*
    Initialize a multidimensional minimzer without relying on derivatives of function
  */
  // Select minimization type
  const gsl_multimin_fminimizer_type *minimizer = gsl_multimin_fminimizer_nmsimplex2rand;
  // const gsl_multimin_fminimizer_type *minimizer = gsl_multimin_fminimizer_nmsimplex2;
  gsl_multimin_fminimizer * fmin = gsl_multimin_fminimizer_alloc(minimizer,_ini->size);



  
  // for ( std::map<std::string, momVals>::const_iterator it = rawPseudo.data.disps[0].moms.begin();
  // 	it != rawPseudo.data.disps[0].moms.end(); ++it )
  //   {
  //     std::cout << it->first << std::endl;
  //   }
  


  
  // Iterate through each z^2
  for ( std::map<int, zvals>::iterator z = rawPseudo.data.disps.begin();
  	z != rawPseudo.data.disps.end(); ++z )
    {
      std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++\n"
  		<< "+++  Performing polynomial fits for z = " << z->first << " data  +++\n"
  		<< "+++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;

      // Initial time of this collection of zsep data
      auto zTimeI = std::chrono::steady_clock::now();




      // Run over the jackknife samples (w/o an iterator for now)
      for ( int J = 0; J < gauge_configs; J++ )
  	{

	  // Make a new fitStruc to manipulate in fit
	  // ... assigning reality & invCov for this z
	  // fitStruc jfit(true, rawPseudo.data.covsRInv[z->first]);
	  fitStruc jfit(false, rawPseudo.data.covsIInv[z->first]);


	  // Run over the momentum combinations
	  // Map appears to be order st. { pz1, pz2, ..., pz6 } are in order
	  for ( std::map<std::string, momVals>::iterator mj = z->second.moms.begin();
		mj != z->second.moms.end(); ++mj )
	    {
	      jfit.IT.push_back( mj->second.IT );
	      // jfit.M.push_back( mj->second.mat[J].real() );
	      jfit.M.push_back( mj->second.mat[J].imag() );
	    }

	  
  	  // Initial time of this jackknife fit
  	  auto jTimeI = std::chrono::steady_clock::now();
  	  std::cout << "............. J = " << J << std::endl;


  	  // Define the gsl_multimin_function
  	  gsl_multimin_function Chi2;
  	  // Dimension of the system
  	  Chi2.n = _ini->size;
  	  // Function to minimize
  	  Chi2.f = &chi2Func;
  	  Chi2.params = &jfit;

  	  std::cout << "Establishing initial state for minimizer..." << std::endl;
  	  // Set the state for the minimizer
  	  // Repeated call the set function to ensure nelder-mead random minimizer
  	  // starts w/ a different random simplex for each jackknife sample
  	  int status = gsl_multimin_fminimizer_set(fmin,&Chi2,_ini,_iniSteps);
  	  std::cout << "Minimizer established..." << std::endl;


  	  // Iteration count
  	  int k = 1;
  	  double tolerance = 0.0000001; // 0.0001
  	  int maxIters = 10000;         // 1000

  	  while ( gsl_multimin_test_size( gsl_multimin_fminimizer_size(fmin), tolerance) == GSL_CONTINUE )
  	    {
  	      // End after maxIters
  	      if ( k > maxIters ) { break; }

  	      // Iterate
  	      gsl_multimin_fminimizer_iterate(fmin);

  	      std::cout << "................ Current params  (" << J << "," << k << ") ::"
  			<< std::setprecision(14)
  			<< "  a = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),0)
  			<< "  b = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),1)
  			<< "  c = " << gsl_vector_get(gsl_multimin_fminimizer_x(fmin),2)
  			<< std::endl;

  	      k++;
  	    }



  	  // Return the best fit parameters
  	  gsl_vector *bestFitParams = gsl_vector_alloc(_ini->size);
  	  bestFitParams = gsl_multimin_fminimizer_x(fmin);

  	  // Return the best correlated Chi2
  	  double chiSq = gsl_multimin_fminimizer_minimum(fmin);
  	  // Determine the reduced chi2
  	  double reducedChiSq = chiSq / (z->second.moms.size() - _ini->size - rawPseudo.data.svsR[z->first]);

	  // int dum;
	  // ss << _ini->size;
	  // ss >> dum;
	  // std::cout << "SOME INFO " << std::endl;
	  // std::cout << z->second.moms.size() << " " << dum
	  // 	    << " "  << rawPseudo.data.svsR[z->first] << std::endl;
	  // exit(7);



#define BEST(i) (gsl_vector_get(bestFitParams,(i)))


  	  // Final time of this jackknife fit
  	  auto jTimeF = std::chrono::steady_clock::now();
  	  std::chrono::duration<double> jTimeTot = jTimeF - jTimeI;
  	  std::cout << "........ Time to complete J = " << J << " fit = " << jTimeTot.count() << "s\n";
	  std::cout << "         [ BEST --> (a,b,c) = ( "
		    << BEST(0) << " , " << BEST(1) << " , " << BEST(2) << " ) <-- "
		    << " rchi2 = " << reducedChiSq << " ] " << std::endl;
	  

  	} // end jackknife fit loop

      // Final time of this collection of zsep data
      auto zTimeF = std::chrono::steady_clock::now();
      std::chrono::duration<double> zTimeTot = zTimeF - zTimeI;
      std::cout << "........ Time to complete all zsep = " << z->first
  		<< " jackknife fits = " << zTimeTot.count() << "s\n";

    } // end loop over z ( std::map<int, zvals>::const_iterator )



//       double nu;
//       double chisq;
      
//       gsl_matrix *evenNU, *oddNU, *covParams;
//       gsl_vector *pITDreal, *pITDreal_weights, *pITDimag, *pITDimag_weights, *fitParams;
      
//       // Initialize vectors to hold real/imag matelems + errors
//       pITDreal = gsl_vector_alloc(datumPerZ);
//       pITDreal_weights = gsl_vector_alloc(datumPerZ);
//       pITDimag = gsl_vector_alloc(datumPerZ);
//       pITDimag_weights = gsl_vector_alloc(datumPerZ);
      
//       evenNU = gsl_matrix_alloc(datumPerZ,order); // +1 to hold nu^0 term too
//       oddNU  = gsl_matrix_alloc(datumPerZ,order);
      
      
//       // Only instantiate one set of fit/cov params - results of real fit will be reset by imag fit
//       fitParams = gsl_vector_alloc(order);
//       covParams = gsl_matrix_alloc(order,order);
      

// #define BEST(i) (gsl_vector_get(fitParams,(i)))
// #define COV(i,j) (gsl_matrix_get(covParams,(i),(j)))
      

//       // Start by assuming data is uncorrelated
//       // Cycle through all z^2 and Real for this jackknife sample
//       auto it = j->real.disps.begin(); // it++;
//       while ( it != j->real.disps.end() )
// 	{
// 	  std::cout << "        --- Fitting polynomial to z^2 = " << pow(it->first,2) << " REAL data"
// 		    << std::endl;
// 	  // Set the data/error/fit params
// 	  for ( int i = 0; i < datumPerZ; i++ )
// 	    {
// 	      nu = it->second.ensem.IT[i];
	      
// 	      // Set the evenNU matrices
// 	      gsl_matrix_set(evenNU, i, 0, pow(nu,2));
// 	      gsl_matrix_set(evenNU, i, 1, pow(nu,4));
// 	      gsl_matrix_set(evenNU, i, 2, pow(nu,6));
// 	      // Set the data/variance vectors
// 	      // Subtract 1.0 from Real data to enforce nu=0 normalization of 1.0
// 	      gsl_vector_set(pITDreal, i, it->second.ensem.avgM[i]-1);
// 	      gsl_vector_set(pITDreal_weights, i, 1.0/pow(it->second.ensem.errM[i],2));
// 	    }
	  
// 	  gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(datumPerZ,order);
// 	  gsl_multifit_linear_svd(evenNU, work);
// 	  gsl_multifit_wlinear(evenNU, pITDreal_weights, pITDreal, fitParams, covParams, &chisq, work);
// 	  gsl_multifit_linear_free(work);
	  
// 	  printf ("# best REAL fit: Y = 1.0 + %g NU^2 + %g NU^4 + %g NU^6\n",
// 		  BEST(0), BEST(1), BEST(2));
// 	  printf ("# REAL covariance matrix:\n");
// 	  printf ("[ %+.9e, %+.9e, %+.9e \n",
// 		  COV(0,0), COV(0,1), COV(0,2));
// 	  printf ("  %+.9e, %+.9e, %+.9e \n",
// 		  COV(1,0), COV(1,1), COV(1,2));
// 	  printf ("  %+.9e, %+.9e, %+.9e ]\n",
// 		  COV(2,0), COV(2,1), COV(2,2));
	  
// 	  printf ("# chisq REAL = %g\n\n\n", chisq);
	  
	  
// 	  // Write the fit parameter central values
// 	  // Do a hard write of z=0 fit parameters to avoid a -nan in dat file
// 	  if ( it->first == 0 )
// 	    {
// 	      OUT << std::setprecision(15) << j - jack.begin() << " 0" << " " << it->first << " 0 0 0" << "\n";
// 	    }
// 	  else {
// 	    OUT << std::setprecision(15) << j - jack.begin() << " 0" << " " << it->first << " " << BEST(0)
// 		<< " " << BEST(1) << " " << BEST(2) << "\n";
// 	  }
	  
// 	  it++;
// 	} // end iterator over all real displacements

//       // Now cycle through all z^2 and Imag for this jackknife sample
//       it = j->imag.disps.begin();
//       while ( it != j->imag.disps.end() )
// 	{
// 	  std::cout << "        --- Fitting polynomial to z^2 = " << pow(it->first,2) << " IMAG data"
// 		    << std::endl;
// 	  // Set the data/error/fit params
// 	  for ( int i = 0; i < datumPerZ; i++ )
// 	    {
// 	      nu = it->second.ensem.IT[i];

// 	      // Set the oddNU matrices
// 	      gsl_matrix_set(oddNU, i, 0, pow(nu,1));
// 	      gsl_matrix_set(oddNU, i, 1, pow(nu,3));
// 	      gsl_matrix_set(oddNU, i, 2, pow(nu,5));
// 	      // Set the data/variance vectors
// 	      gsl_vector_set(pITDimag, i, it->second.ensem.avgM[i]);
// 	      gsl_vector_set(pITDimag_weights, i, 1.0/pow(it->second.ensem.errM[i],2));
// 	    }

// 	  gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc(datumPerZ,order);
// 	  gsl_multifit_linear_svd(oddNU, work);
// 	  gsl_multifit_wlinear(oddNU, pITDimag_weights, pITDimag, fitParams, covParams, &chisq, work);
// 	  gsl_multifit_linear_free(work);

// 	  printf ("# best IMAG fit: Y = %g NU + %g NU^3 + %g NU^5\n",
// 		  BEST(0), BEST(1), BEST(2));
// 	  printf ("# IMAG covariance matrix:\n");
// 	  printf ("[ %+.9e, %+.9e, %+.9e \n",
// 		  COV(0,0), COV(0,1), COV(0,2));
// 	  printf ("  %+.9e, %+.9e, %+.9e \n",
// 		  COV(1,0), COV(1,1), COV(1,2));
// 	  printf ("  %+.9e, %+.9e, %+.9e ]\n",
// 		  COV(2,0), COV(2,1), COV(2,2));

// 	  printf ("# chisq IMAG = %g\n\n\n", chisq);

	  
// 	  // Write the fit parameter central values
// 	  // Do a hard write of z=0 fit parameters to avoid a -nan in dat file
// 	  if ( it->first == 0 )
// 	    {
// 	      OUT << std::setprecision(15) << j - jack.begin() << " 1 " << it->first << " 0 0 0" << "\n";
// 	    }
// 	  else {
// 	    OUT << std::setprecision(15) << j - jack.begin() << " 1 " << it->first << " " << BEST(0)
// 		  << " " << BEST(1) << " " << BEST(2) << "\n";
// 	  }

// 	  it++;
// 	} // end iterator over all imag displacements

//       // Free associated memory for this jackknife sample
//       gsl_matrix_free(evenNU);
//       gsl_matrix_free(oddNU);
//       gsl_vector_free(pITDreal);
//       gsl_vector_free(pITDimag);
//       gsl_vector_free(pITDreal_weights);
//       gsl_vector_free(pITDimag_weights);
//       gsl_vector_free(fitParams);
//       gsl_matrix_free(covParams);


//     } // end jack iterator
//   OUT.close();

  return 0;
}
