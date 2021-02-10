/*
  EVOLVE REDUCED PSEUDO IOFFE-TIME DATA AT DIFFERENT Z^2 TO A COMMON Z0^2
  WHERE Z0^2 = EXP(-2GAMMA_E-1)*MU^-2
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
/*
  Grab headers for gsl function calls
*/
#include <gsl/gsl_sf_gamma.h> // Evaluaton of Gamma/Beta functions
#include <gsl/gsl_rng.h> // Random numbers
#include <gsl/gsl_integration.h> // Numerical integration
#include <gsl/gsl_multimin.h> // multidimensional minimization
#include <gsl/gsl_vector.h> // allocating/accessing gsl_vectors for passing params to minimizer
#include <gsl/gsl_matrix.h> // matrix routines for inversion of data covariance
#include <gsl/gsl_permutation.h> // permutation header for matrix inversions
#include <gsl/gsl_linalg.h> // linear algebra
#include <gsl/gsl_sf_hyperg.h> // hypergeometric fun
#include <gsl/gsl_sf_expint.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>

/*
  Grab arb headers for generalized hypergeometric function evaluation
*/
// #include "arb.h"
// #include "arb_hypgeom.h"
// #include "acb.h"
// #include "acb_hypgeom.h"

/*
  Headers for threading
*/
#include<omp.h>

#include "hdf5.h"

// #include "pitd_util_NEW_TEST.h"
#include "pitd_util.h"

using namespace PITD;


// // SU(3) invariant
// const double Cf = 4.0/3.0;
// const double hbarc = 0.1973269804; // GeV * fm
// const double aLat = 0.094; // fm
// const double muRenorm = 2.0; // GeV
// const double MU = (aLat*muRenorm)/hbarc; // fm^-1
// const double alphaS = 0.303;


// Structure to hold all needed parameters to perform convolution of DGLAP kernel and polyfit of reduced pITD
struct convolParams_t
{
  double nu, matelem;
  int z, comp;
  polyFitParams_t p;
  // Constructor with initializer list
  convolParams_t(polyFitParams_t _p, double _n, double _m, int _z, int _c) : p(_p), nu(_n), matelem(_m),
									     z(_z), comp(_c) {}
};

// Convolve DGLAP kernel w/ polynomial fit of pITD
double polyFitDGLAPConvo(double u, void * p)
{
  convolParams_t * cp = (convolParams_t *)p;
  // Friendly local copies
  double ioffeTime        = (cp->nu);
  // double matelem          = (cp->matelem);
  int z                   = (cp->z);
  polyFitParams_t polyfit = (cp->p);
  int comp                = (cp->comp);

  // Track the value of polynomial fits evaluated for u != 1 and u = 1
  double polyFit, polyFitUnity;

  /*
    Evaluate the polynomials
  */
  if ( comp == 0 )
    {
      polyFit      = polyfit.func(true, u*ioffeTime); // real polynomial fit for u != 1
      polyFitUnity = polyfit.func(true, ioffeTime);   // real polynomial fit for u = 1
    }
  if ( comp == 1 )
    {
      polyFit      = polyfit.func(false, u*ioffeTime); // imag polynomial fit for u != 1
      polyFitUnity = polyfit.func(false, ioffeTime);   // imag polynomial fit for u = 1
    }

  /*
    Return PLUS-PRESCRIPTION of Altarelli-Parisi kernel multiplied by polynomial fit of reduced pITD
  */
  if ( u == 1 )
    return 0;
  else
    return ((1+pow(u,2))/(1-u))*log( (exp(2*M_EULER+1)/4)*pow(MU*z,2) ) *(polyFit - polyFitUnity);
}

// Convolve matching kernel w/ polynomial fit of pITD
double polyFitMatchingConvo(double u, void *p)
{
  convolParams_t * cp = (convolParams_t *)p;
  // Friendly local copies
  double ioffeTime        = (cp->nu);
  // double matelem          = (cp->matelem);
  int z                   = (cp->z);
  polyFitParams_t polyfit = (cp->p);
  int comp                = (cp->comp);

  // Track the value of polynomial fits evaluated for u != 1 and u = 1
  double polyFit, polyFitUnity;

  /*
    Evaluate the polynomials
  */
  if ( comp == 0 )
    {
      polyFit      = polyfit.func(true, u*ioffeTime); // real polynomial fit for u != 1
      polyFitUnity = polyfit.func(true, ioffeTime);   // real polynomial fit for u = 1
    }
  if ( comp == 1 )
    {
      polyFit      = polyfit.func(false, u*ioffeTime); // imag polynomial fit for u != 1
      polyFitUnity = polyfit.func(false, ioffeTime);   // imag polynomial fit for u = 1
    }

  /*
    Return PLUS-PRESCRIPTION of MSbar matching kernel multiplied by polynomial fit of reduced pITD
  */
  if ( u == 1.0 )
    return 0;
  else
    return (((4*log(1-u))/(1-u))-2*(1-u))*(polyFit - polyFitUnity);
}



// Perform numerical convolution of DGLAP kernel & reduced pITD data using "+" prescription
double convolutionDGLAP(polyFitParams_t &poly, double nu, double matelem, int z, int comp)
{
  /*
    Compute approximation to definite integral of form

    I = \int_a^b f(x)w(x)dx

    Where w(x) is a weight function ( = 1 for general integrands)
    Absolute/relative error bounds provided by user
  */

  // Collect and package the parameters to do the convolution
  convolParams_t cParams(poly, nu, matelem, z, comp); 

  
  // Define the function to integrate
  gsl_function F;
  F.function = &polyFitDGLAPConvo;
  F.params = &cParams;
  
  // hold n double precision intervals of integrand, their results and error estimates
  size_t n = 10000;

  /*
    Allocate workspace sufficient to hold data for size_t n intervals
  */
  gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(n);

  // Set some parameters for the integration
  double result, abserr;
  size_t nevals; // evaluations to solve
  double epsabs=0.0000000000001; // Set absolute error of integration
  double epsrel = 0.0; // To compute a specified absolute error, setting epsrel to zero
  /*
    Perform the numerical convolution here
    Pass success boolean as an int
  */
  // int success = gsl_integration_cquad(&F,0.0,0.99999999999999,epsabs,epsrel,w,&result,&abserr,&nevals);
  int success = gsl_integration_cquad(&F,0.0,1.0,epsabs,epsrel,w,&result,&abserr,&nevals);

  // Free associated memory
  gsl_integration_cquad_workspace_free(w);

  return result;  
}

// Perform numerical convolution of MSbar matching kernel & reduced pITD data using "+" prescription
double convolutionMATCH(polyFitParams_t &poly, double nu, double matelem, int z, int comp)
{
  /*
    Compute approximation to definite integral of form

    I = \int_a^b f(x)w(x)dx

    Where w(x) is a weight function ( = 1 for general integrands)
    Absolute/relative error bounds provided by user
  */

  // Collect and package the parameters to do the convolution
  convolParams_t cParams(poly, nu, matelem, z, comp);
 

  // Define the function to integrate
  gsl_function F;
  F.function = &polyFitMatchingConvo;
  F.params = &cParams;
  
  // hold n double precision intervals of integrand, their results and error estimates
  size_t n = 10000;

  /*
    Allocate workspace sufficient to hold data for size_t n intervals
  */
  gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(n);

  // Set some parameters for the integration
  double result, abserr;
  size_t nevals; // evaluations to solve
  double epsabs=0.0000000000001; // Set absolute error of integration
  double epsrel = 0.0; // To compute a specified absolute error, setting epsrel to zero
  /*
    Perform the numerical convolution here
    Pass success boolean as an int
  */
  // int success = gsl_integration_cquad(&F,0.0,0.99999999999999,epsabs,epsrel,w,&result,&abserr,&nevals);
  int success = gsl_integration_cquad(&F,0.0,1.0,epsabs,epsrel,w,&result,&abserr,&nevals);

  // Free associated memory
  gsl_integration_cquad_workspace_free(w);

  return result;
}

#if 0
// Fill outRaw*,outEvo*,outMatch*,outEvoMatch*  std::vector<reducedPITD>
void fill_reducedPITD(compPITD &c, int z, double IT, double avgM, double errM)
{
  // Prepare the raw lattice data for reference
  c.disps[z].ensem.IT.push_back(IT);
  c.disps[z].ensem.avgM.push_back(avgM);
  c.disps[z].ensem.errM.push_back(errM);
}


// Method to determine a ensAvg/err reducedPITD
// from passed jackknife reducedPITD
reducedPITD jkReducedPITD(std::vector<reducedPITD> &jkr)
{
  // Catch some parameters 
  int zmax  = jkr[0].real.disps.size();
  int itMax = jkr[0].real.disps[0].ensem.IT.size();
  // Initialize the reducedPITD that will be returned
  reducedPITD a(jkr.size(),zmax,itMax);
  for ( int z = 0; z < zmax; ++z )
    {
      for ( int m = 0; m < itMax; ++m )
	{
	  a.real.disps[z].ensem.IT[m] = jkr[0].real.disps[z].ensem.IT[m];
	  a.imag.disps[z].ensem.IT[m] = jkr[0].imag.disps[z].ensem.IT[m];
	  double dumr = 0.0;
	  double dumi = 0.0;
	  for ( auto j = jkr.begin(); j != jkr.end(); j++ )
	    {
	      dumr+=j->real.disps[z].ensem.avgM[m];
	      dumi+=j->imag.disps[z].ensem.avgM[m];
	    }
	  a.real.disps[z].ensem.avgM[m] = dumr/jkr.size();
	  a.imag.disps[z].ensem.avgM[m] = dumi/jkr.size();

	  // Now that we have the central value, determine the error
	  dumr = 0.0; dumi = 0.0;
	  for ( auto j = jkr.begin(); j != jkr.end(); j++ )
	    {
	      dumr += pow(j->real.disps[z].ensem.avgM[m] - a.real.disps[z].ensem.avgM[m], 2);
	      dumi += pow(j->imag.disps[z].ensem.avgM[m] - a.imag.disps[z].ensem.avgM[m], 2);
	    }

	  // Now finish with the error
	  a.real.disps[z].ensem.errM[m] = sqrt(((jkr.size() - 1)/(1.0*jkr.size()))*dumr);
	  a.imag.disps[z].ensem.errM[m] = sqrt(((jkr.size() - 1)/(1.0*jkr.size()))*dumi);

	} // end m
    } // end z

  return a;
}

// Method to write out EnsAvg reducedPITD objects
void rpitdWrite(std::ofstream &o, reducedPITD &r)
{
  for ( auto d = r.real.disps.begin(); d != r.real.disps.end(); ++d )
    {
      for ( int m = 0; m < d->second.ensem.IT.size(); m++ )
	{
	  o << std::setprecision(15) << d->second.ensem.IT[m] << " 0 "
	    << d->second.ensem.avgM[m] << " " << d->second.ensem.errM[m]
	    << " " << d->first << "\n";
	}
    }

  for ( auto d = r.imag.disps.begin(); d != r.imag.disps.end(); ++d )
    {
      for ( int m = 0; m < d->second.ensem.IT.size(); m++ )
	{
	  o << std::setprecision(15) << d->second.ensem.IT[m] << " 1 "
	    << d->second.ensem.avgM[m] << " " << d->second.ensem.errM[m]
	    << " " << d->first << "\n";
	}
    }
  o.close();
}
#endif

int main( int argc, char *argv[] )
{

  if ( argc != 9 )
    {
      std::cout << "Usage: $0 <pITD poly fit txt> <h5 file> <matelemType> <gauge_configs> <zmin> <zmax> <pmin> <pmax>" << std::endl;
      std::cout << "--- Note: should pass zmin = 0, pmin = 1, pmax = 6" << std::endl;
      exit(1);
    }

  std::string polyFit_pITD, matelemType;
  int gauge_configs, zmin, zmax, pmin, pmax;
  

  std::stringstream ss;
  ss << argv[1]; ss >> polyFit_pITD;  ss.clear(); ss.str(std::string());
  ss << argv[3]; ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[4]; ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[5]; ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[6]; ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[7]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[8]; ss >> pmax;          ss.clear(); ss.str(std::string());


  // reducedPITD *ensem = new reducedPITD(gauge_configs,9,6); // 9zseps, 6moms
  // std::vector<reducedPITD > jack(gauge_configs,reducedPITD(gauge_configs-1,9,6));//, new reducedPITD());


  reducedPITD rawPseudo(gauge_configs);
  /*
    ACCESS FULL DATASET OF FITTED IOFFE-TIME PSEUDO-STRUCTURE FUNCTIONS
  */
  H5Read(argv[2], &rawPseudo, gauge_configs, zmin, zmax, pmin, pmax);

  /*
    ACCESS THE Z^2 POLYNOMIAL FIT PARAMETERS FOR ALL JACKKNIFE SAMPLES
    Organized according to:
             <z> <jack> <comp> <a> <b> <c> <chi2>
  */
  std::ifstream IN;
  IN.open(polyFit_pITD);
  if (IN.is_open())
    {
      while ( ! IN.eof() )
	{
	  // Variables to read polynomial fit
	  int z, j, comp;
	  double a, b, c, chi2;
	  IN >> std::setprecision(15) >> z >> j >> comp >> a >> b >> c >> chi2;
	  // std::cout << z << " " << j << " " << comp << " " << a << " " << b << " "
	  // 	    << c << " " << chi2 << std::endl;
	  if ( comp == 0 )
	    rawPseudo.data.disps[z].polyR.push_back(polyFitParams_t(a,b,c));
	  else if ( comp == 1 )
	    rawPseudo.data.disps[z].polyI.push_back(polyFitParams_t(a,b,c));
	  else {
	    std::cerr << "Unable to parse polynomial fit coefficients!" << std::endl;
	    exit(2);
	  }
	}
    }
  std::cout << "Done reading polynomial fit results" << std::endl;
  IN.close();

  for ( int J = 0; J < gauge_configs; J++ ) 
    {
      std::cout << rawPseudo.data.disps[1].polyR[J] << std::endl;
    }
  

  // Get the number of unique displacements from read in unevolved data
  int znum = rawPseudo.data.disps.size();


  /*
    For each z^2, perform convolution of polyFit_pITD and Altarelli-Parisi kernel + Matching kernel
  */
  reducedPITD outRaw(gauge_configs);
  reducedPITD evoKernel(gauge_configs);
  reducedPITD matchingKernel(gauge_configs);
  reducedPITD theITD(gauge_configs);





  for ( auto zi = rawPseudo.data.disps.begin(); zi != rawPseudo.data.disps.end(); ++zi )
    {
#pragma omp parallel //for num_threads(16)
      for ( auto ji = zi->second.polyR.begin(); ji != zi->second.polyR.end(); ++ji )
	{
	  
	  auto jkNum = std::distance(zi->second.polyR.begin(),ji);
	  std::cout << "Evolving Z = " << zi->first << " data for JACK = "
		    << jkNum << std::endl;

	  // Each momentum with same z, receives same evolution/matching
	  for ( auto mi = zi->second.moms.begin(); mi != zi->second.moms.end(); ++mi )
	    {
	      
	      // DGLAP: <polyFitParams for this jk> <ioffe time> <mat jk> <zsep> < real --> 0 >
	      double dglap = convolutionDGLAP(*ji, mi->second.IT, mi->second.mat[jkNum].real(),
					      zi->first, 0);
	      double match = convolutionMATCH(*ji, mi->second.IT, mi->second.mat[jkNum].real(),
					      zi->first, 0);
	      

	      std::cout << "----> DGLAP (RE) = " << dglap << std::endl;
	      std::cout << "----> MATCH (RE) = " << match << std::endl;

	    } // mi
	} // ji
    } // zi


    // 	      // Prepare the raw lattice data for reference
    // 	      fill_reducedPITD(outRaw[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
    // 			       *it,J->real.disps[z].ensem.errM[place]);
    // 	      if ( z == 0 )
    // 		{
    // 		  fill_reducedPITD(evoKernel[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],0.0,-1.0);
    // 		  // evoKernel[J-jackReal.begin()].disps[z].ensem.errM.push_back(J->disps[z].ensem.errM[place]);

    // 		  fill_reducedPITD(matchingKernel[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],0.0,-1.0);
    // 		  // matchingKernel[J-jackReal.begin()].disps[z].ensem.errM.push_back(J->disps[z].ensem.errM[place]);

    // 		  fill_reducedPITD(theITD[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],1.0,-1.0);
    // 		  // theITD[J-jackReal.begin()].disps[z].ensem.errM.push_back(J->disps[z].ensem.errM[place]);
    // 		}
    // 	      else
    // 		{
    // 		  // fill_reducedPITD(evoKernel[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
    // 		  // 		   ((alphaS*Cf)/(2*M_PIl))*dglap,-1);
    // 		  fill_reducedPITD(evoKernel[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
    // 				   dglap,-1);
		  
    // 		  // fill_reducedPITD(matchingKernel[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
    // 		  // 		   ((alphaS*Cf)/(2*M_PIl))*match,-1);
    // 		  fill_reducedPITD(matchingKernel[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
    // 				   match,-1);

    // 		  fill_reducedPITD(theITD[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
    // 				   (*it)+((alphaS*Cf)/(2*M_PIl))*(dglap+match),-1);
    // 		}
    // 	    } // end [REAL] *it disps

    // 	  // [IMAG] Iterate over all ioffe-times stored for this z
    //       for ( auto it = J->imag.disps[z].ensem.avgM.begin(); it != J->imag.disps[z].ensem.avgM.end(); ++it )
    //         {
    //           auto place = it - J->imag.disps[z].ensem.avgM.begin();

    //           double dglap = convolutionDGLAP(J->imag.disps[z],J->imag.disps[z].ensem.IT[place],(*it),z,1);
    //           double match = convolutionMATCH(J->imag.disps[z],J->imag.disps[z].ensem.IT[place],(*it),z,1);

    // 	      std::cout << "----> DGLAP (IM) = " << dglap << std::endl;
    // 	      std::cout << "----> MATCH (IM) = " << match << std::endl;


    //           // Prepare the raw lattice data for reference
    //           fill_reducedPITD(outRaw[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],
    //                            *it,J->imag.disps[z].ensem.errM[place]);
    //           if ( z == 0 )
    //             {
    //               fill_reducedPITD(evoKernel[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],0.0,-1.0);
    //               // evoKernel[J-jackReal.begin()].disps[z].ensem.errM.push_back(J->disps[z].ensem.errM[place]);

    //               fill_reducedPITD(matchingKernel[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],0.0,-1.0);
    //               // matchingKernel[J-jackReal.begin()].disps[z].ensem.errM.push_back(J->disps[z].ensem.errM[place]);

    //               fill_reducedPITD(theITD[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],0.0,-1.0);
    //               // theITD[J-jackReal.begin()].disps[z].ensem.errM.push_back(J->disps[z].ensem.errM[place]);
    //             }
    //           else
    //             {
    //               // fill_reducedPITD(evoKernel[J-jack.begin()].real,z,J->imag.disps[z].ensem.IT[place],
    //               //               ((alphaS*Cf)/(2*M_PIl))*dglap,-1);
    //               fill_reducedPITD(evoKernel[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],
    //                                dglap,-1);
                  
    //               // fill_reducedPITD(matchingKernel[J-jack.begin()].real,z,J->imag.disps[z].ensem.IT[place],
    //               //               ((alphaS*Cf)/(2*M_PIl))*match,-1);
    //               fill_reducedPITD(matchingKernel[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],
    //                                match,-1);

    //               fill_reducedPITD(theITD[J-jack.begin()].imag,z,J->imag.disps[z].ensem.IT[place],
    //                                (*it)+((alphaS*Cf)/(2*M_PIl))*(dglap+match),-1);
    //             }
    //         } // end [IMAG] *it disps

    // 	} //end z
    // } // end *J jackReal
  

//   // Push the outRaw, evoKernel, matchingKernel, theITD to H5 files
//   reducedPITD *dummy = new reducedPITD(gauge_configs,znum,6);

//   std::string outevoh5 =  "b_b0xDA__J0_A1pP." + matelemType + ".EVO.h5";
//   std::string outmatchh5 = "b_b0xDA__J0_A1pP." + matelemType + ".MATCH.h5";
//   std::string outevomatchh5 = "b_b0xDA__J0_A1pP." + matelemType + ".EVO-MATCH.h5";

//   char *evoKernelH5 = &outevoh5[0];
//   char *matchingKernelH5 = &outmatchh5[0];
//   char *theITDH5 = &outevomatchh5[0];


//   /*
//     WRITE EVO,MATCH,EVO-MATCH DATA FOR EACH JACKKNIFE SAMPLE
//     ERROR IS HARD SET TO -1 PER JACKKNIFE SAMPLE, AS AN ERROR WILL BE DETERMINED
//     BY ANALYZING ALL JACKKNIFE SAMPLES AND THUS FORMING A COVARIANCE MATRIX
    
//     IMAGINARY EVOLUTIONS NOT POSSIBLE YET - POLYFITS ON IMAGINARY DATA NOT DONE (7/1/2020)

//     WILL NOT DETERMINE CENTRAL VALUE/ERROR HERE, SO DUMMY IS PASSED
//     AS SECOND & THIRD ARGS WHICH WOULD BE ENSEMBLE AVERAGE REDUCEDPITDS

//     LIKEWISE dumI IS PASSED FOR IMAGINARY EVOLUTION PER JACKKNIFE SAMPLE
//   */
//   H5Write(evoKernelH5,dummy,evoKernel,gauge_configs,6,2,9);
//   H5Write(matchingKernelH5,dummy,matchingKernel,gauge_configs,6,2,9);
//   H5Write(theITDH5,dummy,theITD,gauge_configs,6,2,9);




//   /*
//     Determine central value/error of outRaw,evoKernel,matchingKernel,theITD data
//     from jackknife samples
//   */
//   // Open output files to catch these values
//   std::ofstream OUT_evomatch, OUT_evo, OUT_match, OUT_raw;
//   std::string output_evomatch = "b_b0xDA__J0_A1pP."+matelemType+".EVO-MATCH.txt";
//   std::string output_evo = "b_b0xDA__J0_A1pP."+matelemType+".EVO.txt";
//   std::string output_match = "b_b0xDA__J0_A1pP."+matelemType+".MATCH.txt";
//   std::string output_raw = "b_b0xDA__J0_A1pP."+matelemType+".RAW.txt";
//   OUT_evomatch.open(output_evomatch.c_str(), std::ofstream::trunc);
//   OUT_evo.open(output_evo.c_str(), std::ofstream::trunc);
//   OUT_match.open(output_match.c_str(), std::ofstream::trunc);
//   OUT_raw.open(output_raw.c_str(), std::ofstream::trunc);




//   reducedPITD outRawEnsAvg = jkReducedPITD(outRaw);
//   reducedPITD evoKernelEnsAvg = jkReducedPITD(evoKernel);
//   reducedPITD matchingKernelEnsAvg = jkReducedPITD(matchingKernel);
//   reducedPITD theITDEnsAvg = jkReducedPITD(theITD);



//   rpitdWrite(OUT_raw,outRawEnsAvg);
//   rpitdWrite(OUT_evo,evoKernelEnsAvg);
//   rpitdWrite(OUT_match,matchingKernelEnsAvg);
//   rpitdWrite(OUT_evomatch,theITDEnsAvg);



  return 0;
}
