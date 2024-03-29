/*
  EVOLVE REDUCED PSEUDO IOFFE-TIME DATA AT DIFFERENT Z^2 TO A COMMON Z0^2
  WHERE Z0^2 = EXP(-2GAMMA_E-1)*MU^-2

  USING EITHER A POLYNOMIAL FIT OF rPITD OR COSINE/SINE TRANSFORMS OF MODEL PSEUDO-PDFS

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
#include <gsl/gsl_integration.h> // Numerical integration

/*
  Headers for threading
*/
#include<omp.h>

#include "hdf5.h"

// #include "pitd_util_NEW_TEST.h"
#include "pitd_util.h"
//#include "kernel.h"

using namespace PITD;



// Structure to hold all needed parameters to perform convolution of DGLAP kernel and fit of reduced pITD
struct convolParams_t
{
  double nu, matelem;
  int z, comp;
  rpitdFitParams_t p;
  // Constructor with initializer list
  convolParams_t(rpitdFitParams_t _p, double _n, double _m, int _z, int _c) : p{_p}, nu(_n), matelem(_m),
									     z(_z), comp(_c) {}
};

// Convolve DGLAP kernel w/ reduced-pITD fit of pITD
double rpitdFitDGLAPConvo(double u, void * p)
{
  convolParams_t * cp = (convolParams_t *)p;
  // Friendly local copies
  double ioffeTime        = (cp->nu);
  // double matelem          = (cp->matelem);
  int z                   = (cp->z);
  rpitdFitParams_t rpitdfit = (cp->p);
  int comp                = (cp->comp);

  // Track the value of reduced-pITD fits evaluated for u != 1 and u = 1
  double rpitdFit, rpitdFitUnity;

  /*
    Evaluate the reduced-pITD fits
  */
  if ( comp == 0 )
    {
      rpitdFit      = rpitdfit.func(true, u*ioffeTime); // real reduced-pITD fit for u != 1
      rpitdFitUnity = rpitdfit.func(true, ioffeTime);   // real reduced-pITD fit for u = 1
    }
  if ( comp == 1 )
    {
      rpitdFit      = rpitdfit.func(false, u*ioffeTime); // imag reduced-pITD fit for u != 1
      rpitdFitUnity = rpitdfit.func(false, ioffeTime);   // imag reduced-pITD fit for u = 1
    }

  /*
    Return PLUS-PRESCRIPTION of Altarelli-Parisi kernel multiplied by reduced-pITD fit of reduced pITD
  */
  if ( u == 1 )
    return 0;
  else
    return ((1+pow(u,2))/(1-u))*log( (exp(2*M_EULER+1)/4)*pow(MU*z,2) ) *(rpitdFit - rpitdFitUnity);
}

// Convolve matching kernel w/ reduced-pITD fit of pITD
double rpitdFitMatchingConvo(double u, void *p)
{
  convolParams_t * cp = (convolParams_t *)p;
  // Friendly local copies
  double ioffeTime        = (cp->nu);
  // double matelem          = (cp->matelem);
  int z                   = (cp->z);
  rpitdFitParams_t rpitdfit = (cp->p);
  int comp                = (cp->comp);

  // Track the value of reduced-pITD fits evaluated for u != 1 and u = 1
  double rpitdFit, rpitdFitUnity;

  /*
    Evaluate the reduced-pITD fits
  */
  if ( comp == 0 )
    {
      rpitdFit      = rpitdfit.func(true, u*ioffeTime); // real reduced-pITD fit for u != 1
      rpitdFitUnity = rpitdfit.func(true, ioffeTime);   // real reduced-pITD fit for u = 1
    }
  if ( comp == 1 )
    {
      rpitdFit      = rpitdfit.func(false, u*ioffeTime); // imag reduced-pITD fit for u != 1
      rpitdFitUnity = rpitdfit.func(false, ioffeTime);   // imag reduced-pITD fit for u = 1
    }

  /*
    Return PLUS-PRESCRIPTION of MSbar matching kernel multiplied by reduced-pITD fit of reduced pITD
  */
  if ( u == 1.0 )
    return 0;
  else
    return (((4*log(1-u))/(1-u))-2*(1-u))*(rpitdFit - rpitdFitUnity);
}



// Perform numerical convolution of DGLAP kernel & reduced pITD data using "+" prescription
double convolutionDGLAP(rpitdFitParams_t &rpitd, double nu, double matelem, int z, int comp)
{
  /*
    Compute approximation to definite integral of form

    I = \int_a^b f(x)w(x)dx

    Where w(x) is a weight function ( = 1 for general integrands)
    Absolute/relative error bounds provided by user
  */

  // Collect and package the parameters to do the convolution
  convolParams_t cParams(rpitd, nu, matelem, z, comp); 

  
  // Define the function to integrate
  gsl_function F;
  F.function = &rpitdFitDGLAPConvo;
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
double convolutionMATCH(rpitdFitParams_t &rpitd, double nu, double matelem, int z, int comp)
{
  /*
    Compute approximation to definite integral of form

    I = \int_a^b f(x)w(x)dx

    Where w(x) is a weight function ( = 1 for general integrands)
    Absolute/relative error bounds provided by user
  */

  // Collect and package the parameters to do the convolution
  convolParams_t cParams(rpitd, nu, matelem, z, comp);
 

  // Define the function to integrate
  gsl_function F;
  F.function = &rpitdFitMatchingConvo;
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

  // pfq_t dum;
  // // dum.real = pow(2,-3-0.125-3.0)*gsl_sf_gamma(2+0.125)*gsl_sf_gamma(1+3.0)*M_PI*0.25*2.0*pseudoPDFSineTransform(0.125,3.0,-1.0*pow(0.25,2)/4).real;
  // dum.real = pseudoPDFSineTransform(0.125,3.0,-1.0*pow(0.25,2)/4).real;
  // // dum.imag = pseudoPDFSineTransform(0.125,3.0,-1.0*pow(0.25,2)/4).imag;
  // dum.imag = 0.0;
  
  // std::cout << std::setprecision(10) << "Sine-transform 2F3 = " << dum.real << " " << dum.imag << std::endl;
  // dum = pseudoPDFCosineTransform(0.125,3.0,-1.0*pow(0.25,2)/4);
  // dum.imag = 0.0;
  // std::cout << std::setprecision(10) << "Cosine-transform 2F3 = " << dum.real << " " << dum.imag << std::endl;
  // // dum = GenHypGeomEval(1.5);
  // // std::cout << std::setprecision(10) << dum.real << " " << dum.imag << std::endl;
  // exit(8);

  if ( argc != 10 )
    {
      std::cout << "Usage: $0 <polynomial(0) -or- pseudo-PDF (1) fit of rPITD> <pITD fit txt> <h5 file> <matelemType> <gauge_configs> <zmin> <zmax> <pmin> <pmax>" << std::endl;
      std::cout << "--- Note: should pass zmin = 0, pmin = 1, pmax = 6" << std::endl;
      exit(1);
    }

  std::string rpitdFit, matelemType;
  int gauge_configs, zmin, zmax, pmin, pmax;
  int type; // type of fit of reduced pITD
  // int rpitdReality;
  

  std::stringstream ss;
  ss << argv[1]; ss >> type;          ss.clear(); ss.str(std::string());
  ss << argv[2]; ss >> rpitdFit;      ss.clear(); ss.str(std::string());
  ss << argv[4]; ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[5]; ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[6]; ss >> zmin;          ss.clear(); ss.str(std::string());
  ss << argv[7]; ss >> zmax;          ss.clear(); ss.str(std::string());
  ss << argv[8]; ss >> pmin;          ss.clear(); ss.str(std::string());
  ss << argv[9]; ss >> pmax;          ss.clear(); ss.str(std::string());
  // ss << argv[10]; ss >> rpitdReality; ss.clear(); ss.str(std::string());


  reducedPITD rawPseudo(gauge_configs);
  /*
    ACCESS FULL DATASET OF FITTED IOFFE-TIME PSEUDO-STRUCTURE FUNCTIONS
  */
  H5Read(argv[3], &rawPseudo, gauge_configs, zmin, zmax, pmin, pmax, "pitd");

  /*
    ACCESS THE Z^2 REDUCED-PITD FIT PARAMETERS FOR ALL JACKKNIFE SAMPLES
    Organized according to:
         --> polynomial fit:   <z> <jack> <comp> <a> <b> <c> <chi2>
	 --> pseudo-PDF fit:   <z> <comp> <chi2> <norm> <alpha> <beta> <gamma> <delta>
  */
  std::ifstream IN;
  IN.open(rpitdFit);
  if (IN.is_open())
    {
      while ( ! IN.eof() )
	{
	  // Variables to read reduced-pITD fit
	  int z, j, comp;
	  double a, b, c, chi2, norm, d;

	  if ( type == 0 )
	    {
	      IN >> std::setprecision(15) >> z >> j >> comp >> a >> b >> c >> chi2;
	      // std::cout << z << " " << j << " " << comp << " " << a << " " << b << " "
	      // 	    << c << " " << chi2 << std::endl;
	      if ( comp == 0 )
		rawPseudo.data.disps[z].rpitdR.push_back(rpitdFitParams_t(a,b,c));
	      else if ( comp == 1 )
		rawPseudo.data.disps[z].rpitdI.push_back(rpitdFitParams_t(a,b,c));
	      else {
		std::cerr << "Unable to parse reduced-pITD fit coefficients!" << std::endl;
		exit(2);
	      }
	    }
	  else if ( type == 1 )
	    {
	      // std::cout << "Parsing pseudo-PDF cosine/sine transforms" << std::endl;
	      IN >> std::setprecision(15) >> z >> comp >> chi2 >> norm >> a >> b >> c >> d;
	      if ( comp == 0 )                                          // hard set a = 0.2 to match AR (06/01) 
		rawPseudo.data.disps[z].rpitdR.push_back(rpitdFitParams_t(0.2,b,c)); // c is zero (05/21/2021)
	      else if ( comp == 1 )                                     // hard set a = 0.2 to match AR (06/29)
		rawPseudo.data.disps[z].rpitdI.push_back(rpitdFitParams_t(0.2,b,norm)); // pass norm (05/21/2021)
	      else {
		std::cerr << "Unable to parse reduced-pITD fit coefficients!" << std::endl;
		exit(2);
	      }
	    }
	}
    }
  std::cout << "Done reading reduced-pITD fit results" << std::endl;
  IN.close();

  // for ( int J = 0; J < gauge_configs; J++ ) 
  //   {
  //     std::cout << rawPseudo.data.disps[1].rpitdR[J] << std::endl;
  //   }
  

  // Get the number of unique displacements from read in unevolved data
  int znum = rawPseudo.data.disps.size();
  // std::cout << "ZNUM = " << znum << std::endl;


  /*
    For each z^2, perform convolution of rpitdFit and Altarelli-Parisi kernel + Matching kernel
  */
  reducedPITD outRaw(gauge_configs);
  // reducedPITD evoKernel(gauge_configs);
  // reducedPITD matchingKernel(gauge_configs);
  // reducedPITD theITD(gauge_configs);


  reducedPITD evoKernel = rawPseudo;
  reducedPITD matchingKernel = rawPseudo;
  reducedPITD theITD = rawPseudo;




  for ( auto zi = rawPseudo.data.disps.begin(); zi != rawPseudo.data.disps.end(); ++zi )
    {
#pragma omp parallel for // num_threads(2)
      for ( int ji = 0; ji < gauge_configs; ji ++ ) // gauge_configs
	// N.B. Would like to iterate over a std::vec of rpitdFitParams_t, but then stuck with same component
      // for ( auto ji = zi->second.rpitdR.begin(); ji != zi->second.rpitdR.end(); ++ji )
	{
	  
	  std::cout << "Evolving Z = " << zi->first << " data for JACK = " << ji << std::endl;

	  // Each momentum with same z, receives same evolution/matching
	  for ( auto mi = zi->second.moms.begin(); mi != zi->second.moms.end(); ++mi )
	    {

	      // Don't forget the Ioffe-time
	      evoKernel.data.disps[zi->first].moms[mi->first].IT      = mi->second.IT;
	      matchingKernel.data.disps[zi->first].moms[mi->first].IT = mi->second.IT;
	      theITD.data.disps[zi->first].moms[mi->first].IT         = mi->second.IT;
	      if ( zi->first == 0 )
		{
		  evoKernel.data.disps[zi->first].moms[mi->first].mat[ji] = 0.0; //.push_back(ji); //0.0);
                  matchingKernel.data.disps[zi->first].moms[mi->first].mat[ji] = 0.0; //.push_back(ji); //0.0);
                  theITD.data.disps[zi->first].moms[mi->first].mat[ji] = 1.0; //.push_back(ji); //1.0);
		}
	      else
		{
	      
		  std::complex<double> dglap, match;

		  // Run over each component of pITD
		  for ( int comp = 0; comp != 2; comp++ )
		    {

		      // DGLAP/MATCH: <rpitdFitParams for this jk> <ioffe time> <mat jk> <zsep> < real --> 0 >
		      if ( comp == 0 )
			// if ( rpitdReality == 0 )
			{
			  std::cout << "(RE) Starting evaluation" << std::endl;
			  dglap.real( convolutionDGLAP(zi->second.rpitdR[ji], mi->second.IT,
						       mi->second.mat[ji].real(), zi->first, 0) );
			  match.real( convolutionMATCH(zi->second.rpitdR[ji], mi->second.IT,
			  			       mi->second.mat[ji].real(), zi->first, 0) );

			  std::cout << "(RE) Running ? = " << dglap.real() << " " << match.real() << std::endl;
			}
		      if ( comp == 1 )
		      	// if ( rpitdReality == 1 )
		      	{
			  std::cout << "(IM) Starting evaluation" << std::endl;
		      	  dglap.imag( convolutionDGLAP(zi->second.rpitdI[ji], mi->second.IT,
		      				       mi->second.mat[ji].imag(), zi->first, 1) );
		      	  match.imag( convolutionMATCH(zi->second.rpitdI[ji], mi->second.IT,
		      				       mi->second.mat[ji].imag(), zi->first, 1) );
			  std::cout << "(IM) Running ? = " << dglap.imag() << " " << match.imag() << std::endl;
		      	}
	
		    } // comp
		  
		  // std::cout << "----> DGLAP = " << dglap << "          ----> MATCH = " << match << std::endl;
		  
		  // // Prepare the raw lattice data for reference
		  // fill_reducedPITD(outRaw[J-jack.begin()].real,z,J->real.disps[z].ensem.IT[place],
		  // 		       *it,J->real.disps[z].ensem.errM[place]);
		  
		  evoKernel.data.disps[zi->first].moms[mi->first].mat[ji] = dglap; //.push_back(dglap);
		  matchingKernel.data.disps[zi->first].moms[mi->first].mat[ji] = match; //.push_back(match);
		  theITD.data.disps[zi->first].
		    moms[mi->first].mat[ji] = mi->second.mat[ji] + ((alphaS*Cf)/(2*M_PI))*(dglap+match);

		  // std::cout << "PRIOR = " << mi->second.mat[ji] << std::endl;
		  // std::cout << "RESULT = " << mi->second.mat[ji] + ((alphaS*Cf)/(2*M_PI))*(dglap+match)
		  // 	    << std::endl;

    		} // if-else

	    } // end mi
	} // end ji
    } // end zi


  // Push the outRaw, evoKernel, matchingKernel, theITD to H5 files
  // reducedPITD *dummy = new reducedPITD(gauge_configs,znum,6);

  std::string outevoh5 =  "b_b0xDA__J0_A1pP." + matelemType + ".zmin" + std::to_string(zmin) +"_zmax" + std::to_string(zmax) + ".imagiscorrect.EVO.h5";
  std::string outmatchh5 = "b_b0xDA__J0_A1pP." + matelemType + ".zmin" + std::to_string(zmin) +"_zmax" + std::to_string(zmax) + ".imagiscorrect.MATCH.h5";
  std::string outevomatchh5 = "b_b0xDA__J0_A1pP." + matelemType + ".zmin" + std::to_string(zmin) +"_zmax" + std::to_string(zmax) + ".imagiscorrect.EVO-MATCH.h5";
  char *evoKernelH5 = &outevoh5[0];
  char *matchingKernelH5 = &outmatchh5[0];
  char *theITDH5 = &outevomatchh5[0];


  /*
    WRITE EVO,MATCH,EVO-MATCH DATA FOR EACH JACKKNIFE SAMPLE
  */
  H5Write(evoKernelH5,&evoKernel,gauge_configs,zmin,zmax,pmin,pmax,"evoKernel");
  H5Write(matchingKernelH5,&matchingKernel,gauge_configs,zmin,zmax,pmin,pmax,"matchingKernel");
  H5Write(theITDH5,&theITD,gauge_configs,zmin,zmax,pmin,pmax,"itd");




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
