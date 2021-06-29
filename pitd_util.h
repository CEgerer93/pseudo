/*
  Define classes/structs/methods needed to handle the reduced pseudo-ioffe-time distributions
*/

#ifndef __pitd_util_h__
#define __pitd_util_h__

#include<vector>
#include<map>
#include<iostream>
#include<iomanip>
#include<complex>

#include<gsl/gsl_matrix.h>
#include <gsl/gsl_sf_gamma.h> // Evaluaton of Gamma/Beta functions

#include "hdf5.h"
#include "kernel.h" // To allow potential for 2F3 evolution w/ pseudo-PDF

/*
 * Define some global constants
 */
const double Cf = 4.0/3.0;
const double hbarc = 0.1973269804;       // GeV * fm
const double aLat = 0.094;               // fm
const double muRenorm = 2.0;             // GeV
const double MU = (aLat*muRenorm)/hbarc; // fm^-1
const double alphaS = 0.303;


namespace PITD
{
  // Operator to allow direct multiplication of long double and std::complex<double>
  std::complex<double> operator*(long double ld, std::complex<double> c);

  /*
    Structure to hold reduced-pITD fit parameters, and fit function for Re/Im component of pITD
  */
  struct rpitdFitParams_t
  {
    double a, b, c;
    double func(bool reality, double ioffe) // return (Re/Im) reduced-pITD evaluated at ioffe
    {
#ifndef PSEUDOCONVOL
      if ( reality )
	return 1.0 + a*pow(ioffe, 2) + b*pow(ioffe, 4) + c*pow(ioffe,6); // + d*pow(ioffe,8);
      if ( !reality )
	return a*pow(ioffe, 1) + b*pow(ioffe, 3) + c*pow(ioffe, 5); // +d*pow(ioffe, 7);
#else
      if ( reality )
	{
	  double dum = pseudoPDFCosineTransform(a,b,-1.0*pow(ioffe,2)/4).real;
	  return dum; //pow(2,-1-a-b)*M_PI*gsl_sf_gamma(2+a+b)*dum;
	}
      if ( !reality )
	{
	  double dum = pseudoPDFSineTransform(a,b,-1.0*pow(ioffe,2)/4).real;
	  return pow(2,-3-a-b)*gsl_sf_gamma(2+a)*gsl_sf_gamma(1+b)*M_PI*ioffe*c*dum;
	  // c should be N+ of q+ pseudo-PDF
	}
#endif
    }
    // Def/Param constructors, with initializer lists
  rpitdFitParams_t() : a(0.0), b(0.0), c(0.0) {}
  rpitdFitParams_t(double _a = 0.0, double _b = 0.0, double _c = 0.0) : a(_a), b(_b), c(_c) {}
  };  
  

  // Easy printing of rpitdFitParams_t
  std::ostream& operator<<(std::ostream& os, const rpitdFitParams_t& p);  



  struct momVals
  {
    double                             IT;
    std::vector<std::complex<double> > mat;
    std::complex<double>               matAvg;
  momVals(int g_ = 1) : mat(g_) {}
  };


  struct zvals
  {
    std::map<std::string, momVals> moms;
    // Reduced-PITD fit results to pITD
    std::vector<rpitdFitParams_t>   rpitdR;
    std::vector<rpitdFitParams_t>   rpitdI;
  };


  // PITD info for each of Real/Imag components
  struct pitd
  {
    std::map<int, zvals> disps;                     // data for each zsep
    std::map<int, gsl_matrix *> covsR, covsI;       // covariances for each zsep
    std::map<int, gsl_matrix *> covsRInv, covsIInv; // inverse of covariances for each zsep
    std::map<int, int> svsR, svsI;

    gsl_matrix *covR, *covI;        // Full data covariances
    gsl_matrix *invCovR, *invCovI; // Inverses of full data covariances
    int svsFullR, svsFullI;        // # singular values removed from full data covariance
  };


  /*
    Master class for the reducedPITD
  */
  class reducedPITD
  {
  public:
    // Hold real/imaginary components
    pitd data;


    // Default
    reducedPITD() {}
    // Parametrized
    reducedPITD(int g) { gauge_configs = g; }
    reducedPITD(int g, int zmin, int zmax, int pmin, int pmax)
      {
	gauge_configs = g;
	zminCut = zmin; zmaxCut = zmax;
	pminCut = pmin; pmaxCut = pmax;
      }
    // Destructor
    virtual ~reducedPITD() {};


    // Quickly return the number of configs/jackknife samples
    int getCfgs() { return gauge_configs; }

    // Print all ensemble avg data with the same z value
    void ensemPrintZ(int zf, int comp);
    // Print the reduced-pITD fit coefficients for a specified z
    void rpitdFitPrint(int zf, int comp);


    // Determine the data covariance for each zsep
    void calcCovPerZ();
    // Determine the inverse of data covariance for each zsep
    void calcInvCovPerZ();

    // Determine the full data covariance
    void calcCov();
    void calcInvCov();

    // Cut on Z's and P's to exclude from fit
    void cutOnPZ(int minz, int maxz, int minp, int maxp);

    // View a covariance matrix
    void viewZCovMat(int zsep);
    // View inverse of a covariance matrix
    void viewZCovInvMat(int zsep);


    /*
      Isolate a single jackknife sample
    */

    
  private:
    int gauge_configs;
    int zminCut, zmaxCut, pminCut, pmaxCut;
  };


  /* // Print real/imag (comp) ensemble average data for provided zsep */
  /* void reducedPITD::ensemPrintZ(int zf, int c) {} */

  /* // Print the real/imag (comp) reduced-pITD fit parameters to ensemble average data for provided zsep */
  /* void reducedPITD::rpitdFitPrint(int zf, int comp) {} */

  
  /*
    READER FOR PASSED H5 FILES
  */
  void H5Read(char *inH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin,
	      int pmax, std::string dTypeName);

  
  /*
    WRITER FOR MAKING NEW H5 FILES - E.G. EVOLVED/MATCHED DATASETS
  */
  void H5Write(char *outH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin,
	       int pmax, std::string dTypeName);

  void printMat(gsl_matrix *g);

}
#endif
