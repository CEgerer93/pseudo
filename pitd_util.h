/*
  Define classes/structs/methods needed to handle the reduced pseudo-ioffe-time distributions
*/

#ifndef __pitd_util_h__
#define __pitd_util_h__

#include<vector>
#include<map>
#include<iostream>
#include<iomanip>
#include<cmath>
#include<complex>

#include<gsl/gsl_matrix.h>
#include<gsl/gsl_eigen.h>

#include "hdf5.h"

/*
 * Define some global constants
 */
const double Cf       = 4.0/3.0;
const double hbarc    = 0.1973269804;       // GeV * fm
const double aLat     = ALAT; //0.094; // 0.0749;                // fm
const double muRenorm = INPUTSCALE; //2.0; //4.0;             // GeV
const double MU       = (aLat*muRenorm)/hbarc; // fm^-1
const double alphaS   = ALPHAS; //0.303; // 0.2;


namespace PITD
{
  /* // Scale a vector */
  /* template<typename T> */
  /*   std::vector<T>& operator*=(std::vector<T>& v, T s); */

  std::vector<double>& operator*=(std::vector<double>& v, double s);

  // Operator to allow direct multiplication of long double and std::complex<double>
  std::complex<double> operator*(long double ld, std::complex<double> c);

  /*
    Structure to hold polynomial fit parameters, and fit function for Re/Im component of pITD
  */
  struct polyFitParams_t
  {
    double a, b, c;
    double func(bool reality, double ioffe) // return (Re/Im) polynomial evaluated at ioffe
    {
      double ret(0.0);
      if ( reality )
	ret = 1.0 + a*pow(ioffe, 2) + b*pow(ioffe, 4) + c*pow(ioffe,6); // + d*pow(ioffe,8);
      if ( !reality )
	ret = a*pow(ioffe, 1) + b*pow(ioffe, 3) + c*pow(ioffe, 5); // +d*pow(ioffe, 7);
      return ret;
    }
    // Def/Param constructors, with initializer lists
  polyFitParams_t() : a(0.0), b(0.0), c(0.0) {}
  polyFitParams_t(double _a = 0.0, double _b = 0.0, double _c = 0.0) : a(_a), b(_b), c(_c) {}
  };  
  

  // Easy printing of polyFitParams_t
  std::ostream& operator<<(std::ostream& os, const polyFitParams_t& p);  



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
    // Polynomial fit results to pITD
    std::vector<polyFitParams_t>   polyR;
    std::vector<polyFitParams_t>   polyI;
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
    
    /* ~pitd() */
    /* { */
    /*   gsl_matrix_free(covR); gsl_matrix_free(invCovR); */
    /*   gsl_matrix_free(covI); gsl_matrix_free(invCovI); */
    /* } */
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
  reducedPITD(int g, int zmin, int zmax, int pmin, int pmax) : gauge_configs(g), zminCut(zmin),
      zmaxCut(zmax), pminCut(pmin), pmaxCut(pmax)
      {
	numPZ = (zmax - zmin + 1)*(pmax - pmin + 1);

	data.covR    = gsl_matrix_calloc(numPZ,numPZ);
	data.covI    = gsl_matrix_calloc(numPZ,numPZ);
	data.invCovR = gsl_matrix_calloc(numPZ,numPZ);
	data.invCovI = gsl_matrix_calloc(numPZ,numPZ);
      }
    /* // Destructor */
    /* virtual ~reducedPITD() */
    /*   { */
    /* 	gsl_matrix_free(data.covR); gsl_matrix_free(data.covI); */
    /* 	gsl_matrix_free(data.invCovR); gsl_matrix_free(data.invCovI); */
    /*   }; */
#warning "FIX ME - Need a suitable destructor for pitdEvo case, wherein gsl_matrices are not allocated!"
    /* virtual ~reducedPITD() {}  */


    // Quickly return the number of configs/jackknife samples
    int getCfgs() { return gauge_configs; }

    // Print all ensemble avg data with the same z value
    void ensemPrintZ(int zf, int comp);
    // Print the polynomial fit coefficients for a specified z
    void polyFitPrint(int zf, int comp);


    // Determine the data covariance for each zsep
    void calcCovPerZ();
    // Determine the inverse of data covariance for each zsep
    void calcInvCovPerZ();

    // Determine the full data covariance
    void calcCov();
    void calcInvCov();

    // Modify full data covariance to include a systematic error
    // ---> ex) squared diff. btwn. matelem fits w/ diff. Tmin added to diagonal of data covariance
    void addSystematicCov(reducedPITD *sysDist);
    // Modify data covariance per z to include a systematic error
    void addSystematicCovPerZ(reducedPITD *sysDist);

    // Cut on Z's and P's to exclude from fit
    void cutOnPZ(int minz, int maxz, int minp, int maxp);

    // View a covariance matrix
    void viewZCovMat(int zsep);
    // View inverse of a covariance matrix
    void viewZCovInvMat(int zsep);

    
  private:
    int gauge_configs;
    int zminCut, zmaxCut, pminCut, pmaxCut;
    int numPZ;
  };

  
  /*
    READER FOR PASSED H5 FILES
  */
  void H5Read(char *inH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin,
	      int pmax, std::string dTypeName, int dirac);

  
  /*
    WRITER FOR MAKING NEW H5 FILES - E.G. EVOLVED/MATCHED DATASETS
  */
  void H5Write(char *outH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin,
	       int pmax, std::string dTypeName, int dirac);

  void printVec(gsl_vector *g);

  void printMat(gsl_matrix *g);

  double computeDet(gsl_matrix * m);

  int matrixInv(gsl_matrix * M, std::map<int, gsl_matrix *> &mapInvs, int zi);

}
#endif
