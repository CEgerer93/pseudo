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

#include "hdf5.h"

namespace PITD
{
  struct momVals
  {
    /* struct */
    /* { */
    /*   std::vector<std::complex<double> > mat;       // Matelem - for avg or each jk */
    /*   std::array<double,3>               polFitParams; // Polynomial fit in ioffe-time for reduced pITD data */

    /*   // Data covariances for this zsep */
    /*   gsl_matrix                         *zCovR, *zCovI; */
    /* } amom; */

    
    double                             IT;
    std::vector<std::complex<double> > mat;
    std::complex<double> matAvg;
    /* std::string                       tag = "pz";                 // a verbose label */
    /* momVals(int _g = 0, std::string _s = "") { amom.mat.resize(_g), tag += _s; }; // Parametrized */

  momVals(int _g = 0) : mat(_g) {};

  /* momVals(int _g, std::string _s) : amom.mat(_g), tag = "pz"+_s {}; */
  };

  
  /* // Try to implement a weak-ordering for the zvals.moms map */
  /* bool operator<(const std::string ml, const std::string mr) */
  /* { */
  /*   return std::atoi(ml.substr(2)) < std::atoi(ml.substr(2)); */
  /* } */
    

  struct zvals
  {
    std::map<std::string, momVals> moms;
    // Data covariances for this zsep
//    gsl_matrix *zCovR, *zCovI;
  };


  // PITD info for each of Real/Imag components
  struct pitd
  {
    std::map<int, zvals> disps; // data for each zsep
    std::map<int, gsl_matrix *> covsR, covsI; // covariances for each zsep
    std::map<int, gsl_matrix *> covsRInv, covsIInv; // inverse of covariances for each zsep
    std::map<int, int> svsR, svsI;

    /* std::map<int, momVals> disps;  // Match a zsep to momenta */
    gsl_matrix *invCovR, *invCovI; // Inverses of entire data covariances
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
    // Destructor
    virtual ~reducedPITD() {};


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

    // View a covariance matrix
    void viewZCovMat(int zsep);
    // View inverse of a covariance matrix
    void viewZCovInvMat(int zsep);
    
  private:
    int gauge_configs;
  };


  /* // Print real/imag (comp) ensemble average data for provided zsep */
  /* void reducedPITD::ensemPrintZ(int zf, int c) {} */

  /* // Print the real/imag (comp) polynomial fit parameters to ensemble average data for provided zsep */
  /* void reducedPITD::polyFitPrint(int zf, int comp) {} */

  
  /*
    READER FOR PASSED H5 FILES
  */
  void H5Read(char *inH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin, int pmax);

  
  /* /\* */
  /*   WRITER FOR MAKING NEW H5 FILES - E.G. EVOLVED/MATCHED DATASETS */
  /* *\/ */
  /* void H5Write(char *outH5, reducedPITD *ens, std::vector<reducedPITD> &jack, */
  /* 	       int cfgs, int pmin, int pmax, int zmin, int zmax, int ReIm) {} */

  void printMat(gsl_matrix *g);

}
#endif
