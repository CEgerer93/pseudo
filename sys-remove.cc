/*
  Utility to remove from pseudo-ITD systematic effects determined with jacobi polynomial expansion fit
*/
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>

#include "pitd_util.h"
#include "kernel.h"

using namespace PITD;

// Hard set of zmin/zmax/pmin/pmax
const int zmin(1), zmax(12), pmin(1), pmax(6);
int gauge_configs;




// Remove the specified systematic effects from original pseudo-ITD
reducedPITD correct(reducedPITD &r, int truncs[], std::vector<std::vector<double> > &c, int comp)
{
  // std::cout << truncs[0] << "," << truncs[1] << "," << truncs[2] << std::endl;


  // Initialize the reducedPITD object to hold corrected data
  reducedPITD mod = r;


  for ( auto z = mod.data.disps.begin(); z != mod.data.disps.end(); ++z )
    {
      for ( auto p = z->second.moms.begin(); p != z->second.moms.end(); ++p )
	{
	  for ( int g = 0; g < gauge_configs; ++g )
	    {
	      
	      if ( comp == 0 )
		{
		  for ( int n = 0; n < truncs[0]; ++n )
		    {
		      // 0th & 1st entries of 'c' are alpha & beta; sigma_n sum starts at n=1 for real comp!
		      double loc = (1.0/z->first)*c[2+n][g]*\
			pitd_texp_sigma_n_treelevel(1+n, 75, c[0][g], c[1][g], p->second.IT);

		      // Make a complex number to then subtract from orig mat.
		      std::complex<double> plx(loc,0.0);
		      p->second.mat[g] -= plx;
		    }
		}
	      if ( comp == 1 )
		{
		  for ( int n = 0; n < truncs[0]; ++n )
		    {
		      // 0th & 1st entires of 'c' are alpha & beta; eta_n sum starts at n=0 for imag comp!
		      double loc = (1.0/z->first)*c[2+n][g]*\
			pitd_texp_eta_n_treelevel(n, 75, c[0][g], c[1][g], p->second.IT);

		      // Make a complex number to then subtract from orig mat.
		      std::complex<double> plx(0.0,loc);
		      p->second.mat[g] -= plx;
		    }
		}
	    } // g
	} // p
    } // z

  return mod;
}


int main( int argc, char *argv[] )
{
  if ( argc != 7 )
    {
      std::cout << "Usage: $0 <h5 file> <configs> <matelemType> <comp to correct> <corrections truncations> <alpha/beta + correction coefficients> " << std::endl;
      exit(1);
    }

  int comp;
  std::string matelemType, coeffsFile;

  std::stringstream ss;
  ss.clear(); ss.str(std::string());
  ss << argv[2]; ss >> gauge_configs; ss.clear(); ss.str(std::string());
  ss << argv[3]; ss >> matelemType;   ss.clear(); ss.str(std::string());
  ss << argv[4]; ss >> comp;          ss.clear(); ss.str(std::string());
  // ss << argv[5]; ss >> orders;        ss.clear(); ss.str(std::string());
  ss << argv[6]; ss >> coeffsFile;        ss.clear(); ss.str(std::string());
  

  // Parse the truncation orders
  ss << argv[5];
  int orders[3];
  std::string line;
  int cnt = 0;
  // Now fetch the truncation orders
  while(std::getline(ss, line, '.'))
    {
      orders[cnt] = std::stoi(line);
      cnt++;
    }
  

  // Total number of corrections
  int tot = 0;
  for ( int i = 0; i < sizeof(orders)/sizeof(orders[0]); i++ )
    tot += orders[i];


  // Now that we have the truncation orders and gauge configs, initialize vectors to hold coefficients
  std::vector<std::vector<double> > coeffs(2+tot);


  // Open the file with correction coefficients - ALPHA & BETA ARE FIRST TWO ENTRIES ALWAYS!
  std::ifstream in; in.open(coeffsFile);
  if ( in.is_open() )
    {
      while ( ! in.eof() )
	{

	  // Read line by line
	  getline(in, line, '\n');
	  // std::cout << line << std::endl;

	  // Put the read line into a stringstream
	  std::stringstream iss(line);
	  std::string chunk;
	  
	  // For this line, split according to spaces
	  int dum(0);
	  while ( std::getline(iss, chunk, ' ') )
	    {
	      coeffs[dum].push_back(std::stod(chunk)); // parse coeff as double and push back coeff vector
	      dum++;
	    }

	}
    }



  // std::cout << std::setprecision(10) << coeffs[0] << std::endl;



  
  // The original pseudo-ITD w/ any and all systematic effects
  reducedPITD rawPseudo(gauge_configs);
  H5Read(argv[1], &rawPseudo, gauge_configs, zmin, zmax, pmin, pmax, "pitd");



  // Determine the corrected reduced pseudo-ITD from original
#warning "Have only implemented corrections originating from discretization effects"
  reducedPITD correctedPseudo = correct(rawPseudo, orders, coeffs, comp);
  
   

  /*
    Write out the modified reduced pseudo-ITD
  */
  std::string output;
  ss.clear(); ss.str(std::string());
  ss << argv[1]; ss >> output;


  output.replace(output.end()-2,output.end(),"disc-corrected.h5");
  char * out_h5 = &output[0];
  H5Write(out_h5, &correctedPseudo, gauge_configs, zmin, zmax, pmin, pmax, "pitd");  

  return 0;
}
