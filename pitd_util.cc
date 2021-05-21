/*
  Define classes/structs/methods needed to handle the reduced pseudo-ioffe-time distributions
*/
#include "pitd_util.h"

#include<gsl/gsl_permutation.h> // permutation header for matrix inversions
#include<gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h> // linear algebra

namespace PITD
{
  // Operator to allow direct multiplication of long double and std::complex<double>
  std::complex<double> operator*(long double ld, std::complex<double> c)
  {
    c.real()*ld; c.imag()*ld;
    return c;
  }

  // Easy printing of rpitdFitParams_t
  std::ostream& operator<<(std::ostream& os, const rpitdFitParams_t& p)
  {
    os << p.a << " " << p.b << " " << p.c;
    return os;
  }


  // Generic gsl_matrix viewer
  void printMat(gsl_matrix *g)
  {
    std::cout << "{";
    for ( size_t i = 0; i < g->size1; i++ ) {
      std::cout << "{";
      for ( size_t j = 0; j < g->size1; j++ ) {
	std::cout << gsl_matrix_get(g,i,j) << ",";
      }
      std::cout << "},\n";
    }    
  }


  /*
    PERFORM INVERSION OF PASSED MATRIX - RETURN # OF SVs REMOVED
  */
  int matrixInv(gsl_matrix * M, std::map<int, gsl_matrix *> &mapInvs, int zi) 
  // int matrixInv(gsl_matrix * M, gsl_matrix* MInv, size_t dataDim)
  {
    size_t dataDim = M->size1;
    gsl_matrix * toInvert = gsl_matrix_alloc(dataDim,dataDim);
    gsl_matrix_memcpy(toInvert,M); // make a copy of M, as toInvert is modified below
    gsl_matrix * MInv = gsl_matrix_alloc(dataDim,dataDim);
    gsl_matrix * V = gsl_matrix_alloc(dataDim,dataDim);
    gsl_vector *S = gsl_vector_alloc(dataDim);
    gsl_vector *work = gsl_vector_alloc(dataDim);


    // int sig;
    // gsl_permutation *perm = gsl_permutation_alloc(dataDim);
    // int luRet = gsl_linalg_LU_decomp(M, perm, &sig);
   
    // int luInvert = gsl_linalg_LU_invert(M, perm, MInv);
    // return 0;



    /*
      PERFORM THE SINGULAR VALUE DECOMPOSITION OF DATA COVARIANCE MATRIX (A)
      
      A = USV^T
          ---> A is an MxN matrix
    	  ---> S is the singular value matrix (diagonal w/ singular values along diag - descending)
    	  ---> V is returned in an untransposed form
    */
    gsl_linalg_SV_decomp(toInvert,V,S,work); // SVD decomp;  'toInvert' replaced w/ U on return

    // Define an svd cut
    double svdCut = 1e-16;
    // Initialize the inverse of the S diag
    gsl_vector *pseudoSInv = gsl_vector_alloc(dataDim);
    gsl_vector_set_all(pseudoSInv,0.0);

    // Vector of singular values that are larger than specified cut
    std::vector<double> aboveCutVals;
    
    std::cout << "The singular values above SVD Cut = " << svdCut << " are..." << std::endl;
    for ( int s = 0; s < dataDim; s++ )
      {
    	double dum = gsl_vector_get(S,s);
    	if ( dum >= svdCut ) { aboveCutVals.push_back(dum); }
      }
    
    // Assign the inverse of aboveCutVals to the pseudoSInv vector
    for ( std::vector<double>::iterator it = aboveCutVals.begin(); it != aboveCutVals.end(); ++it )
      {
    	gsl_vector_set(pseudoSInv,it-aboveCutVals.begin(),1.0/(*it));
      }

    // Promote this pseudoSInv vector to a matrix, where entries are placed along diagonal
    gsl_matrix * pseudoSInvMat = gsl_matrix_alloc(dataDim,dataDim);
    gsl_matrix_set_zero(pseudoSInvMat);
    for ( int m = 0; m < dataDim; m++ )
      {
    	gsl_matrix_set(pseudoSInvMat,m,m,gsl_vector_get(pseudoSInv,m));
    	std::cout << gsl_vector_get(pseudoSInv,m) << std::endl;
      }

    /*
      With singular values that are zero, best we can do is construct a pseudo-inverse

      In general, the inverse we are after is
                               VS^(-1)U^T
		     with S the diagonal matrix of singular values
    */
    // In place construct the transpose of U ('toInvert' was modified in place to U above)
    gsl_matrix_transpose(toInvert);

    gsl_matrix * SinvUT = gsl_matrix_alloc(dataDim,dataDim); gsl_matrix_set_zero(SinvUT);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,pseudoSInvMat,toInvert,0.0,SinvUT);

    // Now make the inverse of 'toInvert'
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,V,SinvUT,0.0,MInv);



    gsl_matrix * id = gsl_matrix_alloc(dataDim,dataDim); gsl_matrix_set_zero(id);
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1.0,M,MInv,0.0,id);

    // std::cout << "Check ID" << std::endl;
    // for ( int i = 0; i < dataDim; i++ )
    //   {
    // 	for ( int j = 0; j < dataDim; j++ )
    // 	  {
    // 	    std::cout << gsl_matrix_get(id,i,j) << " ";
    // 	  }
    // 	std::cout << "\n";
    //   }
    // std::cout <<"$$$$$" << std::endl;

    // Insert the MInv into std::map<int, gsl_matrix *> mapInvs container
    // & Return the number of singular values removed
    mapInvs[zi] = MInv;
    return pseudoSInv->size - aboveCutVals.size();
  }


  /*
    Calculation data covariance for each zsep
  */
  void reducedPITD::calcCovPerZ()
  {
    for ( std::map<int, zvals>::iterator d = data.disps.begin(); d != data.disps.end(); ++d )
      {

	// Instantiate pointers to gsl_matrices
	gsl_matrix *cR = gsl_matrix_alloc(d->second.moms.size(),d->second.moms.size());
	gsl_matrix *cI = gsl_matrix_alloc(d->second.moms.size(),d->second.moms.size());


  	for ( std::map<std::string, momVals>::const_iterator mi = d->second.moms.begin();
  	      mi != d->second.moms.end(); mi++ )
  	  {

  	    for ( std::map<std::string, momVals>::const_iterator mj = d->second.moms.begin();
  		  mj != d->second.moms.end(); mj++ )
  	      {

  		double _r(0.0), _i(0.0);

		for ( int J = 0; J < gauge_configs; J++ )
		  {

		    _r += ( mi->second.mat[J].real() - mi->second.matAvg.real() )
		      *( mj->second.mat[J].real() - mj->second.matAvg.real() );
		    _i += ( mi->second.mat[J].imag() - mi->second.matAvg.imag() )
		      *( mj->second.mat[J].imag() - mj->second.matAvg.imag() );
		  
		  } // end J


		// Set entry and proceed
		gsl_matrix_set(cR,
			       std::distance<std::map<std::string, momVals>::const_iterator>
			       (d->second.moms.begin(),mi),
			       std::distance<std::map<std::string, momVals>::const_iterator>
			       (d->second.moms.begin(),mj),
			       (( gauge_configs - 1 )/(1.0*gauge_configs))*_r);
		
		gsl_matrix_set(cI,
			       std::distance<std::map<std::string, momVals>::const_iterator>
			       (d->second.moms.begin(),mi),
			       std::distance<std::map<std::string, momVals>::const_iterator>
			       (d->second.moms.begin(),mj),
			       (( gauge_configs - 1 )/(1.0*gauge_configs))*_i);

		// std::cout << "XXX = " << (( gauge_configs - 1 )/(1.0*gauge_configs))*_i << std::endl;

	      } // end mj
	  } // end mi

	// Associate a z with freshly computed data covariances for data w/ said z value
	std::pair<int, gsl_matrix *> zcovR(d->first,cR);
	std::pair<int, gsl_matrix *> zcovI(d->first,cI);

	// Insert these int-gsl_matrix pairs into the pitd struct
	data.covsR.insert( zcovR );
	data.covsI.insert( zcovI );

      } // d
  }


  // Calculate the inverse of covariance matrices for each zsep
  //    --> covsR, covsI gsl_matrices replaced in place with inverse
  void reducedPITD::calcInvCovPerZ()
  {
    for ( auto it = data.covsR.begin(); it != data.covsR.end(); ++it )
      {
	// Make an int-int map, mapping a zsep to the singular values removed for this data.covsR inversion
	std::pair<int, int> zs( it->first, matrixInv(it->second, data.covsRInv, it->first) );
	data.svsR.insert( zs ); // pack the SVs removed
      }
    for ( auto it = data.covsI.begin(); it != data.covsI.end(); ++it )
      {
	// Make an int-int map, mapping a zsep to the singular values removed for this data.covsI inversion
	std::pair<int, int> zs( it->first, matrixInv(it->second, data.covsIInv, it->first) );
    	data.svsI.insert( zs ); // pack the SVs removed
      }
  }


  /*
    Calculate the full data covariance
  */
  void reducedPITD::calcCov()
  {
    // Instantiate pointers to gsl matrices and set all entries to zero
    data.covR = gsl_matrix_calloc(data.disps.size()*data.disps.begin()->second.moms.size(),
				  data.disps.size()*data.disps.begin()->second.moms.size());
    data.covI = gsl_matrix_calloc(data.disps.size()*data.disps.begin()->second.moms.size(),
				  data.disps.size()*data.disps.begin()->second.moms.size());
    for ( std::map<int, zvals>::iterator di = data.disps.begin(); di != data.disps.end(); ++di )
      {
	for ( std::map<std::string, momVals>::const_iterator mi = di->second.moms.begin();
	      mi != di->second.moms.end(); mi++ )
	  {
	    // Get the Ith index
	    int I = std::distance<std::map<std::string, momVals>::const_iterator>
	      (di->second.moms.begin(), mi) + (di->first-zminCut)*di->second.moms.size();

	    for ( auto dj = data.disps.begin(); dj != data.disps.end(); ++dj )
	      {
		for ( auto mj = dj->second.moms.begin(); mj != dj->second.moms.end(); mj++ )
		  {
		    // Get the Jth index
		    int J = std::distance(dj->second.moms.begin(), mj) + (dj->first-zminCut)*dj->second.moms.size();

		    double _r(0.0), _i(0.0);

		    for ( int J = 0; J < gauge_configs; J++ )
		      {

			_r += ( mi->second.mat[J].real() - mi->second.matAvg.real() )*
			  ( mj->second.mat[J].real() - mj->second.matAvg.real() );
			_i += ( mi->second.mat[J].imag() - mi->second.matAvg.imag() )*
			  ( mj->second.mat[J].imag() - mj->second.matAvg.imag() );
		      
		      } // end J


		    // Set the covariance entry and proceed
#ifdef UNCORRELATED
		    if ( I == J )
		      {
			gsl_matrix_set(data.covR, I, J, (( gauge_configs - 1 )/(1.0*gauge_configs))*_r );
			gsl_matrix_set(data.covI, I, J, (( gauge_configs - 1 )/(1.0*gauge_configs))*_i );
		      }
#else
		    gsl_matrix_set(data.covR, I, J, (( gauge_configs - 1 )/(1.0*gauge_configs))*_r );
		    gsl_matrix_set(data.covI, I, J, (( gauge_configs - 1 )/(1.0*gauge_configs))*_i );
#endif
		  } // end mj
	      } // end dj
	  } // end mi
      } // end di
  }

  /*
    Calculate the inverse of full data covariance
  */
  void reducedPITD::calcInvCov()
  {
    int dumZi = -1;
    std::map<int, gsl_matrix *> dumInvsMap;

    // Get the inverse
    data.svsFullR = matrixInv(data.covR, dumInvsMap, dumZi);
    // Rip out the matrix inverse
    data.invCovR = dumInvsMap.begin()->second;

    // Get the inverse
    data.svsFullI = matrixInv(data.covI, dumInvsMap, dumZi);
    // Rip out the matrix inverse
    data.invCovI = dumInvsMap.begin()->second;
  }

  /*
    Cut on Z's and P's to exclude from fit
  */
  void reducedPITD::cutOnPZ(int minz, int maxz, int minp, int maxp)
  {
    for ( std::map<int, zvals>::iterator di = data.disps.begin(); di != data.disps.end(); ++di )
      {
        for ( std::map<std::string, momVals>::const_iterator mi = di->second.moms.begin();
              mi != di->second.moms.end(); mi++ )
          {
            // Get the Ith index
            int I = std::distance<std::map<std::string, momVals>::const_iterator>
              (di->second.moms.begin(), mi) + di->first*di->second.moms.size();

            for ( auto dj = data.disps.begin(); dj != data.disps.end(); ++dj )
              {
                for ( auto mj = dj->second.moms.begin(); mj != dj->second.moms.end(); mj++ )
                  {
                    // Get the Jth index
                    int J = std::distance(dj->second.moms.begin(), mj) + dj->first*dj->second.moms.size();


		    if ( di->first < minz || di->first > maxz || 
			 atoi(&mi->first[2]) < minp || atoi(&mi->first[2]) > maxp )
		      {
			gsl_matrix_set(data.covR, I, J, 0.0);
			gsl_matrix_set(data.covI, I, J, 0.0);
		      }
		    else if ( dj->first < minz || dj->first > maxz ||
			      atoi(&mj->first[2]) < minp || atoi(&mj->first[2]) > maxp )
		      {
			gsl_matrix_set(data.covR, I, J, 0.0);
			gsl_matrix_set(data.covI, I, J, 0.0);
		      }
		    else
		      continue;

		  } // mj
	      } // dj
	  } // mi
      } // di
  }


  /*
    View a covariance matrix
  */
  void reducedPITD::viewZCovMat(int z)
  {
    std::cout << "PRINTING ZCOV FOR ZSEP = " << z << " DATA...." << std::endl;
    // printMat(data.covsR[z]);
    printMat(data.covsI[z]);
  }

  void reducedPITD::viewZCovInvMat(int z)
  {
    std::cout << "PRINTING ZCOV INV FOR ZSEP = " << z << " DATA...." << std::endl;
    // printMat(data.covsRInv[z]);
    printMat(data.covsIInv[z]);
  }
  
  
  
  /*
    READER FOR PASSED H5 FILES
  */
  void H5Read(char *inH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin,
	      int pmax, std::string dTypeName)
  {
    /*
    OPEN THE H5 FILE CONTAINING ENSEM/JACK RESULTS OF pPDF
    */
    std::cout << "     READING H5 FILE = " << inH5 << std::endl;
    hid_t h5File = H5Fopen(inH5,H5F_ACC_RDONLY,H5P_DEFAULT);
    /*
    Some h5 handles for accessing data
    */
    hid_t space, h5Pitd;
    herr_t h5Status;
    // The name of data actually stored in h5 file
    const char * DATASET = &dTypeName[0];

    // Access the first group entry within root - i.e. the current
    hid_t h5Current = H5Gopen(h5File, "/b_b0xDA__J0_A1pP", H5P_DEFAULT);

    // Other group headings w/in h5 file
    const char * const momenta[] = {"pz0","pz1","pz2","pz3","pz4","pz5","pz6"};
    const char * const comp[] = {"Re", "Im"}; // {"1","2"};
    const char * const zsep[] = {"zsep0","zsep1","zsep2","zsep3","zsep4","zsep5","zsep6","zsep7","zsep8",
				 "zsep9","zsep10","zsep11","zsep12","zsep13","zsep14","zsep15","zsep16"};


    // Iterator through stored displacements
    for ( int z = zmin; z <= zmax; z++ )
      {
	zvals dumZ;

	hid_t h5Zsep = H5Gopen(h5Current, zsep[z], H5P_DEFAULT);

	// Iterate through each momentum stored
	for ( int m = pmin; m <= pmax; m++ )
	  {
	    hid_t h5Mom = H5Gopen(h5Zsep, momenta[m], H5P_DEFAULT);


	    // // Start by grabbing handle to ensemble group
	    // hid_t h5Ensem = H5Gopen(h5Mom, "ensemble", H5P_DEFAULT);
	    
	    // for ( int c = 0; c < ReIm; c++ )
	    //   {
	    //     // Get the component handle
	    //     hid_t h5Comp = H5Gopen(h5Ensem,comp[c], H5P_DEFAULT);
	    
	    //     for ( int z = zmin; z <= zmax; z++ )
	    //       {
	    // 	// Get the zsep handle
	    // 	hid_t h5Zsep = H5Gopen(h5Comp, zsep[z], H5P_DEFAULT);
	    
	    // 	double ensemPitd[3];
	    // 	// Grab the dataset handle
	    // 	h5Pitd  = H5Dopen(h5Zsep, DATASET, H5P_DEFAULT);
	    // 	// Grab the data!
	    // 	h5Status = H5Dread(h5Pitd, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, ensemPitd);
	    
	    
	    // 	if ( c == 0 )
	    // 	  {
	    // 	    dat->data.real.disps[z].ensem.IT[m-pmin]=ensemPitd[0];
	    // 	    ens->real.disps[z].ensem.avgM[m-pmin]=ensemPitd[1];
	    // 	    ens->real.disps[z].ensem.errM[m-pmin]=ensemPitd[2];
	    // 	  }
	    // 	if ( c == 1 )
	    // 	  {
	    // 	    ens->imag.disps[z].ensem.IT[m-pmin]=ensemPitd[0];
	    // 	    ens->imag.disps[z].ensem.avgM[m-pmin]=ensemPitd[1];
	    // 	    ens->imag.disps[z].ensem.errM[m-pmin]=ensemPitd[2];
	    // 	  }
	    //       } // end z
	    //     H5Gclose(h5Comp);
	    //   } // end c
	    // H5Gclose(h5Ensem); // Finished parsing the ensemble groups
	    
	    // Now grab a handle to the jack group
	    hid_t h5Jack = H5Gopen(h5Mom, "jack", H5P_DEFAULT);


	    // A container for this {p,z} data
	    momVals dumMomVal(gauge_configs);

	
	    // for ( int c = 1; c > -1; c-- )
	    for ( int c = 0; c < 2; c++ )
	      {
		// Get the component handle
		hid_t h5Comp = H5Gopen(h5Jack,comp[c], H5P_DEFAULT);
		

		std::cout << "/b_b0xDA__J0_A1pP/" << zsep[z] << "/" << momenta[m]
			  << "jack/" << comp[c] << "/" << dTypeName << std::endl;
		
		// // Get the zsep handle
		// hid_t h5Zsep = H5Gopen(h5Comp, zsep[z], H5P_DEFAULT);
		
		// Initialize a buffer to read jackknife dataset
		// double read_buf[gauge_configs*3];
		double read_buf[gauge_configs*2];
		/* double *read_buf = NULL; */
		/* read_buf = (double*) malloc(sizeof(double)*gauge_configs*3); */
		
		
		hid_t dset_id = H5Dopen(h5Comp, DATASET, H5P_DEFAULT);
		herr_t status = H5Dread(dset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, read_buf);
		
		
		
		// // A container for this {p,z} data
		// momVals dumMomVal(gauge_configs);
		dumMomVal.IT = read_buf[0];    // Same Ioffe-time
		double _avgR(0.0), _avgI(0.0); // Talley the averages
		
		// Fill the dumMomVal Ioffe-time data for each jackknife sample
		for ( int J = 0; J < gauge_configs; J++ )
		  {
		    if ( c == 0 )
		      {
			dumMomVal.mat[J].real(read_buf[1+2*J]);
			_avgR += read_buf[1+2*J];
		      }
		    if ( c == 1 )
		      {
			dumMomVal.mat[J].imag(read_buf[1+2*J]);
			_avgI += read_buf[1+2*J];
		      }
		  }
		// Compute the averge Mat for the {p, z}
		if ( c == 0 )
		  dumMomVal.matAvg.real( _avgR / gauge_configs );
		if ( c == 1 )
		  dumMomVal.matAvg.imag( _avgI / gauge_configs );


		H5Gclose(h5Comp);
	      } // end c jack

		// Associate a std::string with this collection of IT data
		std::pair<std::string, momVals> amom ( momenta[m], dumMomVal );
		
		dumZ.moms.insert(amom);
		
		// // Associate an int w/ zvals object
		// std::pair<int, zvals> az ( z, dumZ );
		// // Insert this zvals map into the pitd.disps map
		// dat->data.disps.insert( az );

		// std::cout << "XXX = " << _avgI / gauge_configs << std::endl;
		// std::cout << "XXX = " << dumZ.moms["pz1"].mat[0].real() << std::endl;

		
		// End of parsing jack data from read_buf


		
	      // 	H5Gclose(h5Comp);
	      // } // end c jack



	    H5Gclose(h5Jack);
	    H5Gclose(h5Mom);

	  } // end mom

	// Associate an int w/ zvals object
	std::pair<int, zvals> az ( z, dumZ );
	// Insert this zvals map into the pitd.disps map
	dat->data.disps.insert( az );
	
      } // end z
    H5Gclose(h5Current);
    H5Fclose(h5File);

    std::cout << "READ H5 FILE SUCCESS" << std::endl;
  } // end H5Read

  
  /*
    WRITER FOR MAKING NEW H5 FILES - E.G. EVOLVED/MATCHED DATASETS
  */
  void H5Write(char *outH5, reducedPITD *dat, int gauge_configs, int zmin, int zmax, int pmin,
	       int pmax, std::string dTypeName)
  {
    /*
      OPEN THE H5 FILE FOR WRITING
    */
    std::cout << "    WRITING H5 FILE = " << outH5 << std::endl;
    hid_t h5File = H5Fcreate(outH5,H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    /*
      Some h5 handles for writing data
    */
    hid_t space, h5Pitd;
    herr_t h5Status;
    // The name of data actually stored in h5 file
    const char * DATASET = &dTypeName[0];

    // Make the first group entry within root - i.e. the current
    hid_t h5Current = H5Gcreate(h5File, "/b_b0xDA__J0_A1pP", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // Other group headings w/in h5 file
    const char * const momenta[] = {"pz0","pz1","pz2","pz3","pz4","pz5","pz6"};
    const char * const comp[] = {"Re", "Im"}; // {"1","2"};
    const char * const zsep[] = {"zsep0","zsep1","zsep2","zsep3","zsep4","zsep5","zsep6","zsep7","zsep8",
  				 "zsep9","zsep10","zsep11","zsep12","zsep13","zsep14","zsep15","zsep16"};
    // const char * const zsep[] = {"0","1","2","3","4","5","6","7","8",
    // 				 "9","10","11","12","13","14","15","16"};


    // Iterator through displacements to store
    for ( int z = zmin; z <= zmax; z++ )
      {
	zvals dumZ;

	hid_t h5Zsep = H5Gcreate(h5Current, zsep[z], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	// Iterate through each momentum to store
	for ( int m = pmin; m <= pmax; m++ )
	  {

	    hid_t h5Mom = H5Gcreate(h5Zsep, momenta[m], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	    // if ( H5Aexists(h5Current, momenta[m] ) )
	    //   {
	    // 	std::cout << "Group exists, opening" << std::endl;
	    // 	h5Mom = H5Gopen1(h5Current, momenta[m]);
	    //   }
	    // else
	    //   {
	    // 	std::cout << "Group doesn't exist, creating ---- > " << momenta[m] << std::endl;
	    // 	h5Mom = H5Gcreate(h5Current, momenta[m], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	    //   }
	

	    // Now grab a handle to the jack group
	    hid_t h5Jack = H5Gcreate(h5Mom, "jack", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	    // if ( H5Aexists(h5Mom, "jack") )
	    //   {
	    // 	std::cout << "jack exists, opening" << std::endl;
	    // 	h5Jack = H5Gopen1(h5Mom, "jack");
	    //   }
	    // else
	    //   {
	    // 	std::cout << "jack doesn't exist, creating" << std::endl;
	    // 	h5Jack = H5Gcreate(h5Mom, "jack", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	    //   }



	    // This set of {p,z} data
	    momVals dumMomVal = dat->data.disps[z].moms[momenta[m]];
	    std::cout << "Got to dumMomVal" << std::endl;

	    for ( int c = 0; c != 2; c++ )
	    // for ( int c = 1; c > -1; c-- )
	      {
		// Get the component handle
		hid_t h5Comp = H5Gcreate(h5Jack,comp[c], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		// if ( H5Aexists(h5Jack, comp[c]) )
		//   {
		//     std::cout << "Comp exists, opening" << std::endl;
		//     h5Comp = H5Gopen1(h5Jack, comp[c]);
		//   }
		// else
		//   {
		//     std::cout << "Comp doesn't exist, creating" << std::endl;
		//     H5Gcreate(h5Jack,comp[c], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		//   }


  		// // Get the zsep handle
  		// hid_t h5Zsep;
		// if ( H5Aexists(h5Comp, zsep[z]) )
		//   {
		//     std::cout << "zsep exists, opening" << std::endl;
		//     h5Zsep = H5Gopen1(h5Comp, zsep[z]);
		//   }
		// else
		//   {
		//     std::cout << "zsep doesn't exist, creating" << std::endl;
		//     H5Gcreate(h5Comp, zsep[z], H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		//   }


  		// Prepare buffer for writing
  		double jackBuff[2*gauge_configs];
		// jackBuff[0] = dumMomVal.IT;
  		for ( int J = 0; J < gauge_configs; J++ )
  		  {
		    jackBuff[2*J] = dumMomVal.IT;
  		    if ( c == 0 )
		      jackBuff[2*J+1]=dumMomVal.mat[J].real();
  		    if ( c == 1 )
		      jackBuff[2*J+1]=dumMomVal.mat[J].imag();
  		  } // end J

  		// Grab the dataset handle
  		hsize_t dims[2] = {gauge_configs,2};
		std::cout << "BEFORE DATASPACE SET" << std::endl;
  		hid_t DATASPACE = H5Screate_simple(2,dims,NULL);
  		hid_t dset_id = H5Dcreate(h5Comp, DATASET, H5T_IEEE_F64LE, DATASPACE,
  					  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  		// Push data to file
  		herr_t status = H5Dwrite(dset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, jackBuff);


		H5Gclose(h5Comp);
	      } // end c jack

	    H5Gclose(h5Jack);
	    H5Gclose(h5Mom);
	  } // end mom

	H5Gclose(h5Zsep);
      } // end z
    H5Gclose(h5Current);
    H5Fclose(h5File);

    std::cout << "WRITE H5 FILE SUCCESS" << std::endl;
  } // end H5Write

}

