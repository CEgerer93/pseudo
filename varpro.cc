/*
  Define classes/structs/methods needed for variable projection
*/
#include "varpro.h"

namespace VarPro
{
  void varPro::makeBasis(gsl_vector *d, double a, double b)
  {
    int s;
    for ( s = 0; s < numLT; s++ )
      gsl_vector_set(basis, s, pitd_texp_sigma_n(s, 85, a, b, *NU*, *Z*));
    for ( s = numLT; s < numFunc; s++ )
      gsl_vector_set(basis, s, pitd_texp_sigma_n_treelevel(s-numLT, 85, a, b, *NU*));
  }

}
