#include "sonne.hpp"

#include <armadillo>

class fmmnode{

    private:
    std::vector<sonne> bodies;
    std::vector<double> center;
    std::vector<fmmnode> children; 
    long size;
    
     

    public:

    arma::mat get_force_matrix(std::vector<sonne> &bodies, double theta, double G, double dt);