#include "sonne.hpp"

#include <armadillo>
#include <vector>


class fmm{

    private:
    std::vector<sonne> bodies;
    std::vector<double> center;
    std::vector<fmm> children; 
    long size;
    
     

    public:

    arma::Mat<double> get_force_matrix(std::vector<sonne> &bodies, double theta, double G, double dt);


    //void compute_forces(std::vector<sonne> &bodies, double theta, double G, double dt);


};