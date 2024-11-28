#include "fmm.hpp"
#include <armadillo>


class fmmnode{

    private:
    bool is_empty;
    bool is_leaf;
    
    std::vector<sonne> bodies;
    std::vector<double> center;
    std::vector<fmmnode> children;
    
    long size;
    
     
    public:

    fmmnode();
    fmmnode(std::vector<sonne> &bodies);

};

    //arma::mat get_force_matrix(std::vector<sonne> &bodies, double theta, double G, double dt);